"""RAG 생성 단계: 학습한 검색기가 고른 자료만 LLM(Ollama)에 주고 답을 쓰게 한 뒤, 근거를 코드로 검증한다.

- LLM은 이 PC의 Ollama(http://127.0.0.1:11434)에서만 실행. 밖으로 나가는 요청 없음.
- 검증 규칙(하나라도 어기면 답을 버리고 사이트는 정해진 틀 답변만 보여 줌)
  1) 자료에 없는 가게·시설 이름을 말하면 안 됨
  2) 전화번호·시간(00:00)·금액(…원)은 자료에 그대로 있는 것만
  3) [번호] 인용은 자료 번호 안에서만, 가게를 말할 때는 인용 필수
  4) 350자 이내
"""
import os, re, json, time, threading, urllib.request

OLLAMA = os.getenv('GINEUNG_OLLAMA', 'http://127.0.0.1:11434')
PREFER = [m for m in [os.getenv('GINEUNG_LLM', '')] if m] + ['exaone3.5:7.8b', 'exaone3.5:2.4b', 'qwen2.5:7b', 'qwen2.5:14b', 'qwen2.5:3b', 'gemma2:9b', 'llama3.1:8b']
SYSTEM = (
    "너는 전북 임실군 주민을 돕는 '기능상속' 안내원이다. 반드시 아래 [자료]에 적힌 내용만 사용해 답한다.\n"
    "- 자료에 없는 가게 이름, 전화번호, 운영시간, 가격, 품목을 절대 지어내지 않는다.\n"
    "- 가게나 시설을 말할 때는 자료의 이름을 그대로 쓰고 바로 뒤에 [번호]를 붙인다. 번호가 작을수록 가깝다.\n"
    "- 가장 가까운 1곳을 먼저, 필요하면 1곳만 더 말한다. 상품권 가맹·운영시간·제공 기능처럼 자료에 있는 쓸모 있는 정보가 있으면 한 가지만 덧붙인다.\n"
    "- 전체 2문장 이내, 어르신이 알아듣기 쉬운 짧은 존댓말. 같은 말을 되풀이하지 않는다. 목록·표·굵은 글씨는 쓰지 않는다.\n"
    "- '가기 전에 전화로 확인해 보세요' 같은 확인 권유는 마지막에 한 번만, 짧게 쓴다.\n"
    "- 자료로 답할 수 없으면 '자료에서 찾지 못했어요'라고 말하고 행복택시(이웃 장보기 대행)를 권한다."
)


class LLM:
    def __init__(self):
        self.model, self.available, self.error = None, [], ''
        self.sem = threading.Semaphore(int(os.getenv('GINEUNG_LLM_PARALLEL', '1')))
        self.detect()

    def _req(self, path, body=None, timeout=5):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(OLLAMA + path, data=data, headers={'Content-Type': 'application/json'}, method='POST' if data else 'GET')
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return json.loads(f.read().decode('utf-8'))

    def detect(self):
        try:
            self.available = [m['name'] for m in self._req('/api/tags').get('models', [])]
        except Exception as e:
            self.available, self.model, self.error = [], None, f'Ollama에 연결할 수 없음 ({e.__class__.__name__})'
            return None
        for want in PREFER:
            hit = next((n for n in self.available if n == want or n.split(':')[0] == want.split(':')[0] and want.endswith(n.split(':')[-1])), None)
            if hit: self.model = hit; break
        else:
            self.model = self.available[0] if self.available else None
        self.error = '' if self.model else '설치된 LLM 모델이 없음 (setup_llm.bat 실행)'
        return self.model

    def warmup(self):
        if not self.model: return
        try: self._req('/api/chat', {'model': self.model, 'messages': [{'role': 'user', 'content': '안녕'}], 'stream': False, 'keep_alive': '60m', 'options': {'num_predict': 1}}, timeout=120)
        except Exception as e: self.error = f'모델 준비 실패 ({e.__class__.__name__})'

    def chat(self, messages, timeout=30):
        if not self.sem.acquire(timeout=3):
            raise TimeoutError('다른 답변을 만드는 중')
        try:
            out = self._req('/api/chat', {'model': self.model, 'messages': messages, 'stream': False, 'keep_alive': '60m',
                                          'options': {'temperature': 0.2, 'top_p': 0.9, 'num_predict': 160, 'num_ctx': 4096}}, timeout=timeout)
            return out.get('message', {}).get('content', '').strip()
        finally:
            self.sem.release()


def build_context(items, programs, cats):
    lines = ['[자료] (번호가 작을수록 질문한 분과 가까운 곳)']
    for n, it in enumerate(items, 1):
        parts = [it['name'], it.get('category', ''), it.get('address') or it.get('dong', '')]
        if it.get('voucher'): parts.append('임실사랑상품권 가맹')
        if it.get('hours'): parts.append('운영 ' + it['hours'])
        if it.get('functions'): parts.append('제공 ' + ', '.join(it['functions']))
        if it.get('docs'): parts.append('발급 ' + it['docs'])
        if it.get('phone'): parts.append('전화 ' + it['phone'])
        if it.get('evidence'): parts.append('근거 ' + it['evidence'])
        lines.append(f'[{n}] ' + ' | '.join(p for p in parts if p))
    for p in programs[:2]:
        lines.append(f"[제도] {p['name']}: " + ' / '.join(p['items'][:4]))
    if cats:
        lines.append('[검색기가 판단한 관련 업종] ' + ', '.join(c['name'] for c in cats[:3]))
    return '\n'.join(lines)


_norm = lambda s: re.sub(r'\s+', '', str(s or ''))


def verify(answer, items, ctx, all_names):
    why = []
    a = answer.replace('**', '').strip()
    if not a: return False, ['빈 답변'], a
    if len(a) > 260: why.append('너무 김')
    na, nctx = _norm(a), _norm(ctx)
    ctx_names = {_norm(i['name']) for i in items}
    foreign = [n for n in all_names if len(n) >= 3 and n in na and n not in ctx_names and not any(n in c for c in ctx_names)]
    if foreign: why.append('자료에 없는 이름: ' + ', '.join(foreign[:3]))
    # 데이터베이스에도 없는 '지어낸' 가게 이름: …상회·마트·약국처럼 생긴 말이 자료에 없으면 실패
    shaped = re.findall(r'([가-힣A-Za-z0-9]{1,12}(?:상회|상점|마트|슈퍼|약국|농약사|종묘사|식당|센터|의원|병원|한의원|치과|정육점|철물|미용실|세탁소|하나로마트|주유소|경로당))', a)
    made = [m for m in shaped if _norm(m) not in nctx and not any(_norm(m) in c or c in _norm(m) for c in ctx_names)]
    if made: why.append('자료에 없는 가게: ' + ', '.join(made[:3]))
    for pat, lbl in ((r'\d{2,4}-\d{3,4}-\d{4}|\d{3}-\d{4}', '전화번호'), (r'\d{1,2}:\d{2}', '시간'), (r'\d[\d,]*\s*원', '금액')):
        bad = [m for m in re.findall(pat, a) if _norm(m) not in nctx]
        if bad: why.append(f'자료에 없는 {lbl}: ' + ', '.join(bad[:3]))
    cites = [int(x) for x in re.findall(r'\[(\d+)\]', a)]
    if any(c < 1 or c > len(items) for c in cites): why.append('없는 자료 번호 인용')
    # 이름 바로 뒤의 [번호]가 그 이름의 자료 번호와 같은지
    for n, it in enumerate(items, 1):
        for m in re.finditer(re.escape(it['name']) + r'\s*(?:에서|은|는|이|가|을|를|에|도|의)?\s*\[(\d+)\]', a):
            if int(m.group(1)) != n: why.append(f"인용 번호 불일치: {it['name']}→[{m.group(1)}]"); break
    mentions = [n for n in ctx_names if len(n) >= 2 and n in na]
    if mentions and not cites: why.append('가게를 말했지만 인용 번호 없음')
    if not mentions and not cites and '찾지 못' not in a: why.append('자료를 쓰지 않음')
    return not why, why, a


def answer(llm, question, items, programs, cats, all_names):
    ctx = build_context(items, programs, cats)
    t = time.time()
    text = llm.chat([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'{ctx}\n\n[질문] {question}'}])
    ok, why, clean = verify(text, items, ctx, all_names)
    return {'answer': clean if ok else '', 'verified': ok, 'reasons': why, 'model': llm.model, 'ms': round((time.time() - t) * 1000), 'raw_len': len(text)}
