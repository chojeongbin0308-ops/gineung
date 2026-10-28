"""기능상속 AI 상담 — 질문 이해 모델 학습 데이터 생성기

문장 틀(템플릿) × 가게·마을·물건 × 말투 변형으로 학습 문장을 만든다.
일반화 성능을 정직하게 재기 위해 시험 세트는 두 종류로 나눈다.
  - test_seen   : 학습과 같은 틀·어휘에서 뽑았지만 학습에 쓰지 않은 문장
  - test_unseen : 학습에 한 번도 나오지 않은 문장 틀 + 처음 보는 물건·가게·마을
사람이 직접 쓴 문장(data/human_test.csv)이 있으면 그것이 최종 시험지다.

실행: python3 model/make_dataset.py  →  model/data/*.jsonl
"""
import json, random, re, os, collections

SEED = 20261028
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# ── 어휘 ─────────────────────────────────────────────────────────────
# (표면형, 분류, 'train' | 'test')  test = 학습에 절대 나오지 않는 물건
ITEMS = [
 # 생필품
 ('세제','daily','train'),('가루세제','daily','train'),('섬유유연제','daily','test'),('휴지','daily','train'),('두루마리 휴지','daily','train'),
 ('키친타월','daily','test'),('라면','daily','train'),('컵라면','daily','train'),('간장','daily','train'),('된장','daily','train'),('고추장','daily','train'),
 ('식용유','daily','train'),('설탕','daily','train'),('소금','daily','train'),('참기름','daily','train'),('밀가루','daily','train'),('밀가리','daily','train'),
 ('비누','daily','train'),('샴푸','daily','train'),('치약','daily','train'),('칫솔','daily','train'),('고무장갑','daily','test'),('쓰레기봉투','daily','train'),
 ('통조림','daily','train'),('참치캔','daily','test'),('커피믹스','daily','train'),('과자','daily','train'),('국수','daily','test'),
 # 신선식품
 ('두부','fresh','train'),('계란','fresh','train'),('달걀','fresh','train'),('우유','fresh','train'),('치즈','fresh','train'),('요구르트','fresh','test'),
 ('콩나물','fresh','train'),('대파','fresh','train'),('양파','fresh','train'),('감자','fresh','train'),('사과','fresh','train'),('배추','fresh','train'),
 ('무','fresh','train'),('생선','fresh','train'),('고등어','fresh','test'),('돼지고기','fresh','train'),('소고기','fresh','train'),('닭고기','fresh','train'),
 ('어묵','fresh','train'),('부추','fresh','train'),('정구지','fresh','test'),('버섯','fresh','test'),('바나나','fresh','train'),('떡','fresh','train'),
 ('식빵','fresh','test'),('모짜렐라','fresh','test'),
 # 무거운 물건
 ('쌀','heavy','train'),('쌀 20키로','heavy','train'),('생수','heavy','train'),('생수 한 박스','heavy','train'),('부탄가스','heavy','train'),
 ('연탄','heavy','train'),('등유','heavy','test'),('밀가루 한 포대','heavy','train'),
 # 급한 물건
 ('건전지','urgent','train'),('전구','urgent','train'),('형광등','urgent','test'),('음료수','urgent','train'),('콜라','urgent','train'),('사이다','urgent','train'),
 ('담배','urgent','train'),('소주','urgent','train'),('막걸리','urgent','test'),('라이터','urgent','train'),
 # 농약·비료
 ('농약','agri','train'),('고추약','agri','train'),('살충제','agri','train'),('살균제','agri','train'),('제초제','agri','train'),('풀약','agri','test'),
 ('비료','agri','train'),('복합비료','agri','train'),('요소비료','agri','test'),('퇴비','agri','train'),('상토','agri','train'),
 # 종자·농자재
 ('종자','seed','train'),('씨앗','seed','train'),('볍씨','seed','train'),('고추 모종','seed','train'),('모종','seed','train'),('비닐','seed','train'),
 ('멀칭 비닐','seed','test'),('호미','seed','train'),('장화','seed','train'),('고무호스','seed','train'),('차광망','seed','test'),
 # 병해충 상담
 ('병해충','advice','train'),('탄저병','advice','train'),('진딧물','advice','test'),('벌레','advice','train'),('고추 병','advice','train'),('잎마름병','advice','train'),
 # 택배
 ('택배','pickup','train'),('소포','pickup','train'),('등기','pickup','test'),('주문한 물건','pickup','train'),
]
SHOPS = [('상리상회','train'),('덕곡슈퍼','train'),('구룡상회','train'),('하리농약사','train'),('새터상회','train'),('동네 가게','train'),('마을 가게','train'),
         ('우리 동네 슈퍼','train'),('점빵','train'),('구판장','train'),('할매네 가게','train'),('농약방','train'),('농약사','train'),
         ('솔메슈퍼','test'),('아랫마을 점방','test'),('김씨네 가게','test'),('새터농약방','test')]
VILLAGES = [('상리','train'),('하리','train'),('덕곡리','train'),('구룡리','train'),('새터','train'),('솔메리','train'),('우리 마을','train'),
            ('윗마을','train'),('아랫마을','train'),('읍내','train'),('신평리','test'),('월곡리','test'),('앞마을','test'),('봉동','test')]

# 의도별로 그럴듯한 물건 분류
INTENT_CATS = {
 'buy': ['daily','fresh','heavy','urgent','agri','seed'], 'find': ['daily','fresh','heavy','urgent','agri','seed'],
 'taxi': ['daily','fresh','heavy','urgent'], 'license': ['agri'], 'supply': ['agri','seed','daily','fresh'],
}

# ── 문장 틀 ──────────────────────────────────────────────────────────
# {S}=가게 {V}=마을 {I}=물건.  '#'으로 시작하는 틀은 시험 전용(학습에 안 씀)
T = {
'buy': [
 '{S} 문 닫으면 {I}는 어디서 사?', '{S} 없어지면 {I} 어떻게 사요', '{V}에서 {I} 이제 어디서 사야 돼?', '{S} 닫은 뒤에 {I} 살 데 있어?',
 '{I} 사려면 어떻게 해야 해', '{I} 어디서 사요?', '{S}가 없어지면 {I}는 누가 팔아?', '{V} 사는데 {I} 살 곳이 없어',
 '앞으로 {I} 어디서 구해야 돼요', '{S} 폐업하면 {I} 사러 어디 가?', '{I} 살라면 어디로 가야 되나', '{S} 그만두면 {I}는 어쩌나',
 '#{S} 장사 접으면 {I}는 뭐 어디 가서 사', '#{I} 구할 데가 이제 어디여', '#{V} 사람들은 {I} 어디서 사게 되나요',
],
'find': [
 '{I} 찾아줘', '근처에 {I} 파는 데 있어?', '{I} 파는 가게 알려줘', '주변에 {I} 살 수 있는 곳', '가까운 데 {I} 있나', '{V}에 {I} 파는 곳 있어?',
 '여기서 제일 가까운 {I} 파는 데', '{I} 어디 파는지 찾아 줘요', '내 근처 {I} 가게', '1키로 안에 {I} 파는 곳',
 '#{I} 파는 집 가까이 있능가', '#지금 있는 데서 {I} 살 만한 가게 좀', '#{I} 근방에 파는 데 없나',
],
'taxi': [
 '{I} 좀 사다 줘', '{I} 사다 줄 사람 있어?', '읍내 가는 사람 {I} 좀 부탁해요', '{I} 배달 부탁할게', '{I} 하나만 대신 사다 주세요',
 '{V}로 {I} 좀 갖다 줄 수 있어?', '돌아오는 길에 {I} 사다 줄 수 있나', '행복택시로 {I} 부탁해', '{I} 심부름 해 줄 사람',
 '#{I} 좀 대신 사 오면 수고비 드릴게', '#나가는 김에 {I} 하나 사다 주쇼', '#{I} 사서 {V}까지 좀 가져다줘',
],
'loss': [
 '{S} 문 닫으면 뭐가 문제야?', '{S} 없어지면 어떤 기능이 사라져', '{S} 폐업하면 {V}에 뭐가 없어지나', '{S}가 하던 역할이 뭐야',
 '{S} 닫으면 주민들 뭐가 불편해져?', '{V} 가게 없어지면 뭐가 사라져요', '{S} 그만두면 어떤 생활기능이 없어지나요', '{S} 문 닫으면 생기는 문제',
 '#{S}마저 문 닫으면 우리는 뭐가 힘들어지나', '#{S} 없어지면 동네에서 뭘 못 하게 돼',
],
'plan': [
 '{S} 상속안 추천해 줘', '{S} 대안 알려줘', '{S} 문 닫으면 어떻게 이어가?', '{S} 기능 누가 이어받아?', '{V} 상속안 보여 줘',
 '{S} 폐업 대책 뭐 있어', '{S} 대신할 방법 추천', '{S} 없어지면 어떻게 해결해', '{V} 생활기능 어떻게 유지해?', '{S} 상속 방안 짜 줘',
 '#{S} 빈자리 뭘로 메우면 좋을까', '#{S} 닫아도 안 불편하게 하려면',
],
'gap': [
 '{S}는 어디가 공백이야?', '{V} 공백 생기는 거 뭐야', '{S} 닫으면 못 잇는 기능', '{V}에서 부족한 게 뭐야', '{S} 공백 위험 알려줘',
 '{V} 기존 자원으로 안 되는 거', '{S} 없어지면 해결 안 되는 부분', '{V} 생활서비스 빈 곳',
 '#{S} 대신할 데가 없는 게 뭐냐', '#{V}는 뭐가 제일 구멍이야',
],
'resources': [
 '{V} 주변에 쓸 수 있는 자원 알려 줘', '{V} 근처 이동장터 있어?', '{V}에 배송 되는 데 어디야', '{V} 마을회관 있어?', '{V} 주변 생활자원',
 '{V}에 오는 이동장터 언제야', '{V} 복지관 서비스 있나', '{V} 근처 콜택시 돼?', '{V}에서 이용할 수 있는 거 뭐 있어',
 '#{V} 쪽에 들어오는 차나 배달 뭐 있소', '#{V}에서 쓸 만한 시설',
],
'supply': [
 '{S} 유통망 알려줘', '{S} 거래처 어디야', '{S} 공급처는 누가 이어받아?', '{S}는 {I} 어디서 받아 왔어?', '{S} 도매상 어디야',
 '{S} 물건 들여오던 데', '{S} 납품 받던 곳 이어받으려면', '{S} 유통망 누가 이어받으면 좋아', '{I} 공급처 알려줘', '{S} 거래하던 대리점',
 '#{S} 사장님이 {I} 떼 오던 데가 어디여', '#{S} 물건 받던 길 그대로 하려면',
],
'license': [
 '{I} 팔려면 무슨 자격 필요해?', '{S} 이어받으려면 어떤 자격이 필요해', '{I} 판매 등록 어떻게 해', '{I} 가게 하려면 허가 필요해?',
 '{S} 인수하려면 등록해야 돼?', '{I} 판매관리인 교육 받아야 해?', '{S} 이어서 하려면 인허가 뭐 필요해', '{I} 장사 시작하려면 뭐 해야 돼',
 '#{I} 장사하려면 신고해야 되나', '#{S} 내가 맡아서 하면 서류 뭐 필요해',
],
'process': [
 '기능상속은 어떻게 진행돼?', '기능상속 절차 알려줘', '기능상속이 뭐야', '어떻게 진행되는지 알려 줘', '{S} 기능상속 절차', '기능상속 하려면 어떻게 해',
 '진행 과정 설명해 줘', '기능상속 순서가 어떻게 돼', '{S} 상속 절차 어떻게 돼요',
 '#기능상속 처음부터 끝까지 어떻게 하는 거여', '#이거 신청하면 그다음엔 뭐 해',
],
'other': [
 '안녕', '안녕하세요', '뭐 할 수 있어?', '도움말', '고마워', '너는 누구야', '오늘 날씨 어때', '사용법 알려줘', '처음이야', '감사합니다',
 '#반갑소', '#이거 어떻게 쓰는 거요', '#심심하다',
],
}

# ── 말투 변형 ────────────────────────────────────────────────────────
ENDING_SWAP = [  # (문장 끝 패턴, 바꿀 후보들)  — 마지막 개체 뒤쪽만 바꾼다
 (r'사\?$', ['사?', '사요?', '사능가', '산당가', '사야 되능가', '사냐']),
 (r'줘$', ['줘', '줘요', '주세요', '주쇼', '주랑께', '줄래']),
 (r'알려 ?줘$', ['알려줘', '알려 주세요', '알려 주쇼', '갈쳐 줘', '좀 알려줘']),
 (r'있어\?$', ['있어?', '있어요?', '있능가', '있당가', '있나', '있소?']),
 (r'돼\?$', ['돼?', '돼요?', '되능가', '된당가', '되나']),
 (r'야\?$', ['야?', '예요?', '여?', '이당가']),
 (r'해$', ['해', '해요', '혀', '하쇼', '해 줘']),
]
def splice(text, spans, new_suffix_from, new_suffix):
    return text[:new_suffix_from] + new_suffix, spans

def vary(text, spans, rng):
    last = max([e for _,e,_ in spans], default=0)
    # 1) 문장 끝 말투
    for pat, cands in ENDING_SWAP:
        m = re.search(pat, text)
        if m and m.start() >= last and rng.random() < .7:
            text = text[:m.start()] + rng.choice(cands); break
    # 2) 물음표·마침표
    r = rng.random()
    if text.endswith('?') and r < .25: text = text[:-1]
    elif not text.endswith(('?','.')) and r < .15: text += rng.choice(['?', '.', '~', '..'])
    # 3) 앞말 붙이기 (모든 구간이 뒤로 밀린다)
    if rng.random() < .25:
        pre = rng.choice(['저기 ', '근데 ', '아이고 ', '혹시 ', '나 ', '할매가 ', '그 ', '좀 ', '우리 ', '이봐 '])
        text = pre + text; spans = [(s+len(pre), e+len(pre), l) for s,e,l in spans]
    # 4) 띄어쓰기 일부 없애기 (어르신 입력·음성인식 결과 흉내) — 글자 위치를 다시 계산
    if rng.random() < .3:
        keep, mp, j = [], {}, 0
        for i,ch in enumerate(text):
            mp[i] = j
            if ch == ' ' and rng.random() < .5: continue
            keep.append(ch); j += 1
        mp[len(text)] = j
        text = ''.join(keep); spans = [(mp[s], mp[e], l) for s,e,l in spans]
        spans = [(s,e,l) for s,e,l in spans if e > s]
    return text, spans

def josa_fix(text, spans):
    # 받침에 맞게 조사 고치기: 는/은, 가/이, 를/을 — 개체 바로 뒤 한 글자만 다룬다
    def has_final(ch):
        c = ord(ch) - 0xAC00
        return 0 <= c <= 11171 and c % 28 != 0
    pair = {'는':'은','은':'는','가':'이','이':'가','를':'을','을':'를'}
    out = list(text)
    for s, e, _ in spans:
        if e < len(out) and out[e] in pair and (e+1 == len(out) or out[e+1] in ' ?.,'):
            prev = out[e-1]
            if '가' <= prev <= '힣' and has_final(prev) != (out[e] in '은이을'): out[e] = pair[out[e]]
    return ''.join(out)

def render(tpl, fill):
    text, spans, i = '', [], 0
    for m in re.finditer(r'\{(S|V|I)\}', tpl):
        text += tpl[i:m.start()]
        val, lab = fill[m.group(1)]
        spans.append((len(text), len(text)+len(val), lab))
        text += val; i = m.end()
    text += tpl[i:]
    return text, spans

def build(split_mode, n_per_tpl, rng):
    rows = []
    want = 'test' if split_mode == 'unseen' else 'train'
    items = [x for x in ITEMS if x[2] == want]; shops = [x for x in SHOPS if x[1] == want]; vills = [x for x in VILLAGES if x[1] == want]
    for intent, tpls in T.items():
        for tpl in tpls:
            held = tpl.startswith('#')
            if (split_mode == 'unseen') != held: continue
            tpl = tpl.lstrip('#')
            for _ in range(n_per_tpl):
                cats = INTENT_CATS.get(intent)
                if intent in ('gap','loss','plan','resources','process','other') or '{I}' not in tpl:
                    pool = items
                elif intent == 'license':
                    pool = [x for x in items if x[1] in ('agri',)] or items
                else:
                    pool = [x for x in items if (cats is None or x[1] in cats)]
                it = rng.choice(pool)
                if '{I}' not in tpl and intent == 'buy': pass
                if 'advice' in [x[1] for x in [it]] and intent not in ('buy',): pass
                fill = {'S': (rng.choice(shops)[0], 'SHOP'), 'V': (rng.choice(vills)[0], 'VILLAGE'), 'I': (it[0], 'ITEM')}
                text, spans = render(tpl, fill)
                text, spans = vary(josa_fix(text, spans), spans, rng)
                cat = it[1] if '{I}' in tpl else 'none'
                rows.append({'text': text, 'intent': intent, 'entities': [{'start':s,'end':e,'label':l} for s,e,l in spans], 'item_cat': cat})
    # 병해충 상담: 구매 의도로 묻는 질문 (예: 진딧물 생기면 무슨 약 사?)
    adv = [x for x in items if x[1]=='advice']
    adv_t = ['{I} 생기면 무슨 약 사야 돼?', '{I} 약 어디서 사', '{I} 생겼는데 어떻게 해', '{I} 잡는 약 살 데'] if split_mode!='unseen' else ['{I} 들었는데 약 뭐 써야 되나']
    for tpl in adv_t:
        for _ in range(n_per_tpl):
            it = rng.choice(adv); text, spans = render(tpl, {'I':(it[0],'ITEM')}); text, spans = vary(josa_fix(text, spans), spans, rng)
            rows.append({'text':text,'intent':'buy','entities':[{'start':s,'end':e,'label':l} for s,e,l in spans],'item_cat':'advice'})
    return rows

def dedup(rows):
    seen, out = set(), []
    for r in rows:
        if r['text'] in seen: continue
        seen.add(r['text']); out.append(r)
    return out

def main():
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    seen_rows = dedup(build('seen', 70, rng)); rng.shuffle(seen_rows)
    n = len(seen_rows); n_val = n_test = int(n*.08)
    splits = {'train': seen_rows[n_val+n_test:], 'valid': seen_rows[:n_val], 'test_seen': seen_rows[n_val:n_val+n_test]}
    tr = set(r['text'] for r in splits['train'])
    splits['valid'] = [r for r in splits['valid'] if r['text'] not in tr]
    splits['test_seen'] = [r for r in splits['test_seen'] if r['text'] not in tr]
    unseen = dedup(build('unseen', 12, rng)); rng.shuffle(unseen)
    splits['test_unseen'] = [r for r in unseen if r['text'] not in tr]
    for k,v in splits.items():
        with open(os.path.join(OUT, f'{k}.jsonl'),'w',encoding='utf-8') as f:
            for r in v: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    stats = {k: {'n': len(v), 'intents': dict(collections.Counter(r['intent'] for r in v))} for k,v in splits.items()}
    with open(os.path.join(OUT,'stats.json'),'w',encoding='utf-8') as f: json.dump(stats, f, ensure_ascii=False, indent=1)
    for k,v in stats.items(): print(k, v['n'])
    for r in splits['train'][:8]: print(' ', r['text'], r['intent'], [(r['text'][e['start']:e['end']], e['label']) for e in r['entities']], r['item_cat'])
    print('unseen 예시:')
    for r in splits['test_unseen'][:6]: print(' ', r['text'], r['intent'], [(r['text'][e['start']:e['end']], e['label']) for e in r['entities']], r['item_cat'])

if __name__ == '__main__':
    main()
