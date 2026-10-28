"""기능상속 AI 서버 — ko-sroberta 의미 검색 (임실군 상가)

방문자 질문 문장(예: "세재가튼거 피료헌디 엇다 팔어")을 한국어 문장 임베딩 모델로 벡터화하고,
임실군 상가의 업종·상호 설명과 의미가 가까운 순으로 돌려준다. 거리 계산은 방문자 기기에서 한다
(위치정보를 서버로 받지 않기 위해).

보안 원칙
- 열린 기능은 /health(상태) 와 /search(검색) 두 개뿐. 파일·명령 접근 없음.
- 우리 사이트 주소에서 온 요청만 허용(CORS), 접속지별 분당 요청 수 제한, 질문 길이 제한.
- 질문 문장은 저장·기록하지 않는다. 처리 건수만 센다.
"""
import os, glob, time, hashlib, threading, collections, datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
MODELS_DIR = os.path.join(HERE, 'models')
BASE_MODEL = 'jhgan/ko-sroberta-multitask'


def pick_model():
    """models/current.txt 에 적힌 버전 폴더(models/v2 등)를 쓰고, 없으면 기본 ko-sroberta를 쓴다."""
    if os.getenv('GINEUNG_MODEL'):
        return os.getenv('GINEUNG_MODEL'), 'env', {}
    cur = os.path.join(MODELS_DIR, 'current.txt')
    if os.path.exists(cur):
        ver = open(cur, encoding='utf-8').read().strip()
        path = os.path.join(MODELS_DIR, ver)
        if ver and os.path.isdir(path):
            meta = {}
            mp = os.path.join(path, 'metrics.json')
            if os.path.exists(mp):
                try:
                    import json
                    meta = json.load(open(mp, encoding='utf-8'))
                except Exception:
                    meta = {}
            return path, ver, meta
        print(f'[주의] models/current.txt 의 "{ver}" 폴더가 없어 기본 모델을 씁니다.')
    return BASE_MODEL, 'v1-base', {}


MODEL_NAME, MODEL_VER, MODEL_META = pick_model()
REGION = os.getenv('GINEUNG_REGION', '임실군')
ORIGINS = ['https://chojeongbin0308-ops.github.io', 'http://localhost:8787', 'http://127.0.0.1:8787']
RATE_PER_MIN = int(os.getenv('GINEUNG_RATE', '30'))          # 접속지 하나당 분당
GLOBAL_PER_MIN = int(os.getenv('GINEUNG_GLOBAL_RATE', '300'))  # 서버 전체 분당 (PC 보호)
MAX_BODY = 2048                                               # 요청 크기 상한(바이트)
MAX_CHARS = 200
VERSION = '1.1'

COLS = {'name': '상호명', 'big': '상권업종대분류명', 'mid': '상권업종중분류명', 'small': '상권업종소분류명',
        'addr': '도로명주소', 'lat': '위도', 'lng': '경도', 'gu': '시군구명', 'dong': '행정동명', 'id': '상가업소번호'}


def read_csv(path):
    for enc in ('utf-8-sig', 'cp949', 'utf-8'):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f'CSV 인코딩을 읽을 수 없습니다: {path}')


def load_stores():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    if not files:
        raise SystemExit(f'[오류] {DATA_DIR} 폴더에 상가(상권)정보 CSV 파일을 넣어 주세요.')
    frames = []
    for f in files:
        df = read_csv(f)
        miss = [c for c in ('name', 'small', 'lat', 'lng', 'gu') if COLS[c] not in df.columns]
        if miss:
            print(f'[건너뜀] {os.path.basename(f)}: 열 없음 {[COLS[m] for m in miss]}')
            continue
        frames.append(df[df[COLS['gu']] == REGION])
    if not frames:
        raise SystemExit('[오류] 읽을 수 있는 상가 CSV가 없습니다.')
    df = pd.concat(frames, ignore_index=True)
    for k in ('big', 'mid', 'small', 'name', 'addr', 'dong', 'id'):
        if COLS[k] not in df.columns:
            df[COLS[k]] = ''
    df = df.fillna('')
    df['lat_f'] = pd.to_numeric(df[COLS['lat']], errors='coerce')
    df['lng_f'] = pd.to_numeric(df[COLS['lng']], errors='coerce')
    df = df.dropna(subset=['lat_f', 'lng_f']).drop_duplicates(subset=[COLS['name'], COLS['addr']]).reset_index(drop=True)
    df['desc'] = (df[COLS['big']] + ' ' + df[COLS['mid']] + ' ' + df[COLS['small']] + ' ' + df[COLS['name']]).str.strip()
    df['cat'] = (df[COLS['big']] + ' ' + df[COLS['mid']] + ' ' + df[COLS['small']]).str.strip()
    return df, files


def load_knowledge():
    """train/build_knowledge.py 가 만든 data/knowledge.json (상가 외 생활기능·시설, 근거, 상품권, 제도). 없으면 상가만."""
    p = os.path.join(DATA_DIR, 'knowledge.json')
    if not os.path.exists(p):
        return None, None
    import json
    return json.load(open(p, encoding='utf-8')), p


_norm = lambda s: __import__('re').sub(r'[\s()\[\]·.\-]|주식회사|\(주\)', '', str(s or '')).lower()


class Engine:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch
        df, files = load_stores()
        kn, kn_path = load_knowledge()
        vouchers = set(kn['vouchers']) if kn else set()
        cs = (kn or {}).get('commerce_source', {})
        # 상가 + 생활기능·시설을 하나의 목록으로
        items = []
        for _, r in df.iterrows():
            items.append({'name': r[COLS['name']], 'category': r[COLS['small']], 'kind': '상가', 'address': r[COLS['addr']], 'dong': r[COLS['dong']],
                          'lat': round(float(r['lat_f']), 6), 'lng': round(float(r['lng_f']), 6), 'approx': '', 'phone': '', 'hours': '', 'docs': '', 'functions': [],
                          'voucher': f"{r[COLS['dong']]}|{_norm(r[COLS['name']])}" in vouchers,
                          'evidence': f"{cs.get('name', '상가(상권)정보')} {cs.get('basis', '')} 등재 · {cs.get('caution', '현재 영업은 확인 필요')}",
                          '_cat': r['cat'], '_desc': r['desc']})
        cat_display = {r['cat']: r[COLS['small']] for _, r in df.iterrows()}
        self.programs = []
        if kn:
            label_text = {l['name']: l['text'] for l in kn['labels']}
            for t, n in ((v, k) for k, v in label_text.items()): cat_display[t] = n
            small2cat = {v: k for k, v in cat_display.items()}
            for r in kn['resources']:
                t = label_text.get(r['label']) or small2cat.get(r['label'])
                if not t: continue
                items.append({'name': r['name'], 'category': r['label'], 'kind': '시설·서비스', 'address': r['address'], 'dong': r['emd'], 'lat': r['lat'], 'lng': r['lng'],
                              'approx': r['approx'], 'phone': r['phone'], 'hours': r['hours'], 'docs': r['docs'], 'functions': r['functions'],
                              'voucher': f"{r['emd']}|{_norm(r['name'])}" in vouchers,
                              'evidence': ' · '.join(x for x in (f"{r['source']} ({r['basis']})" if r['basis'] else r['source'], r['caution']) if x), 'url': r.get('url', ''),
                              '_cat': t, '_desc': f"{t} {r['name']}"})
            self.programs = kn['programs']
        self.items = items
        self.cat_display = cat_display
        self.n_store = sum(1 for i in items if i['kind'] == '상가'); self.n_facility = len(items) - self.n_store
        mps = getattr(torch.backends, 'mps', None)
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if mps is not None and mps.is_available() else 'cpu')
        self.gpu = torch.cuda.get_device_name(0) if self.device == 'cuda' else ('Apple Silicon (MPS)' if self.device == 'mps' else '')
        print(f'[모델] {MODEL_VER} · {MODEL_NAME} 불러오는 중 ({self.device}{" · " + self.gpu if self.gpu else ""})')
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        self.cats = sorted(cat_display)
        srcs = files + ([kn_path] if kn_path else [])
        key = hashlib.sha1((MODEL_NAME + MODEL_VER + REGION + '|'.join(f'{f}:{os.path.getmtime(f)}' for f in srcs)).encode()).hexdigest()[:12]
        cache = os.path.join(DATA_DIR, f'cache_{key}.npz')
        if os.path.exists(cache):
            z = np.load(cache)
            self.store_emb, self.cat_emb = z['store'], z['cat']
            print('[색인] 저장된 색인을 불러왔습니다.')
        else:
            print(f'[색인] {REGION} 상가 {self.n_store}곳 + 생활기능·시설 {self.n_facility}곳, 검색 대상 {len(self.cats)}개를 색인하는 중…')
            self.store_emb = self.encode([i['_desc'] for i in items])
            self.cat_emb = self.encode(self.cats)
            np.savez_compressed(cache, store=self.store_emb, cat=self.cat_emb)
        self.cat_index = {c: i for i, c in enumerate(self.cats)}
        self.store_cat = np.array([self.cat_index[i['_cat']] for i in items])
        print(f'[준비 완료] 상가 {self.n_store}곳 · 생활기능·시설 {self.n_facility}곳 · 검색 대상 {len(self.cats)}개 · 제도 {len(self.programs)}개')

    def encode(self, texts):
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def search(self, text, k_cat=5, k_store=60):
        q = self.encode([text])[0]
        cs = self.cat_emb @ q
        ss = self.store_emb @ q
        # 점수 = 업종·생활기능 의미 유사도 60% + 개별 자원(이름 포함) 유사도 40% — 이름에만 끌려가는 추천을 줄임
        score = 0.6 * cs[self.store_cat] + 0.4 * ss
        top_c = np.argsort(-cs)[:k_cat]
        top_s = np.argsort(-score)[:k_store]
        names = [self.cat_display[self.cats[i]] for i in top_c[:2]]
        progs = [p for p in self.programs if any(n in p['labels'] for n in names)][:2]
        pub = lambda it: {k: v for k, v in it.items() if not k.startswith('_')}
        return {
            'categories': [{'name': self.cat_display[self.cats[i]], 'path': self.cats[i], 'score': round(float(cs[i]), 4)} for i in top_c],
            'stores': [{**pub(self.items[i]), 'score': round(float(score[i]), 4)} for i in top_s],
            'programs': [{'name': p['name'], 'items': p['items'][:4], 'url': p.get('url', '')} for p in progs],
        }


engine = None
started = time.time()
stats = {'total': 0, 'today': 0, 'day': datetime.date.today().isoformat(), 'ms': collections.deque(maxlen=200)}
hits = collections.defaultdict(collections.deque)
lock = threading.Lock()

app = FastAPI(title='gineung-ai', docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=['GET', 'POST'],
                   allow_headers=['Content-Type', 'ngrok-skip-browser-warning'], max_age=600)


@app.middleware('http')
async def guard(request: Request, call_next):
    # 열린 주소는 /health, /search 두 개뿐. 큰 요청은 본문을 읽기 전에 거절
    if request.url.path not in ('/health', '/search'):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail': 'not found'}, status_code=404)
    try:
        if int(request.headers.get('content-length') or 0) > MAX_BODY:
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail': 'too large'}, status_code=413)
    except ValueError:
        pass
    resp = await call_next(request)
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['Cache-Control'] = 'no-store'
    resp.headers['Referrer-Policy'] = 'no-referrer'
    return resp


@app.on_event('startup')
def _load():
    global engine
    engine = Engine()


def client_ip(req: Request):
    return req.headers.get('cf-connecting-ip') or (req.headers.get('x-forwarded-for') or '').split(',')[0].strip() or (req.client.host if req.client else '?')


allq = collections.deque()


def rate_ok(ip):
    now = time.time()
    with lock:
        while allq and now - allq[0] > 60:
            allq.popleft()
        if len(allq) >= GLOBAL_PER_MIN:
            return False
        q = hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_PER_MIN:
            return False
        q.append(now); allq.append(now)
        if len(hits) > 5000:
            hits.clear()
        return True


class Ask(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)


@app.get('/health')
def health():
    ms = list(stats['ms'])
    return {'ok': engine is not None, 'service': 'gineung-ai', 'version': VERSION, 'model': BASE_MODEL if MODEL_VER == 'v1-base' else os.path.basename(MODEL_NAME), 'model_version': MODEL_VER,
            'model_score': {k: MODEL_META[k] for k in ('top1', 'top3', 'dialect_top1', 'test_set') if k in MODEL_META},
            'device': engine.device if engine else '', 'gpu': engine.gpu if engine else '', 'region': REGION,
            'stores': engine.n_store if engine else 0, 'facilities': engine.n_facility if engine else 0, 'categories': len(engine.cats) if engine else 0,
            'uptime_s': int(time.time() - started), 'requests_total': stats['total'], 'requests_today': stats['today'],
            'avg_ms': round(sum(ms) / len(ms), 1) if ms else None}


@app.post('/search')
def search(body: Ask, req: Request):
    if engine is None:
        raise HTTPException(503, '준비 중')
    if not rate_ok(client_ip(req)):
        raise HTTPException(429, '요청이 너무 많습니다')
    t = time.time()
    out = engine.search(body.text.strip())
    dt = (time.time() - t) * 1000
    with lock:
        today = datetime.date.today().isoformat()
        if stats['day'] != today:
            stats['day'], stats['today'] = today, 0
        stats['total'] += 1; stats['today'] += 1; stats['ms'].append(dt)
    out['ms'] = round(dt, 1)
    return out
