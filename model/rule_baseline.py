"""현재 사이트의 규칙 기반 상담(키워드 매칭)을 파이썬으로 옮겨 같은 시험지로 채점한다.
학습 모델과 비교할 기준선 점수를 만든다.  실행: python3 model/rule_baseline.py
"""
import json, re, os, collections
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# 사이트의 키워드 규칙을 schema v0.2 분류로 옮긴 것 (키워드 수준은 v0.1과 같게 유지 — 공정한 기준선)
ITEM_KW = [(r'농약|살충|살균|제초|비료|상토|퇴비', 'agri_chem'), (r'종자|씨앗|모종|비닐|농자재', 'agri_supply'),
           (r'병해충|병이|벌레|처방|어떤 약|무슨 약', 'advice'), (r'택배|소포|주문품', 'pickup'),
           (r'두부|계란|달걀|우유|치즈|신선', 'fresh'), (r'채소|야채|과일|콩나물', 'produce'), (r'고기|정육', 'meat'), (r'생선|수산', 'seafood'),
           (r'쌀|잡곡', 'grain'), (r'부탄|가스|연탄|등유', 'fuel'), (r'생수|음료|담배|술', 'drink'),
           (r'건전지|전구', 'hardware'), (r'휴지|세제|라면|조미료|생필품|통조림|생활용품', 'grocery'), (r'약국|상비약|파스', 'pharmacy')]
def item_of(q):
    for pat, cat in ITEM_KW:
        if re.search(pat, q): return cat
    return None
def intent_of(q):
    if re.search(r'(사려면|살 수|사고 싶|사야|사러|살 곳|살까|구매|구입|어디서 사|어떻게 사|사\?|사$|사요|받으려면|받아|받지|받을|받나|맡기|써야|어떻게 해)', q) and item_of(q) and not re.search(r'이어받|인수|팔려면|판매하려', q): return 'buy'
    if item_of(q) == 'advice' and not re.search(r'이어받|인수', q): return 'buy'
    if re.search(r'자격|등록|허가|면허|인허가|교육|판매관리인', q): return 'license'
    if re.search(r'기능상속(은|이|을|는)?\s*(뭐|무엇|어떻게|절차|과정|진행|하려면)|절차|진행 과정|어떻게 진행', q): return 'process'
    if re.search(r'유통|공급|도매|들여|거래처|납품|어디서 (사|받)|이어받|인수|물려', q): return 'supply'
    if re.search(r'공백|부족|위험|못 (잇|이어)|안 되|빈', q): return 'gap'
    if re.search(r'자원|주변|이동장터|배송|마을회관|복지|콜택시', q): return 'resources'
    if re.search(r'상속|대안|어떻게|추천|대체|이어|방법|해결|유지율', q): return 'plan'
    if re.search(r'사라|잃|문제|기능|폐업|문 닫|닫으면|없어지', q): return 'loss'
    return 'other'

def score(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    ok = sum(intent_of(r['text']) == r['intent'] for r in rows)
    with_item = [r for r in rows if r['item_cat'] != 'none']
    cat_ok = sum(item_of(r['text']) == r['item_cat'] for r in with_item)
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r['intent']][1] += 1; per[r['intent']][0] += intent_of(r['text']) == r['intent']
    return {'n': len(rows), 'intent_acc': round(ok/len(rows), 4), 'item_cat_acc': round(cat_ok/max(1,len(with_item)), 4),
            'per_intent': {k: round(a/b, 3) for k, (a, b) in sorted(per.items())}}

if __name__ == '__main__':
    out = {}
    for sp in ['test_seen', 'test_unseen', 'human_test']:
        p = os.path.join(D, f'{sp}.jsonl')
        if os.path.exists(p): out[sp] = score(p)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    json.dump(out, open(os.path.join(D, 'baseline_scores.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
