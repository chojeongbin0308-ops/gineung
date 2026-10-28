"""사람이 쓴 시험 문장(CSV)을 학습 데이터와 같은 JSONL로 바꾼다. 개체 위치는 문장에서 찾아 넣는다.
CSV 칸: text,intent,shop,village,item,item_cat,writer   (text·intent 외에는 비워도 됨)
item_cat 이 비어 있으면 lexicon/items.csv 에서 물건 이름으로 찾아 채운다.
실행: python3 model/human_to_jsonl.py"""
import csv, json, os
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, 'data')
LEX = {r['item'].replace(' ', ''): r['cat'] for r in csv.DictReader(open(os.path.join(H, 'lexicon', 'items.csv'), encoding='utf-8'))}
CATS = set(json.load(open(os.path.join(H, 'schema.json'), encoding='utf-8'))['item_categories'])
rows, miss, nocat = [], 0, []
for r in csv.DictReader(open(os.path.join(D, 'human_test.csv'), encoding='utf-8-sig')):
    if not (r.get('text') or '').strip(): continue
    ents = []
    for col, lab in (('shop','SHOP'), ('village','VILLAGE'), ('item','ITEM')):
        v = (r.get(col) or '').strip()
        if not v: continue
        i = r['text'].find(v)
        if i < 0: miss += 1; continue
        ents.append({'start': i, 'end': i+len(v), 'label': lab})
    item = (r.get('item') or '').strip()
    cat = (r.get('item_cat') or '').strip() or (LEX.get(item.replace(' ', ''), '') if item else 'none')
    if cat not in CATS: nocat.append(r['text']); continue
    rows.append({'text': r['text'], 'intent': r['intent'].strip(), 'entities': sorted(ents, key=lambda e: e['start']), 'item_cat': cat, 'writer': r.get('writer','')})
with open(os.path.join(D, 'human_test.jsonl'), 'w', encoding='utf-8') as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + '\n')
print(len(rows), 'rows,', miss, 'entity values not found in text')
if nocat: print('물건 분류를 못 정한 문장 (item_cat 칸을 채워 주세요):', *nocat, sep='\n  ')
