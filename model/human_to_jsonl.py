"""사람이 쓴 시험 문장(CSV)을 학습 데이터와 같은 JSONL로 바꾼다. 개체 위치는 문장에서 찾아 넣는다.
CSV 칸: text,intent,shop,village,item,item_cat,writer   실행: python3 model/human_to_jsonl.py"""
import csv, json, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
rows, miss = [], 0
for r in csv.DictReader(open(os.path.join(D, 'human_test.csv'), encoding='utf-8')):
    ents = []
    for col, lab in (('shop','SHOP'), ('village','VILLAGE'), ('item','ITEM')):
        v = (r.get(col) or '').strip()
        if not v: continue
        i = r['text'].find(v)
        if i < 0: miss += 1; continue
        ents.append({'start': i, 'end': i+len(v), 'label': lab})
    rows.append({'text': r['text'], 'intent': r['intent'], 'entities': sorted(ents, key=lambda e: e['start']), 'item_cat': r['item_cat'] or 'none', 'writer': r.get('writer','')})
with open(os.path.join(D, 'human_test.jsonl'), 'w', encoding='utf-8') as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + '\n')
print(len(rows), 'rows,', miss, 'entity values not found in text')
