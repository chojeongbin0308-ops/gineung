"""기능상속 AI 상담 — 질문 이해 모델 학습·평가·내보내기

하나의 한국어 인코더(기본 klue/roberta-small) 위에 머리 3개를 얹어 함께 학습한다.
  1) 의도 분류      (문장 → buy / find / taxi / …)
  2) 개체 태깅      (글자 구간 → 가게 SHOP / 마을 VILLAGE / 물건 ITEM, BIO)
  3) 물건 분류      (문장 → 생필품 / 신선식품 / 농약 …)

실행 예 (Colab GPU):
  python model/train_intent.py --model klue/roberta-small --epochs 5 --out runs/roberta-small
결과: runs/…/  best.pt · metrics.json · onnx/model_quantized.onnx · tokenizer.json · labels.json
"""
import argparse, json, os, random, time, math
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = json.load(open(os.path.join(HERE, 'schema.json'), encoding='utf-8'))
INTENTS = list(SCHEMA['intents'])
ENT = list(SCHEMA['entities'])
TAGS = ['O'] + [f'{p}-{e}' for e in ENT for p in ('B', 'I')]
CATS = list(SCHEMA['item_categories'])
I2 = {k: i for i, k in enumerate(INTENTS)}; T2 = {k: i for i, k in enumerate(TAGS)}; C2 = {k: i for i, k in enumerate(CATS)}

def read(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')] if os.path.exists(path) else []

class Enc(Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        for r in rows:
            e = tok(r['text'], truncation=True, max_length=max_len, return_offsets_mapping=True)
            tags, started = [], set()
            for (s, t) in e['offset_mapping']:
                if s == t: tags.append(-100); continue          # 특수 토큰
                lab = 'O'
                for j, ent in enumerate(r['entities']):
                    if s < ent['end'] and t > ent['start']:      # 토큰이 개체 구간과 겹침
                        lab = ('I-' if j in started else 'B-') + ent['label']; started.add(j); break
                tags.append(T2[lab])
            self.items.append((e['input_ids'], e['attention_mask'], tags, I2[r['intent']], C2.get(r.get('item_cat', 'none'), C2['none']), e['offset_mapping']))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch, pad_id):
    L = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), L), pad_id); m = torch.zeros((len(batch), L), dtype=torch.long); tg = torch.full((len(batch), L), -100)
    for i, (a, b, c, _, _, _) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a); m[i, :len(b)] = torch.tensor(b); tg[i, :len(c)] = torch.tensor(c)
    return ids, m, tg, torch.tensor([b[3] for b in batch]), torch.tensor([b[4] for b in batch])

class Joint(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.intent = nn.Linear(h, len(INTENTS)); self.tag = nn.Linear(h, len(TAGS)); self.cat = nn.Linear(h, len(CATS))
    def forward(self, input_ids, attention_mask):
        x = self.drop(self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        c = x[:, 0]
        return self.intent(c), self.tag(x), self.cat(c)

def spans_from_tags(tag_ids, offsets):
    out, cur = [], None
    for t, (s, e) in zip(tag_ids, offsets):
        if s == e: continue
        lab = TAGS[t]
        if lab.startswith('B-') or (lab.startswith('I-') and (cur is None or cur[2] != lab[2:])):
            if cur: out.append(tuple(cur))
            cur = [s, e, lab[2:]]
        elif lab.startswith('I-') and cur: cur[1] = e
        else:
            if cur: out.append(tuple(cur)); cur = None
    if cur: out.append(tuple(cur))
    return out

@torch.no_grad()
def evaluate(model, ds, rows, device, bs=64, pad_id=0):
    model.eval()
    dl = DataLoader(ds, batch_size=bs, collate_fn=lambda b: collate(b, pad_id))
    pi, pc, pt = [], [], []
    for ids, m, tg, yi, yc in dl:
        a, b, c = model(ids.to(device), m.to(device))
        pi += a.argmax(-1).tolist(); pc += c.argmax(-1).tolist(); pt += b.argmax(-1).tolist()
    n = len(rows); ok_i = ok_c = n_c = 0; tp = fp = fn = 0
    per = {k: [0, 0] for k in INTENTS}
    for k, r in enumerate(rows):
        gi = r['intent']; per[gi][1] += 1
        if INTENTS[pi[k]] == gi: ok_i += 1; per[gi][0] += 1
        if r.get('item_cat', 'none') != 'none':
            n_c += 1; ok_c += CATS[pc[k]] == r['item_cat']
        offs = ds.items[k][5]
        pred = set(spans_from_tags(pt[k][:len(offs)], offs))
        gold = set((e['start'], e['end'], e['label']) for e in r['entities'])
        tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {'n': n, 'intent_acc': round(ok_i / n, 4), 'item_cat_acc': round(ok_c / max(1, n_c), 4),
            'entity_f1': round(2 * prec * rec / max(1e-9, prec + rec), 4),
            'per_intent': {k: round(a / b, 3) for k, (a, b) in per.items() if b}}

def export_onnx(model, tok, out, max_len):
    os.makedirs(os.path.join(out, 'onnx'), exist_ok=True)
    model = model.eval().cpu()
    ex = tok('하리농약사 닫으면 고추약 어디서 사능가', return_tensors='pt')
    path = os.path.join(out, 'onnx', 'model.onnx')
    kw = dict(input_names=['input_ids', 'attention_mask'], output_names=['intent_logits', 'tag_logits', 'cat_logits'],
              dynamic_axes={'input_ids': {0: 'b', 1: 's'}, 'attention_mask': {0: 'b', 1: 's'}, 'intent_logits': {0: 'b'}, 'tag_logits': {0: 'b', 1: 's'}, 'cat_logits': {0: 'b'}},
              opset_version=14)
    try: torch.onnx.export(model, (ex['input_ids'], ex['attention_mask']), path, dynamo=False, **kw)
    except TypeError: torch.onnx.export(model, (ex['input_ids'], ex['attention_mask']), path, **kw)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    qpath = os.path.join(out, 'onnx', 'model_quantized.onnx')
    quantize_dynamic(path, qpath, weight_type=QuantType.QInt8)
    import onnxruntime as ort
    sess = ort.InferenceSession(qpath)
    o = sess.run(None, {'input_ids': ex['input_ids'].numpy(), 'attention_mask': ex['attention_mask'].numpy()})
    with torch.no_grad(): ref = model(ex['input_ids'], ex['attention_mask'])
    same = int(np.argmax(o[0]) == int(ref[0].argmax()))
    tok.save_pretrained(out)
    json.dump({'intents': INTENTS, 'tags': TAGS, 'cats': CATS, 'max_len': max_len}, open(os.path.join(out, 'labels.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return {'onnx_mb': round(os.path.getsize(path) / 1e6, 1), 'quantized_mb': round(os.path.getsize(qpath) / 1e6, 1), 'quantized_intent_matches_torch': bool(same)}

def predict_fn(model, tok, device):
    @torch.no_grad()
    def predict(text):
        model.eval()
        e = tok(text, return_offsets_mapping=True, return_tensors='pt')
        a, b, c = model(e['input_ids'].to(device), e['attention_mask'].to(device))
        offs = e['offset_mapping'][0].tolist()
        ents = [(text[s:t], l) for s, t, l in spans_from_tags(b[0].argmax(-1).tolist(), offs)]
        pi = torch.softmax(a[0], -1); pc = torch.softmax(c[0], -1)
        return {'intent': INTENTS[int(pi.argmax())], 'intent_p': round(float(pi.max()), 3), 'entities': ents,
                'item_cat': CATS[int(pc.argmax())], 'cat_p': round(float(pc.max()), 3)}
    return predict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='klue/roberta-small'); ap.add_argument('--data', default=os.path.join(HERE, 'data'))
    ap.add_argument('--out', default='runs/intent'); ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--bs', type=int, default=32); ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--max_len', type=int, default=64); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_export', action='store_true')
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device, '| model:', a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows = {k: read(os.path.join(a.data, f'{k}.jsonl')) for k in ['train', 'valid', 'test_seen', 'test_unseen', 'human_test']}
    ds = {k: Enc(v, tok, a.max_len) for k, v in rows.items() if v}
    print({k: len(v) for k, v in ds.items()})
    model = Joint(a.model).to(device)
    dl = DataLoader(ds['train'], batch_size=a.bs, shuffle=True, collate_fn=lambda b: collate(b, pad_id))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(dl) * a.epochs; sch = get_linear_schedule_with_warmup(opt, int(.06 * steps), steps)
    ce = nn.CrossEntropyLoss(); ce_t = nn.CrossEntropyLoss(ignore_index=-100)
    os.makedirs(a.out, exist_ok=True); best = -1; log = []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for ids, m, tg, yi, yc in dl:
            ids, m, tg, yi, yc = ids.to(device), m.to(device), tg.to(device), yi.to(device), yc.to(device)
            li, lt, lc = model(ids, m)
            loss = ce(li, yi) + ce_t(lt.reshape(-1, lt.size(-1)), tg.reshape(-1)) + ce(lc, yc)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            tot += loss.item()
        v = evaluate(model, ds['valid'], rows['valid'], device, pad_id=pad_id)
        score = v['intent_acc'] + v['entity_f1'] + v['item_cat_acc']
        log.append({'epoch': ep + 1, 'loss': round(tot / len(dl), 4), 'valid': v, 'sec': round(time.time() - t0, 1)})
        print(f"epoch {ep+1}  loss {tot/len(dl):.4f}  valid intent {v['intent_acc']}  ent F1 {v['entity_f1']}  cat {v['item_cat_acc']}  ({time.time()-t0:.0f}s)")
        if score > best: best = score; torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
    model.load_state_dict(torch.load(os.path.join(a.out, 'best.pt'), map_location=device))
    res = {'model': a.model, 'epochs': a.epochs, 'train_log': log, 'test': {}}
    for k in ['test_seen', 'test_unseen', 'human_test']:
        if k in ds: res['test'][k] = evaluate(model, ds[k], rows[k], device, pad_id=pad_id)
    base_p = os.path.join(a.data, 'baseline_scores.json')
    if os.path.exists(base_p): res['rule_baseline'] = json.load(open(base_p, encoding='utf-8'))
    if not a.no_export: res['export'] = export_onnx(model, tok, a.out, a.max_len); model.to(device)
    json.dump(res, open(os.path.join(a.out, 'metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n=== 시험 결과 (모델 vs 규칙) ===')
    for k, v in res['test'].items():
        b = res.get('rule_baseline', {}).get(k, {})
        print(f"{k:12s} n={v['n']:4d}  의도 {v['intent_acc']:.3f} (규칙 {b.get('intent_acc','-')})  물건분류 {v['item_cat_acc']:.3f} (규칙 {b.get('item_cat_acc','-')})  개체F1 {v['entity_f1']:.3f}")
    if 'export' in res: print('ONNX:', res['export'])
    pred = predict_fn(model, tok, device)
    for q in ['하리농약사 닫으믄 고추약은 어디서 사능가', '근처에 모짜렐라 파는 데 있소', '오는 길에 계란 좀 사다 주쇼', '구룡리는 뭐가 제일 구멍이여']:
        print(q, '→', pred(q))

if __name__ == '__main__':
    main()
