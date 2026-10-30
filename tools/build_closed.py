"""인허가이력 시트 → page.html의 CLOSED 데이터 (폐업 이력 조회용).

사용: python tools/build_closed.py <imsil_data_catalog.xlsx>
- 담배소매업·식품 영업 인허가 공개자료 중 폐업·말소 기록만 담는다 (영업 중 기록 제외).
- 개인 이름으로 된 상호(예: 2~4글자 이름)는 이름을 지우고 건수에만 넣는다.
- 주소는 읍면까지만 남긴다.
"""
import json, os, re, sys
import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, 'src/page.html')
BIZ = re.compile(r'(마트|슈퍼|수퍼|상회|상점|식당|가게|약국|편의|농협|축협|휴게|주유|철물|건강|식품|수산|정육|제과|다방|커피|카페|여관|모텔|회관|센터|클럽|PC|피씨|게임|당구|노래|주점|호프|치킨|통닭|대리점|안경|전자|[점사회당원관소장실집방]$)')
UPJONG = {'담배소매업': '담배', '식품자동판매기영업': '식품자판기', '유통전문판매업': '식품유통',
          '집단급식소 식품판매업': '급식 식품판매', '기타식품판매업': '식품판매'}


def is_person(name):
    return bool(re.fullmatch(r'[가-힣]{2,4}', name)) and not BIZ.search(name)


def main(path):
    ws = openpyxl.load_workbook(path, read_only=True)['인허가이력']
    rows = list(ws.iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(rows) if r and r[0] == '출처 ID')
    hdr = rows[hi]
    out = []
    for r in rows[hi + 1:]:
        if not r or not r[2] or r[0] == '출처 ID':
            continue
        d = dict(zip(hdr, r))
        st = str(d['원문 상태'] or '')
        if st.startswith('영업'):
            continue
        name = re.sub(r'\s+', ' ', str(d['상호'])).strip()
        m = re.search(r'임실군\s*(\S+?[읍면])\b', str(d['원문 주소'] or ''))
        area = m.group(1) if m else ''
        y = re.match(r'(\d{4})', str(d['폐업일 원문'] or ''))
        out.append([area, UPJONG.get(d['허가 업종'], d['허가 업종']), 'c' if st.startswith('폐업') else 'x',
                    int(y.group(1)) if y else 0, '' if is_person(name) else name])
    out.sort(key=lambda x: -x[3])
    js = 'const CLOSED = ' + json.dumps(out, ensure_ascii=False, separators=(',', ':')) + ';'
    src = open(PAGE, encoding='utf-8').read()
    a, b = '/*CLOSED-DATA*/', '/*END-CLOSED-DATA*/'
    i, j = src.index(a) + len(a), src.index(b)
    open(PAGE, 'w', encoding='utf-8').write(src[:i] + js + src[j:])
    print(len(out), 'rows,', sum(1 for x in out if not x[4]), 'names hidden,', len(js), 'bytes')


if __name__ == '__main__':
    main(sys.argv[1])
