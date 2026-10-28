# 질문 이해 모델 (AI 상담)

AI 상담 질문에서 **의도 · 개체(가게/마을/물건) · 물건 분류**를 함께 뽑는 한국어 모델입니다.
숫자 계산(적합도, 상속안, 거리)은 지금처럼 분석 엔진이 하고, 모델은 "무엇을 묻는지"만 알아듣습니다.

## 바로 학습하기 (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chojeongbin0308-ops/gineung/blob/main/model/train_intent_colab.ipynb)

런타임을 **T4 GPU**로 바꾸고 위에서부터 실행하면 20~30분 안에 학습·평가·내보내기가 끝납니다.

## 파일

| 파일 | 내용 |
|---|---|
| `schema.json` | 의도 11개, 개체 3종, 물건 분류 9종 정의 |
| `make_dataset.py` | 문장 틀 × 어휘 × 말투 변형(전라도 어미, 띄어쓰기 생략 등)으로 학습 데이터 생성 |
| `data/human_test.csv` | **사람이 쓴 시험 문장.** 팀원이 계속 추가해 주세요 (현재 30개는 예시) |
| `human_to_jsonl.py` | 위 CSV를 시험 형식으로 변환 |
| `rule_baseline.py` | 지금 사이트의 키워드 규칙 방식을 같은 시험지로 채점 (비교 기준) |
| `train_intent.py` | 학습 · 평가 · ONNX 내보내기·8비트 경량화 |
| `train_intent_colab.ipynb` | Colab용 실행 노트북 |

## 시험지 설계

- `test_seen`: 학습과 같은 문장 틀 (쉬움, 참고용)
- `test_unseen`: **학습에 없는 문장 틀 + 처음 보는 물건·가게·마을** (예: 모짜렐라, 정구지, 풀약, 봉동)
- `human_test`: 사람이 쓴 문장 — 최종 성능은 이것으로 판단합니다

## 현재 기준선 (규칙 방식)

| 시험지 | 의도 정확도 | 물건 분류 정확도 |
|---|---|---|
| test_seen | 0.577 | 0.481 |
| test_unseen | 0.122 | 0.124 |
| human_test (예시 30문장) | 0.433 | 0.688 |

규칙 방식에는 `find`(주변 가게 찾기), `taxi`(행복택시 부탁) 의도가 아예 없어서 점수가 더 낮게 나옵니다.

## 사람이 쓴 시험 문장 추가하는 법

`data/human_test.csv`에 한 줄씩 추가합니다. 칸: `text,intent,shop,village,item,item_cat,writer`
- `intent`: buy · find · taxi · loss · plan · gap · resources · supply · license · process · other
- `shop`·`village`·`item`: 문장 속에 **그대로 나온 글자**만 적고, 없으면 비워 둡니다
- `item_cat`: daily · fresh · heavy · urgent · agri · seed · advice · pickup · none
- 어르신 말투, 사투리, 오타, 띄어쓰기 없는 문장을 많이 넣을수록 좋습니다
