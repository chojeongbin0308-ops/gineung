# 기능상속 AI 서버 (전용 PC용)

ko-sroberta 문장 임베딩으로 임실군 상가를 **의미 검색**하는 서버입니다. 사이트(AI 상담·상태 페이지)가 이 서버를 부르고, 서버가 꺼져 있으면 사이트는 자동으로 기기 안 방식으로 답합니다.

## 맥북에서
터미널에서 `bash setup_server.sh` → `data` 폴더에 CSV → `bash start_server.sh` (켜져 있는 동안 맥이 잠들지 않음, 끄기 Ctrl+C). 애플 칩 GPU(MPS)를 자동으로 씁니다.

## 처음 한 번 (Windows 전용 PC)
1. Python 3.11 설치 (python.org, 설치 화면에서 **Add python.exe to PATH** 체크)
2. `setup_server.bat` 더블클릭 — 가상환경, PyTorch(CUDA), 서버 패키지, cloudflared 설치. 마지막 줄에 `GPU: True`가 나오면 성공
3. `data` 폴더에 `소상공인시장진흥공단_상가(상권)정보_전북_*.csv` 넣기 (GitHub에 올리지 않습니다)

## 켜기
- `start_server.bat` 더블클릭 → 처음엔 모델 다운로드·색인에 몇 분, 이후엔 수십 초
- 공개 주소가 나오면 자동으로 복사되고 관리자 페이지가 열립니다 → **서버 주소 저장**
- 창을 닫으면 서버가 꺼집니다. 오류로 멈추면 10초 뒤 자동으로 다시 켜집니다
- 재부팅 후 자동으로 켜지게 하려면 `register_autostart.bat` 한 번 실행

## 열린 기능
| 주소 | 내용 |
|---|---|
| `GET /health` | 상태(모델, GPU, 가동 시간, 처리 건수, 평균 응답 시간) |
| `POST /search` `{"text": "..."}` | 업종 상위 5개 + 의미가 가까운 가게 60곳(점수 포함). 거리 정렬은 방문자 기기에서 |

보안: 우리 사이트에서 온 요청만 허용, 접속지별 분당 30회 제한, 질문 200자 제한, 질문 문장은 저장하지 않음, 파일·명령 접근 기능 없음.

## 모델 바꾸기
- 학습한 모델 폴더를 `models/v2`처럼 넣고(안에 `metrics.json`이 있으면 점수가 상태 페이지에 표시됨), `models/current.txt`에 `v2` 한 줄을 적은 뒤 서버 창을 닫았다 다시 켭니다.
- 되돌리기: `current.txt`를 이전 버전으로 바꾸고 재시작. `current.txt`가 없으면 기본 `jhgan/ko-sroberta-multitask`(v1-base)를 씁니다.
- 모델이 바뀌면 가게 색인은 자동으로 새로 만듭니다. 모델 폴더는 GitHub에 올리지 않습니다.
