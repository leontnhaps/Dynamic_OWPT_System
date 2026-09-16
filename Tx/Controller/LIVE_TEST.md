# 실전 · YOLO + SAC 탭

기존 Server / Raspberry Pi 연결과 `Com_main.py` GUI를 그대로 사용한다. 새 탭은 PC에서 YOLO 검출과 고정된 SAC 정책 추론만 수행하며 온라인 학습은 하지 않는다. M0 문서는 변경하지 않는다.

## 실행

Windows VSCode 터미널, 프로젝트 루트:

```powershell
& "C:\Users\gmlwn\anaconda3\envs\PTCamera_wavehsare_gpu\python.exe" Tx/Controller/Com_main.py --server <서버IP>
```

기존 서버를 실행하고 Pi는 기존처럼 `python Tx/RaspberryPi/Rasp_main.py --server <서버IP> --servo-port /dev/serial0`으로 연결한다. 기존 CUDA 환경의 torch를 재설치할 필요 없다. GUI에 필요한 Pillow와 학습에 사용한 numpy/gymnasium/stable-baselines3, YOLO에 사용한 ultralytics가 같은 PC 환경에 있어야 한다. 실전 탭의 모델 불러오기 전에는 ML 패키지를 import하지 않으므로 기존 탭은 그대로 사용할 수 있다.

1. Camera 탭: 선택할 SAC 모델 config의 해상도(현재 1296×972)로 영상 수신. fps 목표 10. 레이저 기준점도 config 값을 그대로 사용한다. 실측 기준점과 다르면 해당 config 사본에 실측 좌표를 넣어 별도로 선택한다. 잘못된 해상도를 자동 확대/축소해서 정책에 넣지 않는다.
2. Pan / Tilt 탭: 상태 조회 → 운용 범위 입력·적용 → Pan/Tilt 0°, 0° 입력·이동 완료 응답 확인. 현재 모델의 범위는 Pan -180~180, Tilt -15~40. 실제로 적용한 더 좁은 범위를 벗어나는 추적 명령은 자동 제한해 계속 추적하지 않고 중단한다. SPD/ACC는 이 탭의 입력값을 사용한다.
3. 실전 탭: YOLO best.pt, SAC model.zip(또는 체크포인트), **그 SAC를 학습한 config.json** 선택. SAC 선택 시 인접 config 또는 체크포인트 상위 폴더 config를 자동 찾는다. cpu/cuda, PV class ID, confidence를 설정하고 모델 불러오기. class 이름 목록과 해상도/레이저 기준점이 표시된다. 다른 학습 run의 config를 섞지 않는다.
4. 검출 + SAC 출력 보기: 추론을 워밍업하고 녹색 PV 박스/중심, 빨간 레이저 기준점, 오차와 원래/적용 Δ각도를 확인. 이동은 전송하지 않는다. 직전 명령이 없을 때 미리보기 계산만 (0,0)을 가정한다.
5. 추적 시작: 현재 알려진 **직전 명령각**에서 시작하며 자동 영점 이동은 없다. YOLO와 SAC가 함께 동작하고 실제 명령을 전송한다. 첫 테스트는 레이저 OFF 상태에서 Pan/Tilt 움직임을 확인한다. 학습 모델의 추적 성능/실제 축 부호는 보장하지 않는다.
6. 정지: 추가 명령 전송을 중단한다. 추론 중인 이전 결과도 폐기한다. 이미 전송한 명령을 취소하거나 기계적으로 즉시 정지시키는 기능은 아니다. Pi serial write 응답도 실제 각도/도달 피드백이 아니다.

## 제어 및 기록

- 원본 영상 좌표 → PV 중심−laser → `simulation.control.encode_observation` → deterministic SAC → `absolute_command` 재사용. Δ ±5°, 1° 반올림과 누적 명령은 학습과 동일하다. 관측은 영상 오차/차분/직전 명령 6개이다.
- 제어 목표 간격은 config.dt(0.1초). 실제 간격은 영상 수신/추론/응답 지연에 따라 길어진다. 최대 한 건만 추론하고, 한 건의 이동 응답을 기다린 뒤 새 프레임으로 다음 추론을 한다. 응답은 실제 도달을 의미하지 않는다.
- 미검출/동일 class 다중 검출, 0.5초 이상 지난 수신 프레임/추론 결과, 연결 변경, 서보 오류/응답 시간 초과, 운용 범위 이탈은 자동 중단한다. 재검출되더라도 사용자가 다시 시작해야 한다. 영상 나이는 PC 수신 기준이며 Pi 촬영부터의 전체 지연은 아니다.
- 수동 이동/범위/영상 설정 명령은 실전 추적을 중단한다. 이동 응답 대기 중에는 다른 이동을 막는다. 레이저를 자동 ON하지 않는다.
- `captures/M3/live/<실행시각>/session.json`: 선택 모델 경로, config, 운용 설정.
- `frames.csv`: 프레임 ID, 수신 후 경과시간, 추론시간, 오차, 원래 Δ각도, 적용 Δ각도, 목표 명령각, request ID, sent/hold/missing 상태.
- 기존 GUI events JSONL에는 명령 및 응답, live_start/stop/error가 기록된다. 정지 이유는 여기에 있다. 위치 오차는 검출 좌표 기준이며 실측 서보각/전력은 기록하지 않는다.

## 검증 범위

`python -m unittest discover -s tests -p test_live_tracking.py -v`

단일 표적 선택/해상도 및 모델 입력 계약/명령 변환, 미리보기 전송 금지, 오래된 결과 폐기, 미검출 중단, 운용 범위 및 종료 후 결과 폐기를 검증한다. 실제 YOLO 가중치·카메라·서보·Windows GUI의 통합 운용 검증은 현장에서 수행해야 한다.
