## M3-2. Train SAC for Stationary PV Alignment and Maintenance

### 1. 실행 준비

기존 Server와 Raspberry Pi 실행 방법을 유지한다. Pi는 `--servo-port /dev/serial0`을 사용한다.

노트북 저장소 루트의 PowerShell에서 Controller를 실행한다.

```powershell
& "C:\Users\gmlwn\anaconda3\envs\PTCamera_wavehsare_gpu\python.exe" Tx/Controller/Com_main.py --server <서버IP>
```

패키지가 없는 경우 같은 Python 환경에 설치한다. 기존 CUDA PyTorch가 설치되어 있으면 그대로 사용한다.

```powershell
& "C:\Users\gmlwn\anaconda3\envs\PTCamera_wavehsare_gpu\python.exe" -m pip install -r Tx/Controller/requirements-m3-2.txt
```

### 2. 신규 학습

1. Server·Raspberry Pi·GUI를 실행한 뒤 **M3-2 (정지 PV · SAC)** 탭으로 바로 이동한다. 이전 탭에서 영상을 시작하거나 서보 범위를 적용할 필요가 없다.
2. 신규 학습과 제어주기를 선택한다. YOLO 기본 경로는 M1-3의 `y26n_v1/weights/best.pt`이다.
3. **자동 준비 (영상 · 서보 · 모델)**를 누른다. 모델 로딩 후 1296×972 / 30 FPS / quality 80 / 자동 노출·gain으로 영상을 시작하고 서보 운용 범위를 적용한다. 응답과 새 영상 확인 후 PV 검출 미리보기가 자동 시작된다.
4. 신규 학습의 운용 범위는 이미 적용된 범위가 있으면 유지하고, 없으면 기존 기본값 Pan −180~180°, Tilt −15~40°를 적용한다. 초기각 범위와 운용 범위는 구분하며 적용값을 화면에 표시한다. 준비와 미리보기는 이동 명령을 보내지 않는다.
5. 고정 PV의 상자와 레이저 기준점 (696,384)을 확인한다. 최고 confidence PV 하나를 사용한다. 필요하면 같은 탭의 **Pan / Tilt 수동 이동**에서 Pan −17°, Tilt 0°를 입력하고 **입력 각도로 이동**을 누른다. Pan ± / Tilt ±는 직전 전송 완료 명령을 기준으로 지정한 Step만큼 움직인다. 기본 Step은 1°이며 정수 단위로 입력한다.
6. 최대 step, 반복 횟수, 초기 이동 대기 시간을 정하고 **학습 / 평가 시작**을 누른다. 각 episode의 초기 무작위 범위 기본값은 **Pan −27~−7°, Tilt −10~5°**이다. 시작을 누르면 이 범위 안의 초기 목표각으로 이동한다. 시작 전 PV가 미검출이어도 실제 최신 영상과 장치 준비가 확인되면 초기 이동 후 검출을 시도한다.

수동 이동은 자동 준비 완료 후 사용할 수 있으며, 준비·학습·평가·저장·명령 응답 대기 중에는 잠긴다. 학습을 중지한 경우 저장과 명령 응답 처리가 끝난 뒤 사용한다. 목표각은 실제 측정 각도가 아니며 초기 무작위 범위와 별개로 적용된 서보 운용 범위를 따른다.

`100 step × 10 episode`는 변경 가능한 GUI 시작값이다. 유효 경험이 1,000개 미만이면 업데이트 없이 수집 상태로 끝날 수 있다. 이후 checkpoint로 이어서 학습한다. 조준 수렴을 보장하는 횟수가 아니다.

기본 제어주기는 720 ms이고 비교 실험은 별도 신규 모델로 800 ms를 선택한다. 각 episode는 첫 적중 후에도 최대 step까지 계속한다. 초기 이동 후 2초 대기, 새 유효 PV 검출 최대 3초 대기를 적용한다.

### 3. 학습 흐름 및 중단

첫 업데이트 전에는 축별 ±5° 균일 무작위 행동을 사용한다. 유효 transition 1,000개에 도달한 episode 종료 후 그 episode의 신규 유효 transition 수만큼 업데이트하며, 다음 episode부터 SAC 행동을 샘플링한다. 초기 수집 기간의 업데이트를 소급하지 않는다.

보상은 `−조준 거리/1000 + bounding-box intensity mean`이다. Intensity mean은 고정 빔 모델값의 픽셀 평균이다. 실제 밝기·수신전력을 사용하지 않으며 별도 적중 보너스·명령 변화량 벌점을 더하지 않는다.

미검출·오래된 결과·새 프레임 없음에서는 목표각을 유지한다. 해당 transition은 제외하며 재검출 시 오차 변화량을 0으로 초기화한다. 학습 도중 3초간 유효 관측이 없으면 `target_lost`로 해당 episode를 종료하고 업데이트·저장 후 다음 episode로 진행한다.

초기 이동 후 대기를 마치고 3초 안에 유효 PV를 찾지 못한 경우에도 `initial_target_lost`로 episode 1회를 기록하고 저장 후 다음 초기각으로 이동한다. 초기 미검출 episode의 transition·업데이트 수는 0이며 관측하지 못한 오차와 보상은 생성하지 않는다. 예를 들어 반복 횟수 10은 미검출로 끝난 episode를 포함한 총 10회이다. 초기 영상 획득 실패, 명령 전송·응답 오류, 연결 오류, 사용자 중지는 전체 반복을 중단한다.

**전체 중지 · 저장**은 추가 명령과 자동 반복을 중단한다. 이미 전송된 목표각으로의 이동을 즉시 정지하는 기능은 아니다. 자동 원점 복귀는 수행하지 않는다. 업데이트 중 중지하면 진행 중인 업데이트 1회를 마친 뒤 저장한다.

### 4. 이어서 학습 및 평가

실행 폴더의 `latest.json`에 마지막으로 완성된 checkpoint의 상대 경로가 기록된다. `checkpoints/episode_.../` 아래 `manifest.json`, `model.zip`, `replay.npz`, `state.json`, `rng.npz`가 함께 있는 폴더를 선택한다.

- **이어서 학습:** 모델·optimizer·alpha·replay·카운터·난수 상태를 복원하고 저장된 제어주기를 자동 적용한다. 실제 장비는 새 초기각으로 reset한다.
- **고정 모델 평가:** 평균 정책의 결정적 행동을 사용한다. 모델·alpha 업데이트와 replay 추가를 하지 않는다. 제어주기는 기본 `저장값`을 사용하며 비교 실험에서는 0.720 / 0.800 / 0.600 / 0.500 s를 선택할 수 있다.
- 두 모드 모두 저장된 보상·정규화·운용 범위를 유지한다. 자동 준비에서 checkpoint의 운용 범위를 장치에 적용하므로 이전 탭을 방문할 필요가 없다.
- 종료한 실행을 계속하려면 checkpoint를 선택하고 자동 준비를 다시 실행한다. 재개·평가 결과는 새 폴더에 저장된다.

#### 평가 제어주기 비교

1. 현재 작업을 종료한 뒤 **고정 모델 평가** 모드와 평가할 checkpoint를 선택한다.
2. **제어주기 s**에서 `저장값`으로 기준 평가를 실행한다. 다음 실행에서는 `0.600`, 이후 필요하면 `0.500`을 선택한다.
3. 주기를 변경할 때마다 같은 탭에서 **자동 준비**를 다시 누른다. 완료 메시지의 **학습 주기 / 실행 주기**가 의도한 값인지 확인한다.
4. 같은 PV 위치와 초기각 범위, Seed, 최대 step으로 비교한다. 설정한 주기는 최소 명령 간격이며 실제 간격은 처리 지연만큼 길어질 수 있다. 같은 step 수라도 주기가 짧아지면 실행 시간도 짧아진다.

M1-5의 tilt 5° 이동은 영상 안정화 시간 P95 약 0.61 s로 측정됐다. 0.600 / 0.500 s는 정착이 끝나기 전에 다음 명령이 들어갈 수 있는 비교 조건이다. 관측의 오차 변화량도 이전 관측과의 차이이므로, 학습 주기와 다른 평가에서 입력 분포와 동작이 달라질 수 있다. 짧은 주기가 더 정확한 추적을 보장하지 않는다.

평가 중 주기는 고정하며 관측 유효 시간에도 같은 실행 주기를 적용한다. 지연된 명령을 연속으로 몰아 보내지 않는다. 기존 모델 파일을 수정하거나 평가 주기로 학습 checkpoint를 만들지 않는다. 결과는 `config.json`의 `control_timing`과 실제 간격이 포함된 `steps.csv`로 구분한다.

임시 M2 사진·Calibration 탭과 M4 실전 학습 탭 및 전용 코드는 삭제했다. 정지 PV 학습은 M3-2 탭에서 진행하며, 구형 M4의 6차원 모델은 새 설계와 호환되지 않는다. 기존 촬영 결과와 모델 파일은 삭제하지 않는다.

### 5. 결과 확인

모든 결과는 `captures/M3-2/<실행별 폴더>/`에 저장한다.

이번 실행 로그는 `config.json`의 `schema=tx-stationary-run-v3`이다. v2의 초기 미검출 반복 규칙을 유지하고 `control_timing.training_period_s`, `execution_period_s`, `evaluation_override`로 학습 주기와 평가 실행 주기를 구분한다. `tracking.dt`는 실제 실행 주기이다. Checkpoint 형식은 `tx-stationary-v1`을 유지하므로 기존 M3-2 checkpoint로 이어서 학습할 수 있다. 초기각 범위와 반복 횟수는 현재 GUI의 실행 설정을 사용하며, 학습 재개에는 저장 주기를 적용한다.

| 파일 | 내용 |
|---|---|
| `config.json` | 카메라·YOLO 식별, 실행·SAC·보상·정규화 설정, 학습·실행 제어주기 |
| `detections.csv` | 실제 처리한 프레임의 수신·처리 시각과 검출 결과 |
| `steps.csv` | 실행 action, 전송 목표각, 결과 관측, 두 보상 항, 저장·제외 여부 |
| `episodes.csv` | RMS/P95, 최초 적중 시간, 적중 비율, 관측 유효 비율, 종료·업데이트 기록 |
| `updates.csv` | 업데이트별 actor/critic loss, alpha 및 alpha loss |
| `events.jsonl` | 초기화, 명령, 미검출·재검출, 중단, 저장 상태 |
| `summary.csv` | 실행 모드·종료 사유별 episode 지표 평균·표본 표준편차·개수 |
| `checkpoints/`, `latest.json` | 완성된 모델·replay·학습 상태와 최신 위치 |

초기 검출과 재검출 직후 관측은 action 결과 기반 RMS·적중 비율에서 제외한다. 미검출 값은 공란이며 0으로 채우지 않는다. 시간은 동일 노트북의 monotonic clock을 사용한다. 목표각은 실제 서보 각도 feedback이 아니다.

첫 실험 검토에는 `config.json`, `steps.csv`, `episodes.csv`, `events.jsonl`, `summary.csv`를 전달한다. 검출 문제를 확인할 때 `detections.csv`도 함께 사용한다. 학습을 재개할 때는 checkpoint 폴더 전체를 유지한다.

### 6. 소프트웨어 검증

```powershell
& "C:\Users\gmlwn\anaconda3\envs\PTCamera_wavehsare_gpu\python.exe" -m unittest discover -s tests -p "test_stationary_tracking.py" -v
```

시간·미검출·보상·업데이트 횟수 검증은 실제 장비 없이 수행한다. PyTorch와 Stable-Baselines3가 있으면 실제 SAC 업데이트 및 checkpoint 복원 검사도 실행한다. 패키지가 없으면 그 검사는 명시적으로 skip한다. 이 테스트는 실제 장비 학습 수렴 검증을 대신하지 않는다.
