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
5. 고정 PV의 상자와 레이저 기준점 (696,384)을 확인한다. 최고 confidence PV 하나를 사용한다.
6. 최대 step, 반복 횟수, 초기 이동 대기 시간을 정하고 **학습 / 평가 시작**을 누른다. 이때 초기 무작위 목표각으로 이동한다.

`100 step × 10 episode`는 변경 가능한 GUI 시작값이다. 유효 경험이 1,000개 미만이면 업데이트 없이 수집 상태로 끝날 수 있다. 이후 checkpoint로 이어서 학습한다. 조준 수렴을 보장하는 횟수가 아니다.

기본 제어주기는 720 ms이고 비교 실험은 별도 신규 모델로 800 ms를 선택한다. 각 episode는 첫 적중 후에도 최대 step까지 계속한다. 초기 이동 후 2초 대기, 새 유효 PV 검출 최대 3초 대기를 적용한다.

### 3. 학습 흐름 및 중단

첫 업데이트 전에는 축별 ±5° 균일 무작위 행동을 사용한다. 유효 transition 1,000개에 도달한 episode 종료 후 그 episode의 신규 유효 transition 수만큼 업데이트하며, 다음 episode부터 SAC 행동을 샘플링한다. 초기 수집 기간의 업데이트를 소급하지 않는다.

보상은 `−조준 거리/1000 + bounding-box intensity mean`이다. Intensity mean은 고정 빔 모델값의 픽셀 평균이다. 실제 밝기·수신전력을 사용하지 않으며 별도 적중 보너스·명령 변화량 벌점을 더하지 않는다.

미검출·오래된 결과·새 프레임 없음에서는 목표각을 유지한다. 해당 transition은 제외하며 재검출 시 오차 변화량을 0으로 초기화한다. 3초간 유효 관측이 없으면 해당 episode를 종료하고 업데이트·저장 후 다음 episode로 진행한다. 초기화 실패, 연결 오류, 사용자 중지는 전체 반복을 중단한다.

**전체 중지 · 저장**은 추가 명령과 자동 반복을 중단한다. 이미 전송된 목표각으로의 이동을 즉시 정지하는 기능은 아니다. 자동 원점 복귀는 수행하지 않는다. 업데이트 중 중지하면 진행 중인 업데이트 1회를 마친 뒤 저장한다.

### 4. 이어서 학습 및 평가

실행 폴더의 `latest.json`에 마지막으로 완성된 checkpoint의 상대 경로가 기록된다. `checkpoints/episode_.../` 아래 `manifest.json`, `model.zip`, `replay.npz`, `state.json`, `rng.npz`가 함께 있는 폴더를 선택한다.

- **이어서 학습:** 모델·optimizer·alpha·replay·카운터·난수 상태를 복원한다. 실제 장비는 새 초기각으로 reset한다.
- **고정 모델 평가:** 평균 정책의 결정적 행동을 사용한다. 모델·alpha 업데이트와 replay 추가를 하지 않는다.
- 두 모드 모두 저장된 제어주기·보상·정규화·운용 범위를 유지한다. 자동 준비에서 checkpoint의 운용 범위를 장치에 적용하므로 이전 탭을 방문할 필요가 없다.
- 종료한 실행을 계속하려면 checkpoint를 선택하고 자동 준비를 다시 실행한다. 재개·평가 결과는 새 폴더에 저장된다.

구형 M4의 6차원 모델은 새 설계와 호환되지 않는다. 기존 M4 탭의 시뮬레이션 의존 코드를 이 학습에 사용하지 않는다.

### 5. 결과 확인

모든 결과는 `captures/M3-2/<실행별 폴더>/`에 저장한다.

| 파일 | 내용 |
|---|---|
| `config.json` | 카메라·YOLO 식별, 실행·SAC·보상·정규화 설정 |
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
