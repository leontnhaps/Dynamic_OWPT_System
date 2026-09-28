## Tx PV 추적 시뮬레이터와 SAC 사전학습

이 폴더는 **5 m 부근에서 PV 중심을 레이저 기준점에 맞추는 Tx Pan/Tilt 사전학습용 코드**다. 카메라나 Raspberry Pi, 서버 연결 없이 PC에서 실행한다. 사진 자체를 학습시키는 YOLO 코드가 아니라, 사진에서 측정한 좌표 배율을 이용하는 좌표 기반 제어 시뮬레이터다. 이번 구현은 M0 완료 판정이나 실기 검증을 대신하지 않는다.


### 변경된 기본 학습: 한 방향 지속 이동

기존 ±6° 모델을 새 범위에 그대로 이어 학습하지 말고 새 출력 폴더에서 처음부터 학습한다. `--resume`은 저장된 설정을 유지하므로 이전 모델을 지정하면 이전 ±6°/왕복 환경으로 이어진다. 구 config의 `angle_limit_deg`는 호환용으로만 보존한다.

```bash
python -m simulation.train --steps 100000 --device cuda --out captures/M3/sac_straight_gpu
python -m simulation.evaluate --model captures/M3/sac_straight_gpu/model.zip --episodes 20 --out captures/M3/eval_straight
```

30초에 총 3.6 m를 이동하므로 카메라가 고정되어 있으면 PV가 화면 밖으로 나간다. B1과 SAC는 영상 오차를 이용해 따라가야 한다. 아래로 가는 궤적은 Tilt -15°에서 PV가 아래쪽 화면 밖으로 계속 이동하면 즉시 에피소드를 종료한다. 이는 제어 실패와 구분해서 해석할 물리 범위 한계다. 등속 직선 이동이라고 Pan/Tilt 각속도까지 일정한 것은 아니다. 세계 Z=5 m의 전방 평면 궤적은 Pan ±180° 전체를 훈련하지 않으며, 전 범위 대응 성능을 주장하지 않는다.

명령 속도 제한은 제거했다. 기본 `slew_deg_s=null`, `command_step_deg=1`로 정수 각도 목표를 바로 전달한다. 범위 확대에 따른 SAC 탐색/수렴은 별도로 평가해야 하며 100,000 step을 성능 보장 횟수로 해석하지 않는다. `smoke_result.json`의 과거 수치는 이전 국소 환경 기록이다.

### 현재 기본 정책: ±5° 조정량

사용자가 세미나 전 코드 변경을 승인했다. M0 문서 수정은 세미나 이후로 보류하며, 이번 구현에서는 SAC의 출력 의미를 절대 목표각에서 직전 명령각 기준 조정량으로 변경한다. `action_mode="delta"`, `delta_limit_deg=5`, `command_step_deg=1`이 기본이다. 입력 6개, SAC 구조, 보상 가중치, PV 궤적, 한계 방향 종료는 유지한다.

$$ \Delta\theta_{j,t}=\operatorname{round}_{1^\circ}(5a_{j,t}),\qquad \theta_{j,t}=\operatorname{clip}(\theta_{j,t-1}+\Delta\theta_{j,t},\theta_{j,min},\theta_{j,max}) $$

정확한 절반은 0에서 먼 방향으로 반올림한다. 각 축은 -5,-4,…,0,…,+4,+5° 중 하나를 선택한다. 0은 유지이며 직전 명령각에 누적한 절대 목표각을 모터에 전달한다. 한계에서 잘린 변화량은 다음 step에 이월하지 않는다. 직전 명령각은 feedback이 아니며 관측 정규화는 기존 절대 명령각 기준 그대로다. 전체 범위는 Pan ±180°, Tilt -15~40°이고 시작은 (0,0)이다. B1도 오차에서 조정량을 계산하고 같은 ±5°/1° 변환을 거친다. B0의 delta action은 (0,0)이다.

```bash
python -m simulation.train --steps 100000 --device cuda --out captures/M3/sac_delta5_gpu
python -m simulation.evaluate --model captures/M3/sac_delta5_gpu/model.zip --episodes 20 --out captures/M3/eval_delta5
```

`--resume` 없이 새 학습을 시작한다. 이전 config에 action_mode가 없으면 `absolute`로 읽어 기존 모델의 의미를 보존한다. 새 delta 모델은 저장/재개/평가 시 delta 설정을 유지한다. ±5°는 step당 조정 범위이며 기존 0.2° slew limiter를 다시 사용하는 것은 아니다. 실제 모터 응답은 여전히 기본적으로 즉시 도달 가정이다. 이번 검증은 동작 검증이며 학습 성능 개선을 보장하지 않는다.

### 1. M0와 이번 구현의 관계

근거는 첨부된 M0-1 연구 시나리오, M0-2 구조, M0-3 변수/정보 흐름(DRAFT), M0-4 RL 관측/행동/보상, M0-5 baseline, M0-6 지표, M0-7 하드웨어 제약이다. 제목에 DRAFT가 있는 문서는 최종 확정본으로 승격하지 않는다. 이번 README는 구현 설명이며 새로운 최종 마일스톤 문서가 아니다.

M0의 원래 모델은 Tx 두 축과 PV gimbal 두 축을 포함한다. 이번에는 사용자가 먼저 진행하려는 **Tx 두 축 조준 단계만 구현**했다. 따라서 관측은 8→6개, 행동은 4→2개이고, PV 입사각 보상은 비활성이다. 향후 네 축을 사용할 때는 관측/행동 공간을 확장하고 별도 모델을 학습해야 한다. 수신 전력·충전량은 정책 입력과 보상 모두에 넣지 않는다.

### 2. 사진에서 회전당 픽셀 변화를 추정한 방법

`calibration.py`에 `simulation_data.zip`의 수동 클릭 좌표와 보정 이력을 기록했다. Y +0.32 m는 +0.05 m, X ±0.275 m는 ±0.225 m로 정정했고, **오른쪽 +0.225 m 행은 사용자 지시에 따라 적합에서 제외**했다. 원본 ZIP은 수정하지 않았다. 수치는 기존 좌표 보고서의 소수 셋째 자리 값을 사용한다.

기준 사진의 PV 위치 (654.974, 387.552)는 물체 이동량 계산의 원점이다. **조준 목표인 레이저 기준점 (696, 384)와는 다르다.** 세계 X는 오른쪽, Y는 위쪽, 영상 v는 아래쪽을 양의 방향으로 둔다.

$$ \Delta u=k_x X,\qquad \Delta v=-k_y Y,\qquad k_x=\frac{\sum X_i\Delta u_i}{\sum X_i^2},\qquad k_y=-\frac{\sum Y_i\Delta v_i}{\sum Y_i^2} $$

| 항목 | 가로 | 세로 |
|---|---:|---:|
| 추정 국소 배율 | 7.8321 px/cm | 7.9886 px/cm |
| 5 m에서 역산한 유효 초점거리 | 3916.03 px | 3994.31 px |
| 중심 부근 1° 회전 시 이동량 크기 | 68.35 px | 69.71 px |
| 0.1° 회전 시 근사 이동량 | 6.83 px | 6.97 px |
| 0.2° 회전 시 근사 이동량 | 13.67 px | 13.94 px |

회전 근사는 `f = k × Z`와 작은 각도의 `tan(δθ) ≈ δθ`에서 나온다. θ는 식 안에서 rad 단위다.

$$ f_x=k_x Z,\qquad f_y=k_y Z,\qquad |\Delta u|\approx f_x|\Delta\theta_{pan}|,\qquad |\Delta v|\approx f_y|\Delta\theta_{tilt}| $$

5 m에서 1°는 평면 위 약 8.73 cm 이동에 해당하고, 여기에 약 7.8 px/cm를 곱하면 약 68 px가 된다. **직접 측정한 서보 응답이 아니라 핀홀 모델로부터의 추정**이다. 광축에서 벗어나면 국소 기울기에 sec² 항이 붙고 두 축 결합도 생긴다. 환경에서는 위 선형식 대신 아래 회전행렬 투영을 실제로 계산한다.

$$ R=R_y(\theta_{pan})R_x(-\theta_{tilt}),\qquad P_c=R^T P_w,\qquad u=c_x+f_xX_c/Z_c,\qquad v=c_y-f_yY_c/Z_c $$

`(cx,cy)=(648,486)`은 미보정 주점의 초기 가정으로만 사용한다. 제어 목표는 항상 `(uL,vL)=(696,384)`다. 카메라/레이저는 함께 회전하고, 회전 중심과 카메라 중심 사이 병진 및 거리별 시차 변화는 이 초기 모델에서 무시한다. 레이저 기준점은 현재 5 m, 1296×972 설정에서의 잠정값이다. 거리를 바꾼다고 이 기준점의 유효성이 자동으로 보장되지 않는다.

M0-2의 일반적인 기준점은 각도별 `f_cal(θTx,pan, θTx,tilt)`이다. 여기서는 사용자가 제시한 5 m 부근 고정 기준점 가정으로 이 함수를 상수 근사했으며, 각도별 보정이 불필요하다고 확정한 것은 아니다.

남은 가로 자료 중 왼쪽 22.5 cm는 적합값과 약 13.29 px 차이가 있고, 세로 이동 사진에도 약 15 px의 가로 변화가 있다. 축 교차항·렌즈 왜곡을 정확히 식별할 자료는 부족하다. 이를 정밀 카메라 보정으로 간주하지 않으며, 기본 학습에서는 각 episode의 fx/fy를 각각 ±10% 범위로 바꾼다. 이 범위 역시 불확실성 대응용 가정이며 통계적 신뢰구간이 아니다.

### 3. 관측, 행동, 보상

PV 검출기가 제공할 위치로 영상 오차를 구한다. 초기 환경은 이상적인 PV 중심 투영에 1 px 표준편차의 측정 노이즈를 더한다. 실제 모델 연결 시 YOLO 등의 검출 좌표를 원래 영상 좌표계로 환산한 뒤 같은 계산을 해야 한다.

$$ e_{u,t}=u_{PV,t}-u_L,\qquad e_{v,t}=v_{PV,t}-v_L,\qquad \Delta e_t=e_t-e_{t-1} $$

$$ o_t=(e_{u,t}/E_u,e_{v,t}/E_v,\Delta e_{u,t}/E_u,\Delta e_{v,t}/E_v,(\theta_{pan,t-1}-m_{pan})/s_{pan},(\theta_{tilt,t-1}-m_{tilt})/s_{tilt}) $$

기본 `Eu=648`, `Ev=486`, 각도는 degree다. Δe는 시간으로 나누지 않는 프레임 차분이며 episode 시작에는 0이다. 직전 각도는 **제한 후 전송한 명령값**이다. 실제 각도 피드백, 깊이, 세계 위치, 타깃 속도는 정책 입력에 없다.

현재 기본 SAC의 `a`는 위 절에서 정의한 **±5° 조정량**이다. 아래 절대 목표각 식은 `action_mode="absolute"`인 기존 모델의 호환 경로에만 적용된다.

$$ \theta^*_{j,t}=\frac{\theta_{j,max}-\theta_{j,min}}{2}a_{j,t}+\frac{\theta_{j,max}+\theta_{j,min}}{2},\qquad a_{j,t}\in[-1,1] $$

사용자 지정 범위는 Pan [-180°,180°], Tilt [-15°,40°]이다. 매 episode의 시작 명령과 가상 실제각은 (0°,0°)이다. 정규화의 s는 축별 반범위 (180°,27.5°), m은 중간값 (0°,12.5°)이다. 기존 absolute 모드에서만 정규화 action (0,0)은 반올림 전 (0°,12.5°), 전달 명령은 (0°,13°)이며 B0는 별도 역변환으로 (0°,0°)를 유지한다. 이 범위의 실제 하드웨어 응답을 검증했다는 의미는 아니다. 명령 전달 전 다음 각도 양자화를 적용한다.

$$ \theta_{j,t}=\operatorname{clip}(\operatorname{round}_{1^\circ}(\theta^*_{j,t}),\theta_{j,min},\theta_{j,max}) $$

기본 dt=0.1 s이며, 1° 격자로 반올림한다(정확한 절반은 0에서 먼 방향). absolute 호환 모드에는 step당 변화량 상한이 없다. 새 delta 모드는 정책 조정량 범위가 축당 ±5°다. 기본 가상 actuator는 즉시 응답하며 실제 모터의 이동 시간까지 제거되었다는 뜻은 아니다. B1과 SAC 모두 같은 변환을 사용한다. 이전 config를 로드한 경우에만 기존 slew 제한/연속 명령이 유지된다.

$$ r_{point,t}=-[(e_{u,t+1}/E_u)^2+(e_{v,t+1}/E_v)^2],\qquad r_{cmd,t}=\sum_{j\in\{pan,tilt\}}[(\theta_{j,t}-\theta_{j,t-1})/\Delta\theta_{j,ref}]^2 $$

$$ r_t=w_{point}r_{point,t}-w_{cmd}r_{cmd,t} $$

M0의 조준/명령 변화 비용을 그대로 사용하며, 이번 Tx 단계에서는 입사각 항을 생략했다. 초기 가중치 `w_point=10`, `w_cmd=0.02`, `Δθ_ref=1°`는 조정 가능한 구현 기본값이다. 정답으로 확정한 실험 결과가 아니다.

가시 영역 안에서는 노이즈를 포함한 측정 오차로 보상을 계산한다. 후방/투영 특이점은 미검출로 처리하고, 화면 밖 보상 오차는 축당 ±10×영상 너비로 제한해 유한하게 유지한다. 후방의 영상 GT 좌표는 NaN으로 기록되며 해당 영상 RMS는 정의되지 않는다. 화면 밖에서는 정책에 마지막 측정값만 유지하고, 학습 보상만 내부 GT 오차로 계속 계산한다. 이것은 **미검출 상태의 임시 학습 처리**다. 최대 300 step(30 s)까지 진행하되, 축 한계 방향 이탈 조건이 충족되면 즉시 종료한다. 실제 미검출/재탐색과 온라인 보상 처리는 별도 구현이 필요하다.

### 4. 왜 SAC인가

M0-4에서 정한 SAC를 사용했다. SAC는 연속 각도 출력에 적합한 off-policy 알고리즘이며 replay buffer로 과거 전이를 재사용한다. 학습 시 엔트로피 항으로 탐색하고 평가 시에는 `deterministic=True`로 실행한다. 구현은 Stable-Baselines3 `SAC("MlpPolicy")`, 은닉층 [128,128], 자동 엔트로피 계수, twin Q critic을 사용한다. 원시 사진이 아닌 여섯 개 숫자가 입력이므로 MLP를 사용한다.

SAC 자체가 흔들림을 방지하는 것은 아니다. 명령 변화 비용과 변화율 제한을 함께 사용한다. 실제 관측/응답이 다르면 재조정이 필요하다. 다른 알고리즘보다 우수하다는 실험적 결론은 아직 없다.

### 5. 실행 순서

저장소 루트에서 Python 3.10~3.12 가상환경을 만들고 아래 명령을 실행한다. 개발 검증은 Python 3.12/CPU에서 수행했다. CPU 전용 PyTorch가 필요하면 requirements 설치 전에 해당 운영체제용 PyTorch CPU wheel을 설치한다.

```bash
python -m venv .venv-sim
# Linux/macOS
source .venv-sim/bin/activate
# Windows PowerShell은 .venv-sim\Scripts\Activate.ps1
python -m pip install -r simulation/requirements.txt

# 보정 수치 및 잔차 확인
python -m simulation.calibration

# 실제 학습이 가능한지 짧게 확인 (성능 검증용 학습량 아님)
python -m simulation.train --steps 3000 --out captures/M3/smoke

# 본 학습: 100,000 step은 초기 실행 예산, 수렴 보장 횟수 아님
python -m simulation.train --steps 100000 --out captures/M3/sac_straight

# 학습하지 않은 seed들에서 B0/B1/SAC를 같은 조건으로 평가
python -m simulation.evaluate --model captures/M3/sac_straight/model.zip --episodes 20 --out captures/M3/eval_5m

# 세미나용 좌표 애니메이션
python -m simulation.demo captures/M3/eval_5m
```

GUI 없는 서버에서는 `demo`에 `--gif captures/M3/demo.gif`를 붙여 저장한다. 점은 PV 중심을 나타내며 PV 크기나 광학 충전 영역을 표현하지 않는다. `--controller B1`로 비교 제어기도 재생할 수 있다.

학습 출력: `model.zip`, `replay_buffer.pkl`, `config.json`, `manifest.json`, `monitor.csv`, `checkpoints/`. 매 10,000 step 모델 체크포인트를 저장하고 종료 시 최신 모델과 replay buffer를 저장한다. 중간 체크포인트는 모델만 있으므로 아래 resume은 최종 `model.zip`과 버퍼 쌍을 대상으로 한다. 신뢰할 수 있는 자체 생성 모델/버퍼만 로드한다.

```bash
# 저장한 모델/버퍼에서 추가 학습 (새 결과 폴더 사용)
python -m simulation.train --resume captures/M3/sac_straight/model.zip --steps 100000 --out captures/M3/sac_straight_more

# 미실측 지연 가정을 넣은 별도 스트레스 평가
python -m simulation.evaluate --model captures/M3/sac_straight/model.zip --stress --out captures/M3/eval_stress
```

resume은 가중치/optimizer/버퍼를 이어서 사용하는 새 실행이다. 환경 episode와 RNG를 다시 초기화하므로 중단 시점에서 bitwise 동일하게 이어지는 실행은 아니다. 설정이 다른 버퍼를 실환경 데이터로 간주해 사용하지 않는다.

설정 변경 시 `config.json`을 복사해 편집하고 `train --config 경로`로 새 학습을 시작한다. 평가에는 저장된 학습 설정을 자동으로 사용한다. `--stress`는 tau=0.15 s, 1 step 명령 지연, 측정 노이즈 3 px라는 명시적 가정만 추가한다. 기본 모델은 tau=0, 명령 지연=0인 이상적 actuator다. 미측정 동역학을 측정한 것으로 가장하지 않는다.

### 6. 비교 결과 읽기

- B0: 0° 고정 조준. B1: M0의 영상 오차 비례 제어. SAC: 학습한 정책(기본 delta, 기존 모델 absolute).
- M0의 B2는 Tx/PV 공동 보정 제어이므로 Tx만 있는 이번 단계에서 별도의 비교군으로 꾸미지 않았다.
- 기본 운동 `straight`는 에피소드 시작에 방향 φ를 [-π,π)에서 균일하게 한 번 뽑고 속도 벡터 0.12(cosφ,sinφ) m/s를 30초 동안 유지한다. X/Y는 초기 위치+속도×시간이며 ±30 cm는 초기 위치 범위일 뿐 이동 경계가 아니다. 반사, 방향 재추첨, 화면 경계에서의 되돌림은 없다. `sine`, `linear`(경계 반사), `mixed`(기존 두 운동 혼합)는 선택 옵션으로 남긴다. 세계 Z=5 m 및 PV 자세는 고정하지만 Tx–PV 직선 거리는 이동에 따라 달라진다. `--scenario stationary`로 고정 PV도 평가할 수 있다.
- 같은 seed는 제어기와 관계없이 같은 초기 타깃/궤적/focal을 생성한다. 학습 seed 기본42, 평가 episode seed 기본10000부터다. 성능 판단은 여러 독립 학습 seed에서도 반복해야 한다.
- `frames.csv`: 프레임별 오차/명령/내부 실제각/세계 좌표. GT 필드는 평가·진단용이다.
- `episodes.csv`: episode별 pointing RMS, p95, 명령 변화, 가시 비율, 유효 투영 비율, return. RMS/p95는 전방의 유한한 투영만 사용하며 화면 밖 좌표도 포함한다. 카메라 뒤 좌표는 CSV에 NaN으로 표시되고 픽셀 지표에서는 제외하므로, 가시/유효 투영 비율을 반드시 함께 읽는다. 유효 투영이 하나도 없는 episode의 RMS/p95는 JSON에서 null이고 집계의 valid_episodes로 산출 가능한 episode 수를 표시한다. 화면에서 사라진 경우를 작은 오차의 성공으로 해석하지 않는다.
- `summary.json`: episode 지표의 평균과 표본 표준편차. `comparison.png`: 첫 평가 seed의 오차/Pan/Tilt 시계열.
- 20 px 이내 비율은 임시 공학적 비교 지표다. PV 크기/광점 크기/실제 수신전력이 없으므로 충전 성공률로 해석하지 않는다.

최초 실행 검증 수치는 `smoke_result.json`에 있다. 3,000 step만 학습한 모델을 5개 episode로 평가했을 때 평균 RMS는 B0 194.75 px, B1 24.06 px, SAC 62.01 px였다. SAC는 아직 B1보다 오차와 명령 변화가 크므로 이 짧은 학습을 수렴 또는 실기 투입 가능한 결과로 해석하지 않는다. 이 파일은 실행 검증 기록이며 학습 모델 가중치는 포함하지 않는다.

M0의 명령 변화 지표는 다음처럼 전달 명령의 축별 범위 `Rpan=360°`, `Rtilt=55°`로 정규화한다. 실제 모터 각속도나 진동을 측정한 지표가 아니다.

$$ C_{var}=\frac{1}{T-1}\sum_{t=2}^{T}\sum_j[(\theta_{j,t}-\theta_{j,t-1})/R_j]^2 $$

### 7. 코드의 주요 함수

| 파일/함수 | 역할 |
|---|---|
| `calibration.fit_scale()` | 보정된 수동 좌표로 가로/세로 배율 적합, 잔차 반환 |
| `Config` | 실측 참조값과 가정 파라미터 저장/검증/JSON 입출력 |
| `env.project()` | 세계 좌표를 Tx 회전 후 영상 좌표로 투영 |
| `TrackingEnv.reset()` | episode 궤적·초기 명령·난수·관측 초기화 |
| `target_position()` | t에 따른 5 m 평면의 PV 위치 생성 |
| `TrackingEnv.step()` | 명령 제한→가정 응답→타깃 이동→측정→보상→다음 관측 |
| `encode_observation()` | 오차/차분/직전 전달 명령을 M0 순서로 정규화 |
| `absolute_command()` | SAC의 delta/absolute 출력을 해석해 정수 절대 명령각 생성 |
| `reward_terms()` | 제어 후 pointing 보상과 명령 변화 비용 계산 |
| `proportional_action()` | B1 비례 제어기의 절대 목표각 생성 |
| `train.main()` | SAC 생성/재개, 학습, 모델·버퍼·설정 저장 |
| `evaluate.rollout()` | 제어기 하나의 episode 실행과 M0 지표 계산 |
| `evaluate.main()` | 동일 조건 비교, CSV/JSON/그래프 저장 |
| `demo.main()` | 평가 CSV의 영상 좌표 애니메이션 또는 GIF |

실제 연결할 때는 `encode_observation()`과 `absolute_command()`의 입력 순서/단위/설정을 동일하게 유지해야 한다. 실제 시작 명령을 알고 있는 상태에서 초기화하고, 화면 좌표 스케일 및 축 방향을 먼저 확인해야 한다. 현재 파일들은 어떤 UART·모터·레이저 제어 명령도 보내지 않는다.

### 8. 아직 검증하지 않은 부분

실제 Pan/Tilt 부호, 영점, 회전당 픽셀 변화, 명령-움직임 지연, backlash, 카메라/서보 회전 중심 차이, 5 m 외의 거리별 레이저 기준점은 미측정이다. YOLO 오검출·누락·영상 지연 및 실제 UAV 자세 변화도 본 모델로 검증하지 않는다. 그래서 결과는 **시뮬레이션 사전학습**으로 보고하고, 하드웨어 안정 추적이나 충전 성공을 주장하지 않는다. 실환경 추가 학습 루프는 이 폴더에 포함하지 않았다.

### 9. 수학·알고리즘 출처

- [OpenCV camera calibration: pinhole projection](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [Stable-Baselines3 SAC: algorithm/API](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)

외부 문서는 모델 형태와 API 근거다. 68~70 px/degree, 명령 제한, 주기, 가중치는 이 프로젝트 자료에서 추정하거나 초기 설정한 값이며 외부 논문의 실측값이 아니다.


### 10. 한계 방향 이탈 종료와 기존 모델 호환

`end_on_limit_exit=true`에서 세 조건을 모두 만족하면 `terminated=True`다: (1) 가상 실제 축이 min/max에 도달, (2) 그 한계가 가리키는 쪽 화면 경계 밖에 PV가 존재, (3) PV 자체가 그 바깥 방향으로 이동. 현재 카메라 자세로 직전/현재 PV 위치를 모두 투영하여 카메라 회전으로 생긴 화면 이동은 제외한다. 방향 부호가 반대인 설정도 반영한다. 화면 안, 한계 도달 전, 안쪽으로 복귀, 정지, 단순 후방 조준은 이 조건으로 종료하지 않는다. 이 판정은 simulator GT이며 정책 입력은 여전히 6개다. 최대 시간 종료는 `truncated=True`다.

평가는 종료 즉시 중단하고 frames.csv의 `termination_reason`, episodes.csv의 `episode_steps`, `duration_s`, `limit_exit`를 저장한다. 제어기마다 평가 길이가 달라질 수 있으므로 return/RMS만으로 순위를 정하지 말고 지속 시간과 화면 유지율도 비교한다. 기존 음수 보상에서는 조기 종료를 선호하는 정책이 생길 가능성이 있어 종료 빈도도 함께 점검한다. 별도의 종료 보너스나 추가 벌점은 도입하지 않았다.

이전 config에는 새 필드가 없으므로 `command_step_deg=0`, `end_on_limit_exit=false`로 읽어 원래 동작을 보존한다. 새 조건 학습은 이전 모델에 --resume을 붙이지 않고 새 폴더에서 시작한다:

```bash
python -m simulation.train --steps 100000 --device cuda --out captures/M3/sac_delta5_gpu
python -m simulation.evaluate --model captures/M3/sac_delta5_gpu/model.zip --episodes 20 --out captures/M3/eval_delta5
```
