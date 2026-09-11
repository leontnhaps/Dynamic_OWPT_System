# M1-2 GUI 사용 방법

기존 M1-1 Camera 탭과 영상 경로를 유지하고, M1-2 Pan / Tilt 탭을 추가했습니다.

## 실행

저장소 최상위 폴더에서 실행합니다. 노트북에는 Pillow와 Tkinter가 필요합니다.
Pi는 기존 Picamera2 환경에 pyserial이 추가로 필요합니다(`python -m pip install pyserial`,
또는 해당 Python 환경에 맞는 OS 패키지). 서버는 기존 파일을 그대로 사용합니다.

서버:

```bash
python Server/Server_main.py
```

Raspberry Pi (`SERVER_IP`와 실제 ESP32 장치 경로로 교체):

```bash
python Tx/RaspberryPi/Rasp_main.py --server SERVER_IP --servo-port /dev/ttyUSB0
```

기존 IR-CUT 옵션이 필요하면 확인된 배선에 맞는 `--ir-pin`을 함께 사용합니다.
`--servo-port`를 생략하면 영상은 사용할 수 있지만 실제 서보 이동은 거부됩니다.
포트를 자동으로 탐색하거나 실행 직후 영점 이동을 하지 않습니다.

노트북 GUI:

```bash
python Tx/Controller/Com_main.py --server SERVER_IP --output captures/m1_2
```

모의 실행은 Pi 명령 대신 아래 명령을 사용합니다. 실제 GPIO/서보는 작동하지 않습니다.

```bash
python Tx/RaspberryPi/Rasp_main.py --server 127.0.0.1 --simulate
```

## 실험 순서

1. 레이저를 OFF 상태로 두고 고정된 표식이 보이게 배치합니다. 이 프로그램은 레이저 ON/OFF 명령을 추가하지 않았습니다.
2. Camera 탭에서 기존 방식으로 영상을 시작합니다.
3. M1-2 탭에서 Pan/Tilt min/max를 입력하고 **운용 범위 적용**을 누릅니다. 실제 장착 간섭·배선을 확인하며 범위를 설정합니다. 시작값은 공란입니다.
4. Pan/Tilt 목표 명령값을 직접 입력하고 **입력 각도로 이동**을 누릅니다. 현재 실제 위치를 자동으로 읽거나 추정하지 않습니다.
5. 영상으로 정면 자세를 확인한 뒤 **직전 명령을 기준 자세로 저장**을 누릅니다. 이는 기준 명령을 기록하는 기능이며, 서보 내부 영점을 바꾸지 않습니다.
6. Step을 설정하고 Pan ±, Tilt ±로 각 축을 시험합니다. 마지막으로 전송 완료된 두 축 명령에서 한 축만 변경합니다. 화면에서 편집 중인 다른 축 값을 함께 보내지 않습니다.
7. 시험 종류를 선택하고 관찰 내용을 적은 뒤 **관찰 + JPEG/JSON 저장**을 누릅니다. 예: `Pan + → 장치 우회전, 영상 표식은 왼쪽으로 이동`.
8. 기준 자세 복귀를 반복하고 표식 위치를 관찰·저장합니다. 도달/정착 판단은 영상 관찰로 수행하며, 자동 반복 이동은 없습니다.
9. **설정 저장**으로 입력 범위·기준 자세·Step·SPD/ACC를 저장합니다. 불러오기는 자동 이동을 하지 않으며 범위를 다시 적용해야 합니다.

Pan/Tilt 범위 검사는 제품의 명목 범위 Pan [-180,180], Tilt [-45,90] 안에서 설정된 운용 범위를 사용합니다.
명목 범위 전체가 실제 조립 상태에서 사용 가능하다는 뜻은 아닙니다.
범위 밖 명령은 자동으로 잘라서 실행하지 않고 거부합니다.
SPD=100, ACC=1은 참고 코드의 기본값이며 실측 속도·가속도나 검증된 최적값이 아닙니다.
명령 각도는 소수 입력을 허용하지만 실제 분해능은 아직 측정되지 않았습니다.

## 상태와 증빙 해석

- GUI `command` 이벤트: 네트워크로 요청 전송.
- Pi `servo` / `serial_written`: ESP32 시리얼로 명령 바이트를 썼음. 펌웨어 수신 확인이나 실제 도달 확인이 아님.
- `simulated`: 모의 명령이며 실물 시험 증빙으로 사용하지 않음.
- 실제각은 `null`, `arrival_verified`는 `false`로 기록합니다.
- 응답 대기 중에는 다음 이동 요청을 막고, 5초간 응답이 없으면 실행 여부 불명으로 표시합니다. 자동 재전송하지 않습니다. 연결 상태 및 실물 위치를 확인한 뒤 상태 조회/범위 적용을 수행합니다.
- Pi 제어 연결이 새로 만들어지면 운용 범위와 직전 명령 기록을 초기화합니다. 모터가 실제로 초기 위치에 복귀했다는 의미가 아닙니다.
- JPEG 옆 JSON의 `m1_2`: GUI 저장 시점의 시험 종류, 관찰 내용, 직전 명령, 적용 범위, 기준 자세.
- 프레임의 `servo_at_capture_start`: Pi 촬영 호출 시작 시점의 명령 기록. 명령과 촬영의 동시 진행 가능성이 있어 실제 자세/정착을 보증하지 않습니다.
- `events_*.jsonl`: 명령 요청, Pi 결과/오류, 설정 변경, 증빙 파일명을 누적 기록합니다.
- 영상의 표식 이동 확인은 서보 시험용입니다. 영상 중앙을 laser reference로 정의하거나 calibration을 대체하지 않습니다.

## 적용 파일

- 수정: `Tx/Controller/Com_main.py`, `Tx/RaspberryPi/Rasp_main.py`
- 추가: `Tx/Controller/servo_panel.py`, `common/servo.py`
- 추가: `tests/test_m1_2.py`, `tests/test_relay_m1_2.py`, 이 안내문
- 서버·네트워크 프로토콜은 기존 경로를 유지합니다.

## 참고 및 검증

적용 기준: Dynamic_OWPT_System `ea76e072602a9d8bdf5aae3b895fe9e26f827079`.
1순위 참고: DLC_system_model `Raspberrypi/Rasp_main.py` 및 `Com/app/window.py`.
`move(pan,tilt,speed,acc)`를 115200 baud의 ESP32 JSON
`{"T":133,"X":pan,"Y":tilt,"SPD":speed,"ACC":acc}`와 줄바꿈으로 전송하는 기존 방식을 따릅니다.
2순위 실험 코드는 사용하지 않았습니다.

```bash
python -m unittest discover -s tests -v
```

검증: 명령 유효성·범위 초과·비정상 수치·부분 시리얼 쓰기 실패·한 축 이동 처리,
실제 relay + 모의 Pi를 통한 영상 수신/명령 응답/거부/재접속을 포함한 9개 테스트 통과.
통합 테스트 실행 시 localhost 7500/7501/7600/7601 포트를 비워두세요.
실제 서보 구동과 GUI 화면의 육안 검증은 이 실행 환경에서 수행하지 못했습니다.
