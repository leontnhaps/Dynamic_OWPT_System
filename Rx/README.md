# Rx — M4-1 Pimoroni Pan-Tilt 구동 확인

현재 단계는 **Pro Mini → I²C → Pimoroni Pan-Tilt HAT → 서보**의 수동 구동 확인이다. OOK 수신·명령 프레임·속도 측정은 M4 후속 구현 범위이며 이 스케치에는 아직 없다. 실제 장비 구동은 미검증이다.

## 대상과 파일

- MCU: Arduino Pro Mini, ATmega328P **3.3V / 8MHz**.
- 드라이버: Pimoroni Pan-Tilt HAT(PIM183 키트/PIM250 보드), I²C `0x15`. PCA9685용 ServoKit 코드를 사용하지 않는다.
- 스케치: `Arduino/pimoroni_pantilt_test/pimoroni_pantilt_test.ino`.
- 추가 Arduino 라이브러리는 필요 없고 AVR 코어의 `Wire`를 사용한다.
- PV가 장착되고 장치가 뒤집힌 상태이므로 물리적 영점·방향·운용 한계는 미확정이다. Tx 보정값을 복사하지 않는다.

## 전원과 배선

**배터리를 분리한 상태에서 배선한다.** 시험에는 안정된 5V 서보 전원을 준비한다. 제조사는 Pi와 함께 사용하는 구성에 2.5A급 전원을 권장한다. 이는 이 조립체의 실측 소비 전류가 아니다. 현재 3.7V 배터리를 HAT의 5V 패드에 직결해 잠깐 움직인 결과만으로 정상 동작이나 전원 용량을 판정하지 않는다. 승압기 또는 적절한 별도 5V 전원이 준비되기 전에는 코드 업로드까지만 진행할 수 있다.

| 연결할 곳 | 연결 대상 |
|---|---|
| Pro Mini A4 | HAT 옆면 SDA |
| Pro Mini A5 | HAT 옆면 SCL |
| Pro Mini GND | HAT GND 및 별도 5V 전원 GND |
| 별도 안정된 5V 전원 + | HAT 옆면 5V |
| Pro Mini VCC | 안정된 3.3V MCU 전원 |
| SDA, SCL 각각 | 각각 4.7kΩ으로 MCU 3.3V에 풀업(이미 적절한 풀업이 있으면 중복 추가하지 않음) |

공식 회로도에서 HAT의 PIC와 서보 전원은 5V이며, `3V` 패드는 별도 3.3V 레일이다. 이 시험에서는 HAT의 `3V` 패드를 연결하지 않는다. `3V`를 서보 전원 입력이나 5V에서 생성되는 출력으로 사용하지 않는다. HAT의 EEPROM 기능도 사용하지 않는다. Pi 없이 운용하므로 SDA/SCL 풀업을 확인한다. **I²C 풀업은 5V가 아니라 3.3V**에 연결한다. HAT 옆면 TX/RX는 이번 제어 배선에 쓰지 않는다.

USB 시리얼 변환기는 **신호 레벨도 3.3V인지 확인**하고 TX→Pro Mini RXI, RX→TXO, GND→GND로 연결한다. 변환기의 3.3V 전원 출력이 MCU를 감당할 때만 VCC 전원으로 사용하며, 다른 MCU 전원과 병렬로 연결하지 않는다. DTR은 Pro Mini의 업로드용 DTR/GRN 단자에 연결한다. DTR이 없으면 업로드 시 수동 리셋이 필요할 수 있다. HAT 서보 전류를 Pro Mini VCC, RAW 또는 시리얼 변환기에서 공급하지 않는다.

## 업데이트와 업로드

기존 저장소에서 로컬 변경을 보존한 뒤:

```sh
git fetch origin
git switch M4-1
git pull --ff-only origin M4-1
```

1. Arduino IDE에서 위 `.ino`를 연다.
2. 보드 `Arduino Pro or Pro Mini`, 프로세서 `ATmega328P (3.3V, 8 MHz)`, 해당 시리얼 포트를 선택한다.
3. `Wire` 타임아웃을 지원하는 Arduino AVR Boards 코어를 사용한다. 지원하지 않으면 의도적으로 컴파일 오류가 발생하므로 보드 패키지를 업데이트한다.
4. 서보 전원을 분리한 채 업로드한다. 이때 HAT 미연결에 따른 I²C 오류는 예상된다.
5. 전원을 끄고 위 배선을 완성한다. PV를 받쳐 두고 MCU와 HAT 전원을 켠다. 시리얼 모니터는 **9600 baud**, 줄바꿈은 `Newline` 또는 `Both NL & CR`로 설정한다.

## 최초 시험 순서

명령을 **하나씩** 입력하고 응답 및 실제 움직임을 확인한다.

| 순서 | 명령 | 확인 사항 |
|---|---|---|
| 1 | `off` | `OK PWM off` 확인. 실패하면 전원·공통 GND·풀업·SDA/SCL 확인 |
| 2 | `probe` | `ACK 0x15` 확인. ACK는 주소 응답이며 제품 식별 검증은 아님 |
| 3 | `status` | fault=0, mask=0. 최초 명령각은 unknown |
| 4 | `pan 0` | PV를 받친 채 Pan 하나의 동작 확인, 2초 뒤 PWM 해제 |
| 5 | `pan 5` | 앞 동작을 관찰한 후 작은 명령 변화 확인 |
| 6 | `pan 0` | 기준 명령으로 복귀 여부 관찰 |
| 7 | `tilt 0`, 이후 `tilt 5`, 이후 `tilt 0` | 각각 따로 입력하고 Tilt 동작 확인 |
| 8 | `off` | 시험 종료 후 출력 해제, 전원 분리 |

`0`은 공식 라이브러리의 1450µs 목표 펄스에 대응하는 **명령각**이며 장치의 현재 위치나 물리적 수평이 아니다. 첫 명령은 현재 위치에서 크게 움직일 수 있다. ±10° 제한은 초기 시험용 소프트웨어 제한일 뿐 충돌 없는 범위를 보증하지 않는다. 혼 장착·기구 방향을 확인하고 필요하면 PV 하중을 분리한 상태에서 영점을 잡는다. 전원이 켜진 서보 축을 손으로 강제로 돌리지 않는다.

`pan N`, `tilt N`은 정수 -10~10만 허용한다. 한 번에 한 축만 활성화하며 반대 축은 PWM이 해제된다. 마지막 이동 명령 후 2초가 지나면 활성 축도 해제한다. **PWM 해제는 전원 차단이 아니며, 유지 토크가 사라지면 PV가 처질 수 있다.** 받침 없이 장착 상태 유지에 사용하지 않는다. 명령을 연속 전송하면 2초 타이머가 갱신된다.

부팅 때에는 출력을 해제하는 명령만 보내며 자동 중앙 정렬·왕복은 하지 않는다. MCU가 시작하기 전 또는 I²C 장애 때의 움직임까지 소프트웨어로 막을 수는 없다. I²C 오류는 fault로 남겨 이동을 거부한다. 배선/전원 복구 후 `off`가 성공하면 재시도할 수 있다. `status`는 마지막 ACK 기반 상태이며 실제 PWM, 실제 각도, 도착 또는 전압을 측정하지 않는다.

## 기록과 검증

각 시험 결과를 `captures/M4-1/<날짜_시간>/`의 새 폴더에 보관한다. 시리얼 모니터의 전체 출력을 `serial.txt`로 복사하고, `notes.md`에 보드 버전, 커밋, 전원 사양, 실제 배선, 명령 순서, 축별 실제 움직임 방향, 떨림/처짐/발열 여부를 적는다. 확보하지 못한 전압·전류·각도는 미측정으로 남긴다. 가능하면 짧은 동작 영상도 함께 전달한다. 자동 파일 저장 기능은 없다.

호스트 회귀 시험은 아래처럼 실행한다. 가짜 Wire/Serial로 명령 파싱, I²C 바이트 순서, 오류 잠금, 부팅 시 비활성, 유휴 해제를 확인하며 AVR 빌드나 실장비 시험을 대체하지 않는다.

```sh
g++ -std=c++11 -Wall -Wextra -Werror -I Rx/tests Rx/tests/pantilt_host_test.cpp -o /tmp/pantilt_host_test
/tmp/pantilt_host_test
```

## 근거

- [Pimoroni 공식 드라이버](https://github.com/pimoroni/pantilthat-python/blob/a209e8eb58dfe24569fd17680957c8c7e57a219f/library/pantilthat/pantilt.py): config `0x00`, Pan `0x01`, Tilt `0x03`, little-endian 펄스 폭, 575~2325µs 변환을 참조했다.
- [제조사 제품 안내](https://shop.pimoroni.com/products/pan-tilt-hat)
- [Pimoroni 회로도 사본](https://cdn.robotshop.com/media/P/Pim/RB-Pim-234/pdf/pan-tilt-hat--without-pan-tilt-module-schematics.pdf)
