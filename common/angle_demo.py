"""M6-4 single-frame demonstration for the receiver's two servo channels."""

from common.ook_experiment import HEADER, PAYLOAD_BITS, TRAILING_ZEROS, make_frame


HOME_PAN = 103
HOME_TILT = 20
RX_VIEWPOINT = '셀 뒤에서 빛이 오는 쪽을 보는 기준'
MAX_PAN = 180
MAX_TILT = 35
BIT_MS = 8
# Bit periods actually exercised in the Pi reference and limit runs at M6-4.
ANGLE_BIT_PERIODS = (100, 50, 30, 20, 15, 14, 12, 10, 8, 6, 5, 4, 3, 2, 1)
DECODE_SAMPLE_MS = 1
VOLTAGE_SAMPLE_MS = 5
VOLTAGE_SETTLE_NS = 500_000_000
VOLTAGE_WINDOW_NS = 10_000_000_000
EXPECTED_SAMPLES = VOLTAGE_WINDOW_NS // (VOLTAGE_SAMPLE_MS * 1_000_000)
MIN_VOLTAGE_SAMPLES = EXPECTED_SAMPLES * 9 // 10
MAX_VOLTAGE_GAP_NS = 50_000_000
FRAME_LEN = len(HEADER) + PAYLOAD_BITS + len(TRAILING_ZEROS)


def validate_angle_bit_ms(value):
    if type(value) is not int or value not in ANGLE_BIT_PERIODS:
        raise ValueError('각도 OOK는 속도 시험에서 측정한 비트 길이를 선택하세요.')
    return value


def validate_angles(pan, tilt):
    if type(pan) is not int or not 0 <= pan <= MAX_PAN:
        raise ValueError(f'Pan 서보각은 0~{MAX_PAN}° 정수입니다.')
    if type(tilt) is not int or not 0 <= tilt <= MAX_TILT:
        raise ValueError(f'Tilt 서보각은 0~{MAX_TILT}° 정수입니다.')


def encode_angles(pan, tilt):
    """Transmit absolute whole-degree commands as two unsigned big-endian words."""
    validate_angles(pan, tilt)
    return make_frame(f'{pan:016b}{tilt:016b}')


def payload_from_frame(frame):
    payload = frame[len(HEADER):len(HEADER) + PAYLOAD_BITS]
    if make_frame(payload) != frame:
        raise ValueError('OOK 프레임의 헤더 또는 끝 비트가 일치하지 않습니다.')
    return payload


def decode_angles(payload):
    """Unpack the actual decisions, including out-of-range values for diagnostics."""
    if not isinstance(payload, str) or len(payload) != PAYLOAD_BITS or set(payload) - {'0', '1'}:
        raise ValueError('OOK 각도 payload는 결정된 32비트여야 합니다.')
    return int(payload[:16], 2), int(payload[16:], 2)


def validate_jog(pan_step, tilt_step):
    """Allow only small whole-degree relative moves for the manual Rx check."""
    if any(type(step) is not int or step not in (-5, -1, 0, 1, 5)
           for step in (pan_step, tilt_step)) or (pan_step, tilt_step) == (0, 0):
        raise ValueError('수동 점검은 한 축 이상을 ±1° 또는 ±5° 움직일 수 있습니다.')
