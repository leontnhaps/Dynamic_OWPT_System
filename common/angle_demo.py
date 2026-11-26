"""M6-4 single-frame demonstration for the receiver's two servo channels."""

from common.arc_protocol import VoltageFrameDecoder, crc8


HOME_PAN = 103
HOME_TILT = 20
RX_VIEWPOINT = '셀 뒤에서 빛이 오는 쪽을 보는 기준'
MAX_PAN = 180
MAX_TILT = 35
BIT_MS = 8
DECODE_SAMPLE_MS = 1
VOLTAGE_SAMPLE_MS = 5
VOLTAGE_SETTLE_NS = 500_000_000
VOLTAGE_WINDOW_NS = 1_000_000_000
EXPECTED_SAMPLES = VOLTAGE_WINDOW_NS // (VOLTAGE_SAMPLE_MS * 1_000_000)
MIN_VOLTAGE_SAMPLES = EXPECTED_SAMPLES * 9 // 10
MAX_VOLTAGE_GAP_NS = 50_000_000
# The distinct sync word and CRC discriminator identify version 1 of this demo.
SYNC = '1010101011010011'
DISCRIMINATOR = 0xA1
FRAME_LEN = len(SYNC) + 8 + 8 + 8 + 1


def encode_angles(pan, tilt):
    """Encode absolute servo angles, each expressed in whole degrees."""
    if type(pan) is not int or not 0 <= pan <= MAX_PAN:
        raise ValueError(f'Pan 서보각은 0~{MAX_PAN}° 정수입니다.')
    if type(tilt) is not int or not 0 <= tilt <= MAX_TILT:
        raise ValueError(f'Tilt 서보각은 0~{MAX_TILT}° 정수입니다.')
    check = crc8(DISCRIMINATOR, pan, tilt)
    # A final bright bit keeps idle darkness from completing a truncated zero tail.
    return f'{SYNC}{pan:08b}{tilt:08b}{check:08b}1'


class AngleDecoder(VoltageFrameDecoder):
    def __init__(self):
        super().__init__(BIT_MS, SYNC, FRAME_LEN, sample_s=DECODE_SAMPLE_MS / 1000)

    def decode_word(self, word, start):
        if not word & 1:
            return None
        pan, tilt, check = (word >> 1).to_bytes(3, 'big')
        if pan > MAX_PAN or tilt > MAX_TILT or check != crc8(DISCRIMINATOR, pan, tilt):
            return None
        return pan, tilt


def validate_jog(pan_step, tilt_step):
    """Allow only small whole-degree relative moves for the manual Rx check."""
    if any(type(step) is not int or step not in (-5, -1, 0, 1, 5)
           for step in (pan_step, tilt_step)) or (pan_step, tilt_step) == (0, 0):
        raise ValueError('수동 점검은 한 축 이상을 ±1° 또는 ±5° 움직일 수 있습니다.')
