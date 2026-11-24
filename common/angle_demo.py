"""M6-4 single-frame demonstration for the receiver's two servo channels."""

from common.arc_protocol import VoltageFrameDecoder, crc8


HOME_PAN = 103
HOME_TILT = 20
MAX_PAN = 180
MAX_TILT = 35
BIT_MS = 50
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
        super().__init__(BIT_MS, SYNC, FRAME_LEN)

    def decode_word(self, word, start):
        if not word & 1:
            return None
        pan, tilt, check = (word >> 1).to_bytes(3, 'big')
        if pan > MAX_PAN or tilt > MAX_TILT or check != crc8(DISCRIMINATOR, pan, tilt):
            return None
        return pan, tilt
