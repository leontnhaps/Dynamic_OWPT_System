"""Read the Pi OOK speed analysis for display in the station experiment tab."""
import csv
import json
import math
from pathlib import Path
from typing import NamedTuple

from common.ook_experiment import PAYLOAD_BITS, PI_DECODER, PI_RECEIVER, PI_TRIALS_PER_RATE, RULES


SPEED_COLUMNS = ('Tx ms/bit', '요청 bit/s', 'Rx 표본 ms', '시도/계획', 'Tx 완료',
                 'Tx 미완료', 'Rx 일치/Tx 완료', '동기 실패', '오류/판정 비트', '판정 누락 비트', 'BER (%)')
_COUNTS = ('planned', 'not_started', 'incomplete_tx', 'evaluated_blocks', 'correct_blocks',
           'sync_failed', 'decided_bits', 'error_bits', 'erased_bits')


class _SpeedCondition(NamedTuple):
    bitrate: float
    sample_ms: int
    planned: int
    attempted: int
    full_success: bool
    display: tuple


def _rate_row(row):
    period = float(row['bit_ms'])
    bitrate = float(row['bitrate_bps'])
    sample_ms = int(row['sample_ms'])
    if not (math.isfinite(period) and period > 0 and sample_ms > 0 and
            math.isclose(bitrate, 1000 / period, rel_tol=1e-7)):
        raise ValueError('속도 분석의 비트 시간 또는 Rx 표본 간격이 잘못되었습니다.')
    counts = [int(row[key]) for key in _COUNTS]
    planned, not_started, incomplete, completed, correct, sync_failed, decided, errors, erased = counts
    attempted = planned - not_started
    if (min(counts) < 0 or planned == 0 or incomplete + completed != attempted
            or correct + sync_failed > completed or errors > decided
            or decided + erased != completed * PAYLOAD_BITS
            or correct * PAYLOAD_BITS > decided - errors):
        raise ValueError('속도 분석의 프레임 횟수 또는 비트 집계가 서로 맞지 않습니다.')
    display = (f'{period:g}', f'{bitrate:.2f}'.rstrip('0').rstrip('.'), str(sample_ms),
               f'{attempted}/{planned}', str(completed), str(incomplete), f'{correct}/{completed}',
               str(sync_failed), f'{errors}/{decided}', str(erased),
               f'{100 * errors / decided:.4f}'.rstrip('0').rstrip('.') if decided else '—')
    return _SpeedCondition(bitrate, sample_ms, planned, attempted,
                           correct == planned == PI_TRIALS_PER_RATE, display)


def load_speed_results(analysis_dir):
    """Return (headline, table rows); bit/s is the Tx setting of the measured run."""
    directory = Path(analysis_dir)
    with (directory / 'provenance.json').open(encoding='utf-8-sig') as stream:
        meta = json.load(stream)
    if (not isinstance(meta, dict) or meta.get('receiver') != PI_RECEIVER
            or meta.get('decoder_version') != PI_DECODER or meta.get('source_rules') != RULES):
        raise ValueError('현재 Pi + INA219 OOK 분석 형식의 폴더를 선택하세요.')
    with (directory / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set((*_COUNTS, 'bit_ms', 'bitrate_bps', 'sample_ms')).issubset(reader.fieldnames or ()):
            raise ValueError('속도 분석 요약에 필요한 열이 없습니다.')
        try:
            rows = [_rate_row(row) for row in reader]
        except (TypeError, ValueError) as exc:
            raise ValueError(f'속도 분석 요약을 읽을 수 없습니다. {exc}') from exc
    if not rows:
        raise ValueError('분석에 속도별 측정 결과가 없습니다.')
    if (len({(r.bitrate, r.sample_ms) for r in rows}) != len(rows)
            or sum(r.planned for r in rows) != meta.get('planned_trials')
            or sum(r.attempted for r in rows) != meta.get('saved_trials')):
        raise ValueError('속도별 집계와 분석 출처의 시험 횟수가 서로 맞지 않습니다.')
    rows.sort(key=lambda r: (r.sample_ms, r.bitrate))
    passed = [r for r in rows if r.full_success]
    if passed:
        fastest = max(passed, key=lambda r: r.bitrate)
        samples = ', '.join(str(r.sample_ms) for r in passed if r.bitrate == fastest.bitrate)
        headline = (f'30/30 프레임이 모두 일치한 최고 시험 속도 {fastest.display[1]} bit/s '
                    f'(Tx {fastest.display[0]} ms/bit · Rx {samples} ms)')
    else:
        headline = '이번 측정에서 30/30 프레임이 모두 일치한 속도 조건이 없습니다.'
    return headline, [r.display for r in rows]
