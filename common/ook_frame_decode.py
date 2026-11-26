"""INA219 header-only phase fit and whole-bit decisions for a fixed OOK frame."""

import math
from bisect import bisect_left, bisect_right
from collections import deque
from itertools import accumulate

from common.ook_experiment import HEADER, PAYLOAD_BITS, TRAILING_ZEROS


HEADER_EDGES = (0, *(i for i in range(1, len(HEADER)) if HEADER[i] != HEADER[i-1]), len(HEADER))
FRAME_BITS = len(HEADER) + PAYLOAD_BITS + len(TRAILING_ZEROS)
MIN_CONTRAST_V = .024  # Six INA219 voltage counts.


def header_residual(n0, n1, y0, y1, squares, total_y):
    """Least-squares two-level fit, scalar on the Pi or vectorized in the PC analysis."""
    lo, hi = y0 / n0, y1 / n1
    sst = squares - total_y ** 2 / (n0 + n1)
    sse = squares - y0 * lo - y1 * hi
    return lo, hi, sse, sst


def bit_decisions(counts, sums, threshold):
    decisions = []
    for count, value in zip(counts, sums):
        mean = float(value / count) if count else None
        decisions.append(dict(samples=int(count), mean_v=mean,
                              decision='?' if mean is None else str(int(mean >= threshold))))
    return decisions


def fit_header(times, values, bit, lo, hi):
    """Find the best header origin in a short live window using the speed test's SSE/SST rule."""
    offsets = [i * bit for i in HEADER_EDGES]
    # A sample can change header assignment only when it crosses a run boundary.
    cuts = {lo, hi}
    for offset in offsets:
        left, right = bisect_right(times, lo + offset), bisect_left(times, hi + offset)
        cuts.update(t - offset for t in times[left:right])
    cuts = sorted(cuts)
    centered = [v-values[0] for v in values]
    prefix = list(accumulate(centered, initial=0.0))
    squares = list(accumulate((v*v for v in centered), initial=0.0))
    scores = []
    on = [i for i, edge in enumerate(HEADER_EDGES[:-1]) if HEADER[edge] == '1']
    for left, right in zip(cuts, cuts[1:]):
        origin = (left + right) / 2
        edges = [bisect_left(times, origin + offset) for offset in offsets]
        begin, end = edges[0], edges[-1]
        n1 = sum(edges[i+1] - edges[i] for i in on)
        n0 = end - begin - n1
        score = math.inf
        if n0 and n1:
            total_y = prefix[end] - prefix[begin]
            y1 = sum(prefix[edges[i+1]] - prefix[edges[i]] for i in on)
            low, high, sse, sst = header_residual(n0, n1, total_y-y1, y1,
                                                   squares[end]-squares[begin], total_y)
            if high > low and sst > 0:
                score = max(0., sse) / sst
        scores.append(score)
    best = min(scores, default=math.inf)
    if not math.isfinite(best):
        return None
    matched = [abs(score-best) <= 1e-12 + 1e-12*abs(best) for score in scores]
    first = matched.index(True)
    end = first + 1
    while end < len(matched) and matched[end]:
        end += 1
    return (cuts[first] + cuts[end]) / 2


class PayloadDecoder:
    """Stream one observed payload at a time. No expected payload enters this decoder."""

    def __init__(self, bit_ms):
        self.bit = bit_ms / 1000
        if not math.isfinite(self.bit) or self.bit <= 0:
            raise ValueError('Invalid bit period.')
        self.history = deque()
        self.next_check = -math.inf

    def feed(self, t_seconds, volts):
        if not math.isfinite(t_seconds) or not math.isfinite(volts):
            raise ValueError('Invalid INA219 time or voltage.')
        bit = self.bit
        history = self.history
        if history:
            gap = t_seconds - history[-1][0]
            if gap <= 0:
                raise ValueError('Rx timestamps must increase.')
            if gap > bit:
                history.clear()  # Do not infer bit levels across an unobserved bit.
        history.append((t_seconds, volts))
        while history[0][0] < t_seconds - (FRAME_BITS+4)*bit:
            history.popleft()
        if t_seconds < self.next_check or t_seconds - history[0][0] < (FRAME_BITS-1)*bit:
            return None
        self.next_check = t_seconds + bit
        # Work in a local clock to avoid precision loss at long uptimes.
        base = history[0][0]
        times = [t-base for t, _ in history]
        values = [v for _, v in history]
        around = times[-1] - FRAME_BITS*bit
        lo, hi = max(0., around-2*bit), around+2*bit
        # Both sample windows lie well inside the long 0 and 1 training runs for
        # every candidate. Skip the expensive fit on idle voltage.
        def plateau(a, b):
            sample = values[bisect_left(times, around+a*bit):bisect_left(times, around+b*bit)]
            return sum(sample)/len(sample) if sample else None
        dark, bright = plateau(3, 5), plateau(12, 14)
        if dark is None or bright is None or bright-dark < MIN_CONTRAST_V/2:
            return None
        origin = fit_header(times, values, bit, lo, hi)
        if origin is None or origin + FRAME_BITS*bit > times[-1]:
            return None
        edges = [bisect_left(times, origin+i*bit) for i in range(FRAME_BITS+1)]
        counts = [b-a for a, b in zip(edges, edges[1:])]
        if not all(counts):
            return None
        sums = [sum(values[a:b]) for a, b in zip(edges, edges[1:])]
        def level(symbol):
            indices = [i for i, b in enumerate(HEADER) if b == symbol]
            return sum(sums[i] for i in indices) / sum(counts[i] for i in indices)
        low, high = level('0'), level('1')
        if high - low < MIN_CONTRAST_V:
            return None
        decision = bit_decisions(counts, sums, (low + high)/2)
        observed = ''.join(cell['decision'] for cell in decision)
        if not observed.startswith(HEADER) or not observed.endswith(TRAILING_ZEROS):
            return None
        # Permit a subsequent complete frame but never report overlapping fits of this one.
        self.next_check = t_seconds + 3*bit
        return observed[len(HEADER):len(HEADER)+PAYLOAD_BITS]
