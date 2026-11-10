"""Raw OOK test patterns. GPIO command timing is not optical timing."""
import math
import threading
import time


def pattern_settings(cmd):
    bits = cmd.get('bits')
    if not isinstance(bits, str) or not 1 <= len(bits) <= 256 or set(bits) - {'0', '1'}:
        raise ValueError('Pattern must contain 1..256 binary digits')
    ms, repeats = cmd.get('bit_ms'), cmd.get('repeats')
    # No experimental speed floor: actual scheduling is checked by run().
    if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms) or not 0 < ms <= 5000:
        raise ValueError('Raw test bit duration must be > 0 and <= 5000 ms; decimals allowed')
    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ValueError('Repeats must be 1..100')
    if len(bits) * repeats > 256:
        raise ValueError('One test must not exceed 256 bits including repeats')
    if len(bits) * repeats * ms > 120000:
        raise ValueError('One test must not exceed 120 seconds')
    return dict(bits=bits, bit_ms=ms, repeats=repeats)


class LaserPattern:
    def __init__(self, output, finished):
        self.output, self.finished = output, finished
        self.lock = threading.RLock()
        self.job = None

    def status(self):
        with self.lock:
            return dict(pattern_active=self.job is not None,
                        pattern_request_id=self.job['id'] if self.job else None)

    def manual(self, level):
        with self.lock:
            if level == 0:
                if self.job:
                    self.job['cancel'].set()
                self.output(0)
            else:
                if self.job:
                    raise ValueError('Stop the pattern before manual laser ON')
                self.output(level)

    def start(self, cmd):
        settings = pattern_settings(cmd)
        token = cmd.get('request_id')
        if not isinstance(token, str) or not token or len(token) > 128:
            raise ValueError('A request_id is required')
        with self.lock:
            if self.job:
                raise ValueError('A pattern is already running')
            self.output(0)
            job = dict(id=token, cancel=threading.Event(), settings=settings)
            self.job = job
            threading.Thread(target=self.run, args=(job,), daemon=True).start()

    def run(self, job):
        settings = job['settings']
        sequence = settings['bits'] * settings['repeats']
        period = settings['bit_ms'] / 1000
        start = time.monotonic()
        trace, reason, error = [], 'completed', None
        try:
            for index, bit in enumerate(sequence):
                due = start + index * period
                if job['cancel'].wait(max(0, due - time.monotonic())):
                    reason = 'stopped'
                    break
                with self.lock:
                    if job['cancel'].is_set():
                        reason = 'stopped'
                        break
                    if time.monotonic() - due >= period / 2:
                        reason = 'timing_overrun'
                        break  # Do not catch up by emitting compressed symbols.
                    self.output(int(bit))
                    trace.append(dict(index=index, level=int(bit),
                                      scheduled_ms=index * period * 1000,
                                      gpio_call_done_ms=(time.monotonic() - start) * 1000))
            else:
                if job['cancel'].wait(max(0, start + len(sequence) * period - time.monotonic())):
                    reason = 'stopped'
        except Exception as exc:
            reason, error = 'output_error', str(exc)
        finally:
            with self.lock:
                if reason == 'completed' and time.monotonic() - (start + len(sequence) * period) >= period / 2:
                    reason = 'timing_overrun'
                try:
                    self.output(0)
                except Exception as exc:
                    reason, error = 'off_unconfirmed', str(exc)
                off_ms = (time.monotonic() - start) * 1000
                self.job = None
            report = dict(event='laser_pattern', state=reason, request_id=job['id'],
                          settings=settings, trace=trace, off_call_done_ms=off_ms,
                          error=error, timing_source='Pi monotonic GPIO calls; no optical feedback')
            try:
                self.finished(report)
            except OSError:
                pass  # OFF already attempted even if the connection is gone.
