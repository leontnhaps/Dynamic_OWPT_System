"""Host behavior test for the actual sketch; not an AVR build/hardware test."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKETCH = ROOT / "Rx/Arduino/pv_voltage_test/pv_voltage_test.ino"

ARDUINO = r'''
#pragma once
#include <cstdint>
#include <string>
#include <cstring>
#define F(x) x
const int A2=16, INPUT=0, LOW=0, INTERNAL=3;
static unsigned long clockMs=0;
static int analogValue=100;
inline unsigned long millis() { return clockMs; }
inline int analogRead(int) { return analogValue; }
inline void delay(unsigned long n) { clockMs+=n; }
inline void pinMode(int,int) {}
inline void digitalWrite(int,int) {}
inline void analogReference(int) {}
struct SerialStub {
  std::string input, output;
  int credit=64;
  void begin(int) {}
  int available() { return input.size(); }
  int read() { char c=input.front(); input.erase(0,1); return c; }
  int availableForWrite() { return credit; }
  void write(uint8_t c) { output+=char(c); }
  void println(const char* s) { output+=s; output+='\n'; }
} Serial;
'''

CHECKS = r'''
#include <cassert>
#include <cmath>
#include <iostream>
#include "sketch.ino"
void feed(int raw, int n) {
  for(int i=0;i<n;++i) { clockMs+=10; sampleDetector(raw,clockMs); }
}
void baseline() {
  startDetector(clockMs); feed(100,100);
  assert(detectorState==WAIT_RISE && baselineValid && !onValid);
  assert(std::fabs(offAdc-100)<0.001);
}
int main() {
  baseline();
  feed(197,1); // 97 ADC counts are less than 0.6V PV.
  assert(detectorState==WAIT_RISE);
  baseline(); feed(198,10); // 98 counts cross the configured threshold.
  assert(detectorState==TRACK && onValid);
  baseline(); feed(300,1); feed(100,1); // Spike must not acquire ON.
  assert(detectorState==WAIT_RISE && !onValid);
  feed(300,10);
  assert(detectorState==TRACK && highLevel);
  assert(std::fabs(thresholdAdc-200)<0.001);
  feed(300,1000); // Sustained ON must not become ambient.
  assert(std::fabs(offAdc-100)<0.001);
  feed(199,1); assert(highLevel); // Hysteresis around midpoint.
  feed(190,1); assert(!highLevel);
  feed(201,1); assert(!highLevel);
  feed(210,1); assert(highLevel);
  for(int i=0;i<4;++i) {
    feed(100,50); assert(!highLevel && onValid);
    feed(300,50); assert(highLevel && onValid);
  }
  feed(100,201);
  assert(detectorState==WAIT_RISE && !onValid);
  feed(400,10); assert(onValid && std::fabs(onAdc-400)<0.001);
  feed(1023,1); assert(detectorState==CLIPPED && !onValid && !baselineValid);
  feed(100,200); assert(detectorState==CLIPPED); // Requires explicit restart.
  baseline();
  startDetector(clockMs);
  for(int i=0;i<200;++i) feed(i%2 ? 100 : 150,1);
  assert(detectorState==BASELINE); // Unstable startup is not calibration.
  baseline(); feed(110,200); assert(offAdc>100 && offAdc<110);
  // Exercise loop and a stalled UART: ADC must still run.
  startDetector(clockMs); Serial.credit=0; analogValue=100;
  for(int i=0;i<100;++i) { clockMs+=10; loop(); }
  assert(detectorState==WAIT_RISE);
  analogValue=300;
  for(int i=0;i<5;++i) { clockMs+=10; loop(); }
  assert(detectorState==ACQUIRE_ON);
  clockMs+=100; loop(); // Gap invalidates consecutive acquisition samples.
  assert(!onValid);
  for(int i=0;i<9;++i) { clockMs+=10; loop(); }
  assert(onValid);
  // stop must work even with a pending display and retain its response.
  Serial.input="stop\r\n"; loop(); assert(!streaming && detectorState==IDLE);
  Serial.credit=64;
  for(int i=0;i<20;++i) loop();
  assert(Serial.output.find("OK stopped")!=std::string::npos);
  Serial.input="start\n"; loop(); assert(streaming && !onValid);
  for(int i=0;i<20;++i) loop();
  Serial.input="stop extra\n"; loop(); assert(streaming);
  for(int i=0;i<20;++i) loop();
  assert(Serial.output.find("ERR unexpected argument")!=std::string::npos);
  // Voltage formatting is compatible with AVR's non-float printf.
  baseline(); feed(300,10);
  outputLength=outputPosition=0; queueReading(300,clockMs);
  assert(std::strstr(outputBuffer,"pv_V=1.837"));
  assert(std::strstr(outputBuffer,"off_V=0.612"));
  assert(std::strstr(outputBuffer,"level=HIGH"));
  assert(outputLength<sizeof(outputBuffer)-1);
  std::cout << "PV acquisition, noise, levels, rearm, clipping, commands and UART checks passed\n";
}
'''


class PvVoltageTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "requires host g++")
    def test_sketch_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            work = pathlib.Path(folder)
            (work / "Arduino.h").write_text(ARDUINO)
            (work / "sketch.ino").write_text(SKETCH.read_text())
            (work / "check.cpp").write_text(CHECKS)
            subprocess.run(["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
                            "-I", str(work), str(work / "check.cpp"),
                            "-o", str(work / "check")], check=True)
            subprocess.run([str(work / "check")], check=True)


if __name__ == "__main__":
    unittest.main()
