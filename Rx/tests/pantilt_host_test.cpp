#include <cassert>
#include <iostream>
#include "Arduino.h"
#include "Wire.h"
unsigned long fakeMs = 0;
FakeSerial Serial;
FakeWire Wire;
#include "../Arduino/pimoroni_pantilt_test/pimoroni_pantilt_test.ino"

void send(const std::string &s) {
  Serial.input += s;
  while (Serial.available()) loop();
}
int main() {
  setup();
  assert(!fault && enabledMask == 0);
  assert(Wire.writes.size() == 1);
  assert((Wire.writes.back() == std::vector<uint8_t>{0x15, 0, 0}));
  send("pan 0\r\n");
  assert(enabledMask == 1 && commandedValid[0]);
  assert((Wire.writes[2] == std::vector<uint8_t>{0x15, 1, 0xaa, 5})); // 1450us
  assert((Wire.writes[3] == std::vector<uint8_t>{0x15, 0, 1}));
  size_t before = Wire.writes.size();
  send("pan 11\npan -11\npan 1.5\npan 0 extra\npan\nfoo\noff x\n");
  send("pan 999999999999999999999999999999\n");
  send(std::string(60, 'x') + "pan 0\n");
  send(std::string("pan 0\0junk\n", 11));
  assert(Wire.writes.size() == before);
  send("tilt -10\n");
  assert(enabledMask == 2 && commandedDeg[1] == -10);
  assert((Wire.writes[before + 1] == std::vector<uint8_t>{0x15, 3, 0x48, 5})); // 1352us
  fakeMs = 1999; loop(); assert(enabledMask == 2);
  fakeMs = 2000; loop(); assert(enabledMask == 0);
  // Failure at each stage: disable, target, enable. No moves until explicit off.
  for (int stage = 0; stage < 3; ++stage) {
    for (int i = 0; i < stage; ++i) Wire.results.push_back(0);
    Wire.results.push_back(2);
    send("pan 5\n"); assert(fault);
    before = Wire.writes.size();
    send("tilt 0\n"); assert(Wire.writes.size() == before);
    send("off\n"); assert(!fault && enabledMask == 0);
  }
  send("pan 10\n");
  Wire.results = {2};
  send("off\n"); assert(fault && enabledMask == 1); // stop not confirmed
  send("off\n"); assert(!fault && enabledMask == 0);
  // Timer rollover, as on 32-bit AVR (unsigned long is wider on some hosts).
  fakeMs = ~0UL - 1000UL; send("pan 0\n");
  fakeMs = 1000; loop(); assert(enabledMask == 0);
  Wire.results = {2}; send("probe\n"); assert(fault);
  std::cout << "PASS: startup, protocol, parser, axis selection, timeout, I2C faults\n";
}
