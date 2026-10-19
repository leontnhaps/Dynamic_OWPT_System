#include <cassert>
#include <iostream>
#include <vector>
#include "Arduino.h"
#include "../Arduino/pimoroni_pantilt_test/SoftI2c.h"
unsigned long fakeMs = 0, us = 0;
FakeSerial Serial;
uint8_t modes[6] = {};
bool active = false, stuckScl = false, stuckSda = false;
unsigned int clocks = 0, nackByte = 0;
std::vector<int> bits;
void digitalWrite(uint8_t pin, uint8_t value) { assert(pin == 4 || pin == 5); assert(value == LOW); }
void pinMode(uint8_t pin, uint8_t mode) {
  assert(pin == 4 || pin == 5);
  if (pin == 4 && modes[5] == INPUT) {
    if (mode == OUTPUT && modes[4] == INPUT) { active = true; clocks = 0; }
    if (mode == INPUT && modes[4] == OUTPUT) active = false;
  }
  if (pin == 5 && mode == INPUT && modes[5] == OUTPUT && active) {
    ++clocks;
    if (clocks % 9 != 0) bits.push_back(modes[4] == INPUT);
  }
  modes[pin] = mode;
}
int digitalRead(uint8_t pin) {
  if (pin == 5) return stuckScl ? LOW : HIGH;
  if (stuckSda) return LOW;
  if (active && clocks && clocks % 9 == 0)
    return nackByte == clocks / 9 ? HIGH : LOW;
  return modes[4] == INPUT ? HIGH : LOW;
}
unsigned long micros() { return us++; }
void delayMicroseconds(unsigned int n) { us += n; }
int main() {
  SoftI2c bus(4,5); bus.begin();
  bus.beginTransmission(0x15); bus.write(1); bus.write(0xaa); bus.write(5);
  assert(bus.endTransmission() == 0);
  const int expected[] = {0x2a, 1, 0xaa, 5};
  for (int b=0; b<4; ++b) {
    int v=0; for (int i=0; i<8; ++i) v=(v<<1)|bits[b*8+i];
    assert(v == expected[b]);
  }
  assert(modes[4] == INPUT && modes[5] == INPUT);
  nackByte=1; bus.beginTransmission(0x15); assert(bus.endTransmission()==2);
  nackByte=2; bus.beginTransmission(0x15); bus.write(0); assert(bus.endTransmission()==3);
  nackByte=0; stuckSda=true; bus.beginTransmission(0x15); assert(bus.endTransmission()==4);
  stuckSda=false; stuckScl=true; unsigned long before=us;
  assert(bus.endTransmission()==5); assert(us-before < 25100);
  assert(modes[4] == INPUT && modes[5] == INPUT);
  stuckScl=false; bus.beginTransmission(0x15);
  for(int i=0;i<4;++i) bus.write(0);
  assert(bus.endTransmission()==1);
  std::cout << "PASS: D4/D5, open drain, address/data bytes, ACK/NACK, stuck bus, timeout, overflow\n";
}
