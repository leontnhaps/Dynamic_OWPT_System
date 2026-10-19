#pragma once
#include <stdint.h>
#include <vector>
#include <deque>
// Official AVR Wire has the timeout API but no WIRE_HAS_TIMEOUT macro.
struct FakeWire {
  std::vector<std::vector<uint8_t> > writes;
  std::vector<uint8_t> current;
  std::deque<uint8_t> results;
  void begin() {}
  void setClock(unsigned long) {}
  void setWireTimeout(unsigned long, bool) {}
  void beginTransmission(uint8_t address) { current = {address}; }
  void write(uint8_t value) { current.push_back(value); }
  uint8_t endTransmission() {
    writes.push_back(current);
    if (results.empty()) return 0;
    uint8_t r = results.front(); results.pop_front(); return r;
  }
};
extern FakeWire Wire;
