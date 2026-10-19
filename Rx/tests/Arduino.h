#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string>
#include <sstream>
#define F(s) s
extern unsigned long fakeMs;
inline unsigned long millis() { return fakeMs; }
struct FakeSerial {
  std::string input, output;
  void begin(unsigned long) {}
  int available() { return (int)input.size(); }
  int read() { char c = input[0]; input.erase(0, 1); return c; }
  template<typename T> void print(T v) { std::ostringstream s; s << v; output += s.str(); }
  void print(uint8_t v) { print((unsigned int)v); }
  template<typename T> void println(T v) { print(v); output += "\n"; }
};
extern FakeSerial Serial;
#define LOW 0
#define HIGH 1
#define INPUT 0
#define OUTPUT 1
void pinMode(uint8_t pin, uint8_t mode);
void digitalWrite(uint8_t pin, uint8_t value);
int digitalRead(uint8_t pin);
unsigned long micros();
void delayMicroseconds(unsigned int us);

// Arduino AVR standard variant analog pin aliases.
static const uint8_t A0 = 14;
static const uint8_t A1 = 15;
