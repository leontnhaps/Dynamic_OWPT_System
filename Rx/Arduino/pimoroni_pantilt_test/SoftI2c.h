#pragma once
#include <Arduino.h>

// Single-master, write-only I2C for the HAT. No external library required.
// Open drain: drive LOW or release to INPUT, never drive HIGH.
// Fit separate 4.7k resistors from SDA and SCL to MCU 3.3V.
// 10us half-period plus GPIO overhead: <=50kHz, not a measured bus rate.
class SoftI2c {
 public:
  SoftI2c(uint8_t sda, uint8_t scl) : sda_(sda), scl_(scl) {}
  void begin() { release(sda_); release(scl_); }
  void beginTransmission(uint8_t address) {
    address_ = address; length_ = 0; overflow_ = false;
  }
  void write(uint8_t value) {
    if (length_ < sizeof(data_)) data_[length_++] = value;
    else overflow_ = true;
  }
  // Compatible result codes: 0=OK, 1=overflow, 2=address NACK,
  // 3=data NACK, 4=bus stuck, 5=transaction timeout.
  uint8_t endTransmission() {
    if (overflow_) return 1;
    started_ = micros();
    release(sda_);
    if (!clockHigh()) { release(scl_); return 5; }
    if (digitalRead(sda_) == LOW) { release(scl_); return 4; }
    low(sda_); pause(); low(scl_);
    uint8_t result = sendByte((uint8_t)(address_ << 1), 2);
    for (uint8_t i = 0; i < length_ && result == 0; ++i)
      result = sendByte(data_[i], 3);
    // STOP. Even on a failed ACK attempt to release the bus.
    low(scl_); low(sda_); pause();
    bool stopped = clockHigh();
    release(sda_); pause(); release(scl_);
    if (!stopped) return 5;
    if (digitalRead(sda_) == LOW) return 4;
    return result;
  }
 private:
  uint8_t sda_, scl_, address_ = 0, length_ = 0, data_[3];
  bool overflow_ = false;
  unsigned long started_ = 0;
  static void pause() { delayMicroseconds(10); }
  static void release(uint8_t pin) { pinMode(pin, INPUT); digitalWrite(pin, LOW); }
  static void low(uint8_t pin) { digitalWrite(pin, LOW); pinMode(pin, OUTPUT); }
  bool clockHigh() {
    release(scl_);
    do {
      if ((unsigned long)(micros() - started_) >= 25000UL) return false;
    } while (digitalRead(scl_) == LOW);
    pause();
    return true;
  }
  uint8_t sendByte(uint8_t value, uint8_t nack) {
    for (uint8_t mask = 0x80; mask; mask >>= 1) {
      if (value & mask) release(sda_); else low(sda_);
      pause();
      if (!clockHigh()) return 5;
      low(scl_); pause();
    }
    release(sda_); pause();
    if (!clockHigh()) return 5;
    bool ack = digitalRead(sda_) == LOW;
    low(scl_); pause();
    return ack ? 0 : nack;
  }
};
