// M4-1: Arduino Pro Mini ATmega328P, 3.3 V / 8 MHz.
// Protocol: pimoroni/pantilthat-python, commit a209e8eb58dfe24569fd17680957c8c7e57a219f.
// Register values and pulse conversion follow the official PanTilt driver.
// No position feedback: every reported angle is a COMMAND, not a measurement.
#include <Arduino.h>
// Software I2C: board labels A0 = SDA, A1 = SCL. External pull-ups to 3.3V.
#ifndef PANTILT_I2C
#include "SoftI2c.h"
SoftI2c hatBus(A0, A1); // 
#define PANTILT_I2C hatBus
#endif
#include <stdlib.h>
#include <string.h>
#include <errno.h>

const uint8_t HAT_ADDRESS = 0x15;
// Historical Pan sweep: -90..90. Historical standalone Tilt test: -55..55.
// Inverted UAV mounting: test only nonnegative Tilt commands.
// These are command limits, not a measured collision-free mechanical envelope.
const int AXIS_MIN_DEG[2] = {-90, 0};
const int AXIS_MAX_DEG[2] = {90, 55};
const unsigned long IDLE_MS = 2000;
char lineBuffer[40];
uint8_t lineLength = 0;
bool lineOverflow = false;
bool fault = true;
uint8_t enabledMask = 0;
int commandedDeg[2] = {0, 0};
bool commandedValid[2] = {false, false};
unsigned long lastMoveMs = 0;

bool writeRegister(uint8_t reg, uint16_t value, bool word) {
  PANTILT_I2C.beginTransmission(HAT_ADDRESS);
  PANTILT_I2C.write(reg);
  PANTILT_I2C.write((uint8_t)(value & 0xff));
  if (word) PANTILT_I2C.write((uint8_t)(value >> 8)); // SMBus word: low byte first.
  uint8_t result = PANTILT_I2C.endTransmission();
  if (result != 0) {
    Serial.print(F("ERR I2C code="));
    Serial.println(result);
  }
  return result == 0;
}

void tripFault() {
  fault = true;
  // Best effort only: a broken bus cannot guarantee that PWM stops.
  if (writeRegister(0x00, 0, false)) enabledMask = 0;
  Serial.println(F("FAULT: support PV; cut servo power if motion continues; use off to retry"));
}

bool stopOutputs() {
  if (!writeRegister(0x00, 0, false)) {
    fault = true;
    Serial.println(F("ERR stop unconfirmed; disconnect servo power"));
    return false;
  }
  enabledMask = 0;
  fault = false;
  Serial.println(F("OK PWM off (power remains on; holding torque may be lost)"));
  return true;
}

bool angleAllowed(uint8_t axis, long angle) {
  return axis < 2 && angle >= AXIS_MIN_DEG[axis] && angle <= AXIS_MAX_DEG[axis];
}

void printRange(uint8_t axis) {
  if (axis >= 2) { Serial.println(F("ERR invalid axis")); return; }
  Serial.print(axis == 0 ? F("pan_command_range_deg=") : F("tilt_command_range_deg="));
  Serial.print(AXIS_MIN_DEG[axis]); Serial.print(F("..")); Serial.println(AXIS_MAX_DEG[axis]);
}

void moveAxis(uint8_t axis, int angle) {
  if (fault) {
    Serial.println(F("ERR fault; fix wiring/power then off"));
    return;
  }
  if (!angleAllowed(axis, angle)) {
    Serial.println(F("ERR command outside axis range"));
    printRange(axis);
    return;
  }
  // Select ONE axis. Disable both before loading a new target, so an old
  // register value cannot be enabled before the requested target is written.
  if (!writeRegister(0x00, 0, false)) { tripFault(); return; }
  enabledMask = 0;
  uint16_t pulse = 575UL + (uint32_t)(angle + 90) * 1750UL / 180UL;
  if (!writeRegister(axis == 0 ? 0x01 : 0x03, pulse, true)) {
    tripFault(); return;
  }
  if (!writeRegister(0x00, (uint8_t)(1U << axis), false)) {
    tripFault(); return;
  }
  enabledMask = (uint8_t)(1U << axis);
  commandedDeg[axis] = angle;
  commandedValid[axis] = true;
  lastMoveMs = millis();
  Serial.print(F("OK commanded_axis=")); Serial.print(axis == 0 ? F("pan") : F("tilt"));
  Serial.print(F(" commanded_deg=")); Serial.print(angle);
  Serial.print(F(" pulse_us=")); Serial.println(pulse);
}

void status() {
  Serial.print(F("STATUS fault=")); Serial.print(fault);
  Serial.print(F(" last_ack_enabled_mask=")); Serial.println(enabledMask);
  for (uint8_t i = 0; i < 2; ++i) {
    printRange(i);
    Serial.print(i == 0 ? F("pan_commanded_deg=") : F("tilt_commanded_deg="));
    if (commandedValid[i]) Serial.println(commandedDeg[i]);
    else Serial.println(F("unknown"));
  }
  Serial.println(F("Actual position/arrival/power are not measured."));
}

void help() {
  Serial.println(F("M4-1 Pimoroni HAT 0x15; 9600 baud; newline"));
  Serial.println(F("Software I2C: SDA=A0, SCL=A1; pull-ups to 3.3V"));
  Serial.println(F("help | probe | status | off | pan N | tilt N"));
  printRange(0); printRange(1);
  Serial.println(F("N: integer; one axis only; PWM off after 2s"));
  Serial.println(F("Inverted mount: negative Tilt rejected; verify physical direction with tilt 5."));
  Serial.println(F("First target may move far from unknown position. Support PV."));
}

void handleLine(char *line) {
  char *cmd = strtok(line, " \t");
  if (!cmd) return;
  char *arg = strtok(NULL, " \t");
  char *extra = strtok(NULL, " \t");
  bool isPan = strcmp(cmd, "pan") == 0;
  bool isTilt = strcmp(cmd, "tilt") == 0;
  if (isPan || isTilt) {
    if (!arg || extra) { Serial.println(F("ERR usage: pan N or tilt N")); return; }
    char *end;
    errno = 0;
    long angle = strtol(arg, &end, 10);
    if (end == arg || *end != '\0' || errno == ERANGE) {
      Serial.println(F("ERR expected integer degrees")); return;
    }
    uint8_t axis = isPan ? 0 : 1;
    // Validate long before narrowing to AVR's 16-bit int.
    if (!angleAllowed(axis, angle)) {
      Serial.println(F("ERR command outside axis range"));
      printRange(axis); return;
    }
    moveAxis(axis, (int)angle);
    return;
  }
  if (arg) { Serial.println(F("ERR unexpected argument")); return; }
  if (strcmp(cmd, "help") == 0) help();
  else if (strcmp(cmd, "status") == 0) status();
  else if (strcmp(cmd, "off") == 0) stopOutputs();
  else if (strcmp(cmd, "probe") == 0) {
    PANTILT_I2C.beginTransmission(HAT_ADDRESS);
    uint8_t result = PANTILT_I2C.endTransmission();
    Serial.println(result == 0 ? F("ACK 0x15 (identity not verified)") : F("ERR no ACK at 0x15"));
    if (result != 0) tripFault();
  } else Serial.println(F("ERR unknown command; use help"));
}

void setup() {
  Serial.begin(9600);
  PANTILT_I2C.begin();
  stopOutputs(); // No home, centre, sweep, or servo enable on startup.
  help();
}

void loop() {
  if (enabledMask && !fault && (unsigned long)(millis() - lastMoveMs) >= IDLE_MS) {
    stopOutputs();
  }
  // Bounded input per iteration keeps the idle timeout responsive.
  for (uint8_t count = 0; count < 16 && Serial.available(); ++count) {
    char c = (char)Serial.read();
    if (c == '\r' || c == '\n') {
      if (lineOverflow) Serial.println(F("ERR line too long or invalid"));
      else { lineBuffer[lineLength] = '\0'; handleLine(lineBuffer); }
      lineLength = 0;
      lineOverflow = false;
    } else if ((c < 32 && c != '\t') || c > 126) {
      lineOverflow = true;
    } else if (!lineOverflow) {
      if (lineLength < sizeof(lineBuffer) - 1) lineBuffer[lineLength++] = c;
      else lineOverflow = true;
    }
  }
}
