// M4-4: Pro Mini ATmega328P 3.3V / 8MHz; serial 9600 baud.
// PV+ -- 47k -- A2 -- 10k -- GND; PV- -- GND; no 100-ohm load.
// The 57k divider loads the PV: voltages are near-open-circuit estimates.
// A0/A1 stay reserved for the gimbal. No servo, EEPROM or frame decoding.
// Ambient acquisition is inspired by IEEE EDSSC 2012,
// doi:10.1109/EDSSC.2012.6482814. Automatic ON acquisition, midpoint,
// 0.6V rise and timing below are OUR initial test settings, not its algorithm.
#include <Arduino.h>
#include <string.h>
#include <stdio.h>

const uint8_t PV_PIN = A2;
const float ADC_REFERENCE_V = 1.1f;
const float DIVIDER_GAIN = (47000.0f + 10000.0f) / 10000.0f;
const float PV_VOLTS_PER_COUNT = ADC_REFERENCE_V * DIVIDER_GAIN / 1024.0f;
const float RISE_PV_V = 0.6f; // Rise above ambient, NOT absolute PV voltage.
const float RISE_COUNTS = RISE_PV_V / PV_VOLTS_PER_COUNT;
const float STABLE_COUNTS = 0.12f / PV_VOLTS_PER_COUNT;
const float HYSTERESIS_COUNTS = 0.03f / PV_VOLTS_PER_COUNT;
const unsigned long SAMPLE_INTERVAL_MS = 10;
const unsigned long PRINT_INTERVAL_MS = 250;
const uint8_t BASELINE_SAMPLES = 100; // About 1s with laser OFF.
const uint8_t ON_SAMPLES = 10;        // About 100ms stable, elevated signal.
const unsigned long REARM_LOW_MS = 2000;
const float BASELINE_ALPHA = 0.002f; // Slow tracking ONLY while waiting.

enum DetectorState { IDLE, BASELINE, WAIT_RISE, ACQUIRE_ON, TRACK, CLIPPED };
DetectorState detectorState = IDLE;
bool streaming = false;
bool baselineValid = false;
bool onValid = false;
bool highLevel = false;
bool lowTimerActive = false;
float offAdc = 0.0f;
float onAdc = 0.0f;
float thresholdAdc = 0.0f;
uint32_t windowSum = 0;
uint8_t windowCount = 0;
int windowMin = 1023;
int windowMax = 0;
int latestAdc = 0;
unsigned long lowSinceMs = 0;
unsigned long lastSampleMs = 0;
unsigned long latestSampleMs = 0;
unsigned long lastPrintMs = 0;
char commandBuffer[24];
uint8_t commandLength = 0;
bool invalidCommand = false;
// Stream without waiting for the UART to send an entire status line.
char outputBuffer[224];
uint8_t outputLength = 0;
uint8_t outputPosition = 0;
char pendingMessage[80] = "";

void resetWindow() {
  windowSum = 0;
  windowCount = 0;
  windowMin = 1023;
  windowMax = 0;
}

void addToWindow(int raw) {
  windowSum += raw;
  ++windowCount;
  if (raw < windowMin) windowMin = raw;
  if (raw > windowMax) windowMax = raw;
}

void startDetector(unsigned long now) {
  detectorState = BASELINE;
  streaming = true;
  baselineValid = onValid = highLevel = lowTimerActive = false;
  offAdc = onAdc = thresholdAdc = 0.0f;
  resetWindow();
  lastSampleMs = lastPrintMs = now;
}

void sampleDetector(int raw, unsigned long now) {
  latestAdc = raw;
  latestSampleMs = now;
  if (!streaming || detectorState == CLIPPED) return;
  if (raw >= 1023) {
    detectorState = CLIPPED; // Latched: correct illumination, then start again.
    baselineValid = onValid = highLevel = lowTimerActive = false;
    resetWindow();
    return;
  }
  if (detectorState == BASELINE) {
    addToWindow(raw);
    if (windowMax - windowMin > STABLE_COUNTS) {
      resetWindow(); // Require a stable OFF window; keep reporting BASELINE.
      addToWindow(raw);
    }
    if (windowCount >= BASELINE_SAMPLES) {
      offAdc = (float)windowSum / windowCount;
      baselineValid = true;
      detectorState = WAIT_RISE;
      resetWindow();
    }
    return;
  }
  if (detectorState == WAIT_RISE) {
    // Compare BEFORE adaptation so a step is not absorbed into ambient.
    if (raw - offAdc >= RISE_COUNTS) {
      detectorState = ACQUIRE_ON;
      resetWindow();
      addToWindow(raw);
    } else {
      offAdc += BASELINE_ALPHA * (raw - offAdc);
    }
    return;
  }
  if (detectorState == ACQUIRE_ON) {
    // Ambient stays frozen throughout candidate/ON acquisition.
    if (raw - offAdc < RISE_COUNTS) {
      detectorState = WAIT_RISE;
      resetWindow();
      return;
    }
    addToWindow(raw);
    if (windowMax - windowMin > STABLE_COUNTS) {
      resetWindow();
      addToWindow(raw);
    }
    if (windowCount >= ON_SAMPLES) {
      onAdc = (float)windowSum / windowCount;
      thresholdAdc = (offAdc + onAdc) * 0.5f;
      onValid = highLevel = true;
      lowTimerActive = false;
      detectorState = TRACK;
      resetWindow();
    }
    return;
  }
  if (detectorState == TRACK) {
    // Levels are frozen during the burst, including consecutive HIGHs/LOWs.
    if (highLevel && raw < thresholdAdc - HYSTERESIS_COUNTS) highLevel = false;
    else if (!highLevel && raw > thresholdAdc + HYSTERESIS_COUNTS) highLevel = true;
    if (highLevel) lowTimerActive = false;
    else {
      if (!lowTimerActive) {
        lowTimerActive = true;
        lowSinceMs = now;
      }
      if ((unsigned long)(now - lowSinceMs) >= REARM_LOW_MS) {
        // Test-only rearm rule, NOT a finalized packet/end-of-frame rule.
        onValid = false;
        onAdc = thresholdAdc = 0.0f;
        lowTimerActive = false;
        detectorState = WAIT_RISE;
      }
    }
  }
}

const char *stateName() {
  switch (detectorState) {
    case BASELINE: return "BASELINE";
    case WAIT_RISE: return "WAIT";
    case ACQUIRE_ON: return "RISE";
    case TRACK: return "TRACK";
    case CLIPPED: return "CLIPPED";
    default: return "IDLE";
  }
}

void formatVoltage(char *dest, float counts, float voltsPerCount) {
  unsigned int mv = (unsigned int)(counts * voltsPerCount * 1000.0f + 0.5f);
  snprintf(dest, 12, "%u.%03u", mv / 1000, mv % 1000);
}

void queueMessage(const char *message) {
  // Finish an older line first; retain the latest command response.
  if (outputPosition < outputLength) {
    snprintf(pendingMessage, sizeof(pendingMessage), "%s", message);
    return;
  }
  snprintf(outputBuffer, sizeof(outputBuffer), "%s\n", message);
  outputLength = strlen(outputBuffer);
  outputPosition = 0;
}

void queueReading(int raw, unsigned long now) {
  if (outputPosition < outputLength) return; // Drop display, never ADC work.
  char a2[12] = "NA", pv[12] = "NA", off[12] = "NA";
  char on[12] = "NA", threshold[12] = "NA";
  if (raw < 1023) {
    formatVoltage(a2, raw, ADC_REFERENCE_V / 1024.0f);
    formatVoltage(pv, raw, PV_VOLTS_PER_COUNT);
  }
  if (baselineValid) formatVoltage(off, offAdc, PV_VOLTS_PER_COUNT);
  if (onValid) {
    formatVoltage(on, onAdc, PV_VOLTS_PER_COUNT);
    formatVoltage(threshold, thresholdAdc, PV_VOLTS_PER_COUNT);
  }
  const char *level = onValid ? (highLevel ? "HIGH" : "LOW") : "NA";
  snprintf(outputBuffer, sizeof(outputBuffer),
           "ms=%lu adc=%d a2_V=%s pv_V=%s state=%s off_V=%s on_V=%s th_V=%s level=%s%s\n",
           now, raw, a2, pv, stateName(), off, on, threshold, level,
           raw >= 1023 ? " CLIPPED" : "");
  outputLength = strlen(outputBuffer);
  outputPosition = 0;
}

void drainOutput() {
  if (outputPosition >= outputLength && pendingMessage[0]) {
    queueMessage(pendingMessage);
    pendingMessage[0] = '\0';
  }
  int room = Serial.availableForWrite();
  while (room-- > 0 && outputPosition < outputLength)
    Serial.write((uint8_t)outputBuffer[outputPosition++]);
}

void printHelp() {
  Serial.println(F("M4-4 PV voltage + automatic level test; A2; 9600 baud; newline"));
  Serial.println(F("47k/10k divider; no 100 ohm load; INTERNAL nominal 1.1V"));
  Serial.println(F("help | pv | start | stop"));
  Serial.println(F("start: keep laser OFF until WAIT (about 1s stable baseline)."));
  Serial.println(F("Then aim laser: +0.600V PV rise, about 100ms stable ON acquisition."));
  Serial.println(F("ADC 10ms; display 250ms; off/on/th are PV voltage estimates."));
  Serial.println(F("LOW for 2s rearms; start resets all levels. CLIPPED requires restart."));
  Serial.println(F("HIGH/LOW are light levels, NOT decoded bits or laser identification."));
  Serial.println(F("No frame sync, CRC, EEPROM or servo control. Stop before help."));
}

void handleCommand() {
  commandBuffer[commandLength] = '\0';
  char *cmd = strtok(commandBuffer, " \t");
  if (!cmd) return;
  if (strtok(NULL, " \t")) {
    queueMessage("ERR unexpected argument; use help");
    return;
  }
  if (!strcmp(cmd, "help")) {
    if (streaming || outputPosition < outputLength || pendingMessage[0]) queueMessage("ERR stop and wait before help");
    else printHelp(); // Long, blocking text only while sampling is stopped.
  } else if (!strcmp(cmd, "pv")) {
    if (streaming) queueReading(latestAdc, latestSampleMs);
    else queueReading(analogRead(PV_PIN), millis());
  } else if (!strcmp(cmd, "start")) {
    startDetector(millis());
    queueMessage("OK BASELINE: laser OFF; wait for state=WAIT");
  } else if (!strcmp(cmd, "stop")) {
    streaming = false;
    detectorState = IDLE;
    baselineValid = onValid = highLevel = lowTimerActive = false;
    queueMessage("OK stopped");
  } else queueMessage("ERR unknown command; use help");
}

void setup() {
  Serial.begin(9600);
  pinMode(PV_PIN, INPUT);
  digitalWrite(PV_PIN, LOW);
  analogReference(INTERNAL); // Do not drive AREF externally.
  analogRead(PV_PIN);
  delay(10);
  analogRead(PV_PIN);
  printHelp();
}

void loop() {
  unsigned long now = millis();
  if (streaming && (unsigned long)(now - lastSampleMs) >= SAMPLE_INTERVAL_MS) {
    unsigned long gap = now - lastSampleMs;
    lastSampleMs = now; // No catch-up samples with invented timestamps.
    if (gap > 2 * SAMPLE_INTERVAL_MS) {
      // A sampling gap must not count as consecutive calibration samples.
      if (detectorState == BASELINE) resetWindow();
      else if (detectorState == ACQUIRE_ON) {
        resetWindow();
        detectorState = WAIT_RISE;
      }
      lowTimerActive = false;
    }
    sampleDetector(analogRead(PV_PIN), now);
  }
  for (uint8_t i = 0; i < 16 && Serial.available(); ++i) {
    const char c = (char)Serial.read();
    if (c == '\r' || c == '\n') {
      if (invalidCommand) queueMessage("ERR line too long or invalid");
      else handleCommand();
      commandLength = 0;
      invalidCommand = false;
    } else if ((c < 32 && c != '\t') || c > 126) invalidCommand = true;
    else if (!invalidCommand) {
      if (commandLength < sizeof(commandBuffer) - 1) commandBuffer[commandLength++] = c;
      else invalidCommand = true;
    }
  }
  now = millis();
  if (streaming && (unsigned long)(now - lastPrintMs) >= PRINT_INTERVAL_MS) {
    lastPrintMs = now;
    if (!pendingMessage[0]) queueReading(latestAdc, latestSampleMs);
  }
  drainOutput();
}
