// M4-2: Pro Mini ATmega328P 3.3V / 8MHz, serial 9600 baud.
// PV+ -- 47k -- A2 -- 10k -- GND; separate 100-ohm load across PV.
// A0/A1 remain reserved for the gimbal. This sketch does not drive it.
// Voltage estimates use nominal reference/resistors; calibrate before evaluation.
// Serial-only wiring test: no EEPROM writes and no OOK decoding.
#include <Arduino.h>
#include <string.h>

const uint8_t PV_PIN = A2;
const float ADC_REFERENCE_V = 1.1f;
const float DIVIDER_GAIN = (47000.0f + 10000.0f) / 10000.0f;
const unsigned long PRINT_INTERVAL_MS = 250;
bool streaming = false;
unsigned long lastPrintMs = 0;
char commandBuffer[24];
uint8_t commandLength = 0;
bool invalidCommand = false;

void printHelp() {
  Serial.println(F("M4-2 PV voltage test; A2; 9600 baud; newline"));
  Serial.println(F("100 ohm load; divider 47k/10k; INTERNAL nominal 1.1V"));
  Serial.println(F("help | pv (one reading) | start (250ms) | stop"));
  Serial.println(F("Voltages are estimates; CLIPPED means out of ADC range."));
  Serial.println(F("No gimbal commands, EEPROM logging or OOK decoding."));
}

void printPv() {
  const int raw = analogRead(PV_PIN);
  Serial.print(F("ms=")); Serial.print(millis());
  Serial.print(F(" adc=")); Serial.print(raw);
  if (raw >= 1023) {
    // Do not report a saturated conversion as a valid voltage measurement.
    Serial.println(F(" a2_V=NA pv_V=NA CLIPPED"));
    return;
  }
  const float inputV = raw * (ADC_REFERENCE_V / 1024.0f);
  Serial.print(F(" a2_V=")); Serial.print(inputV, 4);
  Serial.print(F(" pv_V=")); Serial.print(inputV * DIVIDER_GAIN, 4);
  Serial.println(F(" ESTIMATE"));
}

void handleCommand() {
  commandBuffer[commandLength] = '\0';
  char *cmd = strtok(commandBuffer, " \t");
  if (!cmd) return;
  if (strtok(NULL, " \t")) {
    Serial.println(F("ERR unexpected argument; use help"));
    return;
  }
  if (!strcmp(cmd, "help")) printHelp();
  else if (!strcmp(cmd, "pv")) printPv();
  else if (!strcmp(cmd, "start")) {
    streaming = true;
    lastPrintMs = millis();
    printPv();
  } else if (!strcmp(cmd, "stop")) {
    streaming = false;
    Serial.println(F("OK stream stopped"));
  } else Serial.println(F("ERR unknown command; use help"));
}

void setup() {
  Serial.begin(9600);
  pinMode(PV_PIN, INPUT);
  digitalWrite(PV_PIN, LOW); // Ensure the input pull-up is disabled.
  analogReference(INTERNAL); // AREF must not be driven by an external supply.
  analogRead(PV_PIN);        // Apply reference selection; discard conversion.
  delay(10);                // Allow the internal reference to settle.
  analogRead(PV_PIN);        // Discard the first post-settling conversion too.
  printHelp();
}

void loop() {
  for (uint8_t i = 0; i < 16 && Serial.available(); ++i) {
    const char c = (char)Serial.read();
    if (c == '\r' || c == '\n') {
      if (invalidCommand) Serial.println(F("ERR line too long or invalid"));
      else handleCommand();
      commandLength = 0;
      invalidCommand = false;
    } else if ((c < 32 && c != '\t') || c > 126) {
      invalidCommand = true;
    } else if (!invalidCommand) {
      if (commandLength < sizeof(commandBuffer) - 1)
        commandBuffer[commandLength++] = c;
      else invalidCommand = true;
    }
  }
  if (streaming && (unsigned long)(millis() - lastPrintMs) >= PRINT_INTERVAL_MS) {
    lastPrintMs = millis(); // No catch-up bursts after delayed iterations.
    printPv();
  }
}
