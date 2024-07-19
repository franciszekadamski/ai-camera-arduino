// const int LED_PIN = 13;
const int SERVO_PIN = 9;
volatile int servo_angle = 90;
String inputString = "";

void write_servo(int angle) {
  int pulse_width = map(angle, 0, 180, 480, 2400);

  // digitalWrite(LED_PIN, HIGH); // control led
  digitalWrite(SERVO_PIN, HIGH);
  delayMicroseconds(pulse_width);
  digitalWrite(SERVO_PIN, LOW);
  delayMicroseconds(20000 - pulse_width);
  // digitalWrite(LED_PIN, LOW); // control led
}

void setup() {
  Serial.begin(9600);
  pinMode(SERVO_PIN, OUTPUT);
}

void loop() {
  while (Serial.available() > 0) {
    String in_message = Serial.readStringUntil('\n');
    if(in_message.length() > 0) {
      servo_angle = in_message.toInt();
    }
  }

  write_servo(servo_angle);
}
