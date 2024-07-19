import serial
import time
import threading


class SerialInterface:
    def __init__(
            self,
            verbose=False,
            baud_rate=9600,
            serial_port='/dev/ttyACM0',
            timeout=1
        ):
        self.verbose = verbose
        try:
            self.ser = serial.Serial(
                serial_port,
                baud_rate,
                timeout=timeout
            )
        except serial.SerialException as e:
            print(f"Exception when initializing serial: {e}")
            raise

    def send(self, angle):
        try:
            self.ser.write(f"{angle}\n".encode())
            if self.verbose:
                print(f'Sending: {angle}')
        except serial.SerialException as e:
            print(f"Serial Exception: {e}")

    def run_sequence(self, sequence):
        sequence.insert(0, 0)
        previous_angle = sequence[0]
        for angle in sequence:
            self.send(angle)
            time.sleep(abs(angle - previous_angle) / 350)
            previous_angle = angle

    def __del__(self):
        self.ser.close()
