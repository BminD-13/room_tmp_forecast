import spidev
import RPi.GPIO as GPIO
import pandas as pd
from time import time

class RasPi:
    spi0_0 = spidev.SpiDev()
    spi0_0.open(0, 0)
    spi0_0.max_speed_hz = 10000

    @staticmethod
    def readMCP3008_AI(iByte, iBit):
        if iByte == 0:
            adc = RaspiGPIO.spi0_0.xfer2([1, (8 + iBit) << 4, 0])
        wert = ((adc[1] & 3) << 8) + adc[2]
        return wert

class Sensor:
    def __init__(self, ad_bit):
        self.ad_bit = ad_bit
        self.ad_in = 0
    
    def measure(self):
        self.ad_in = RaspiGPIO.readMCP3008_AI(0, self.ad_bit)
        return self.ad_in

class TemperaturSensor(Sensor):
    def __init__(self, ad_bit=-1):
        super().__init__(ad_bit)
        self.param_tmp_scale = [
            [-30, -20, -10,  0, 10, 20, 30, 40, 50, 60, 70, 80],
            [-50,  62, 172, 300, 404, 517, 632, 743, 855, 970, 1085, 1200]
        ]
    
    def interpolate(self, value):
        x, y = self.param_tmp_scale
        for i in range(len(x) - 1):
            if y[i] <= value <= y[i + 1]:
                return x[i] + (x[i + 1] - x[i]) * ((value - y[i]) / (y[i + 1] - y[i]))
        return None

    def measure(self):
        raw_value = super().measure()
        return self.interpolate(raw_value)

def fetch_sensor_data(sensors):
    data = []
    timestamp = pd.Timestamp.utcnow().isoformat()
    for sensor in sensors:
        data.append({
            "timestamp": timestamp,
            "sensor_id": id(sensor),
            "temperature": sensor.measure()
        })
    return pd.DataFrame(data)
