import random
import pandas as pd

class RasPiDummy:
    @staticmethod
    def readMCP3008_AI(iByte, iBit):
        return random.randint(300, 900)  # Simulierter ADC-Wert

class Sensor:
    def __init__(self, ad_bit):
        self.ad_bit = ad_bit
    
    def measure(self):
        return RasPiDummy.readMCP3008_AI(0, self.ad_bit)

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
