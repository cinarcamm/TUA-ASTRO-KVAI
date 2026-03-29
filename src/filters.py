import numpy as np

class SatelliteFilter:
    def __init__(self, init_x=400.0, p=1.0, q=0.1, r=10.0):
        self.x = init_x
        self.p = p
        self.q = q 
        self.r = r 

    def kalman_update(self, measurement):
        """Klasik Kalman Filtresi: Gürültüyü temizler."""
        self.p = self.p + self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        return self.x

    def simple_spike_rejection(self, measurement, threshold=100):
        """
        Basit bir koruma: Eğer veri bir anda devasa zıplarsa (Spike), 
        onu reddet ve eski tahmini döndür.
        """
        if abs(measurement - self.x) > threshold:
            return self.x, True # Anomali saptandı
        return self.kalman_update(measurement), False