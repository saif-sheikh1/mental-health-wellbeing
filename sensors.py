import threading
import time
import serial
import collections
import numpy as np

class SensorData:
    def __init__(self):
        self.gsr = 0.0
        self.ppg = 0.0
        self.delta = 0.0
        self.theta = 0.0
        self.low_alpha = 0.0
        self.high_alpha = 0.0
        self.low_beta = 0.0
        self.high_beta = 0.0
        self.low_gamma = 0.0
        self.mid_gamma = 0.0
        
    def to_list(self):
        return [
            self.gsr, self.ppg, self.delta, self.theta,
            self.low_alpha, self.high_alpha, self.low_beta,
            self.high_beta, self.low_gamma, self.mid_gamma
        ]
        
    def to_dict(self):
        return {
            "gsr": self.gsr, "ppg": self.ppg, "delta": self.delta, "theta": self.theta,
            "low_alpha": self.low_alpha, "high_alpha": self.high_alpha, "low_beta": self.low_beta,
            "high_beta": self.high_beta, "low_gamma": self.low_gamma, "mid_gamma": self.mid_gamma
        }

class ArduinoReader(threading.Thread):
    def __init__(self, port="COM3", baud=9600):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = False
        self.connected = False
        self.data_ref = None
        self.ppg_window = collections.deque(maxlen=100)
        self.last_bpm = 75.0
        
    def run(self):
        self.running = True
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            self.connected = True
            print(f"Connected to Arduino on {self.port}")
            
            while self.running:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        # Expecting comma separated values: GSR_RAW, PPG_RAW
                        parts = line.split(',')
                        if len(parts) >= 2:
                            raw_gsr = float(parts[0])
                            raw_ppg = float(parts[1])
                            
                            # Convert GSR (0-1023) to dataset range (0-40952)
                            scaled_gsr = (raw_gsr / 1023.0) * 40952.0
                            
                            # PPG peak detection (simplistic)
                            self.ppg_window.append(raw_ppg)
                            bpm = self._calculate_bpm()
                            
                            if self.data_ref:
                                self.data_ref.gsr = scaled_gsr
                                self.data_ref.ppg = bpm
                except Exception:
                    pass
                time.sleep(0.01)
                
            ser.close()
        except Exception as e:
            print(f"Arduino connection failed: {e}")
            # If no arduino, we'll run a simulation for demonstration
            self._simulate()
            
    def _calculate_bpm(self):
        if len(self.ppg_window) < 50:
            return self.last_bpm
            
        # Very simple peak detection for demonstration
        data = np.array(self.ppg_window)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0: return self.last_bpm
        
        threshold = mean + std
        peaks = 0
        for i in range(1, len(data) - 1):
            if data[i] > threshold and data[i] > data[i-1] and data[i] > data[i+1]:
                peaks += 1
                
        # Assume 100 samples ~ 2 seconds (50Hz)
        # peaks per 2 seconds -> multiply by 30 to get BPM
        calculated_bpm = peaks * 30
        
        # Clamp to reasonable values
        calculated_bpm = max(40.0, min(200.0, float(calculated_bpm)))
        
        # Smooth with previous
        self.last_bpm = 0.8 * self.last_bpm + 0.2 * calculated_bpm
        return self.last_bpm
        
    def _simulate(self):
        print(f"Simulating Arduino data (could not open {self.port})...")
        while self.running:
            if self.data_ref:
                # Add some random walk to a base value
                self.data_ref.gsr = max(0, min(40952, self.data_ref.gsr + np.random.normal(0, 500)))
                if self.data_ref.gsr == 0: self.data_ref.gsr = 5000
                
                self.data_ref.ppg = max(60, min(100, self.data_ref.ppg + np.random.normal(0, 1)))
                if self.data_ref.ppg == 0: self.data_ref.ppg = 75
            time.sleep(0.5)

    def stop(self):
        self.running = False


class EEGReader(threading.Thread):
    def __init__(self, port="COM15", baud=57600):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = False
        self.connected = False
        self.data_ref = None
        
    def _normalize(self, val, max_val):
        return max(0.0, min(1.0, val / max_val))
        
    def run(self):
        self.running = True
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            self.connected = True
            print(f"Connected to TGAM EEG on {self.port}")
            
            while self.running:
                # Based on the TGAM protocol from eeg.py
                b = ser.read(3)
                if len(b) == 3 and b[0] == 170 and b[1] == 170 and b[2] == 32:
                    # Found large packet with EEG powers
                    payload = ser.read(32)
                    if len(payload) == 32:
                        # Extract the 8 bands (each is 3 bytes, big-endian)
                        # Offset varies slightly by packet format, 
                        # using typical TGAM offsets for ASIC_EEG_POWER
                        
                        # Just a safe extraction avoiding out of bounds
                        try:
                            # Delta, Theta, LowAlpha, HighAlpha, LowBeta, HighBeta, LowGamma, MidGamma
                            # Typical max values to normalize against (from dataset structure)
                            delta = (payload[0] << 16) | (payload[1] << 8) | payload[2]
                            theta = (payload[3] << 16) | (payload[4] << 8) | payload[5]
                            la = (payload[6] << 16) | (payload[7] << 8) | payload[8]
                            ha = (payload[9] << 16) | (payload[10] << 8) | payload[11]
                            lb = (payload[12] << 16) | (payload[13] << 8) | payload[14]
                            hb = (payload[15] << 16) | (payload[16] << 8) | payload[17]
                            lg = (payload[18] << 16) | (payload[19] << 8) | payload[20]
                            mg = (payload[21] << 16) | (payload[22] << 8) | payload[23]
                            
                            if self.data_ref:
                                self.data_ref.delta = self._normalize(delta, 1000000)
                                self.data_ref.theta = self._normalize(theta, 500000)
                                self.data_ref.low_alpha = self._normalize(la, 300000)
                                self.data_ref.high_alpha = self._normalize(ha, 300000)
                                self.data_ref.low_beta = self._normalize(lb, 300000)
                                self.data_ref.high_beta = self._normalize(hb, 300000)
                                self.data_ref.low_gamma = self._normalize(lg, 300000)
                                self.data_ref.mid_gamma = self._normalize(mg, 300000)
                        except IndexError:
                            pass
                time.sleep(0.01)
                
            ser.close()
        except Exception as e:
            print(f"TGAM EEG connection failed: {e}")
            self._simulate()
            
    def _simulate(self):
        print(f"Simulating EEG data (could not open {self.port})...")
        while self.running:
            if self.data_ref:
                self.data_ref.delta = max(0, min(1.0, self.data_ref.delta + np.random.normal(0, 0.05)))
                self.data_ref.theta = max(0, min(1.0, self.data_ref.theta + np.random.normal(0, 0.05)))
                self.data_ref.low_alpha = max(0, min(1.0, self.data_ref.low_alpha + np.random.normal(0, 0.05)))
                self.data_ref.high_alpha = max(0, min(1.0, self.data_ref.high_alpha + np.random.normal(0, 0.05)))
                self.data_ref.low_beta = max(0, min(1.0, self.data_ref.low_beta + np.random.normal(0, 0.05)))
                self.data_ref.high_beta = max(0, min(1.0, self.data_ref.high_beta + np.random.normal(0, 0.05)))
                self.data_ref.low_gamma = max(0, min(1.0, self.data_ref.low_gamma + np.random.normal(0, 0.05)))
                self.data_ref.mid_gamma = max(0, min(1.0, self.data_ref.mid_gamma + np.random.normal(0, 0.05)))
                
                # initial seeds if 0
                if self.data_ref.delta == 0:
                    self.data_ref.delta = 0.3
                    self.data_ref.theta = 0.25
                    self.data_ref.low_alpha = 0.4
                    self.data_ref.high_alpha = 0.35
                    self.data_ref.low_beta = 0.2
                    self.data_ref.high_beta = 0.18
                    self.data_ref.low_gamma = 0.12
                    self.data_ref.mid_gamma = 0.1
            time.sleep(1.0)
            
    def stop(self):
        self.running = False
