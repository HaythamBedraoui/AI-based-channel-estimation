import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../WP1_data_generator_python'))

from interpolate import interpolate
from functions import qamdemod

class ChannelEstimator:
    def __init__(self):
        self.interpolator = Interpolator()
    
    def estimate(self, Y, Xp, pilot_pos, Nfft):
        LS_est = np.zeros(len(pilot_pos), dtype=complex)
        for k in range(len(pilot_pos)):
            LS_est[k] = Y[pilot_pos[k]] / Xp[k]
        
        pilot_pos_1based = np.array(pilot_pos) + 1
        H_est = self.interpolator.interpolate(LS_est, pilot_pos_1based, Nfft)
        
        return H_est

class Interpolator:
    def __init__(self, method='linear'):
        self.method = method
    
    def interpolate(self, LS_est, pilot_pos_1based, Nfft):
        H_est = interpolate(LS_est, pilot_pos_1based, Nfft, self.method)
        return H_est


class Equalizer:
    def __init__(self):
        pass
    
    def equalize(self, Y, H_est):
        Y_eq = Y / H_est
        return Y_eq


class DataProcessor: # for demodulation 
    def __init__(self):
        pass
    
    def process(self, Y_eq, pilot_pos, Nfft, M):
        Data_extracted = []
        for k in range(Nfft):
            if k not in pilot_pos:
                Data_extracted.append(Y_eq[k])
        Data_extracted = np.array(Data_extracted)

        rx_bits = qamdemod(Data_extracted, M, mapping='Gray', outputtype='bit', unitaveragepow=True)
        
        return rx_bits


class BERCalculator:
    def __init__(self):
        pass
    
    def calculate(self, rx_bits, tx_bits):
        bit_errors = np.sum(rx_bits != tx_bits)
        BER = bit_errors / len(tx_bits)
        return BER


class BCELossCalculator:
    def __init__(self):
        pass

    def calculate(self, predicted_probs, true_bits):
        eps = 1e-12
        probs = np.clip(predicted_probs, eps, 1 - eps)
        bce = -np.mean(true_bits * np.log(probs) + (1 - true_bits) * np.log(1 - probs))
        return bce


class ReceiverChain:
    
    def __init__(self):
        self.channel_estimator = ChannelEstimator()
        self.equalizer = Equalizer()
        self.data_processor = DataProcessor()
        self.ber_calculator = BERCalculator()
        self.bce_calculator = BCELossCalculator()
    
    def process(self, Y, Xp, pilot_pos, Nfft, M, tx_bits):
        # Channel estimation
        H_est = self.channel_estimator.estimate(Y, Xp, pilot_pos, Nfft)
        
        # Equalization
        Y_eq = self.equalizer.equalize(Y, H_est)
        
        # Data processing
        rx_bits = self.data_processor.process(Y_eq, pilot_pos, Nfft, M)
        
        # BER calculation
        BER = self.ber_calculator.calculate(rx_bits, tx_bits)
        
        return rx_bits, BER, H_est
    
    def calculate_bce_loss(self, predicted_probs, true_bits):
        return self.bce_calculator.calculate(predicted_probs, true_bits)

def rx_chain(Y, Xp, pilot_pos, Nfft, M, tx_bits):
    receiver = ReceiverChain()
    return receiver.process(Y, Xp, pilot_pos, Nfft, M, tx_bits) 