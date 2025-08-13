import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ROOT = os.path.abspath(ROOT)
WP1_PY = os.path.join(ROOT, 'WP1_data_generator', 'WP1_data_generator_python')
RX_TORCH_PATH = os.path.join(ROOT, 'WP3_rx_chain_torch')
sys.path.append(WP1_PY)
sys.path.append(RX_TORCH_PATH)

from rx_chain_torch import ReceiverChain
from functions import qammod, qamdemod, awgn

def generate_test_data(Nfft=256, Nps=4, SNR_dB=20, num_symbols=1000):
    tx_bits = np.random.randint(0, 2, num_symbols * Nfft * 2)
    
    X = np.zeros(Nfft, dtype=complex)
    for k in range(Nfft):
        if k % Nps == 0:
            X[k] = 1.0
        else:
            start_bit = 2 * (k - (k // Nps) * Nps)
            X[k] = qammod(tx_bits[start_bit:start_bit+2], 4, mapping='gray', 
                          inputtype='bit', unitaveragepow=True)
    
    h_true = np.array([1.0, 0.5, 0.3, 0.1])
    H_true = np.fft.fft(h_true, Nfft)
    
    Y = X * H_true
    
    Y_noisy = awgn(Y, SNR_dB, 'measured')
    
    pilot_pos = np.arange(0, Nfft, Nps)
    Xp = X[pilot_pos]
    
    return Y_noisy, Xp, pilot_pos, H_true, tx_bits, X

def classic_receiver(Y, Xp, pilot_pos, Nfft, Nps):
    LS_est = Y[pilot_pos] / Xp
    
    H_est = np.zeros(Nfft, dtype=complex)
    for i in range(Nfft):
        if i in pilot_pos:
            H_est[i] = LS_est[pilot_pos == i][0]
        else:
            left_idx = np.searchsorted(pilot_pos, i) - 1
            left_idx = max(0, min(left_idx, len(pilot_pos) - 1))
            right_idx = min(left_idx + 1, len(pilot_pos) - 1)
            
            if left_idx == right_idx:
                H_est[i] = LS_est[left_idx]
            else:
                d_left = abs(i - pilot_pos[left_idx])
                d_right = abs(pilot_pos[right_idx] - i)
                w_left = d_right / (d_left + d_right)
                w_right = d_left / (d_left + d_right)
                H_est[i] = w_left * LS_est[left_idx] + w_right * LS_est[right_idx]
    
    Y_eq = Y / H_est
    
    rx_bits = []
    for k in range(Nfft):
        if k % Nps != 0:
            real_part = Y_eq[k].real
            imag_part = Y_eq[k].imag
            
            bit0 = 1 if real_part < 0 else 0
            bit1 = 1 if imag_part < 0 else 0
            rx_bits.extend([bit0, bit1])
    
    return np.array(rx_bits), H_est

def evaluate_stage2(Y, Xp, pilot_pos, H_true, Nfft):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = '../checkpoints/stage2_final_checkpoint.pth'
    if not os.path.exists(checkpoint_path):
        return None, None
    
    receiver = ReceiverChain(num_pilots=32, Nfft=256)
    receiver.load_state_dict(torch.load(checkpoint_path, map_location=device))
    receiver.to(device)
    receiver.eval()
    
    Y_t = torch.tensor(Y, dtype=torch.complex64, device=device)
    Xp_t = torch.tensor(Xp, dtype=torch.complex64, device=device)
    pilot_pos_t = torch.tensor(pilot_pos, dtype=torch.int64, device=device)
    
    with torch.no_grad():
        H_est = receiver.channel_estimator.estimate(Y_t, Xp_t, pilot_pos_t, Nfft)
        Y_eq = receiver.equalizer.equalize(Y_t, H_est)
        rx_bits = receiver.data_processor.process(Y_eq, pilot_pos_t, Nfft, 4)
    
    rx_bits = (rx_bits > 0).cpu().numpy().astype(int)
    
    return rx_bits, H_est.cpu().numpy()

def calculate_ber(tx_bits, rx_bits):
    if len(tx_bits) != len(rx_bits):
        min_len = min(len(tx_bits), len(rx_bits))
        tx_bits = tx_bits[:min_len]
        rx_bits = rx_bits[:min_len]
    
    errors = np.sum(tx_bits != rx_bits)
    return errors / len(tx_bits)

def plot_stage2_comparison(Nps_values=[2, 4, 8], SNR_range=np.arange(10, 31, 5)):
    fig, axes = plt.subplots(len(Nps_values), 2, figsize=(15, 5*len(Nps_values)))
    if len(Nps_values) == 1:
        axes = axes.reshape(1, -1)
    
    results = {Nps: {'SNR': SNR_range, 'BER_classic': [], 'BER_ml': []} for Nps in Nps_values}
    
    for idx, Nps in enumerate(Nps_values):
        for SNR_dB in SNR_range:
            Y, Xp, pilot_pos, H_true, tx_bits, _ = generate_test_data(Nfft=256, Nps=Nps, SNR_dB=SNR_dB)
            
            rx_bits_classic, H_est_classic = classic_receiver(Y, Xp, pilot_pos, 256, Nps)
            ber_classic = calculate_ber(tx_bits, rx_bits_classic)
            
            rx_bits_ml, H_est_ml = evaluate_stage2(Y, Xp, pilot_pos, H_true, 256)
            
            if rx_bits_ml is not None:
                ber_ml = calculate_ber(tx_bits, rx_bits_ml)
            else:
                ber_ml = 1.0
            
            results[Nps]['BER_classic'].append(ber_classic)
            results[Nps]['BER_ml'].append(ber_ml)
        
        axes[idx, 0].semilogy(SNR_range, results[Nps]['BER_classic'], 'r--', linewidth=2, 
                              label='Classic LS+ZF', marker='o')
        axes[idx, 0].semilogy(SNR_range, results[Nps]['BER_ml'], 'b-', linewidth=2, 
                              label='Stage 2 (Trained)', marker='s')
        axes[idx, 0].set_title(f'BER vs SNR - Nps={Nps}')
        axes[idx, 0].set_xlabel('SNR (dB)')
        axes[idx, 0].set_ylabel('BER')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)
        axes[idx, 0].set_ylim(1e-4, 1)
        
        Y, Xp, pilot_pos, H_true, _, _ = generate_test_data(Nfft=256, Nps=Nps, SNR_dB=20)
        _, H_est_classic = classic_receiver(Y, Xp, pilot_pos, 256, Nps)
        _, H_est_ml = evaluate_stage2(Y, Xp, pilot_pos, H_true, 256)
        
        if H_est_ml is not None:
            axes[idx, 1].plot(np.abs(H_true), 'k-', linewidth=2, label='True Channel')
            axes[idx, 1].plot(np.abs(H_est_classic), 'r--', linewidth=2, label='Classic LS')
            axes[idx, 1].plot(np.abs(H_est_ml), 'b-', linewidth=2, label='Stage 2 (Trained)')
            axes[idx, 1].set_title(f'Channel Estimation - Nps={Nps}, SNR=20dB')
            axes[idx, 1].set_xlabel('Subcarrier Index')
            axes[idx, 1].set_ylabel('Magnitude')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('../results/stage2_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def main():
    os.makedirs('../results', exist_ok=True)
    plot_stage2_comparison()

if __name__ == '__main__':
    main()
