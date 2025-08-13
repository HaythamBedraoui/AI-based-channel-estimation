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

def generate_test_data(Nfft=256, Nps=8, SNR_dB=20, num_symbols=1000):
    Nd = Nfft - (Nfft // Nps)
    tx_bits = np.random.randint(0, 2, num_symbols * Nd * 2)
    
    X = np.zeros(Nfft, dtype=complex)
    pilot_pos = []
    data_idx = 0
    
    for k in range(Nfft):
        if k % Nps == 0:
            X[k] = 1.0
            pilot_pos.append(k)
        else:
            if data_idx < len(tx_bits) - 1:
                X[k] = qammod(tx_bits[data_idx:data_idx+2], 4, mapping='gray', 
                              inputtype='bit', unitaveragepow=True)
                data_idx += 2
    
    h_taps = np.random.exponential(0.5, 4)
    h_taps[0] = 1.0
    h_taps = h_taps / np.linalg.norm(h_taps)
    h_taps = h_taps * (0.8 + 0.4 * np.random.rand(4))
    H_true = np.fft.fft(h_taps, Nfft)
    
    Y = X * H_true
    Y_noisy = awgn(Y, SNR_dB, 'measured')
    
    Xp = X[pilot_pos]
    
    return Y_noisy, Xp, pilot_pos, H_true, tx_bits, X

def classic_receiver(Y, Xp, pilot_pos, Nfft, Nps):
    LS_est = Y[pilot_pos] / Xp
    
    H_est = np.zeros(Nfft, dtype=complex)
    for i in range(Nfft):
        if i in pilot_pos:
            H_est[i] = LS_est[pilot_pos.index(i)]
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

def evaluate_stage3(Y, Xp, pilot_pos, H_true, Nfft):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = '../checkpoints/stage3_final_checkpoint.pth'
    if not os.path.exists(checkpoint_path):
        return None, None
    
    receiver = ReceiverChain(num_pilots=32, Nfft=256)
    receiver.load_state_dict(torch.load(checkpoint_path, map_location=device))
    receiver.to(device)
    receiver.eval()
    
    Y_t = torch.tensor(Y, dtype=torch.complex64, device=device)
    Xp_t = torch.tensor(Xp, dtype=torch.complex64, device=device)
    pilot_pos_t = torch.tensor(pilot_pos, dtype=torch.long, device=device)
    
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

def plot_stage3_comparison(Nps_values=[2, 4, 8], SNR_range=np.arange(10, 31, 2)):
    fig_ber, axes_ber = plt.subplots(1, len(Nps_values), figsize=(6*len(Nps_values), 5))
    if len(Nps_values) == 1:
        axes_ber = [axes_ber]
    
    fig_ch, axes_ch = plt.subplots(1, len(Nps_values), figsize=(6*len(Nps_values), 5))
    if len(Nps_values) == 1:
        axes_ch = [axes_ch]
    
    results = {}
    
    for idx, Nps in enumerate(Nps_values):
        results[Nps] = {
            'SNR': SNR_range,
            'BER_classic': [],
            'BER_ml': []
        }
        
        for SNR_dB in SNR_range:
            Y, Xp, pilot_pos, H_true, tx_bits, _ = generate_test_data(Nfft=256, Nps=Nps, SNR_dB=SNR_dB)
            
            rx_bits_classic, H_est_classic = classic_receiver(Y, Xp, pilot_pos, 256, Nps)
            ber_classic = calculate_ber(tx_bits, rx_bits_classic)
            
            rx_bits_ml, H_est_ml = evaluate_stage3(Y, Xp, pilot_pos, H_true, 256)
            
            if rx_bits_ml is not None:
                ber_ml = calculate_ber(tx_bits, rx_bits_ml)
            else:
                ber_ml = 1.0
            
            results[Nps]['BER_classic'].append(ber_classic)
            results[Nps]['BER_ml'].append(ber_ml)
        
        ax_ber = axes_ber[idx]
        data = results[Nps]
        SNR = data['SNR']
        BER_classic = data['BER_classic']
        BER_ml = data['BER_ml']
        
        ax_ber.semilogy(SNR, BER_classic, 'r--', linewidth=3, label='Classic LS+ZF', 
                        marker='o', markersize=8, markerfacecolor='red', markeredgecolor='darkred')
        ax_ber.semilogy(SNR, BER_ml, 'b-', linewidth=3, label='Stage 3 (Joint)', 
                        marker='s', markersize=8, markerfacecolor='blue', markeredgecolor='darkblue')
        
        ax_ber.axhline(y=1e-3, color='gray', linestyle=':', linewidth=2, alpha=0.8, label='BER = 1e-3')
        
        ax_ber.set_title(f'BER vs SNR - Pilot Spacing Nps={Nps}', fontsize=16, fontweight='bold', pad=20)
        ax_ber.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
        ax_ber.set_ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
        ax_ber.legend(fontsize=12, framealpha=0.9, loc='upper right')
        ax_ber.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        ax_ber.set_ylim(1e-4, 1)
        ax_ber.set_xlim(SNR.min(), SNR.max())
        
        ax_ber.tick_params(axis='both', which='major', labelsize=12)
        ax_ber.tick_params(axis='both', which='minor', labelsize=10)
        
        if len(idx_classic := np.where(np.array(BER_classic) < 1e-3)[0]) > 0 and len(idx_ml := np.where(np.array(BER_ml) < 1e-3)[0]) > 0:
            snr_classic = SNR[idx_classic[0]]
            snr_ml = SNR[idx_ml[0]]
            improvement = snr_classic - snr_ml
            if improvement > 0:
                ax_ber.text(0.05, 0.95, f'ML Improvement: +{improvement:.1f} dB', 
                           transform=ax_ber.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                           fontsize=11, fontweight='bold')
        
        ax_ch = axes_ch[idx]
        
        Y, Xp, pilot_pos, H_true, _, _ = generate_test_data(Nfft=256, Nps=Nps, SNR_dB=20)
        _, H_est_classic = classic_receiver(Y, Xp, pilot_pos, 256, Nps)
        _, H_est_ml = evaluate_stage3(Y, Xp, pilot_pos, H_true, 256)
        
        if H_est_ml is not None:
            freq = np.arange(256)
            
            ax_ch.plot(freq, np.abs(H_true), 'g-', linewidth=3, label='True Channel', alpha=0.9)
            ax_ch.plot(freq, np.abs(H_est_classic), 'r--', linewidth=2, label='Classic LS', alpha=0.8)
            ax_ch.plot(freq, np.abs(H_est_ml), 'b:', linewidth=2, label='Stage 3 ML', alpha=0.8)
            
            pilot_freq = np.array(pilot_pos)
            ax_ch.scatter(pilot_freq, np.abs(H_true[pilot_freq]), color='red', s=80, 
                         label='Pilot Positions', zorder=5, alpha=0.9, edgecolors='darkred', linewidth=1)
            
            ax_ch.set_title(f'Channel Estimation - Nps={Nps} (SNR=20 dB)', fontsize=16, fontweight='bold', pad=20)
            ax_ch.set_xlabel('Frequency Bin', fontsize=14, fontweight='bold')
            ax_ch.set_ylabel('|H(f)| Magnitude', fontsize=14, fontweight='bold')
            ax_ch.legend(fontsize=12, framealpha=0.9, loc='upper right')
            ax_ch.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax_ch.set_xlim(0, 256)
            
            ax_ch.tick_params(axis='both', which='major', labelsize=12)
            ax_ch.tick_params(axis='both', which='minor', labelsize=10)
            
            mse_classic = np.mean(np.abs(H_true - H_est_classic)**2)
            mse_ml = np.mean(np.abs(H_true - H_est_ml)**2)
            ax_ch.text(0.02, 0.98, f'MSE Classic: {mse_classic:.4f}\nMSE ML: {mse_ml:.4f}', 
                      transform=ax_ch.transAxes, verticalalignment='top', 
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9),
                      fontsize=11, fontweight='bold')
    
    fig_ber.tight_layout()
    fig_ber.savefig('../results/stage3_ber_comparison.png', dpi=300, bbox_inches='tight')
    
    fig_ch.tight_layout()
    fig_ch.savefig('../results/stage3_channel_comparison.png', dpi=300, bbox_inches='tight')
    
    fig_combined, axes_combined = plt.subplots(2, len(Nps_values), figsize=(6*len(Nps_values), 10))
    if len(Nps_values) == 1:
        axes_combined = axes_combined.reshape(2, 1)
    
    for idx, Nps in enumerate(Nps_values):
        ax_ber_combined = axes_combined[0, idx] if len(Nps_values) > 1 else axes_combined[0]
        ax_ber_combined.semilogy(results[Nps]['SNR'], results[Nps]['BER_classic'], 'r--', linewidth=3, 
                                 label='Classic LS+ZF', marker='o', markersize=8)
        ax_ber_combined.semilogy(results[Nps]['SNR'], results[Nps]['BER_ml'], 'b-', linewidth=3, 
                                 label='Stage 3 (Joint)', marker='s', markersize=8)
        ax_ber_combined.axhline(y=1e-3, color='gray', linestyle=':', linewidth=2, alpha=0.8, label='BER = 1e-3')
        ax_ber_combined.set_title(f'BER vs SNR - Nps={Nps}', fontsize=14, fontweight='bold')
        ax_ber_combined.set_xlabel('SNR (dB)')
        ax_ber_combined.set_ylabel('BER')
        ax_ber_combined.legend()
        ax_ber_combined.grid(True, alpha=0.3)
        ax_ber_combined.set_ylim(1e-4, 1)
        
        ax_ch_combined = axes_combined[1, idx] if len(Nps_values) > 1 else axes_combined[1]
        Y, Xp, pilot_pos, H_true, _, _ = generate_test_data(Nfft=256, Nps=Nps, SNR_dB=20)
        _, H_est_classic = classic_receiver(Y, Xp, pilot_pos, 256, Nps)
        _, H_est_ml = evaluate_stage3(Y, Xp, pilot_pos, H_true, 256)
        if H_est_ml is not None:
            freq = np.arange(256)
            ax_ch_combined.plot(freq, np.abs(H_true), 'g-', linewidth=2, label='True Channel')
            ax_ch_combined.plot(freq, np.abs(H_est_classic), 'r--', linewidth=2, label='Classic LS')
            ax_ch_combined.plot(freq, np.abs(H_est_ml), 'b:', linewidth=2, label='Stage 3 ML')
            pilot_freq = np.array(pilot_pos)
            ax_ch_combined.scatter(pilot_freq, np.abs(H_true[pilot_freq]), color='red', s=50, 
                                  label='Pilot Positions', zorder=5)
            ax_ch_combined.set_title(f'Channel Estimation - Nps={Nps}')
            ax_ch_combined.set_xlabel('Frequency Bin')
            ax_ch_combined.set_ylabel('|H(f)|')
            ax_ch_combined.legend()
            ax_ch_combined.grid(True, alpha=0.3)
            ax_ch_combined.set_xlim(0, 256)
    
    fig_combined.tight_layout()
    fig_combined.savefig('../results/stage3_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return results

def main():
    os.makedirs('../results', exist_ok=True)
    results = plot_stage3_comparison()

if __name__ == '__main__':
    main()
