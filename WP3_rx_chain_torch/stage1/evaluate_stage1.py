import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from rx_chain_torch import ReceiverChain
import os

sys.path.append('../../WP1_data_generator/WP1_data_generator_python')
from functions import qammod, qamdemod, awgn

def generate_test_data(Nfft=64, Nps=4, SNR_dB=20, num_symbols=1000):
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

def classic_ls_estimation(Y, Xp, pilot_pos, Nfft):
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
    
    return H_est

def evaluate_stage1(Y, Xp, pilot_pos, H_true, Nfft):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_dir = '../checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, 'stage1_final_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        return None
    
    receiver = ReceiverChain(num_pilots=128, Nfft=256)
    receiver.load_state_dict(torch.load(checkpoint_path, map_location=device))
    receiver.to(device)
    receiver.eval()
    
    Y_t = torch.tensor(Y, dtype=torch.complex64, device=device)
    Xp_t = torch.tensor(Xp, dtype=torch.complex64, device=device)
    pilot_pos_t = torch.tensor(pilot_pos, dtype=torch.int64, device=device)
    
    with torch.no_grad():
        H_est = receiver.channel_estimator.estimate(Y_t, Xp_t, pilot_pos_t, Nfft)
    
    return H_est.cpu().numpy()

def plot_stage1_comparison(Nps_values=[2, 4, 8], SNR_dB=20):
    fig, axes = plt.subplots(len(Nps_values), 2, figsize=(15, 5*len(Nps_values)))
    if len(Nps_values) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, Nps in enumerate(Nps_values):
        Y, Xp, pilot_pos, H_true, _, _ = generate_test_data(Nfft=256, Nps=Nps, SNR_dB=SNR_dB)
        
        H_classic = classic_ls_estimation(Y, Xp, pilot_pos, 256)
        H_trained = evaluate_stage1(Y, Xp, pilot_pos, H_true, 256)
        
        if H_trained is None:
            continue
        
        axes[idx, 0].plot(np.abs(H_true), 'k-', linewidth=2, label='True Channel')
        axes[idx, 0].plot(np.abs(H_classic), 'r--', linewidth=2, label='Classic LS')
        axes[idx, 0].plot(np.abs(H_trained), 'b-', linewidth=2, label='Stage 1 (Trained)')
        axes[idx, 0].set_title(f'Channel Magnitude - Nps={Nps}')
        axes[idx, 0].set_xlabel('Subcarrier Index')
        axes[idx, 0].set_ylabel('Magnitude')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)
        
        axes[idx, 1].plot(np.angle(H_true), 'k-', linewidth=2, label='True Channel')
        axes[idx, 1].plot(np.angle(H_classic), 'r--', linewidth=2, label='Classic LS')
        axes[idx, 1].plot(np.angle(H_trained), 'b-', linewidth=2, label='Stage 1 (Trained)')
        axes[idx, 1].set_title(f'Channel Phase - Nps={Nps}')
        axes[idx, 1].set_xlabel('Subcarrier Index')
        axes[idx, 1].set_ylabel('Phase (rad)')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('../results/stage1_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    os.makedirs('../results', exist_ok=True)
    plot_stage1_comparison()

if __name__ == '__main__':
    main()
