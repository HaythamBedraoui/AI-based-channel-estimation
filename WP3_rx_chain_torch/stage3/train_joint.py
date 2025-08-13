import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ROOT = os.path.abspath(ROOT)
WP1_PY = os.path.join(ROOT, 'WP1_data_generator', 'WP1_data_generator_python')
RX_TORCH_PATH = os.path.join(ROOT, 'WP3_rx_chain_torch')
sys.path.append(WP1_PY)

from functions import qammod, awgn

sys.path.append(RX_TORCH_PATH)
from rx_chain_torch import ReceiverChain


def complex_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a - b
    return torch.mean(diff.real.pow(2) + diff.imag.pow(2))


def build_ofdm_symbol(Nfft: int, Nps: int, tx_bits: np.ndarray, Xp: np.ndarray):
    X = np.zeros(Nfft, dtype=complex)
    pilot_pos = []
    data_idx = 0
    
    for k in range(Nfft):
        if k % Nps == 0:
            X[k] = Xp[k // Nps]
            pilot_pos.append(k)
        else:
            X[k] = qammod(tx_bits[2*data_idx:2*data_idx+2], 4, mapping='gray',
                          inputtype='bit', unitaveragepow=True)
            data_idx += 1
    
    return X, pilot_pos


def main():
    device = torch.device('cpu')
    
    Nfft = 256
    Nps = 8
    Np = Nfft // Nps
    Nd = Nfft - Np
    
    receiver = ReceiverChain(num_pilots=Np, Nfft=Nfft)
    receiver.to(device)
    receiver.train()
    
    ckpt_dir = os.path.join(RX_TORCH_PATH, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, 'stage2_final_checkpoint.pth')
    if os.path.exists(ckpt_path):
        receiver.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    for param in receiver.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(receiver.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    bce = nn.BCEWithLogitsLoss()
    
    num_epochs = 15
    steps_per_epoch = 1200
    snr_list = [10, 15, 20, 25, 30]
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_ch = 0.0
        
        for step in range(steps_per_epoch):
            SNR = float(np.random.choice(snr_list))
            
            h_taps = np.random.exponential(0.5, 4)
            h_taps[0] = 1.0
            h_taps = h_taps / np.linalg.norm(h_taps)
            h_taps = h_taps * (0.8 + 0.4 * np.random.rand(4))
            H_true_np = np.fft.fft(h_taps, Nfft)
            
            rng = np.random.RandomState(step + epoch + 1)
            tx_bits = rng.randint(0, 2, 2 * Nd).astype(np.float32)
            Xp = np.ones(Np)
            
            X, pilot_pos = build_ofdm_symbol(Nfft, Nps, tx_bits.astype(int), Xp)
            
            Y = X * H_true_np
            Y_noisy = awgn(Y, SNR, measured='measured')
            
            Y_t = torch.tensor(Y_noisy, dtype=torch.complex64, device=device)
            Xp_t = torch.tensor(Xp, dtype=torch.complex64, device=device)
            pilot_pos_t = torch.tensor(pilot_pos, dtype=torch.long, device=device)
            tx_bits_t = torch.tensor(tx_bits, dtype=torch.float32, device=device)
            H_true_t = torch.tensor(H_true_np, dtype=torch.complex64, device=device)
            
            optimizer.zero_grad(set_to_none=True)
            
            logits, _, H_est = receiver.process(Y_t, Xp_t, pilot_pos_t, Nfft, 4, tx_bits_t)
            
            loss_bce = bce(logits, tx_bits_t)
            loss_ch = complex_mse(H_est, H_true_t)
            
            loss = loss_bce + 0.1 * loss_ch
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += float(loss.item())
            epoch_bce += float(loss_bce.item())
            epoch_ch += float(loss_ch.item())
        
        final_epoch_loss = epoch_loss / steps_per_epoch
        final_epoch_bce = epoch_bce / steps_per_epoch
        final_epoch_ch = epoch_ch / steps_per_epoch
        
        scheduler.step(final_epoch_loss)
        
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(receiver.state_dict(), os.path.join(ckpt_dir, f'stage3_joint_epoch{epoch+1}.pt'))
    
    torch.save(receiver.state_dict(), os.path.join(ckpt_dir, 'stage3_final_checkpoint.pth'))


if __name__ == '__main__':
    main()


