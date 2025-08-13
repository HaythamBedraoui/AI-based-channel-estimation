import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def torch_interpolate_trainable(H_est, pilot_loc, Nfft, alpha, beta, method='linear'):
    if not isinstance(H_est, torch.Tensor):
        H_est = torch.tensor(H_est, dtype=torch.complex64)
    if not isinstance(pilot_loc, torch.Tensor):
        pilot_loc = torch.tensor(pilot_loc, dtype=torch.float32)
    
    if torch.min(pilot_loc) == 1:
        pilot_loc = pilot_loc - 1
    
    if pilot_loc[0] > 0:
        if len(pilot_loc) > 1:
            slope = (H_est[1] - H_est[0]) / (pilot_loc[1] - pilot_loc[0])
            H_est = torch.cat([H_est[0:1] - slope * pilot_loc[0:1], H_est])
            pilot_loc = torch.cat([torch.tensor([0.0], device=pilot_loc.device), pilot_loc])
        else:
            H_est = torch.cat([H_est[0:1], H_est])
            pilot_loc = torch.cat([torch.tensor([0.0], device=pilot_loc.device), pilot_loc])
    
    if pilot_loc[-1] < Nfft - 1:
        if len(pilot_loc) > 1:
            slope = (H_est[-1] - H_est[-2]) / (pilot_loc[-1] - pilot_loc[-2])
            H_est = torch.cat([H_est, H_est[-1:] + slope * (Nfft - 1 - pilot_loc[-1:])])
            pilot_loc = torch.cat([pilot_loc, torch.tensor([float(Nfft - 1)], device=pilot_loc.device)])
        else:
            H_est = torch.cat([H_est, H_est[-1:]])
            pilot_loc = torch.cat([pilot_loc, torch.tensor([float(Nfft - 1)], device=pilot_loc.device)])

    # Interpolate with trainable parameters
    H_interpolated = torch.zeros(Nfft, dtype=torch.complex64, device=H_est.device)
    
    for i in range(Nfft):
        left_idx = torch.searchsorted(pilot_loc, float(i), right=True) - 1
        left_idx = torch.clamp(left_idx, 0, len(pilot_loc) - 2)
        right_idx = left_idx + 1
        
        X0 = pilot_loc[left_idx]
        X1 = pilot_loc[right_idx]
        Y_beta = H_est[left_idx] 
        Y_alpha = H_est[right_idx]
        
        # Y^i = alpha[i] * Y_alpha + beta[i] * Y_beta
        H_interpolated[i] = alpha[i] * Y_alpha + beta[i] * Y_beta
    
    return H_interpolated

def torch_interpolate(H_est, pilot_loc, Nfft, method='linear'):
    if not isinstance(H_est, torch.Tensor):
        H_est = torch.tensor(H_est, dtype=torch.complex64)
    if not isinstance(pilot_loc, torch.Tensor):
        pilot_loc = torch.tensor(pilot_loc, dtype=torch.float32)
    
    if torch.min(pilot_loc) == 1:
        pilot_loc = pilot_loc - 1
    
    if pilot_loc[0] > 0:
        if len(pilot_loc) > 1:
            slope = (H_est[1] - H_est[0]) / (pilot_loc[1] - pilot_loc[0])
            H_est = torch.cat([H_est[0:1] - slope * pilot_loc[0:1], H_est])
            pilot_loc = torch.cat([torch.tensor([0.0], device=pilot_loc.device), pilot_loc])
        else:
            H_est = torch.cat([H_est[0:1], H_est])
            pilot_loc = torch.cat([torch.tensor([0.0], device=pilot_loc.device), pilot_loc])
    
    if pilot_loc[-1] < Nfft - 1:
        if len(pilot_loc) > 1:
            slope = (H_est[-1] - H_est[-2]) / (pilot_loc[-1] - pilot_loc[-2])
            H_est = torch.cat([H_est, H_est[-1:] + slope * (Nfft - 1 - pilot_loc[-1:])])
            pilot_loc = torch.cat([pilot_loc, torch.tensor([float(Nfft - 1)], device=pilot_loc.device)])
        else:
            H_est = torch.cat([H_est, H_est[-1:]])
            pilot_loc = torch.cat([pilot_loc, torch.tensor([float(Nfft - 1)], device=pilot_loc.device)])

    # Interpolation
    target_indices = torch.arange(Nfft, dtype=torch.float32, device=pilot_loc.device)
    if method.lower().startswith('l'):
        # Use numpy for interpolation and convert back to torch
        pilot_loc_np = pilot_loc.cpu().numpy()
        H_est_real_np = H_est.real.cpu().numpy()
        H_est_imag_np = H_est.imag.cpu().numpy()
        target_indices_np = target_indices.cpu().numpy()
        
        H_real_np = np.interp(target_indices_np, pilot_loc_np, H_est_real_np)
        H_imag_np = np.interp(target_indices_np, pilot_loc_np, H_est_imag_np)
        
        H_real = torch.tensor(H_real_np, dtype=torch.float32, device=pilot_loc.device)
        H_imag = torch.tensor(H_imag_np, dtype=torch.float32, device=pilot_loc.device)
        H_interpolated = torch.complex(H_real, H_imag)
    else:
        # if non-linear cause PyTorch doesn't have cubic interpolation
        pilot_loc_np = pilot_loc.cpu().numpy()
        H_est_real_np = H_est.real.cpu().numpy()
        H_est_imag_np = H_est.imag.cpu().numpy()
        target_indices_np = target_indices.cpu().numpy()
        
        H_real_np = np.interp(target_indices_np, pilot_loc_np, H_est_real_np)
        H_imag_np = np.interp(target_indices_np, pilot_loc_np, H_est_imag_np)
        
        H_real = torch.tensor(H_real_np, dtype=torch.float32, device=pilot_loc.device)
        H_imag = torch.tensor(H_imag_np, dtype=torch.float32, device=pilot_loc.device)
        H_interpolated = torch.complex(H_real, H_imag)
    
    return H_interpolated

def torch_qamdemod(symbols, M, mapping='Gray', outputtype='bit', unitaveragepow=True, soft=False):
    if not isinstance(symbols, torch.Tensor):
        symbols = torch.tensor(symbols, dtype=torch.complex64)
    
    if M == 4 and outputtype == 'bit':  # QPSK
        if unitaveragepow:
            symbols = symbols * torch.sqrt(torch.tensor(2.0, device=symbols.device))
            
        num_symbols = len(symbols)
        
        real_part = symbols.real
        imag_part = symbols.imag

        if soft:
            # Soft demodulation - differentiable!
            # For QPSK: bit0 corresponds to sign of real part, bit1 to sign of imaginary part
            # Use tanh for soft decision (similar to hard limiting but differentiable)
            soft_bit0 = -real_part  # Negative because bit=1 when real<0
            soft_bit1 = -imag_part  # Negative because bit=1 when imag<0
            
            soft_bits = torch.zeros(num_symbols * 2, dtype=torch.float32, device=symbols.device)
            soft_bits[0::2] = soft_bit0
            soft_bits[1::2] = soft_bit1
            
            return soft_bits
        else:
            # Hard demodulation (non-differentiable, for evaluation only)
            bits = torch.zeros(num_symbols * 2, dtype=torch.int32, device=symbols.device)
            bit0 = (real_part < 0).int()
            bit1 = (imag_part < 0).int()
            
            bits[0::2] = bit0
            bits[1::2] = bit1
            
            return bits
    else:
        raise NotImplementedError(f"QAM demodulation for M={M} not implemented")

class ChannelEstimator(nn.Module):
    def __init__(self, num_pilots=8, Nfft=64, GI: float = 1/8, use_dft_denoise: bool = True):
        super().__init__()
        self.num_pilots = num_pilots
        self.Nfft = Nfft
        self.use_dft_denoise = use_dft_denoise
        
        # Use distance-aware interpolation for better generalization
        self.interpolator = DistanceAwareInterpolator(Nfft=Nfft)
        
        # Learnable weights for pilot estimation
        self.estimation_weights = nn.Parameter(torch.ones(1), requires_grad=True)
        
        # Neural network for enhanced pilot processing
        hidden_dim = 128
        self.pilot_processor = nn.Sequential(
            nn.Linear(64, hidden_dim),  # Fixed input size, will pad/truncate as needed
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),  # Fixed output size
            nn.Tanh()  # Bounded output
        )
        
        # Learnable combination weights
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        # DFT-domain denoiser with learnable CP window
        L_cp = int(round(Nfft * GI))
        L_cp = max(1, min(L_cp, Nfft))
        self.dft_denoiser = DFTDenoiser(Nfft=Nfft, L_cp=L_cp)
    
    def estimate(self, Y, Xp, pilot_pos, Nfft):
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.complex64)
        if not isinstance(Xp, torch.Tensor):
            Xp = torch.tensor(Xp, dtype=torch.complex64)
        if not isinstance(pilot_pos, torch.Tensor):
            pilot_pos = torch.tensor(pilot_pos, dtype=torch.long)
        
        # LS estimation at pilot positions
        LS_est = torch.zeros(len(pilot_pos), dtype=torch.complex64, device=Y.device)
        for k in range(len(pilot_pos)):
            LS_est[k] = Y[pilot_pos[k]] / Xp[k]
        
        # Traditional weighted LS estimates
        # Use a single learnable weight for all pilots (more robust)
        weighted_LS = LS_est * self.estimation_weights
        
        # Neural network enhancement
        # Handle variable pilot counts by padding/truncating to fixed size
        if len(pilot_pos) <= 32:  # Max expected pilot count
            # Convert to real-valued input for neural network
            ls_input = torch.stack([LS_est.real, LS_est.imag], dim=-1).flatten()
            # Pad or truncate to fixed size
            if len(ls_input) < 64:
                padding = torch.zeros(64 - len(ls_input), device=ls_input.device)
                ls_input = torch.cat([ls_input, padding])
            else:
                ls_input = ls_input[:64]
            
            # Process through neural network
            nn_output = self.pilot_processor(ls_input)
            
            # Convert back to complex
            nn_real = nn_output[:len(pilot_pos)]
            nn_imag = nn_output[len(pilot_pos):]
            nn_enhanced = torch.complex(nn_real, nn_imag)
            
            # Combine traditional and neural estimates
            alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)
            combined_LS = alpha_clamped * weighted_LS + (1 - alpha_clamped) * nn_enhanced
        else:
            combined_LS = weighted_LS  # Fallback for different pilot counts
        
        pilot_pos_1based = pilot_pos.float() + 1
        H_interp = self.interpolator.interpolate(combined_LS, pilot_pos_1based, Nfft)
        
        if self.use_dft_denoise:
            H_est = self.dft_denoiser.denoise(H_interp)
        else:
            H_est = H_interp
        
        return H_est
    
    def estimate_adaptive(self, Y, Xp, pilot_pos, Nfft, adaptive_weights=None):
        """Enhanced estimation with adaptive weighting based on channel conditions"""
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.complex64)
        if not isinstance(Xp, torch.Tensor):
            Xp = torch.tensor(Xp, dtype=torch.complex64)
        if not isinstance(pilot_pos, torch.Tensor):
            pilot_pos = torch.tensor(pilot_pos, dtype=torch.long)
        
        LS_est = torch.zeros(len(pilot_pos), dtype=torch.complex64, device=Y.device)
        for k in range(len(pilot_pos)):
            LS_est[k] = Y[pilot_pos[k]] / Xp[k]
        
        # Apply adaptive weights if provided, otherwise use trainable weights
        if adaptive_weights is not None:
            weighted_LS = LS_est * adaptive_weights[:len(pilot_pos)]
        else:
            weighted_LS = LS_est * self.estimation_weights[:len(pilot_pos)]
        
        pilot_pos_1based = pilot_pos.float() + 1
        H_est = self.interpolator.interpolate(weighted_LS, pilot_pos_1based, Nfft)
        
        return H_est
    


class Interpolator(nn.Module):
    def __init__(self, method='linear', trainable=True, Nfft=64):
        super().__init__()
        self.method = method
        self.trainable = trainable
        self.Nfft = Nfft
        
        if self.trainable:
            self.interp_alpha = nn.Parameter(torch.ones(Nfft) * 0.5, requires_grad=True)
            self.interp_beta = nn.Parameter(torch.ones(Nfft) * 0.5, requires_grad=True)
    
    def interpolate(self, LS_est, pilot_pos_1based, Nfft):
        # Check if we need to resize - this shouldn't happen during training if initialized correctly
        if self.trainable and self.interp_alpha.size(0) != Nfft:
            raise RuntimeError(f"Interpolator initialized with {self.interp_alpha.size(0)} points but got Nfft={Nfft}. "
                             f"Initialize Interpolator with correct Nfft size.")
            
        if self.trainable:
                H_est = torch_interpolate_trainable(
                LS_est, pilot_pos_1based, Nfft, 
                self.interp_alpha, self.interp_beta,
                self.method
            )
        else:
            H_est = torch_interpolate(LS_est, pilot_pos_1based, Nfft, self.method)
        return H_est


class DistanceAwareInterpolator(nn.Module):
    """Distance-aware linear interpolation with learnable decay.

    Weights are w(d) = exp(-decay * d), normalized between the two nearest pilots.
    This keeps parameter count minimal and generalizes across pilot spacing.
    """
    def __init__(self, Nfft: int, initial_decay: float = 0.5):
        super().__init__()
        self.Nfft = Nfft
        # Parametrize decay via softplus to keep it positive
        self._decay_param = nn.Parameter(torch.tensor(np.log(np.exp(initial_decay) - 1.0), dtype=torch.float32), requires_grad=True)

    @property
    def decay(self) -> torch.Tensor:
        return F.softplus(self._decay_param)

    def interpolate(self, LS_est: torch.Tensor, pilot_pos_1based: torch.Tensor, Nfft: int) -> torch.Tensor:
        if not isinstance(LS_est, torch.Tensor):
            LS_est = torch.tensor(LS_est, dtype=torch.complex64)
        if not isinstance(pilot_pos_1based, torch.Tensor):
            pilot_pos_1based = torch.tensor(pilot_pos_1based, dtype=torch.float32)

        pilot_loc = pilot_pos_1based.to(torch.float32)
        H_est = LS_est

        # Convert to 0-based internally
        if torch.min(pilot_loc) == 1:
            pilot_loc = pilot_loc - 1

        # Endpoint handling similar to torch_interpolate: extrapolate endpoints
        if pilot_loc[0] > 0:
            if len(pilot_loc) > 1:
                slope = (H_est[1] - H_est[0]) / (pilot_loc[1] - pilot_loc[0])
                H_est = torch.cat([H_est[0:1] - slope * pilot_loc[0:1], H_est])
                pilot_loc = torch.cat([torch.tensor([0.0], device=pilot_loc.device), pilot_loc])
            else:
                H_est = torch.cat([H_est[0:1], H_est])
                pilot_loc = torch.cat([torch.tensor([0.0], device=pilot_loc.device), pilot_loc])

        if pilot_loc[-1] < Nfft - 1:
            if len(pilot_loc) > 1:
                slope = (H_est[-1] - H_est[-2]) / (pilot_loc[-1] - pilot_loc[-2])
                H_est = torch.cat([H_est, H_est[-1:] + slope * (Nfft - 1 - pilot_loc[-1:])])
                pilot_loc = torch.cat([pilot_loc, torch.tensor([float(Nfft - 1)], device=pilot_loc.device)])
            else:
                H_est = torch.cat([H_est, H_est[-1:]])
                pilot_loc = torch.cat([pilot_loc, torch.tensor([float(Nfft - 1)], device=pilot_loc.device)])

        # Interpolate with exponential distance weights
        out = torch.zeros(Nfft, dtype=torch.complex64, device=H_est.device)
        decay = self.decay
        for i in range(Nfft):
            left_idx = torch.searchsorted(pilot_loc, float(i), right=True) - 1
            left_idx = torch.clamp(left_idx, 0, len(pilot_loc) - 2)
            right_idx = left_idx + 1

            x0 = pilot_loc[left_idx]
            x1 = pilot_loc[right_idx]
            y0 = H_est[left_idx]
            y1 = H_est[right_idx]

            d_left = torch.abs(torch.tensor(i, dtype=torch.float32, device=pilot_loc.device) - x0)
            d_right = torch.abs(x1 - torch.tensor(i, dtype=torch.float32, device=pilot_loc.device))

            w_left = torch.exp(-decay * d_left)
            w_right = torch.exp(-decay * d_right)
            w_sum = w_left + w_right + 1e-12
            out[i] = (w_left * y0 + w_right * y1) / w_sum

        return out

class Equalizer(nn.Module):
    def __init__(self, Nfft=2048, hidden_dim=256):
        super().__init__()
        # MMSE-like equalizer with learnable noise regularizer (scalar)
        self.Nfft = Nfft
        self.gamma = nn.Parameter(torch.tensor(1e-2, dtype=torch.float32), requires_grad=True)
    
    def equalize(self, Y, H_est):
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.complex64)
        if not isinstance(H_est, torch.Tensor):
            H_est = torch.tensor(H_est, dtype=torch.complex64)
        
        # MMSE-like one-tap equalizer: Y_eq = Y * conj(H) / (|H|^2 + gamma)
        eps = torch.tensor(1e-8, dtype=torch.float32, device=Y.device)
        H_conj = torch.conj(H_est)
        power = (H_est.real * H_est.real + H_est.imag * H_est.imag).to(torch.float32)
        gamma_clamped = torch.clamp(self.gamma, min=0.0, max=5.0)
        denom = power + gamma_clamped + eps
        Y_eq = (Y * H_conj) / denom
        
        # Replace any NaN or Inf values with zeros
        Y_eq = torch.where(torch.isfinite(Y_eq), Y_eq, torch.zeros_like(Y_eq))
        return Y_eq
    


class DataProcessor(nn.Module): # demodulation 
    def __init__(self):
        super().__init__()
        self.training_mode = True  # Controls soft vs hard demodulation
        # Learnable global scale for soft demodulation (matches proposal)
        self.demod_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
    
    def process(self, Y_eq, pilot_pos, Nfft, M):
        if not isinstance(Y_eq, torch.Tensor):
            Y_eq = torch.tensor(Y_eq, dtype=torch.complex64)
        if not isinstance(pilot_pos, torch.Tensor):
            pilot_pos = torch.tensor(pilot_pos, dtype=torch.long)
        
        # Create mask for data positions
        data_mask = torch.ones(Nfft, dtype=torch.bool, device=Y_eq.device)
        data_mask[pilot_pos] = False
        Data_extracted = Y_eq[data_mask]

        # Use soft demodulation during training for gradient flow
        use_soft = self.training and self.training_mode
        rx_bits = torch_qamdemod(Data_extracted, M, mapping='Gray', outputtype='bit', 
                                unitaveragepow=True, soft=use_soft)
        if use_soft and isinstance(rx_bits, torch.Tensor) and rx_bits.dtype.is_floating_point:
            rx_bits = self.demod_scale * rx_bits
        
        return rx_bits

class BERCalculator(nn.Module):
    def __init__(self):
        super().__init__()
    
    def calculate(self, rx_bits, tx_bits):
        if not isinstance(rx_bits, torch.Tensor):
            rx_bits = torch.tensor(rx_bits, dtype=torch.int32)
        if not isinstance(tx_bits, torch.Tensor):
            tx_bits = torch.tensor(tx_bits, dtype=torch.int32)
        
        # Ensure both tensors are on the same device
        device = rx_bits.device if rx_bits.device.type != 'cpu' else tx_bits.device
        rx_bits = rx_bits.to(device)
        tx_bits = tx_bits.to(device)
            
        bit_errors = torch.sum(rx_bits != tx_bits).float()
        BER = bit_errors / len(tx_bits)
        return BER

class BCELossCalculator(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def calculate(self, predicted_probs, true_bits):
        # Here predicted_probs are treated as logits
        if not isinstance(predicted_probs, torch.Tensor):
            predicted_probs = torch.tensor(predicted_probs, dtype=torch.float32)
        if not isinstance(true_bits, torch.Tensor):
            true_bits = torch.tensor(true_bits, dtype=torch.float32)
        return self.loss_fn(predicted_probs, true_bits)

class ReceiverChain(nn.Module):
    
    def __init__(self, num_pilots=8, Nfft=64):
        super().__init__()
        # Use a large enough number to handle all pilot spacing options
        max_pilots = max(num_pilots, 128)  # Ensure we can handle Nps=2 (128 pilots)
        self.channel_estimator = ChannelEstimator(max_pilots, Nfft=Nfft)
        self.equalizer = Equalizer(Nfft=Nfft)
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
        # If rx_bits are logits (float), convert to hard bits for BER via threshold at 0
        if isinstance(rx_bits, torch.Tensor) and rx_bits.dtype.is_floating_point:
            hard_bits = (rx_bits > 0).to(torch.int32)
            BER = self.ber_calculator.calculate(hard_bits, tx_bits)
        else:
            BER = self.ber_calculator.calculate(rx_bits, tx_bits)
        return rx_bits, BER, H_est
    
    def calculate_bce_loss(self, predicted_probs, true_bits):
        return self.bce_calculator.calculate(predicted_probs, true_bits)

def rx_chain(Y, Xp, pilot_pos, Nfft, M, tx_bits):
    num_pilots = len(pilot_pos) if hasattr(pilot_pos, '__len__') else 8
    receiver = ReceiverChain(num_pilots, Nfft=Nfft)
    return receiver.process(Y, Xp, pilot_pos, Nfft, M, tx_bits) 


class DFTDenoiser(nn.Module):
    def __init__(self, Nfft: int, L_cp: int):
        super().__init__()
        self.Nfft = Nfft
        self.L_cp = L_cp
        # Unconstrained parameters; window = sigmoid(logits) in [0,1]
        self.window_logits = nn.Parameter(torch.zeros(L_cp, dtype=torch.float32), requires_grad=True)

    def denoise(self, H_interp: torch.Tensor) -> torch.Tensor:
        if not isinstance(H_interp, torch.Tensor):
            H_interp = torch.tensor(H_interp, dtype=torch.complex64)
        # IFFT to time domain
        h_time = torch.fft.ifft(H_interp)
        # Apply learnable CP-length window
        window = torch.sigmoid(self.window_logits)
        # Ensure correct dtype/device for complex multiplication
        window_c = window.to(h_time.device).to(torch.float32)
        h_win = torch.zeros_like(h_time)
        h_win[: self.L_cp] = h_time[: self.L_cp] * window_c
        # FFT back to frequency domain
        H_hat = torch.fft.fft(h_win)
        return H_hat
