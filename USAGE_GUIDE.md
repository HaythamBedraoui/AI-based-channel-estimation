# AI-based Channel Estimation - Usage Guide

## Quick Start
```python
import torch
from WP3_rx_chain_torch.rx_chain_torch import ReceiverChain

# Initialize receiver chain
receiver = ReceiverChain(num_pilots=8, Nfft=64)

# Process received signal
rx_bits, BER, H_est = receiver.process(Y, Xp, pilot_pos, Nfft, M, tx_bits)
```

## Key Components

### 1. Channel Estimator
```python
from WP3_rx_chain_torch.rx_chain_torch import ChannelEstimator

estimator = ChannelEstimator(
    num_pilots=16,      # Number of pilot symbols
    Nfft=256,           # FFT size
    GI=1/8,            # Guard interval ratio
    use_dft_denoise=True # Enable DFT-based denoising
)

H_est = estimator.estimate(Y, Xp, pilot_pos, Nfft)
```

### 2. Trainable Interpolator
```python
from WP3_rx_chain_torch.rx_chain_torch import DistanceAwareInterpolator

interpolator = DistanceAwareInterpolator(Nfft=64, initial_decay=0.5)
H_interp = interpolator.interpolate(LS_est, pilot_pos_1based, Nfft)
```

### 3. Equalizer
```python
from WP3_rx_chain_torch.rx_chain_torch import Equalizer

equalizer = Equalizer(Nfft=256)
Y_eq = equalizer.equalize(Y, H_est)
```

### 4. Data Processor
```python
from WP3_rx_chain_torch.rx_chain_torch import DataProcessor

processor = DataProcessor()
rx_bits = processor.process(Y_eq, pilot_pos, Nfft, M)
```

## Training Workflow

### Stage 1: Channel Estimator
```bash
cd WP3_rx_chain_torch/stage1
python train_channel_estimator.py
python evaluate_stage1.py
```

### Stage 2: Equalizer
```bash
cd WP3_rx_chain_torch/stage2
python train_equalizer.py
python evaluate_stage2.py
```

### Stage 3: Joint Training
```bash
cd WP3_rx_chain_torch/stage3
python train_joint.py
python evaluate_stage3.py
```

## Input Parameters
- `Y`: Received signal (complex tensor)
- `Xp`: Pilot symbols (complex tensor)
- `pilot_pos`: Pilot positions (integer tensor)
- `Nfft`: FFT size (integer)
- `M`: Modulation order (integer, e.g., 4 for QPSK)
- `tx_bits`: Transmitted bits (integer tensor)

## Output
- `rx_bits`: Received bits (integer or float tensor)
- `BER`: Bit Error Rate (float)
- `H_est`: Channel estimate (complex tensor)

## Dependencies
- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- SciPy >= 1.7.0
