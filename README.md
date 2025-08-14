# AI-based-channel-estimation
AI-based channel estimation for wireless communications using Python and PyTorch

## Overview
This project implements an AI-based channel estimation system for wireless communications using PyTorch. It includes three main work packages:
- **WP1**: Data generation and validation (MATLAB vs Python comparison)
- **WP2**: Receiver chain implementation using NumPy
- **WP3**: Receiver chain implementation using PyTorch with trainable components

## Short Usage Guide

### Quick Start
```python
import torch
from WP3_rx_chain_torch.rx_chain_torch import ReceiverChain, rx_chain

# Initialize receiver chain
receiver = ReceiverChain(num_pilots=8, Nfft=64)

# Process received signal
rx_bits, BER, H_est = receiver.process(Y, Xp, pilot_pos, Nfft, M, tx_bits)
```

### Key Components

#### 1. Channel Estimator
```python
from WP3_rx_chain_torch.rx_chain_torch import ChannelEstimator

# Initialize with custom parameters
estimator = ChannelEstimator(
    num_pilots=16,      # Number of pilot symbols
    Nfft=256,           # FFT size
    GI=1/8,            # Guard interval ratio
    use_dft_denoise=True # Enable DFT-based denoising
)

# Estimate channel response
H_est = estimator.estimate(Y, Xp, pilot_pos, Nfft)
```

#### 2. Trainable Interpolator
```python
from WP3_rx_chain_torch.rx_chain_torch import DistanceAwareInterpolator

# Initialize interpolator
interpolator = DistanceAwareInterpolator(Nfft=64, initial_decay=0.5)

# Interpolate channel estimates
H_interp = interpolator.interpolate(LS_est, pilot_pos_1based, Nfft)
```

#### 3. Equalizer
```python
from WP3_rx_chain_torch.rx_chain_torch import Equalizer

# Initialize equalizer
equalizer = Equalizer(Nfft=256)

# Equalize received signal
Y_eq = equalizer.equalize(Y, H_est)
```

#### 4. Data Processor
```python
from WP3_rx_chain_torch.rx_chain_torch import DataProcessor

# Initialize processor
processor = DataProcessor()

# Process equalized data
rx_bits = processor.process(Y_eq, pilot_pos, Nfft, M)
```

### Training Workflow

#### Stage 1: Channel Estimator Training
```python
# Navigate to stage1 directory
cd WP3_rx_chain_torch/stage1

# Train channel estimator
python train_channel_estimator.py

# Evaluate performance
python evaluate_stage1.py
```

#### Stage 2: Equalizer Training
```python
# Navigate to stage2 directory
cd WP3_rx_chain_torch/stage2

# Train equalizer
python train_equalizer.py

# Evaluate performance
python evaluate_stage2.py
```

#### Stage 3: Joint Training
```python
# Navigate to stage3 directory
cd WP3_rx_chain_torch/stage3

# Train joint model
python train_joint.py

# Evaluate performance
python evaluate_stage3.py
```

### Configuration Parameters

#### Receiver Chain Parameters
- `num_pilots`: Number of pilot symbols (default: 8)
- `Nfft`: FFT size (default: 64)
- `GI`: Guard interval ratio (default: 1/8)

#### Channel Estimator Parameters
- `use_dft_denoise`: Enable DFT-based denoising (default: True)
- `hidden_dim`: Neural network hidden dimension (default: 128)

#### Interpolator Parameters
- `initial_decay`: Initial decay parameter for distance-aware interpolation (default: 0.5)

#### Equalizer Parameters
- `gamma`: Regularization parameter (default: 1e-2)

### Input Requirements

#### Signal Parameters
- `Y`: Received signal (complex tensor)
- `Xp`: Pilot symbols (complex tensor)
- `pilot_pos`: Pilot positions (integer tensor)
- `Nfft`: FFT size (integer)
- `M`: Modulation order (integer, e.g., 4 for QPSK)
- `tx_bits`: Transmitted bits (integer tensor)

#### Data Types
- All tensors should be PyTorch tensors
- Complex tensors for signals and channel estimates
- Integer tensors for positions and bits
- Float tensors for parameters

### Output Format
- `rx_bits`: Received bits (integer or float tensor)
- `BER`: Bit Error Rate (float)
- `H_est`: Channel estimate (complex tensor)

### Performance Metrics
- **BER vs SNR**: Bit Error Rate vs Signal-to-Noise Ratio
- **Channel Estimation Accuracy**: Mean Square Error of channel estimates
- **Constellation Diagrams**: Visual representation of received symbols

### File Structure
```
WP3_rx_chain_torch/
├── rx_chain_torch.py          # Main implementation
├── stage1/                    # Channel estimator training
├── stage2/                    # Equalizer training
├── stage3/                    # Joint training
└── results/                   # Performance results and plots
```

### Dependencies
- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- SciPy >= 1.7.0

### Installation
```bash
# Clone repository
git clone <repository-url>
cd AI-based-channel-estimation

# Install dependencies
pip install torch numpy matplotlib scipy
```

### Examples
See the `results/` directory for performance plots and the training scripts for complete training examples.

## Citation
If you use this code in your research, please cite:
```
AI-based Channel Estimation for Wireless Communications
[Your Name], [Year]
```
