# AI-based Channel Estimation - Step-by-Step Usage Guide

This guide provides step-by-step instructions for running all components of the AI-based channel estimation project.

## Prerequisites

Before running any code, ensure you have the following dependencies installed:

```bash
pip install numpy scipy matplotlib torch
```

## Step 1: Data Generation (WP1)

### 1.1 Generate Data using Python Implementation

```bash
cd WP1_data_generator/WP1_data_generator_python
python data_generator.py
```

This will generate:
- `h_perfect.npy` - Perfect channel response
- `h_ls_estimation.npy` - LS channel estimation
- `x_long.npy` - Transmitted signal
- `y_long.npy` - Received signal

**Note:** The data generator creates a `python_output` directory in the parent folder with the generated files.

### 1.2 Generate Data using MATLAB Implementation (Optional)

If you have MATLAB installed:

```bash
cd WP1_data_generator/WP1_data_generator_matlab
matlab -batch "data_generator"
```

This will generate:
- `h_perfect.mat` - Perfect channel response
- `h_ls_estimation.mat` - LS channel estimation
- `x_long.mat` - Transmitted signal
- `y_long.mat` - Received signal

**Note:** The MATLAB data generator creates a `matlab_output` directory in the parent folder.

### 1.3 Validate Data Generation

To compare MATLAB and Python outputs:

```bash
cd WP1_data_generator
python validation.py
```

This will:
- Load both MATLAB and Python generated data
- Compare statistics (mean, std, correlation)
- Generate a comparison plot saved as `matlab_python_comparison.png`
- Display validation results in the console

## Step 2: Test Receiver Chain with NumPy (WP2)

### 2.1 Run Receiver Chain Test

```bash
cd WP2_rx_chain_numpy
python rx_test.py
```

This will:
- Load the generated data from WP1
- Test the receiver chain across different SNR values (10, 15, 20, 25, 30 dB)
- Calculate BER for each SNR
- Generate channel estimation comparison plots
- Save results in the `results/` directory

### 2.2 View Results

The test generates several result files:
- `BERvsSNR.pdf` - Overall BER vs SNR performance
- `results_Nps2/`, `results_Nps4/`, `results_Nps8/` - Results for different pilot spacings
- Channel estimation plots for each modulation scheme and SNR

## Step 3: Train and Test PyTorch Models (WP3)

### 3.1 Stage 1: Train Channel Estimator

```bash
cd WP3_rx_chain_torch/stage1
python train_channel_estimator.py
```

This will:
- Train the channel estimator for 15 epochs
- Use different pilot spacings (2, 4, 8)
- Train across various SNR levels (10-30 dB)
- Save the trained model

### 3.2 Evaluate Stage 1

```bash
cd WP3_rx_chain_torch/stage1
python evaluate_stage1.py
```

This will:
- Load the trained channel estimator
- Evaluate performance on test data
- Generate comparison plots

### 3.3 Stage 2: Train Equalizer

```bash
cd WP3_rx_chain_torch/stage2
python train_equalizer.py
```

This will:
- Train the equalizer using the trained channel estimator
- Focus on equalization performance
- Save the trained equalizer

### 3.4 Evaluate Stage 2

```bash
cd WP3_rx_chain_torch/stage2
python evaluate_stage2.py
```

This will:
- Load the trained equalizer
- Evaluate equalization performance
- Generate comparison plots

### 3.5 Stage 3: Joint Training

```bash
cd WP3_rx_chain_torch/stage3
python train_joint.py
```

This will:
- Perform joint training of channel estimator and equalizer
- Optimize both components together
- Save the jointly trained model

### 3.6 Evaluate Stage 3

```bash
cd WP3_rx_chain_torch/stage3
python evaluate_stage3.py
```

This will:
- Load the jointly trained model
- Evaluate overall system performance
- Generate comprehensive comparison plots

## Step 4: View Final Results

All training results are saved in the respective stage directories:

- **Stage 1 Results**: `WP3_rx_chain_torch/stage1/`
- **Stage 2 Results**: `WP3_rx_chain_torch/stage2/`
- **Stage 3 Results**: `WP3_rx_chain_torch/stage3/`

The evaluation scripts generate comparison plots showing:
- Channel estimation accuracy
- Equalization performance
- Overall system performance

## Complete Workflow Example

Here's the complete sequence to run the entire project:

```bash
# 1. Generate data
cd WP1_data_generator/WP1_data_generator_python
python data_generator.py

# 2. Validate data (optional)
cd ..
python validation.py

# 3. Test NumPy implementation
cd ../WP2_rx_chain_numpy
python rx_test.py

# 4. Train PyTorch models
cd ../WP3_rx_chain_torch/stage1
python train_channel_estimator.py
python evaluate_stage1.py

cd ../stage2
python train_equalizer.py
python evaluate_stage2.py

cd ../stage3
python train_joint.py
python evaluate_stage3.py
```

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure you're in the correct directory when running scripts
2. **Data Not Found**: Make sure to run the data generator first
3. **CUDA Errors**: The PyTorch models default to CPU. Modify device settings if needed
4. **Memory Issues**: Reduce batch sizes or number of epochs in training scripts

### File Structure Requirements:

```
AI based channel estimation/
├── WP1_data_generator/
│   ├── python_output/          # Generated by data_generator.py
│   └── matlab_output/          # Generated by MATLAB (optional)
├── WP2_rx_chain_numpy/
│   └── results/                # Generated by rx_test.py
└── WP3_rx_chain_torch/
    ├── stage1/                 # Training results
    ├── stage2/                 # Training results
    └── stage3/                 # Training results
```

## Expected Outputs

After running all components, you should have:
- Generated channel data files
- BER vs SNR performance plots
- Channel estimation comparison plots
- Trained PyTorch models
- Performance evaluation results

All results are automatically saved in their respective directories for later analysis.
