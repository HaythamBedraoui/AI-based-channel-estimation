import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

matlab_dir = "../matlab_output"
python_dir = "../python_output"
    
print("Validation Script")
print("-----------------")
    
# Load data
print("Loading files...")

# Load MATLAB data
matlab_perfect = sio.loadmat(os.path.join(matlab_dir, 'h_perfect.mat'))['hEnc']
matlab_ls = sio.loadmat(os.path.join(matlab_dir, 'h_ls_estimation.mat'))['hEncNoise']
        
# Load Python data
python_perfect = np.load(os.path.join(python_dir, 'h_perfect.npy'))
python_ls = np.load(os.path.join(python_dir, 'h_ls_estimation.npy'))

print("\nChecking shapes:")
print(f"  MATLAB perfect: {matlab_perfect.shape}")
print(f"  Python perfect: {python_perfect.shape}")
print(f"  MATLAB LS est.: {matlab_ls.shape}")
print(f"  Python LS est.: {python_ls.shape}")
    
print("\nComparison between matlab and python output:")
    
# Perfect channel stats
m_perfect_mean = np.mean(np.abs(matlab_perfect))
p_perfect_mean = np.mean(np.abs(python_perfect))
m_perfect_std = np.std(np.abs(matlab_perfect))
p_perfect_std = np.std(np.abs(python_perfect))
    
# LS estimation stats
m_ls_mean = np.mean(np.abs(matlab_ls))
p_ls_mean = np.mean(np.abs(python_ls))
m_ls_std = np.std(np.abs(matlab_ls))
p_ls_std = np.std(np.abs(python_ls))
    
print(f"Perfect channel mean - MATLAB: {m_perfect_mean:.6f}, Python: {p_perfect_mean:.6f}")
print(f"Perfect channel std  - MATLAB: {m_perfect_std:.6f}, Python: {p_perfect_std:.6f}")
print(f"LS estimation mean   - MATLAB: {m_ls_mean:.6f}, Python: {p_ls_mean:.6f}")
print(f"LS estimation std    - MATLAB: {m_ls_std:.6f}, Python: {p_ls_std:.6f}")
    
print("\nCorrelation and Mean absolute error:")
    
perfect_corr = np.corrcoef(np.abs(matlab_perfect.flatten()), np.abs(python_perfect.flatten()))[0, 1]
ls_corr = np.corrcoef(np.abs(matlab_ls.flatten()), np.abs(python_ls.flatten()))[0, 1]

perfect_mae = np.mean(np.abs(matlab_perfect - python_perfect))
ls_mae = np.mean(np.abs(matlab_ls - python_ls))
    
print(f"  Perfect channel correlation: {perfect_corr:.6f}")
print(f"  LS estimation correlation: {ls_corr:.6f}")
print(f"  Perfect channel MAE: {perfect_mae:.6f}")
print(f"  LS estimation MAE: {ls_mae:.6f}")

print("\nGenerating comparison plot...")

sample_subcarrier = 0   # First subcarrier
sample_antenna = 0      # First antenna
sample_time_slots = slice(0, 500)  # First 500 time slots

# Extract sample data
matlab_perfect_sample = matlab_perfect[sample_subcarrier, sample_antenna, sample_time_slots]
python_perfect_sample = python_perfect[sample_subcarrier, sample_antenna, sample_time_slots]
matlab_ls_sample = matlab_ls[sample_subcarrier, sample_antenna, sample_time_slots]
python_ls_sample = python_ls[sample_subcarrier, sample_antenna, sample_time_slots]

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot perfect channel comparison
ax1.plot(np.abs(matlab_perfect_sample), 'b-', label='MATLAB Perfect', alpha=0.7, linewidth=1)
ax1.plot(np.abs(python_perfect_sample), 'r--', label='Python Perfect', alpha=0.7, linewidth=1)
ax1.set_title(f'Perfect Channel Comparison (Subcarrier {sample_subcarrier}, Antenna {sample_antenna})')
ax1.set_ylabel('Channel Magnitude')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot LS estimation comparison
ax2.plot(np.abs(matlab_ls_sample), 'g-', label='MATLAB LS Est.', alpha=0.7, linewidth=1)
ax2.plot(np.abs(python_ls_sample), 'm--', label='Python LS Est.', alpha=0.7, linewidth=1)
ax2.set_title(f'LS Estimation Comparison (Subcarrier {sample_subcarrier}, Antenna {sample_antenna})')
ax2.set_xlabel('Time Slot')
ax2.set_ylabel('Channel Magnitude')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('matlab_python_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved as 'matlab_python_comparison.png'")
print("\nValidation complete")
