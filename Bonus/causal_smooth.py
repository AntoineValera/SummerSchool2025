import numpy as np
import matplotlib.pyplot as plt

# Parameters - use bigger sigma for clearer visualization
fs, sigma_sec = 100, 0.15  
sigma_samples = max(1, int(round(sigma_sec * fs)))

# Create kernels
def make_causal_kernel(sigma_samples):
    t = np.arange(4 * sigma_samples + 1)
    k = np.exp(-0.5 * (t / sigma_samples) ** 2)
    return -t, k / np.sum(k)

def make_noncausal_kernel(sigma_samples):
    size = sigma_samples * 8 + 1
    t = np.linspace(-4 * sigma_samples, 4 * sigma_samples, size)
    k = np.exp(-0.5 * (t / sigma_samples) ** 2)
    return t, k / np.sum(k)

# Generate kernels
t_causal, k_causal = make_causal_kernel(sigma_samples)
t_noncausal, k_noncausal = make_noncausal_kernel(sigma_samples)

# Create brief pulse (easier to see than impulse)
step = np.zeros(300)
step[150:155] = 1  # pulse from sample 150-154

# Create triangular peak same size as square pulse
triangle = np.zeros(300)
# Make triangle same width as square pulse (5 samples: 150-154)
triangle[148] = 0.5
triangle[149] = 0.75  
triangle[150] = 1.0    # peak at same location
triangle[151] = 0.75
triangle[152] = 0.5

# Apply filters using corrected causal implementation
def smooth_trace_causal(trace, fs, sigma_sec=1.0):
    sigma_samples = max(1, int(round(sigma_sec * fs)))
    t = np.arange(4 * sigma_samples + 1)
    kernel = np.exp(-0.5 * (t / sigma_samples) ** 2)
    kernel = kernel / np.sum(kernel)
    # Pad with zeros at start, then use 'valid' convolution
    # (np.convolve mode='same' centers kernel = not causal!)
    padded = np.concatenate([np.zeros(len(kernel)-1), trace])
    return np.convolve(padded, kernel, mode='valid')

def smooth_trace_noncausal(trace, fs, sigma_sec=1.0):
    sigma_samples = max(1, int(round(sigma_sec * fs)))
    kernel_size = sigma_samples * 8 + 1
    t = np.linspace(-4 * sigma_samples, 4 * sigma_samples, kernel_size)
    kernel = np.exp(-0.5 * (t / sigma_samples) ** 2)
    return np.convolve(trace, kernel / np.sum(kernel), mode='same')

step_causal = smooth_trace_causal(step, fs, sigma_sec)
step_noncausal = smooth_trace_noncausal(step, fs, sigma_sec)

triangle_causal = smooth_trace_causal(triangle, fs, sigma_sec)
triangle_noncausal = smooth_trace_noncausal(triangle, fs, sigma_sec)

# Plot with better visibility
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# Kernels - fill areas for better visibility
ax1.fill_between(t_noncausal, k_noncausal, alpha=0.3, color='blue', label='Non-causal (uses future)')
ax1.fill_between(t_causal, k_causal, alpha=0.7, color='red', label='Causal (past only)')
ax1.axvline(0, color='black', linewidth=3, label='Present moment')
ax1.set_xlabel('Time offset (samples)', fontsize=12)
ax1.set_ylabel('Kernel weight', fontsize=12) 
ax1.set_title('KERNELS: Red only looks backward!', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Step response - highlight the key difference
t_samples = np.arange(len(step))
ax2.plot(step, 'k-', linewidth=3, alpha=0.7, label='Original pulse input')
ax2.plot(step_noncausal, 'b-', linewidth=3, label='Non-causal (CHEATS - starts early!)')
ax2.plot(step_causal, 'r-', linewidth=3, label='Causal (honest - starts at pulse)')
ax2.axvline(150, color='black', linewidth=2, linestyle='--', alpha=0.8, label='Pulse location')
ax2.set_xlabel('Sample', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_title('PULSE RESPONSE: See how blue starts BEFORE the pulse!', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 300)  # 5x wider window

# Add text annotations
ax2.annotate('Non-causal starts\nrising BEFORE pulse!\n(impossible in real-time)', 
             xy=(130, 0.2), xytext=(80, 0.5),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2),
             fontsize=11, color='blue', fontweight='bold')

ax2.annotate('Causal starts\nEXACTLY at pulse\n(real-time compatible)', 
             xy=(160, 0.2), xytext=(220, 0.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=11, color='red', fontweight='bold')

# Peak response - show how causal shifts peak timing
ax3.plot(triangle, 'k-', linewidth=3, alpha=0.7, label='Original triangular peak')
ax3.plot(triangle_noncausal, 'b-', linewidth=3, label='Non-causal (preserves peak location)')
ax3.plot(triangle_causal, 'r-', linewidth=3, label='Causal (delays peak)')
ax3.axvline(150, color='black', linewidth=2, linestyle='--', alpha=0.8, label='Original peak location')

# Find and mark the shifted peak
causal_peak_idx = np.argmax(triangle_causal)
ax3.axvline(causal_peak_idx, color='red', linewidth=2, linestyle=':', alpha=0.8, label=f'Causal peak (shifted by {causal_peak_idx-150} samples)')

ax3.set_xlabel('Sample', fontsize=12)
ax3.set_ylabel('Amplitude', fontsize=12)
ax3.set_title('PEAK SHIFT: Causal filter delays peak timing!', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 300)  # same as pulse plot

# Add annotations for peak shift
ax3.annotate(f'Peak delayed by\n{causal_peak_idx-150} samples\n(group delay)', 
             xy=(causal_peak_idx, triangle_causal[causal_peak_idx]), xytext=(200, 0.3),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=11, color='red', fontweight='bold')

plt.tight_layout()
plt.show()