import torch
import torchvision.models as models
import time
import subprocess
import psutil

# ==========================================
# CUSTOM HARDWARE TRACKERS
# ==========================================
def get_live_gpu_watts():
    """Asks Windows for the exact live wattage of the RTX 4070 Ti"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
            encoding='utf-8', 
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return float(result.strip())
    except Exception:
        return 26.0 

def get_ram_watts():
    """Uses the industry-standard CodeCarbon estimation: ~3W per 8GB of RAM"""
    try:
        # Get total system RAM in Gigabytes
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        # Calculate standard power draw
        ram_power_watts = (total_ram_gb / 8.0) * 3.0
        return ram_power_watts
    except Exception:
        # Fallback to ~12W assuming a 32GB system
        return 12.0

# ==========================================
# SETUP & WARMUP
# ==========================================
print("Loading FULL VGG-16 Model...")
vgg16 = models.vgg16(weights='DEFAULT').eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vgg16.to(device)

print(f"Model loaded to {device.upper()}.")

dummy_image = torch.randn(1, 3, 224, 224).to(device)

print("\nWarming up RTX 4070 Ti...")
with torch.no_grad():
    for _ in range(10):
        _ = vgg16(dummy_image)
        if device == 'cuda': torch.cuda.synchronize()

# ==========================================
# THE 50-RUN EVALUATION
# ==========================================
print("\nStarting Local PC Baseline (50 Runs)...")

cumulative_gpu_joules = 0.0
cumulative_ram_joules = 0.0
cumulative_time_ms = 0.0

# Calculate the constant RAM wattage once to save processing time
ram_watts = get_ram_watts()
print(f"Detected System RAM. Estimated RAM Power Draw: {ram_watts:.2f} W")

for i in range(1, 51):
    print(f"\n--- RUN {i}/50 ---")
    
    # --- MICRO-TIMER FOR FULL INFERENCE ---
    math_start = time.perf_counter()
    with torch.no_grad():
        final_output = vgg16(dummy_image)
        if device == 'cuda': torch.cuda.synchronize() 
    math_end = time.perf_counter()
    # --------------------------------------
    
    gpu_time_s = (math_end - math_start)
    gpu_time_ms = gpu_time_s * 1000
    
    live_gpu_watts = get_live_gpu_watts()
    
    # Calculate Energy: Joules = Watts * Seconds
    gpu_joules = live_gpu_watts * gpu_time_s
    ram_joules = ram_watts * gpu_time_s
    total_run_joules = gpu_joules + ram_joules
    
    cumulative_gpu_joules += gpu_joules
    cumulative_ram_joules += ram_joules
    cumulative_time_ms += gpu_time_ms
    
    print(f"Actual Math Time: {gpu_time_ms:.2f} ms")
    print(f"Power Draw -> GPU: {live_gpu_watts} W | RAM: {ram_watts:.2f} W")
    print(f"Energy -> GPU: {gpu_joules:.4f} J | RAM: {ram_joules:.4f} J")
    print(f"Total System Energy for this run: {total_run_joules:.4f} Joules")

# ==========================================
# FINAL THESIS METRICS
# ==========================================
total_system_joules = cumulative_gpu_joules + cumulative_ram_joules

print("\n" + "="*50)
print("BASELINE COMPLETE: STANDALONE RTX 4070 Ti + RAM")
print("="*50)
print(f"Average Inference Time: {cumulative_time_ms / 50:.2f} ms")
print(f"Average GPU Energy per Image: {cumulative_gpu_joules / 50:.4f} Joules")
print(f"Average RAM Energy per Image: {cumulative_ram_joules / 50:.6f} Joules")
print("-" * 50)
print(f"AVERAGE TOTAL ENERGY PER INFERENCE: {total_system_joules / 50:.4f} Joules")
print(f"TOTAL CUMULATIVE ENERGY (50 images): {total_system_joules:.4f} Joules")
print("="*50)