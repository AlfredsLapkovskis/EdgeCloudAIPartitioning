import torch
import torchvision.models as models
import time
import subprocess

def get_gpu_power_watts():
    """Reads live RTX 4070 Ti power draw directly from NVIDIA SMI."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        return float(result.strip())
    except Exception as e:
        print(f"Warning: Could not read GPU power: {e}")
        return 0.0

print("Loading FULL MobileNetV2 on RTX 4070 Ti...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
full_model = models.mobilenet_v2(weights='DEFAULT').eval().to(device)
dummy_image = torch.randn(1, 3, 224, 224).to(device)

print("Warming up GPU...")
with torch.no_grad():
    for _ in range(10):
        _ = full_model(dummy_image)

print("\n" + "="*50)
print("STARTING PC 500-RUN BASELINE (MOBILENET-V2)")
print("="*50)

total_latency = 0.0
total_joules = 0.0

for i in range(1, 501):
    # Read GPU power BEFORE math
    power_start = get_gpu_power_watts()
    
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = full_model(dummy_image)
    math_time = time.perf_counter() - start_time
    
    # Read GPU power AFTER math
    power_end = get_gpu_power_watts()
    
    # Average power used * time taken (Joules = Watts * Seconds)
    avg_power_watts = (power_start + power_end) / 2.0
    gpu_joules = avg_power_watts * math_time
    
    total_latency += math_time
    total_joules += gpu_joules
    
    if i % 50 == 0:
        print(f"Run {i}/500 completed... (Latency: {math_time*1000:.2f}ms)")

print("\n" + "="*50)
print("PC CLOUD CONTROL RESULTS (MOBILENET-V2)")
print("="*50)
print(f"Average Latency: {(total_latency / 500) * 1000:.2f} ms")
print(f"Average Energy:  {total_joules / 500:.4f} Joules")
print("="*50)