import torch
import torchvision.models as models
import time

PI_MAX_WATTS = 12.0 

print("Loading FULL VGG-16 on Edge CPU...")
device = torch.device('cpu')
full_model = models.vgg16(weights='DEFAULT').eval().to(device)
dummy_image = torch.randn(1, 3, 224, 224).to(device)

print("Warming up CPU...")
with torch.no_grad():
    for _ in range(5):
        _ = full_model(dummy_image)

print("\n" + "="*50)
print("STARTING PI 500-RUN BASELINE")
print("="*50)

total_latency = 0.0
total_joules = 0.0

for i in range(1, 501):
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = full_model(dummy_image)
    math_time = time.perf_counter() - start_time
    
    # Calculate Energy (Joules = Watts * Seconds)
    joules = PI_MAX_WATTS * math_time
    
    total_latency += math_time
    total_joules += joules
    
    if i % 50 == 0:
        print(f"Run {i}/500 completed...")

print("\n" + "="*50)
print("PI EDGE CONTROL RESULTS")
print("="*50)
print(f"Average Latency: {(total_latency / 500) * 1000:.2f} ms")
print(f"Average Energy:  {total_joules / 500:.4f} Joules")
print("="*50)
