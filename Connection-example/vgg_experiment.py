import torch
import torchvision.models as models
import time
from codecarbon import EmissionsTracker

# 1. Load the pre-trained VGG16 model
print("Loading VGG-16 model...")
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()

# 2. Check if a GPU is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vgg16.to(device)
print(f"Model loaded successfully on: {device.upper()}")

# 3. Create Dummy Data
input_tensor = torch.randn(1, 3, 224, 224).to(device)

# 4. Warm-up Phase
print("Warming up the system...")
with torch.no_grad():
    for _ in range(5):
        _ = vgg16(input_tensor)

# 5. The Actual Experiment: Measuring Latency AND Energy
print("Starting baseline measurement with CodeCarbon...")
num_inferences = 50
total_time = 0.0

tracker = EmissionsTracker(project_name="VGG16_PC_Baseline")
tracker.start()

with torch.no_grad():
    for i in range(num_inferences):
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        output = vgg16(input_tensor)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        inference_time = end_time - start_time
        total_time += inference_time
        print(f"Run {i+1}/{num_inferences} completed in: {inference_time * 1000:.2f} ms")

# Stop the tracker
tracker.stop()

# 6. Calculate Results
average_latency = (total_time / num_inferences) * 1000

# CodeCarbon returns energy in kWh. Convert it to Joules.
# 1 kWh = 3,600,000 Joules
total_energy_kwh = None

if hasattr(tracker, "final_emissions_data") and tracker.final_emissions_data is not None:
    if hasattr(tracker.final_emissions_data, "energy_consumed"):
        total_energy_kwh = tracker.final_emissions_data.energy_consumed

if total_energy_kwh is not None:
    total_energy_joules = total_energy_kwh * 3600000
    average_energy_joules = total_energy_joules / num_inferences
else:
    total_energy_joules = None
    average_energy_joules = None

print("-" * 40)
print("BASELINE COMPLETE")
print(f"Device: {device.upper()}")
print(f"Average End-to-End Latency: {average_latency:.2f} ms")

if total_energy_joules is not None:
    print(f"Total Energy Consumed: {total_energy_joules:.4f} Joules")
    print(f"Average Energy per Inference: {average_energy_joules:.4f} Joules")
else:
    print("Total Energy Consumed: Not available from this CodeCarbon version")
    print("Average Energy per Inference: Not available from this CodeCarbon version")

print("-" * 40)
print("A detailed CSV file named 'emissions.csv' has been saved in your folder.")
