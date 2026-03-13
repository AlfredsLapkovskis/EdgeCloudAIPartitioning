import zmq
import torch
import torchvision.models as models
import io
import time

# --- CHANGE TO THE LINUX LAPTOP'S LOCAL IP ---
FOG_IP = "192.168.10.187" 
# ---------------------------------------------

print("Loading VGG-16 Model...")
vgg16 = models.vgg16(weights='DEFAULT').eval()

# Layer 17 Cut
pi_model = torch.nn.Sequential(*list(vgg16.features.children())[:17])
print("Pi model ready (Layers 1-17).")

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect(f"tcp://{FOG_IP}:5558")

dummy_image = torch.randn(1, 3, 224, 224)

print("Starting partitioned inference...")

# 1. Start the stopwatch
start_time = time.time()

with torch.no_grad():
    intermediate_tensor = pi_model(dummy_image)
    
    buffer = io.BytesIO()
    torch.save(intermediate_tensor, buffer)
    tensor_bytes = buffer.getvalue()
    
    socket.send_multipart([str(start_time).encode('utf-8'), tensor_bytes])

# 2. Stop the stopwatch for local calculation
end_time = time.time()
local_processing_time = end_time - start_time

# 3. Calculate Real Joules (Assuming 8.5 Watts for a Pi under heavy load)
PI_WATTAGE = 8.5 
real_pi_joules = PI_WATTAGE * local_processing_time

print("-" * 40)
print(f"Sent {len(tensor_bytes)} bytes to Linux Laptop.")
print(f"Local Processing Time: {local_processing_time:.3f} seconds")
print(f"Real Pi Energy Consumed: {real_pi_joules:.4f} Joules")
print("Check the Linux Laptop screen for the total Latency!")
print("-" * 40)