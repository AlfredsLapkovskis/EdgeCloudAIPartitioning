import zmq
import torch
import torchvision.models as models
import io
import time
from codecarbon import EmissionsTracker

print("Loading VGG-16 Model...")
vgg16 = models.vgg16(weights='DEFAULT').eval()

# NEW CUT: The Laptop takes Layers 17-31, PLUS the final Pool & Classifier
linux_features = torch.nn.Sequential(*list(vgg16.features.children())[17:])
linux_avgpool = vgg16.avgpool
linux_classifier = vgg16.classifier

device = 'cpu'
linux_features.to(device)
linux_avgpool.to(device)
linux_classifier.to(device)
print("Fog model ready on CPU (Layers 17-End).")

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5558")

tracker = EmissionsTracker(
    project_name="VGG16_Linux_Partition_L17", 
    output_file="linux_partition_emissions.csv"
)

print("Waiting for Raspberry Pi to send data on port 5558...")

while True:
    start_time_bytes, message_bytes = socket.recv_multipart()
    start_time = float(start_time_bytes.decode('utf-8'))
    
    print(f"\nCaught {len(message_bytes)} bytes! Starting CPU inference...")
    
    tracker.start()
    
    buffer = io.BytesIO(message_bytes)
    received_tensor = torch.load(buffer, weights_only=True).to(device)
    
    with torch.no_grad():
        # Pass it through the remaining feature layers first!
        x = linux_features(received_tensor)
        x = linux_avgpool(x)
        x = torch.flatten(x, 1)
        final_output = linux_classifier(x)
        
    tracker.stop()
    end_time = time.time()
    
    total_latency = (end_time - start_time) * 1000
    
    try:
        linux_energy = tracker.final_emissions_data.energy_consumed * 3600000 
        print(f"Linux CPU Energy Consumed: {linux_energy:.4f} Joules")
    except AttributeError:
        pass

    print(f"Total End-to-End Latency: {total_latency:.2f} ms")
    print("-" * 40)