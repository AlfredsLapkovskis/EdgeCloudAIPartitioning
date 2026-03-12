import zmq
import torch
import torchvision.models as models
import io
import time
from codecarbon import EmissionsTracker

print("Loading VGG-16 Model...")
vgg16 = models.vgg16(weights='DEFAULT').eval()

# The Linux Laptop takes the final Pooling and Classifier layers
linux_avgpool = vgg16.avgpool
linux_classifier = vgg16.classifier

device = 'cpu'
linux_avgpool.to(device)
linux_classifier.to(device)
print("Fog model ready on CPU.")

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5558")

# Set up CodeCarbon for the Linux Laptop
tracker = EmissionsTracker(
    project_name="VGG16_Linux_Partition", 
    output_file="linux_partition_emissions.csv"
)

print("Waiting for Raspberry Pi to send data on port 5558...")

while True:
    start_time_bytes, message_bytes = socket.recv_multipart()
    start_time = float(start_time_bytes.decode('utf-8'))
    
    print(f"\nCaught {len(message_bytes)} bytes! Starting CPU inference...")
    
    # Start tracking laptop energy
    tracker.start()
    
    buffer = io.BytesIO(message_bytes)
    received_tensor = torch.load(buffer, weights_only=True).to(device)
    
    with torch.no_grad():
        x = linux_avgpool(received_tensor)
        x = torch.flatten(x, 1)
        final_output = linux_classifier(x)
        
    # Stop tracking laptop energy and stop the stopwatch
    tracker.stop()
    end_time = time.time()
    
    total_latency = (end_time - start_time) * 1000
    
    # CodeCarbon calculates in kWh. Multiply by 3,600,000 to get Joules!
    # (Using a try-except just in case CodeCarbon fails to write to the CSV)
    try:
        linux_energy = tracker.final_emissions_data.energy_consumed * 3600000 
        print(f"Linux CPU Energy Consumed: {linux_energy:.4f} Joules")
    except AttributeError:
        print("Energy logged to CSV. Check the codecarbon INFO prints above!")

    print(f"Total End-to-End Latency: {total_latency:.2f} ms")
    print("-" * 40)
