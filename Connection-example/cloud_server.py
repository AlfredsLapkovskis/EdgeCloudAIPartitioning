import zmq
import torch
import torchvision.models as models
import numpy as np

# Split point (must match Pi)
SPLIT_LAYER = 10

print("Loading VGG16 cloud part...")

model = models.vgg16(pretrained=True)
model.eval()

cloud_model = torch.nn.Sequential(
    *list(model.features.children())[SPLIT_LAYER:],
    model.avgpool,
    torch.nn.Flatten(),
    *list(model.classifier.children())
)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("Cloud server waiting for activations...")

while True:
    msg = socket.recv()

    # receive tensor
    data = np.frombuffer(msg, dtype=np.float32)
    data = data.reshape(1, 128, 56, 56)  # depends on split layer

    tensor = torch.tensor(data)

    with torch.no_grad():
        output = cloud_model(tensor)

    prediction = torch.argmax(output).item()

    socket.send_string(str(prediction))
