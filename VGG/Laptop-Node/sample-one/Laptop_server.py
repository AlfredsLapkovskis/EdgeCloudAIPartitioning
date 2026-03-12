import zmq
import torch
import io

# 1. Set up the ZeroMQ Server
context = zmq.Context()
socket = context.socket(zmq.REP) # REP = Reply pattern
socket.bind("tcp://*:5555")

print("Laptop Server is running.")
print("Waiting for Raspberry Pi to send a tensor on port 5555...")

while True:
    # 2. Wait for the message from the Pi
    message_bytes = socket.recv()
    
    # 3. Deserialize: Convert the raw bytes back into a PyTorch Tensor
    buffer = io.BytesIO(message_bytes)
    received_tensor = torch.load(buffer, weights_only=True)
    
    print(f"Received {len(message_bytes)} bytes. Rebuilt tensor shape: {received_tensor.shape}")
    
    # 4. Instantly send the "Done" signal back so the Pi can stop the stopwatch
    socket.send(b"Done")
