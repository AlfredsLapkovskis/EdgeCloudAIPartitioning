import json
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.models as models
import zmq
from codecarbon import EmissionsTracker


def get_safe_energy_joules(tracker: EmissionsTracker) -> Optional[float]:
    """
    Try to read CodeCarbon energy in kWh and convert to Joules.
    Returns None if not available in the installed version.
    """
    try:
        final_data = getattr(tracker, "final_emissions_data", None)
        if final_data is not None:
            energy_kwh = getattr(final_data, "energy_consumed", None)
            if energy_kwh is not None:
                return energy_kwh * 3_600_000.0
    except Exception:
        pass
    return None


def build_cloud_model(split_index: int) -> torch.nn.Module:
    """
    Build the laptop-side tail of VGG16 after the specified split point.
    """
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval()

    if split_index < 31:
        cloud_model = torch.nn.Sequential(
            *list(vgg16.features.children())[split_index:],
            vgg16.avgpool,
            torch.nn.Flatten(),
            *list(vgg16.classifier.children()),
        )
    else:
        cloud_model = torch.nn.Sequential(
            vgg16.avgpool,
            torch.nn.Flatten(),
            *list(vgg16.classifier.children()),
        )

    return cloud_model.eval()


def reconstruct_tensor(tensor_bytes: bytes, shape: Tuple[int, ...], dtype_str: str) -> torch.Tensor:
    np_dtype = np.dtype(dtype_str)
    array = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)
    return torch.from_numpy(array.copy())


def run_fog_receiver_reqrep(
    bind_port: int,
    expected_split_name: str,
    split_index: int,
    output_csv: str,
) -> None:
    """
    Laptop-side REP server for VGG16 collaborative inference.
    Receives activations and returns ACK JSON with timing, prediction,
    and laptop energy estimate.
    """
    print("=" * 70)
    print(f"Starting laptop REP receiver for: {expected_split_name}")
    print(f"Binding on tcp://*:{bind_port}")

    device = torch.device("cpu")
    cloud_model = build_cloud_model(split_index).to(device).eval()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{bind_port}")

    tracker = EmissionsTracker(
        project_name=f"VGG16_{expected_split_name}_Laptop",
        output_file=output_csv,
    )

    print("Laptop receiver is waiting for activations...")
    print("-" * 70)

    tracker.start()

    try:
        while True:
            header_bytes, tensor_bytes = socket.recv_multipart()

            receive_wall_time = time.time()
            t0 = time.perf_counter()

            header = json.loads(header_bytes.decode("utf-8"))
            split_name = header["split_name"]
            recv_split_index = header["split_index"]
            start_wall_time = header["start_wall_time"]
            dtype_str = header["dtype"]
            shape = tuple(header["shape"])

            if split_name != expected_split_name:
                print(
                    f"WARNING: Expected split_name={expected_split_name}, "
                    f"but received split_name={split_name}"
                )

            if recv_split_index != split_index:
                print(
                    f"WARNING: Expected split_index={split_index}, "
                    f"but received split_index={recv_split_index}"
                )

            intermediate_tensor = reconstruct_tensor(
                tensor_bytes=tensor_bytes,
                shape=shape,
                dtype_str=dtype_str,
            ).to(device)

            t1 = time.perf_counter()

            with torch.no_grad():
                output = cloud_model(intermediate_tensor)

            t2 = time.perf_counter()
            end_wall_time = time.time()

            prediction = torch.argmax(output, dim=1).item()

            reconstruct_ms = (t1 - t0) * 1000.0
            cloud_compute_ms = (t2 - t1) * 1000.0
            fog_total_ms = (t2 - t0) * 1000.0
            approx_pi_to_laptop_receive_ms = (receive_wall_time - start_wall_time) * 1000.0
            approx_pi_to_laptop_end_ms = (end_wall_time - start_wall_time) * 1000.0

            laptop_energy_joules = get_safe_energy_joules(tracker)

            ack = {
                "split_name": split_name,
                "split_index": recv_split_index,
                "receive_wall_time": receive_wall_time,
                "end_wall_time": end_wall_time,
                "prediction": prediction,
                "reconstruct_ms": reconstruct_ms,
                "cloud_compute_ms": cloud_compute_ms,
                "fog_total_ms": fog_total_ms,
                "approx_pi_to_laptop_receive_ms": approx_pi_to_laptop_receive_ms,
                "approx_pi_to_laptop_end_ms": approx_pi_to_laptop_end_ms,
                "laptop_energy_joules": laptop_energy_joules,
            }

            socket.send(json.dumps(ack).encode("utf-8"))

            print(f"Received split: {split_name}")
            print(f"Tensor shape: {shape}")
            print(f"Payload size: {len(tensor_bytes)} bytes")
            print(f"Laptop reconstruction time: {reconstruct_ms:.2f} ms")
            print(f"Laptop cloud compute time: {cloud_compute_ms:.2f} ms")
            print(f"Laptop total handling time: {fog_total_ms:.2f} ms")
            print(f"Approx. Pi start -> Laptop receive: {approx_pi_to_laptop_receive_ms:.2f} ms")
            print(f"Approx. Pi start -> Laptop inference end: {approx_pi_to_laptop_end_ms:.2f} ms")
            print(f"Prediction: {prediction}")

            if laptop_energy_joules is not None:
                print(f"Laptop energy so far: {laptop_energy_joules:.4f} J")
            else:
                print("Laptop energy so far: Not available from this CodeCarbon version")

            print("-" * 70)

    except KeyboardInterrupt:
        print("\nStopping laptop receiver...")

    finally:
        tracker.stop()
        socket.close()
        context.term()
        print(f"Laptop CodeCarbon CSV: {output_csv}")
        print("=" * 70)
