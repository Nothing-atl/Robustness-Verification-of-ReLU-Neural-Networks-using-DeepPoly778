import os
import torch
from verifier import analyze, FullyConnected, Conv, Normalization, DEVICE, INPUT_SIZE
import pandas as pd

# Load ground truth file
gt_file = '../test_cases/gt.txt'
with open(gt_file, 'r') as f:
    lines = [line.strip().split(',') for line in f.readlines()]

total = len(lines)
matched = 0
mismatches = []

for net_name, spec_file, expected in lines:
    spec_path = os.path.join('../test_cases', net_name, spec_file)
    with open(spec_path, 'r') as f:
        file_lines = [line.strip() for line in f.readlines()]
        true_label = int(file_lines[0])
        pixel_values = [float(line) for line in file_lines[1:]]
        eps = float(spec_file[:-4].split('_')[-1])

    # Load network
    if net_name == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif net_name == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif net_name == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif net_name == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif net_name == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif net_name == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif net_name == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif net_name == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net_name == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net_name == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        raise ValueError(f"Unknown network {net_name}")

    net.load_state_dict(torch.load(f'../mnist_nets/{net_name}.pt', map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    pred = net(inputs).max(dim=1)[1].item()

    assert pred == true_label, f"Prediction mismatch in {spec_file}"

    # Use the combined analysis that incorporates both approaches.
    result = analyze(net, inputs, eps, true_label)
    result_str = 'verified' if result else 'not verified'

    if result_str == expected:
        matched += 1
    else:
        mismatches.append(f"{net_name}/{spec_file} — expected: {expected}, got: {result_str}")

print(f"\n✅ Summary: {matched}/{total} matched.")
if mismatches:
    print("\n❌ Mismatches:")
    for mismatch in mismatches:
        print("  " + mismatch)
