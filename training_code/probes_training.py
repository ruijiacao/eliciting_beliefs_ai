import torch
import numpy as np
import argparse
import os
from probe import CCS  # make sure your CCS class is in probe.py
from utils import get_repr  # you define this for loading x0, x1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--position', type=int, required=True)
    parser.add_argument('--save_dir', type=str, default="trained_probes")
    parser.add_argument('--linear', action='store_true')
    args = parser.parse_args()

    layer = args.layer
    position = args.position
    save_path = os.path.join(args.save_dir, f"probe_layer{layer}_pos{position}.pt")

    if os.path.exists(save_path):
        print(f"Skipping (layer={layer}, pos={position}) — already trained.")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    # Get x0 and x1 for this probe (implement this based on your model/output structure)
    x0, x1 = get_repr(layer, position)  # numpy arrays

    ccs = CCS(x0, x1, nepochs=1000, ntries=10, linear=args.linear, device="cuda" if torch.cuda.is_available() else "cpu")
    ccs.repeated_train()

    # Save probe weights (not the full model)
    if args.linear:
        linear_layer = ccs.best_probe[0]
        probe_data = {
            'W': linear_layer.weight.detach().cpu(),
            'b': linear_layer.bias.detach().cpu(),
            'layer': layer,
            'position': position,
            'linear': True
        }
    else:
        # Save full state_dict if nonlinear
        probe_data = {
            'state_dict': ccs.best_probe.state_dict(),
            'layer': layer,
            'position': position,
            'linear': False
        }

    torch.save(probe_data, save_path)
    print(f"Saved probe to {save_path}")

if __name__ == "__main__":
    main()
