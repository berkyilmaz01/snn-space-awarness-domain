#!/usr/bin/env python3
"""Debug samples with sat_pixels=0 to understand why."""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import scipy.io as sio

def main():
    # First, let's look at the raw recordings to understand the structure
    data_root = Path("../ebssa-data-utah/ebssa")
    labelled_dir = data_root / "Labelled Data"
    if not labelled_dir.exists():
        labelled_dir = data_root / "Labelled_Data"

    print(f"Looking in: {labelled_dir}")
    print("="*60)

    mat_files = list(labelled_dir.rglob("*.mat"))
    print(f"Found {len(mat_files)} .mat files\n")

    # Check each one for 'Obj' field
    has_obj = 0
    no_obj = 0

    for mat_path in sorted(mat_files)[:20]:  # First 20
        try:
            mat = sio.loadmat(mat_path, squeeze_me=True)
            keys = [k for k in mat.keys() if not k.startswith('_')]

            if 'Obj' in mat:
                has_obj += 1
                obj = mat['Obj']
                # Check if Obj has valid data
                if hasattr(obj, 'dtype') and obj.dtype.names:
                    fields = list(obj.dtype.names)
                    print(f"✓ {mat_path.name}: Obj fields={fields}")
                else:
                    print(f"? {mat_path.name}: Obj exists but unusual format: {type(obj)}")
            else:
                no_obj += 1
                print(f"✗ {mat_path.name}: NO Obj field. Keys={keys}")

        except Exception as e:
            print(f"ERROR {mat_path.name}: {e}")

    print(f"\n{'='*60}")
    print(f"Files WITH Obj: {has_obj}")
    print(f"Files WITHOUT Obj: {no_obj}")

    # Now check what happens with the dataset
    print(f"\n{'='*60}")
    print("DATASET LOADING CHECK")
    print("="*60)

    from spikeseg.data.datasets import EBSSADataset

    dataset = EBSSADataset(
        root="../ebssa-data-utah/ebssa",
        split="all",
        sensor="all",
        n_timesteps=20,
        height=128,
        width=128,
        normalize=True,
        use_labels=True,
        windows_per_recording=1,
    )

    print(f"Dataset: {len(dataset)} samples")
    print(f"Recordings: {len(dataset.recordings)}")

    # Check recording info
    for i, rec in enumerate(dataset.recordings[:10]):
        if isinstance(rec, dict):
            evt_path = rec.get('event_path', 'unknown')
            name = rec.get('name', Path(evt_path).name if evt_path != 'unknown' else 'unknown')
            print(f"  Recording {i}: {name}")

    # Now check samples
    print(f"\n{'='*60}")
    print("SAMPLE ANALYSIS")
    print("="*60)

    empty_count = 0
    good_count = 0

    for i in range(len(dataset)):
        x, label = dataset[i]
        sat_pixels = (label > 0).sum().item()
        total_events = (x.sum(dim=(0, 1)) > 0).sum().item()

        if sat_pixels == 0:
            empty_count += 1
            rec_idx = i // dataset.windows_per_recording
            rec = dataset.recordings[rec_idx] if rec_idx < len(dataset.recordings) else {}
            name = rec.get('name', 'unknown') if isinstance(rec, dict) else str(rec)
            print(f"Sample {i}: sat_pixels=0, events={total_events}, rec={name}")
        else:
            good_count += 1

    print(f"\n{'='*60}")
    print(f"Samples WITH satellites:    {good_count}")
    print(f"Samples WITHOUT satellites: {empty_count}")
    print(f"Empty ratio: {empty_count/len(dataset)*100:.1f}%")

if __name__ == "__main__":
    main()
