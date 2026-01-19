#!/usr/bin/env python3
"""Check timestamp units between events (TD.ts) and labels (Obj.ts)."""

import numpy as np
import scipy.io as sio
from pathlib import Path

def main():
    # Find a data file
    data_root = Path("/home/ubuntu/ebssa-data-utah/ebssa")
    if not data_root.exists():
        data_root = Path("../ebssa-data-utah/ebssa")

    labelled_dir = data_root / "Labelled Data"
    mat_files = list(labelled_dir.rglob("*.mat"))

    # Filter to *_labelled.mat files (contain both events and labels)
    labelled_files = [f for f in mat_files if '_labelled' in f.stem.lower()]

    print(f"Found {len(labelled_files)} labelled files")

    for mat_file in labelled_files[:3]:  # Check first 3 files
        print(f"\n{'='*60}")
        print(f"File: {mat_file.name}")
        print('='*60)

        mat = sio.loadmat(mat_file, squeeze_me=True)

        # Event timestamps
        if 'TD' in mat:
            td = mat['TD']
            if hasattr(td, 'dtype') and td.dtype.names:
                event_ts = td['ts'].flatten() if 'ts' in td.dtype.names else None
            else:
                event_ts = td[0, 0]['ts'].flatten() if hasattr(td, 'shape') else None

            if event_ts is not None and len(event_ts) > 0:
                event_ts = event_ts.flatten()
                e_min = float(event_ts.min())
                e_max = float(event_ts.max())
                print(f"\nEvent timestamps (TD.ts):")
                print(f"  Count: {len(event_ts)}")
                print(f"  Min: {e_min:,.0f}")
                print(f"  Max: {e_max:,.0f}")
                print(f"  Duration: {(e_max - e_min):,.0f}")
                print(f"  Duration (seconds, if microsec): {(e_max - e_min) / 1e6:.2f}s")
                print(f"  Duration (seconds, if millisec): {(e_max - e_min) / 1e3:.2f}s")

        # Label timestamps
        if 'Obj' in mat:
            obj = mat['Obj']
            if hasattr(obj, 'dtype') and obj.dtype.names:
                label_ts = obj['ts'] if 'ts' in obj.dtype.names else None
                label_x = obj['x'] if 'x' in obj.dtype.names else None
                label_y = obj['y'] if 'y' in obj.dtype.names else None

                # Unwrap if needed
                if label_ts is not None:
                    while hasattr(label_ts, 'shape') and label_ts.shape == ():
                        label_ts = label_ts.item()
                    label_ts = np.asarray(label_ts).flatten()

                if label_ts is not None and len(label_ts) > 0:
                    l_min = float(label_ts.min())
                    l_max = float(label_ts.max())
                    print(f"\nLabel timestamps (Obj.ts):")
                    print(f"  Count: {len(label_ts)}")
                    print(f"  Min: {l_min:,.0f}")
                    print(f"  Max: {l_max:,.0f}")
                    print(f"  Duration: {(l_max - l_min):,.0f}")
                    print(f"  Duration (seconds, if microsec): {(l_max - l_min) / 1e6:.2f}s")
                    print(f"  Duration (seconds, if millisec): {(l_max - l_min) / 1e3:.2f}s")

                    # Compare ranges
                    if event_ts is not None and len(event_ts) > 0:
                        print(f"\nTimestamp comparison:")
                        print(f"  Event range: [{e_min:,.0f}, {e_max:,.0f}]")
                        print(f"  Label range: [{l_min:,.0f}, {l_max:,.0f}]")

                        # Check if they overlap
                        overlap = (l_min <= e_max) and (l_max >= e_min)
                        print(f"  Ranges overlap: {overlap}")

                        if not overlap:
                            l_mean = float(label_ts.mean())
                            e_mean = float(event_ts.mean())
                            ratio = e_mean / l_mean if l_mean != 0 else 0
                            print(f"  Event/Label ratio: {ratio:.2f}x (if ~1000, labels may be in ms)")

if __name__ == "__main__":
    main()
