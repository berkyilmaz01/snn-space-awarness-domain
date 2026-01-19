#!/usr/bin/env python3
"""
Quick diagnostic: Check if input events actually occur at satellite locations.
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spikeseg.data.datasets import EBSSADataset

def main():
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

    print(f"Dataset: {len(dataset)} samples\n")

    total_events_on_satellites = 0
    total_events_on_background = 0
    total_satellite_pixels = 0
    total_background_pixels = 0

    for i in range(min(20, len(dataset))):
        x, label = dataset[i]

        # x shape: (T, C, H, W) - sum over time and channels to get event locations
        event_mask = (x.sum(dim=(0, 1)) > 0)  # (H, W) where events occurred
        satellite_mask = (label > 0)  # (H, W) where satellites are

        # Count events on satellites vs background
        events_on_sat = (event_mask & satellite_mask).sum().item()
        events_on_bg = (event_mask & ~satellite_mask).sum().item()
        sat_pixels = satellite_mask.sum().item()
        bg_pixels = (~satellite_mask).sum().item()

        total_events_on_satellites += events_on_sat
        total_events_on_background += events_on_bg
        total_satellite_pixels += sat_pixels
        total_background_pixels += bg_pixels

        # Event density
        if sat_pixels > 0:
            sat_density = events_on_sat / sat_pixels
        else:
            sat_density = 0
        bg_density = events_on_bg / bg_pixels if bg_pixels > 0 else 0

        print(f"Sample {i:2d}: events_total={event_mask.sum().item():5d}, "
              f"sat_pixels={sat_pixels:4d}, events_on_sat={events_on_sat:4d} "
              f"(density: {sat_density:.2f} vs bg: {bg_density:.4f})")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total events on satellites:  {total_events_on_satellites}")
    print(f"Total events on background:  {total_events_on_background}")
    print(f"Total satellite pixels:      {total_satellite_pixels}")
    print(f"Total background pixels:     {total_background_pixels}")

    if total_satellite_pixels > 0:
        sat_event_rate = total_events_on_satellites / total_satellite_pixels
        bg_event_rate = total_events_on_background / total_background_pixels
        print(f"\nSatellite event density:     {sat_event_rate:.4f}")
        print(f"Background event density:    {bg_event_rate:.4f}")
        print(f"Ratio (sat/bg):              {sat_event_rate/bg_event_rate:.2f}x")

        if sat_event_rate > bg_event_rate * 2:
            print("\n✓ Satellites generate MORE events than background - learning should work!")
        elif sat_event_rate > bg_event_rate:
            print("\n⚠ Satellites generate slightly more events - marginal signal")
        else:
            print("\n✗ Satellites generate FEWER events than background - model can't learn!")

if __name__ == "__main__":
    main()
