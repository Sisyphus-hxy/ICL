#!/usr/bin/env python3
"""Render one demo episode from a prompt/expert HDF5 to mp4.

Frame layout matches evaluate.py's rollout.mp4: agentview | eye_in_hand
side-by-side, vertically flipped (hdf5 stores frames upside down), fps=20.

Usage:
    render_demo.py HDF5 --demo-idx N --out PATH.mp4
"""
import argparse

import h5py
import imageio
import numpy as np


def render(hdf5_path, demo_idx, out_path):
    with h5py.File(hdf5_path, 'r') as f:
        d = f[f'data/demo_{demo_idx}']
        agentview = d['obs/agentview_rgb'][:]
        eih = d['obs/eye_in_hand_rgb'][:]
    writer = imageio.get_writer(str(out_path), fps=20, codec='libx264')
    for i in range(agentview.shape[0]):
        frame = np.concatenate([agentview[i][::-1], eih[i][::-1]], axis=1)
        writer.append_data(frame)
    writer.close()
    return agentview.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('hdf5')
    p.add_argument('--demo-idx', type=int, default=0)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    n = render(args.hdf5, args.demo_idx, args.out)
    print(f'{args.out}  ({n} frames)')


if __name__ == '__main__':
    main()
