#!/usr/bin/env python3
"""Extract the last N snapshots from an ASE trajectory file.

Usage:
    python extract_traj_snapshots.py INPUT.traj [-n N] [-o OUTPUT.traj]

The output defaults to INPUT_lastN.traj in the same directory.
"""

import argparse
import os
import sys


def extract_last_n(input_path, n, output_path=None):
    from ase.io.trajectory import Trajectory

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f'{base}_last{n}{ext}'

    with Trajectory(input_path) as traj:
        total = len(traj)
        if total == 0:
            print(f'ERROR: {input_path} contains no frames.', file=sys.stderr)
            sys.exit(1)

        start = max(0, total - n)
        actual = total - start
        print(f'{input_path}: {total} frames total, extracting last {actual}')

        out = Trajectory(output_path, mode='w')
        for i in range(start, total):
            out.write(traj[i])
        out.close()

    print(f'Written to: {output_path}')
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Extract the last N snapshots from an ASE trajectory file.'
    )
    parser.add_argument('input', help='Input .traj file')
    parser.add_argument(
        '-n', '--num', type=int, default=10,
        help='Number of snapshots to extract from the end (default: 10)'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output .traj file (default: INPUT_lastN.traj)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'ERROR: {args.input} not found.', file=sys.stderr)
        sys.exit(1)

    extract_last_n(args.input, args.num, args.output)


if __name__ == '__main__':
    main()
