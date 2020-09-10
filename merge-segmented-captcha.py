#!/bin/env python3
import argparse
import os

SEGMENTATION_FILE_NAME = "segmentation.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='source directory')
    parser.add_argument('target', help='target directory')
    args = parser.parse_args()
    target_data = {}
    try:
        with open(os.path.join(args.target, SEGMENTATION_FILE_NAME)) as f:
            for line in f:
                line = line.strip()
                key, value = line.split(':')
                target_data[key] = value
    except FileNotFoundError:
        pass
    with open(os.path.join(args.source, SEGMENTATION_FILE_NAME)) as f:
        for line in f:
            line = line.strip()
            key, value = line.split(':')
            target_data[key] = value
            os.rename(os.path.join(args.source, key), os.path.join(args.target, key))
    with open(os.path.join(args.target, SEGMENTATION_FILE_NAME), 'w') as f:
        for key in sorted(target_data.keys()):
            f.write(f'{key}:{target_data[key]}\n')
    os.remove(os.path.join(args.source, SEGMENTATION_FILE_NAME))


if __name__ == '__main__':
    main()
