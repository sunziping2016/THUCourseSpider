#!/bin/env python3
import argparse
import os


SEGMENTATION_FILE_NAME = "segmentation.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='source directory')
    parser.add_argument('targets', nargs='*', help='target directories')
    args = parser.parse_args()
    targets = []
    for target in args.targets:
        path, num = target.split(':')
        targets.append((path, int(num)))
    solved = set()
    try:
        with open(os.path.join(args.source, SEGMENTATION_FILE_NAME)) as f:
            for line in f:
                solved.add(line.strip().split(':')[0])
    except FileNotFoundError:
        pass
    unsolved = []
    for filename in sorted(os.listdir(args.source)):
        name, ext = os.path.splitext(filename)
        name = name.split('.')
        if ext == '.jpeg' and len(name) == 2 and len(name[0]) == 32 and filename not in solved:
            unsolved.append(filename)
    if not targets:
        print(f'{len(unsolved)} unsolved.')
        return
    assert sum([num for path, num in targets]) == len(unsolved), 'size mismatch'
    for path, num in targets:
        os.makedirs(path, exist_ok=True)
        selected = unsolved[:num]
        unsolved = unsolved[num:]
        for filename in selected:
            os.rename(os.path.join(args.source, filename), os.path.join(path, filename))


if __name__ == '__main__':
    main()
