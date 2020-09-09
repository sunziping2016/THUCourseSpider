#!/bin/env python3
import argparse
import json
import os

from PIL import Image
from tqdm import tqdm

SEGMENTATION_FILE_NAME = "segmentation.txt"

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='dataset.config.json', help='path to the config file')
    parser.add_argument('--captcha_dir', default='captcha', help='path to the captcha')
    parser.add_argument('--output_dir', default='dataset/segmented', help='path to the output dataset')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.captcha_dir, SEGMENTATION_FILE_NAME)) as f:
        for line in tqdm(f, desc='Generating'):
            line = line.strip()
            filename, segments = line.split(':')
            segments = [int(i) for i in segments.split(',')]
            filename_without_ext = os.path.splitext(filename)[0]
            code = filename_without_ext.split('.')[1]
            try:
                image = Image.open(os.path.join(args.captcha_dir, filename))
            except FileNotFoundError:
                continue
            image = image.crop((
                config['margin-left'], config['margin-top'],
                IMAGE_WIDTH - config['margin-right'],
                IMAGE_HEIGHT - config['margin-bottom']
            ))
            for left in range(0, image.width - config['character-width'] + 1, config['slide-x']):
                character_image = image.crop((left, 0, left + config['character-width'], image.height))
                center = config['margin-left'] + left + config['character-width'] / 2
                distances = [abs(center - segment) for segment in segments]
                min_distance = min(distances)
                character = 'NAN' if min_distance >= config['blank-distance'] else code[distances.index(min_distance)]
                os.makedirs(os.path.join(args.output_dir, character), exist_ok=True)
                with open(os.path.join(args.output_dir, character, f'{filename}.{round(center)}.jpeg'), 'wb') as image_file:
                    character_image.save(image_file)


if __name__ == '__main__':
    main()
