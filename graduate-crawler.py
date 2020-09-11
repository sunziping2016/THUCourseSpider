#!/bin/env python3
import argparse
import csv
import itertools
import json
import random
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

import constants


reg = re.compile(r'\s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--courses', default='courses.csv', help='path to the courses csv')
    parser.add_argument('--config', default='config.json', help='path to the config file')
    parser.add_argument('--dataset_config', default='dataset.config.json', help='path to the config file')
    parser.add_argument('--temp_file', default='captcha.jpeg', help='temp captcha file')
    parser.add_argument('--generator_save_path', help='path for saving models and codes',
                        default='save/generator')
    parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))),
                        default=[], help="GPU ids separated by `,'")
    parser.add_argument('--load_generator', type=int, default=50,
                        help='load module training at give epoch')
    parser.add_argument('--disable_generator', action='store_true', help='disable neural networks')
    parser.add_argument('--max_try', type=int, default=10, help='max try number for captcha')
    parser.add_argument('--verbose', action='store_true', help='show more messages')
    args = parser.parse_args()
    courses = {}
    with open(args.config) as f:
        config = json.load(f)
    with open(args.courses, newline='') as f:
        reader = iter(csv.reader(f))
        next(reader)  # skip header
        for row in reader:
            assert len(row) == 6, 'Wrong number of columns'
            section = courses.setdefault(tuple(row[:4]), set())
            section.add((int(row[4]), int(row[5])))
    # load model
    if not args.disable_generator:
        from torchvision import transforms

        from data_parallel import get_data_parallel
        from helpers import load_epoch
        from models import CaptchaGenerator_40x40

        with open(args.dataset_config) as f:
            dataset_config = json.load(f)
        model = get_data_parallel(CaptchaGenerator_40x40(
            slide_x=dataset_config['slide-x'],
            total_width=constants.IMAGE_WIDTH - dataset_config['margin-left'] - dataset_config['margin-right']
        ), args.gpu)
        model_state_dict, _ = load_epoch(args.generator_save_path, args.load_generator)
        print('model loaded')
        # noinspection PyUnresolvedReferences
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        model.load_state_dict(model_state_dict)
    else:
        dataset_config = None
        model = None
        transform = None
    while len(courses):
        session = requests.Session()
        session.headers.update(constants.CRAWLER_HEADERS)
        for i in range(args.max_try):
            session.get('http://zhjwxk.cic.tsinghua.edu.cn/xsxk_index.jsp', allow_redirects=False)
            r = session.get('http://zhjwxk.cic.tsinghua.edu.cn/login-jcaptcah.jpg?captchaflag=login1', stream=True)
            with open(args.temp_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)
            if not args.disable_generator:
                import torch
                from PIL import Image

                with open(args.temp_file, 'rb') as f:
                    img = Image.open(f)
                    img = img.crop((
                        dataset_config['margin-left'], dataset_config['margin-top'],
                        constants.IMAGE_WIDTH - dataset_config['margin-right'],
                        constants.IMAGE_HEIGHT - dataset_config['margin-bottom']
                    ))
                img = transform(img)
                _, predicate = model(img.unsqueeze(0), torch.device('cpu'))
                predicate = predicate.squeeze(1)
                code = ''.join([constants.CLASSES[i] for i in predicate.tolist()])
                code = code.replace('#', '').replace('+', '')
            else:
                code = input('Captcha: ')
            r = session.post('https://zhjwxk.cic.tsinghua.edu.cn/j_acegi_formlogin_xsxk.do', {
                'j_username': config['username'],
                'j_password': config['password'],
                'captchaflag': 'login1',
                '_login_image_': code
            })
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                error = soup.select('div[align=center]')
                if len(error) == 0:
                    break
            time.sleep(1 + random.random())
        else:  # for
            break
        print('login successfully!')
        for count, (section, course) in enumerate(itertools.cycle(courses.items())):
            if args.verbose:
                print(f'select {section}')
            response = session.get('http://zhjwxk.cic.tsinghua.edu.cn/xkYjs.vxkYjsXkbBs.do', params={
                'm': section[0],
                'p_xnxq': section[2],
                'tokenPriFlag': section[3],
            })
            content = BeautifulSoup(response.text, 'html.parser')
            try:
                table = content.find('tbody')
                token = content.find_all('input')[1]['value']
            except IndexError:
                print('log out')
                break
            course_ids = []
            for tr in table.find_all('tr'):
                tds = tr.find_all('td')
                course_no = int(reg.sub('', tds[1].text))
                course_id = int(reg.sub('', tds[2].text))
                if (course_no, course_id) in course:
                    left = int(reg.sub('', tds[4].text))
                    if args.verbose:
                        print('course %s:%s has %d left.' % (course_no, course_id, left))
                    if left > 0:
                        course_ids.append(tds[0].find('input')['value'])
            if course_ids:
                print(f'try to get {course_ids}')
                response = session.post('http://zhjwxk.cic.tsinghua.edu.cn/xkYjs.vxkYjsXkbBs.do', data={
                    'm': section[1],
                    'token': token,
                    'p_xnxq': section[2],
                    'tokenPriFlag': section[3],
                    'tabType': '',
                    'page': '',
                    'p_kch': '',
                    'p_kcm': '',
                    'p_xwk_id': course_ids,
                })
                content = BeautifulSoup(response.text, 'html.parser')
                if response.text.find('error') >= 0:
                    message = reg.sub('', content.find('td').text)
                    print(f'Error: {message}', file=sys.stderr)
                with open('response.html', 'wb') as f:
                    f.write(response.content)
                # TODO: process response
            time.sleep(1 + random.random())


if __name__ == '__main__':
    main()
