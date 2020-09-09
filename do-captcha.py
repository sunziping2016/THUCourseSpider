#!/bin/env python3

import argparse
import hashlib
import json
import os

import requests
from bs4 import BeautifulSoup

headers = {
    'Host': 'zhjwxk.cic.tsinghua.edu.cn',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/85.0.4183.83 Safari/537.36',
}


def main():
    parser = argparse.ArgumentParser(description='Recognize some CAPTCHA manually.')
    parser.add_argument('--config', default='config.json', help='path to the config file')
    parser.add_argument('--temp_file', default='captcha.jpeg', help='temp captcha file')
    parser.add_argument('--captcha_dir', default='captcha', help='path to the captcha')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    os.makedirs(args.captcha_dir, exist_ok=True)
    while True:
        s = requests.Session()
        s.headers.update(headers)
        s.get('http://zhjwxk.cic.tsinghua.edu.cn/xsxk_index.jsp', allow_redirects=False)
        r = s.get('http://zhjwxk.cic.tsinghua.edu.cn/login-jcaptcah.jpg?captchaflag=login1', stream=True)
        with open(args.temp_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=128):
                f.write(chunk)
        with open(args.temp_file, 'rb') as f:
            image = f.read()
            m = hashlib.md5()
            m.update(image)
            key = m.hexdigest()
        results = {}
        for filename in os.listdir(args.captcha_dir):
            name, ext = os.path.splitext(filename)
            name = name.split('.')
            if ext == 'jpeg' and len(name) == 2:
                results[name[0]] = name[1]
        code = results.get(key)
        if code is not None:
            print(f'Skip ({code})')
            continue
        code = input('Captcha: ')
        if not code:
            break
        code = code.upper()
        r = s.post('https://zhjwxk.cic.tsinghua.edu.cn/j_acegi_formlogin_xsxk.do', {
            'j_username': config['username'],
            'j_password': config['password'],
            'captchaflag': 'login1',
            '_login_image_': code
        })
        if r.status_code != 200:
            print(f'Wrong status code: {r.status_code}')
            break
        soup = BeautifulSoup(r.text, 'html.parser')
        error = soup.select('div[align=center]')
        if len(error):
            print(f'Error: {error[0].get_text()}')
        else:
            with open(os.path.join(args.captcha_dir, f'{key}.{code}.jpeg'), 'wb') as f:
                f.write(image)
            print(f'Success ({len(results) + 1})!')


if __name__ == '__main__':
    main()
