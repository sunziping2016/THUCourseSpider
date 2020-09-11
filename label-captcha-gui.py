#!/bin/env python3
import argparse
import hashlib
import json
import os
import sys

import requests
from PySide2 import QtWidgets, QtCore, QtGui
from bs4 import BeautifulSoup

import constants
from gui import VLine


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        QtWidgets.QMainWindow.__init__(self)
        self.args = args
        with open(args.config) as f:
            self.config = json.load(f)
        if not self.args.disable_generator:
            from torchvision import transforms

            from data_parallel import get_data_parallel
            from helpers import load_epoch
            from models import CaptchaGenerator_40x40

            with open(args.dataset_config) as f:
                self.dataset_config = json.load(f)
            self.model = get_data_parallel(CaptchaGenerator_40x40(
                slide_x=self.dataset_config['slide-x'],
                total_width=constants.IMAGE_WIDTH - self.dataset_config['margin-left'] - self.dataset_config['margin-right']
            ), self.args.gpu)
            model_state_dict, _ = load_epoch(args.generator_save_path, args.load_generator)
            print('model loaded')
            # noinspection PyUnresolvedReferences
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
            self.model.load_state_dict(model_state_dict)
        else:
            self.dataset_config = None
            self.model = None
            self.transform = None
        os.makedirs(self.args.captcha_dir, exist_ok=True)
        self.image_label = QtWidgets.QLabel()
        self.input_label = QtWidgets.QLabel('Captcha Code:')
        self.input = QtWidgets.QLineEdit()
        self.input.setMaxLength(5)
        validate_regex = '[' + ''.join(constants.CLASSES)[2:] + ']{4,5}'
        self.input.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp(validate_regex, QtCore.Qt.CaseInsensitive)))
        self.refresh_button = QtWidgets.QPushButton('Refresh')
        self.clear_button = QtWidgets.QPushButton('Clear')
        self.ok_button = QtWidgets.QPushButton('Ok')

        self.input_layout = QtWidgets.QHBoxLayout()
        self.input_layout.addWidget(self.input_label)
        self.input_layout.addWidget(self.input)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.refresh_button)
        self.button_layout.addWidget(self.clear_button)
        self.button_layout.addWidget(self.ok_button)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_label, alignment=QtCore.Qt.AlignCenter)
        self.layout.addLayout(self.input_layout)
        self.layout.addLayout(self.button_layout)

        self.setCentralWidget(QtWidgets.QWidget(self))
        self.centralWidget().setLayout(self.layout)

        self.status_label = QtWidgets.QLabel()
        self.count_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status_label)
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.count_label)

        self.refresh_button.clicked.connect(self.update_all)
        self.clear_button.clicked.connect(self.on_clear_click)
        self.ok_button.clicked.connect(self.on_ok_click)
        self.input.textEdited.connect(self.update_buttons)
        self.refresh_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.Refresh, self)
        self.refresh_shortcut.activated.connect(self.update_all)
        self.clear_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.Cancel, self)
        self.clear_shortcut.activated.connect(self.on_clear_click)
        self.ok_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self)
        self.ok_shortcut.activated.connect(self.on_ok_click)

        self.session = None
        self.image = b''
        self.key = ''
        self.results = {}
        self.update_all()

    def update_all(self):
        self.update_data()
        self.update_buttons()
        self.update_count_status()

    def on_ok_click(self):
        code = self.input.text().upper()
        response = self.session.post('https://zhjwxk.cic.tsinghua.edu.cn/j_acegi_formlogin_xsxk.do', {
            'j_username': self.config['username'],
            'j_password': self.config['password'],
            'captchaflag': 'login1',
            '_login_image_': code
        })
        if response.status_code != 200:
            self.status_label.setText(f'Return: {response.status_code}')
            self.status_label.setStyleSheet("QLabel { color: red; }")
        else:
            soup = BeautifulSoup(response.text, 'html.parser')
            error = soup.select('div[align=center]')
            if len(error):
                self.status_label.setText(''.join(error[0].get_text().split()))
                self.status_label.setStyleSheet("QLabel { color: red; }")
            else:
                with open(os.path.join(self.args.captcha_dir, f'{self.key}.{code}.jpeg'), 'wb') as f:
                    f.write(self.image)
                self.status_label.setText(f'Success!')
                self.status_label.setStyleSheet("QLabel { color: green; }")
        self.update_all()

    def on_clear_click(self):
        self.input.setText('')
        self.update_buttons()

    def update_data(self):
        code = ''
        while code is not None:
            self.session = requests.Session()
            self.session.headers.update(constants.CRAWLER_HEADERS)
            self.session.get('http://zhjwxk.cic.tsinghua.edu.cn/xsxk_index.jsp', allow_redirects=False)
            r = self.session.get('http://zhjwxk.cic.tsinghua.edu.cn/login-jcaptcah.jpg?captchaflag=login1', stream=True)
            with open(self.args.temp_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)
            with open(self.args.temp_file, 'rb') as f:
                self.image = f.read()
                m = hashlib.md5()
                m.update(self.image)
                self.key = m.hexdigest()
            self.results = {}
            for filename in os.listdir(self.args.captcha_dir):
                name, ext = os.path.splitext(filename)
                name = name.split('.')
                if ext == '.jpeg' and len(name) == 2 and len(name[0]) == 32:
                    self.results[name[0]] = name[1]
            code = self.results.get(self.key)
            print(f'search result: {code}')
        if not self.args.disable_generator:
            import torch
            from PIL import Image

            with open(self.args.temp_file, 'rb') as f:
                img = Image.open(f)
                img = img.crop((
                    self.dataset_config['margin-left'], self.dataset_config['margin-top'],
                    constants.IMAGE_WIDTH - self.dataset_config['margin-right'],
                    constants.IMAGE_HEIGHT - self.dataset_config['margin-bottom']
                ))
            img = self.transform(img)
            _, predicate = self.model(img.unsqueeze(0), torch.device('cpu'))
            predicate = predicate.squeeze(1)
            code = ''.join([constants.CLASSES[i] for i in predicate.tolist()])
            code = code.replace('#', '').replace('+', '')
        else:
            code = ''
        self.image_label.setPixmap(QtGui.QPixmap(self.args.temp_file))
        self.input.setText(code)

    def update_buttons(self):
        text_len = len(self.input.text())
        self.clear_button.setEnabled(text_len != 0)
        self.ok_button.setEnabled(4 <= text_len <= 5)

    def update_count_status(self):
        self.count_label.setText(f'Total: {len(self.results)}')


def main():
    parser = argparse.ArgumentParser(description='Recognize some CAPTCHA manually.')
    parser.add_argument('--config', default='config.json', help='path to the config file')
    parser.add_argument('--dataset_config', default='dataset.config.json', help='path to the config file')
    parser.add_argument('--temp_file', default='captcha.jpeg', help='temp captcha file')
    parser.add_argument('--captcha_dir', default='captcha', help='path to the captcha')
    parser.add_argument('--generator_save_path', help='path for saving models and codes',
                        default='save/generator')
    parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))),
                        default=[], help="GPU ids separated by `,'")
    parser.add_argument('--load_generator', type=int, default=50,
                        help='load module training at give epoch')
    parser.add_argument('--disable_generator', action='store_true', help='disable neural networks')
    args, unparsed_args = parser.parse_known_args()
    app = QtWidgets.QApplication(sys.argv[:1] + unparsed_args)
    main_window = MainWindow(args)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
