import argparse
import bisect
import os
import sys

from PySide2 import QtWidgets, QtCore, QtGui

SEGMENTATION_FILE_NAME = "segmentation.txt"

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50
IMAGE_SCALE = 3

SCALED_WIDTH = IMAGE_WIDTH * IMAGE_SCALE
SCALED_HEIGHT = IMAGE_HEIGHT * IMAGE_SCALE

SELECT_X_MAX_DISTANCE = 5
LINE_WIDTH = 2 * IMAGE_SCALE


class VLine(QtWidgets.QFrame):
    def __init__(self):
        super(VLine, self).__init__()
        self.setFrameShape(self.VLine)
        self.setFrameShadow(self.Sunken)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        QtWidgets.QMainWindow.__init__(self)
        self.args = args
        self.image_label = QtWidgets.QLabel()
        self.prev_button = QtWidgets.QPushButton('<')
        self.reset_button = QtWidgets.QPushButton('Reset')
        self.ok_button = QtWidgets.QPushButton('Ok')
        self.next_button = QtWidgets.QPushButton('>')

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.prev_button)
        self.button_layout.addWidget(self.reset_button)
        self.button_layout.addWidget(self.ok_button)
        self.button_layout.addWidget(self.next_button)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_label, alignment=QtCore.Qt.AlignCenter)
        self.layout.addLayout(self.button_layout)

        self.setCentralWidget(QtWidgets.QWidget(self))
        self.centralWidget().setLayout(self.layout)

        self.finished_label = QtWidgets.QLabel()
        self.index_label = QtWidgets.QLabel()
        self.pos_label = QtWidgets.QLabel()
        self.results_label = QtWidgets.QLabel()
        self.statusBar().addWidget(self.finished_label)
        self.statusBar().addWidget(VLine())
        self.statusBar().addWidget(self.index_label)
        self.statusBar().addPermanentWidget(self.pos_label)
        self.statusBar().addPermanentWidget(VLine())
        self.statusBar().addPermanentWidget(self.results_label)

        # states
        self.data = {}
        self.unanswered = []
        self.current = 0
        self.answer = []
        self.mouse_x = None
        self.mouse_new = False
        self.mouse_pressed = False
        self.mouse_moved = False

        # connect
        self.prev_button.clicked.connect(self.on_prev_click)
        self.next_button.clicked.connect(self.on_next_click)
        self.prev_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.MoveToPreviousChar, self)
        self.prev_shortcut.activated.connect(self.on_prev_click)
        self.next_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.MoveToNextChar, self)
        self.next_shortcut.activated.connect(self.on_next_click)

        self.load_data()
        self.update_view()

        self.setMouseTracking(True)

    def setMouseTracking(self, flag):
        def recursive_set(parent):
            for child in parent.findChildren(QtCore.QObject):
                if hasattr(child, 'setMouseTracking'):
                    child.setMouseTracking(flag)
                recursive_set(child)
        QtWidgets.QWidget.setMouseTracking(self, flag)
        recursive_set(self)

    def mousePressEvent(self, event):
        relative_x = event.x() - self.image_label.geometry().x()
        relative_x = round(relative_x / IMAGE_SCALE)
        selected_items = [i for i in self.answer if abs(relative_x - i) <= SELECT_X_MAX_DISTANCE]
        if len(selected_items):
            selected = selected_items[0]
            self.answer.remove(selected)
            self.mouse_new = False
            self.mouse_x = selected
        else:
            self.mouse_new = True
            self.mouse_x = relative_x
        self.mouse_pressed = True
        self.mouse_moved = False
        self.update_view()

    def mouseReleaseEvent(self, event):
        relative_x = event.x() - self.image_label.geometry().x()
        relative_x = round(relative_x / IMAGE_SCALE)
        if self.mouse_moved or self.mouse_new:
            selected_items = [i for i in self.answer if abs(relative_x - i) <= SELECT_X_MAX_DISTANCE]
            if len(selected_items) == 0:
                bisect.insort(self.answer, relative_x)
        self.mouse_pressed = False
        self.update_view()

    def mouseMoveEvent(self, event):
        relative_x = event.x() - self.image_label.geometry().x()
        relative_x = round(relative_x / IMAGE_SCALE)
        if relative_x < 0:
            relative_x = 0
        if relative_x >= IMAGE_WIDTH:
            relative_x = IMAGE_WIDTH - 1
        self.mouse_x = relative_x
        self.mouse_moved = True
        self.update_view()

    def on_prev_click(self):
        if self.current - 1 < 0:
            return
        self.current -= 1
        self.answer = self.data.get(self.unanswered[self.current], [])
        self.update_view()

    def on_next_click(self):
        if self.current + 1 >= len(self.unanswered):
            return
        self.current += 1
        self.answer = self.data.get(self.unanswered[self.current], [])
        self.update_view()

    def load_data(self):
        self.data = {}
        try:
            with open(os.path.join(self.args.captcha_dir, SEGMENTATION_FILE_NAME)) as f:
                for line in f:
                    line = line.strip()
                    key, value = line.split(':')
                    value = [int(i) for i in value.split(',')]
                    self.data[key] = value
        except FileNotFoundError:
            pass
        self.unanswered = []
        for filename in sorted(os.listdir(self.args.captcha_dir)):
            name, ext = os.path.splitext(filename)
            name = name.split('.')
            if ext == '.jpeg' and len(name) == 2 and len(name[0]) == 32 and \
                    self.data.get(filename) is None:
                self.unanswered.append(filename)

    def save_data(self):
        with open(os.path.join(self.args.captcha_dir, SEGMENTATION_FILE_NAME), 'w') as f:
            for key in sorted(self.data.keys()):
                f.write(f'{key}:{",".join([str(i) for i in self.data[key]])}')

    def update_view(self):
        self.update_buttons()
        self.update_canvas()
        self.update_status()
        self.update()

    def update_buttons(self):
        answer_len = len(self.answer)
        self.prev_button.setEnabled(self.current > 0)
        self.reset_button.setEnabled(answer_len > 0)
        self.ok_button.setEnabled(4 <= answer_len <= 5)
        self.next_button.setEnabled(self.current < len(self.unanswered) - 1)

    def update_canvas(self):
        if self.current < 0 or self.current >= len(self.unanswered):
            self.image_label.setText("All Finished.")
        else:
            pixmap = QtGui.QPixmap(SCALED_WIDTH, SCALED_HEIGHT)
            painter = QtGui.QPainter(pixmap)
            filename = os.path.join(self.args.captcha_dir, self.unanswered[self.current])
            image = QtGui.QPixmap(filename)
            painter.drawPixmap(0, 0, SCALED_WIDTH, SCALED_HEIGHT, image)
            painter.setPen(QtGui.QPen(QtGui.QColor('green'), LINE_WIDTH))
            for x in self.answer:
                painter.drawLine(x * IMAGE_SCALE, 0, x * IMAGE_SCALE, SCALED_HEIGHT)
            if self.mouse_pressed:
                painter.setPen(QtGui.QPen(QtGui.QColor('red'), LINE_WIDTH))
                painter.drawLine(self.mouse_x * IMAGE_SCALE, 0, self.mouse_x * IMAGE_SCALE, SCALED_HEIGHT)
            painter.end()
            self.image_label.setPixmap(pixmap)

    def update_status(self):
        self.finished_label.setText(f'Prog: {len(self.data)}/{len(self.data) + len(self.unanswered)}')
        self.index_label.setText(f'Idx: {self.current + 1}/{len(self.unanswered)}')
        self.pos_label.setText(f'X: {"N/A" if self.mouse_x is None else self.mouse_x}')
        self.results_label.setText(f'Pos: [{",".join([str(i) for i in self.answer])}]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captcha_dir', default='captcha', help='path to the captcha')
    args, unparsed_args = parser.parse_known_args()
    app = QtWidgets.QApplication(sys.argv[:1] + unparsed_args)
    main_window = MainWindow(args)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
