from PySide2 import QtWidgets


class VLine(QtWidgets.QFrame):
    def __init__(self):
        super(VLine, self).__init__()
        self.setFrameShape(self.VLine)
        self.setFrameShadow(self.Sunken)