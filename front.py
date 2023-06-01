import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from ui import Ui_MainWindow

# import cv2
# from keras.models import load_model
# default_model = load_model('dnnfortitanic.h5')

def predict_result():
    return

def user_train():
    return

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_controller()
    def setup_controller(self):
        self.ui.label.setText('die')
        self.ui.label.setText('survive')

    # def display_img(self):
    #     self.img = cv2.imread(self.img_path)
    #     height, width, channel = self.img.shape
    #     bytesPerline = 3 * width
    #     self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
    #     self.ui.label.setPixmap(QPixmap.fromImage(self.qimg))

