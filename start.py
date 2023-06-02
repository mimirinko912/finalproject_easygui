import sys

from PyQt5 import QtWidgets
from qt_material import apply_stylesheet
from front import MainWindow_controller

# create the application and the main window
app = QtWidgets.QApplication(sys.argv)
window = MainWindow_controller()

extra = {
    'font_size': '15px',
}

apply_stylesheet(app, theme='light_cyan_500.xml', extra=extra)

window.show()
app.exec_()
