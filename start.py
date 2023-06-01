from PyQt5 import QtWidgets
from ui import Ui_MainWindow
from front import MainWindow_controller

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow_controller()
    # ui = Ui_MainWindow()
    # ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
