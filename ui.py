# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pyqt_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(899, 488)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(90, 330, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.radioButton.setFont(font)
        self.radioButton.setCheckable(True)
        self.radioButton.setChecked(False)
        self.radioButton.setObjectName("radioButton")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(260, 330, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.buttonGroup.addButton(self.radioButton_2)
        self.radioButton_3 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_3.setGeometry(QtCore.QRect(170, 330, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setObjectName("radioButton_3")
        self.buttonGroup.addButton(self.radioButton_3)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(140, 380, 121, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(27)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(640, 50, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(140, 260, 121, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.Pclass_box = QtWidgets.QComboBox(self.centralwidget)
        self.Pclass_box.setGeometry(QtCore.QRect(100, 100, 69, 22))
        self.Pclass_box.setObjectName("Pclass_box")
        self.Pclass_box.addItem("")
        self.Pclass_box.addItem("")
        self.Pclass_box.addItem("")
        self.gender_box = QtWidgets.QComboBox(self.centralwidget)
        self.gender_box.setGeometry(QtCore.QRect(100, 140, 69, 22))
        self.gender_box.setObjectName("gender_box")
        self.gender_box.addItem("")
        self.gender_box.addItem("")
        self.Fare_box = QtWidgets.QComboBox(self.centralwidget)
        self.Fare_box.setGeometry(QtCore.QRect(100, 220, 69, 22))
        self.Fare_box.setObjectName("Fare_box")
        self.Fare_box.addItem("")
        self.Fare_box.addItem("")
        self.Fare_box.addItem("")
        self.Fare_box.addItem("")
        self.Pclass = QtWidgets.QLabel(self.centralwidget)
        self.Pclass.setGeometry(QtCore.QRect(40, 100, 47, 16))
        self.Pclass.setObjectName("Pclass")
        self.Sex = QtWidgets.QLabel(self.centralwidget)
        self.Sex.setGeometry(QtCore.QRect(40, 140, 47, 16))
        self.Sex.setObjectName("Sex")
        self.Age = QtWidgets.QLabel(self.centralwidget)
        self.Age.setGeometry(QtCore.QRect(40, 180, 47, 16))
        self.Age.setObjectName("Age")
        self.Fare = QtWidgets.QLabel(self.centralwidget)
        self.Fare.setGeometry(QtCore.QRect(40, 220, 47, 16))
        self.Fare.setObjectName("Fare")
        self.FamilySize = QtWidgets.QLabel(self.centralwidget)
        self.FamilySize.setGeometry(QtCore.QRect(200, 220, 71, 16))
        self.FamilySize.setObjectName("FamilySize")
        self.Embarked_box = QtWidgets.QComboBox(self.centralwidget)
        self.Embarked_box.setGeometry(QtCore.QRect(270, 140, 69, 22))
        self.Embarked_box.setObjectName("Embarked_box")
        self.Embarked_box.addItem("")
        self.Embarked_box.addItem("")
        self.Embarked_box.addItem("")
        self.Title_box = QtWidgets.QComboBox(self.centralwidget)
        self.Title_box.setGeometry(QtCore.QRect(270, 180, 69, 22))
        self.Title_box.setObjectName("Title_box")
        self.Title_box.addItem("")
        self.Title_box.addItem("")
        self.Title_box.addItem("")
        self.Cabin = QtWidgets.QLabel(self.centralwidget)
        self.Cabin.setGeometry(QtCore.QRect(200, 100, 47, 16))
        self.Cabin.setObjectName("Cabin")
        self.Embarked = QtWidgets.QLabel(self.centralwidget)
        self.Embarked.setGeometry(QtCore.QRect(200, 140, 61, 16))
        self.Embarked.setObjectName("Embarked")
        self.Title = QtWidgets.QLabel(self.centralwidget)
        self.Title.setGeometry(QtCore.QRect(200, 180, 47, 16))
        self.Title.setObjectName("Title")
        self.Cabin_box = QtWidgets.QComboBox(self.centralwidget)
        self.Cabin_box.setGeometry(QtCore.QRect(270, 100, 69, 22))
        self.Cabin_box.setObjectName("Cabin_box")
        self.Cabin_box.addItem("")
        self.Cabin_box.addItem("")
        self.Cabin_box.addItem("")
        self.Cabin_box.addItem("")
        self.Cabin_box.addItem("")
        self.Cabin_box.addItem("")
        self.Cabin_box.addItem("")
        self.Cabin_box.addItem("")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(140, 50, 121, 31))
        self.textEdit.setObjectName("textEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(170, 10, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(19)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.result = QtWidgets.QLabel(self.centralwidget)
        self.result.setGeometry(QtCore.QRect(580, 110, 261, 251))
        self.result.setText("")
        self.result.setObjectName("result")
        self.AgeBox = QtWidgets.QSpinBox(self.centralwidget)
        self.AgeBox.setGeometry(QtCore.QRect(100, 180, 71, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(self.AgeBox.sizePolicy().hasHeightForWidth())
        self.AgeBox.setSizePolicy(sizePolicy)
        self.AgeBox.setMinimum(1)
        self.AgeBox.setObjectName("AgeBox")
        self.familySizeBox = QtWidgets.QSpinBox(self.centralwidget)
        self.familySizeBox.setGeometry(QtCore.QRect(270, 220, 71, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(11)
        sizePolicy.setHeightForWidth(self.familySizeBox.sizePolicy().hasHeightForWidth())
        self.familySizeBox.setSizePolicy(sizePolicy)
        self.familySizeBox.setMinimum(1)
        self.familySizeBox.setMaximum(11)
        self.familySizeBox.setObjectName("familySizeBox")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 899, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.radioButton.setText(_translate("MainWindow", "SVM"))
        self.radioButton_2.setText(_translate("MainWindow", "RF"))
        self.radioButton_3.setText(_translate("MainWindow", "DNN"))
        self.pushButton.setText(_translate("MainWindow", "Train"))
        self.label.setText(_translate("MainWindow", "Result"))
        self.pushButton_3.setText(_translate("MainWindow", "Predict"))
        self.Pclass_box.setItemText(0, _translate("MainWindow", "1"))
        self.Pclass_box.setItemText(1, _translate("MainWindow", "2"))
        self.Pclass_box.setItemText(2, _translate("MainWindow", "3"))
        self.gender_box.setItemText(0, _translate("MainWindow", "Male"))
        self.gender_box.setItemText(1, _translate("MainWindow", "Female"))
        self.Fare_box.setItemText(0, _translate("MainWindow", "high"))
        self.Fare_box.setItemText(1, _translate("MainWindow", "midium"))
        self.Fare_box.setItemText(2, _translate("MainWindow", "low"))
        self.Fare_box.setItemText(3, _translate("MainWindow", "beggar"))
        self.Pclass.setText(_translate("MainWindow", "Pclass"))
        self.Sex.setText(_translate("MainWindow", "Sex"))
        self.Age.setText(_translate("MainWindow", "Age"))
        self.Fare.setText(_translate("MainWindow", "Fare"))
        self.FamilySize.setText(_translate("MainWindow", "Family Size"))
        self.Embarked_box.setItemText(0, _translate("MainWindow", "0"))
        self.Embarked_box.setItemText(1, _translate("MainWindow", "1"))
        self.Embarked_box.setItemText(2, _translate("MainWindow", "2"))
        self.Title_box.setItemText(0, _translate("MainWindow", "Mr"))
        self.Title_box.setItemText(1, _translate("MainWindow", "Miss"))
        self.Title_box.setItemText(2, _translate("MainWindow", "Mrs"))
        self.Cabin.setText(_translate("MainWindow", "Cabin"))
        self.Embarked.setText(_translate("MainWindow", "Embarked"))
        self.Title.setText(_translate("MainWindow", "Title"))
        self.Cabin_box.setItemText(0, _translate("MainWindow", "A"))
        self.Cabin_box.setItemText(1, _translate("MainWindow", "B"))
        self.Cabin_box.setItemText(2, _translate("MainWindow", "C"))
        self.Cabin_box.setItemText(3, _translate("MainWindow", "D"))
        self.Cabin_box.setItemText(4, _translate("MainWindow", "E"))
        self.Cabin_box.setItemText(5, _translate("MainWindow", "F"))
        self.Cabin_box.setItemText(6, _translate("MainWindow", "G"))
        self.Cabin_box.setItemText(7, _translate("MainWindow", "T"))
        self.label_2.setText(_translate("MainWindow", "Name"))
