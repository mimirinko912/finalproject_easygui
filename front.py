import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from ui import Ui_MainWindow
import pandas as pd
# import cv2
from keras.models import load_model
default_model = load_model('dnnfortitanic.h5')

def Sex_mapping(data):
    Sex_mapping = {"Male":0,"Female":1}
    return data.map(Sex_mapping)
def Family_mapping(data):
    family_mapping = {1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,7:2.4,8:2.8,9:3.2,10:3.6,11:4}
    return data.map(family_mapping)
def title_mapping(data):
    title_mapping = {"Mr": 0,"Miss" : 1, "Mrs" : 2}
    return data.map(title_mapping)
def Cabin_mapping(data):
    cabin_mapping = {"A":0,"B":0.4,"C":0.8,"D":1.2,"E":1.6,"F":2,"G":2.4,"T":2.8}
    return data.map(cabin_mapping)
def Fare_mapping(data):
    Fare_mapping = {"high":3,"medium":2,"low":1,"beggar":0}
    return data.map(Fare_mapping)
def predict_result(data):
    print(data.items)
    result = default_model.predict(data).flatten()
    return result[0]

def user_train():
    return

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_controller()
    def setup_controller(self):
        self.ui.pushButton_3.clicked.connect(self.predict_user)

    def predict_user(self):
        Pclass  = self.ui.Pclass_box.currentText()
        Sex     = self.ui.gender_box.currentText()
        Age     = self.ui.AgeBox.value()
        Fare    = self.ui.Fare_box.currentText()
        Cabin   = self.ui.Cabin_box.currentText()
        Embarked= self.ui.Embarked_box.currentText()
        Title   = self.ui.Title_box.currentText()
        Famisize= self.ui.familySizeBox.value()
        # print(Pclass,Embarked)
        data = {
            'Pclass':[int(Pclass)],
            'Sex':[Sex],
            'Age':[Age],
            'Fare':[Fare],
            'Cabin':[Cabin],
            'Embarked':[int(Embarked)],
            'Title':[Title],
            'FamilySize':[Famisize]
        }
        data = pd.DataFrame(data)
        data['Title'] = title_mapping(data['Title'])
        data.loc[ data['Age'] <= 16,'Age'] = 0
        data.loc[(data['Age'] > 16 ) & (data['Age'] <= 26),'Age'] = 1
        data.loc[(data['Age'] > 26 ) & (data['Age'] <= 36),'Age'] = 2
        data.loc[(data['Age'] > 36 ) & (data['Age'] <= 62),'Age'] = 3
        data.loc[ data['Age'] > 62,'Age'] = 4
        data['Fare'] = Fare_mapping(data['Fare'])
        data['Cabin'] = Cabin_mapping(data['Cabin'])
        data['Sex'] = Sex_mapping(data['Sex'])
        data['FamilySize'] = Family_mapping(data['FamilySize'])
        # print(data)
        result = predict_result(data)
        print(result)
        if result >= 0.5:
            self.ui.label.setText('survive')
        else:
            self.ui.label.setText('die')

    # def display_img(self,result):
    #     self.img = cv2.imread(self.img_path)
    #     height, width, channel = self.img.shape
    #     bytesPerline = 3 * width
    #     self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
    #     self.ui.label.setPixmap(QPixmap.fromImage(self.qimg))

