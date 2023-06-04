# NCU Introduction to programming skills for artificial intelligence 2023
finalproject_easygui pyQT powered GUI

![1200px-Python_and_Qt svg](https://github.com/mimirinko912/finalproject_easygui/assets/71892273/ca8814d6-e3a9-46b0-8a33-8e03b71effb3)


## guide
run ```start.py``` , after the GUI shows up, simply key in data and values, and select the model you like.

* SVM
* DNN
* RF

click **predict**, to get the result from our **pre-trained model**.

you can also build a model clicking **train**, system will build a model with the type you choosed.

## requirement
python 3.9.15 (not necessary)
``` pip install -r requirements.txt ```

## how to build
after finishing designing UI in QtDesigner, run the ```make_ui.bat``` to build the ```ui.py```.

if not working, copy and paste ```pyuic5 pyqt_gui.ui -o ui.py``` in your terminal.

## files
* ```start.py``` is the main function that calls every other function.
* ```front.py``` get event from the GUI and do things.
* ```ui.py``` is the UI file built by QtDesigner.

## screenshots
![image](https://github.com/mimirinko912/finalproject_easygui/assets/71892273/29b76f0c-b159-4a1c-9565-4926e2f39fda)



