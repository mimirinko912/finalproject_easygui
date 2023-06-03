import pandas as pd
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import Adam
from sklearn import preprocessing

dataset = pd.read_csv('input/titanic.csv')

def data_cleaning(testdata):
    testdata['Title'] = testdata['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 3, 
                     "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                     "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, 
                     "Mme": 3,"Capt": 3,"Sir": 3 }
    testdata['Title'] = testdata['Title'].map(title_mapping)
    
    sex_mapping = {"male":0, "female": 1}
    testdata['Sex'] = testdata['Sex'].map(sex_mapping)
    
    testdata["Age"].fillna(testdata.groupby("Title")["Age"].transform("median"), inplace = True)
    testdata.loc[testdata["Age"]<=16, "Age"]=0
    testdata.loc[(testdata["Age"]> 16)&(testdata["Age"]<= 26), "Age"]=1
    testdata.loc[(testdata["Age"]> 26)&(testdata["Age"]<= 36), "Age"]=2
    testdata.loc[(testdata["Age"]> 36)&(testdata["Age"]<= 62), "Age"]=3
    testdata.loc[testdata["Age"] > 62, "Age"]=4
    
    testdata["Embarked"] = testdata["Embarked"].fillna("S")
    embarked_mapping = {"S":0, "C": 1, "Q": 2}
    testdata['Embarked'] = testdata['Embarked'].map(embarked_mapping)
    
    testdata["Fare"].fillna(testdata.groupby("Pclass")["Fare"].transform("median"), inplace = True)
    testdata.loc[testdata["Fare"]<=17, "Fare"]=0
    testdata.loc[(testdata["Fare"]> 17)&(testdata["Fare"]<= 30), "Fare"]=1
    testdata.loc[(testdata["Fare"]> 30)&(testdata["Fare"]<= 100), "Fare"]=2
    testdata.loc[testdata["Fare"] > 100, "Fare"]=3
    
    testdata["Cabin"] = testdata["Cabin"].str[:1]
    cabin_mapping = {"A":0, "B": 0.4, "C": 0.8, "D": 1.2, 
                 "E":1.6,"F":2,"G":2.4,"T":2.8}
    testdata['Cabin'] = testdata['Cabin'].map(cabin_mapping)
    testdata['Cabin'].fillna(testdata.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
    
    testdata["FamilySize"] = testdata["SibSp"] + testdata["Parch"]+1
    family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 
                  8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
    testdata['FamilySize'] = testdata['FamilySize'].map(family_mapping)
    
    features_drop = ["Name","Ticket", "SibSp", "Parch", "PassengerId"]
    testdata=testdata.drop(features_drop, axis=1)
    
    return testdata

    
def build_model():
    model = Sequential()

    model.add(Dense(input_dim=8,units=40))            #The number of columns for each data
    model.add(Activation("relu"))
    model.add(Dense(units=100))
    model.add(Activation("relu"))
    model.add(Dense(units=10))
    model.add(Activation("relu"))
    model.add(Dense(units=1))
    model.add(Activation("sigmoid"))
    model.summary()
    return model  

    
def trainDNN():
    build_model()
    
    dataset = pd.read_csv('input/titanic.csv')
    dataset_data = data_cleaning(dataset)

    dataset_target2 = dataset[['Survived']]
    dataset_data.shape, dataset_target2.shape

    train_label = dataset[['Survived']]
    dataset_data = dataset_data.drop("Survived", axis = 1)
    
    minmax_scale = preprocessing.MinMaxScaler(feature_range = (0,1))
    scaledFeatures = minmax_scale.fit_transform(dataset_data)

    model = build_model()
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
    model.fit(x = scaledFeatures, y = dataset_target2, validation_split = 0.2, batch_size = 30, epochs = 30)
    
    score = model.evaluate (x = scaledFeatures, y = dataset_target2)
    print ('\nTrain Loss:', score[0])
    print ('\nTrain Acc:', score[1])

    testdata = pd.read_csv('input/test.csv')
    testdata = data_cleaning(testdata)

    survived = model.predict(testdata).flatten().round(0).astype(int)
    testdata_write = pd.read_csv('input/test.csv')
    answer = pd.read_csv("input/answer_passengerID.csv")

    submission = pd.DataFrame({
       "PassengerId": testdata_write['PassengerId'],
       "Survived": survived
    })

    answer_int = answer[['Survived']]
    submission_int = submission[['Survived']]
    print("\nYour score is: ",accuracy_score(answer_int, submission_int))
    
    model.save('DNN_model.h5')
    
    return accuracy_score(answer_int, submission_int)


