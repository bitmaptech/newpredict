import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def Model_Training():
    data = pd.read_csv("roo_data.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    data['Acedamic_percentage_in_Operating_Systems'] = le.fit_transform(data['Acedamic_percentage_in_Operating_Systems'])

    data['percentage_in_Algorithms'] = le.fit_transform(data['percentage_in_Algorithms'])
    data['Percentage_in_Programming_Concepts'] = le.fit_transform(data['Percentage_in_Programming_Concepts'])
    data['Percentage_in_Software_Engineering'] = le.fit_transform(data[' Percentage_in_Software_Engineering'])
    data['Percentage_in_Computer_Networks'] = le.fit_transform(data['Percentage_in_Computer_Networks'])
   
    data['Logical_quotient_rating'] = le.fit_transform(data['Logical_quotient_rating'])
    data['hackathons'] = le.fit_transform(data['hackathons'])
    data['coding_skills_rating'] = le.fit_transform(data['coding_skills_rating'])
    data['self_learning_capability'] = le.fit_transform(data['self_learning_capability'])
    data['memory_capability_score'] = le.fit_transform(data['memory_capability_score'])
    data['interested_career_area'] = le.fit_transform(data['interested_career_area'])
    data['interested_in_games'] = le.fit_transform(data['interested_in_games'])
   
   

    """Feature Selection => Manual"""
    x = data.drop(['Suggested_Job_Role'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Suggested_Job_Role']
    print(type(y))
    x.shape
    

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=1234)

    # from sklearn.svm import SVC
    # svcclassifier = SVC(kernel='linear')
    # svcclassifier.fit(x_train, y_train)
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    svcclassifier = DecisionTreeClassifier()
    #svcclassifier = RandomForestClassifier()
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    import pickle 
    pickle.dump(svcclassifier,open('model1.pkl','wb'))
    
    # print("Confusion Matrix :")
    # cm = confusion_matrix(y_test,y_pred)
    # print(cm)
    # print("\n")
    # from mlxtend.plotting import plot_confusion_matrix

    # fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.show()

Model_Training()



