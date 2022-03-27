from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def algo_predict(input_marks):
    x = np.array([input_marks]).reshape(1, -1)
    root = os.path.dirname(__file__)
    path_df = os.path.join(root, 'recons_dataset/1.csv')
    data = pd.read_csv(path_df)

     
    train, test = train_test_split(data, test_size=0.25)

    X_train = train.drop('CLASS', axis=1)
    Y_train = train['CLASS']

    X_test = test.drop('CLASS', axis=1)
    Y_test = test['CLASS']
    print("ddddddddddddddddddddddddddddddddd")
    print(X_test[0:1])
     
    #clf = RandomForestClassifier()
    clf = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto')

    #clf=linear_model.LogisticRegression(fit_intercept=False)
    #clf = DecisionTreeClassifier(criterion = "entropy", random_state = 1, splitter='best')  
    # Training the classifier
    clf.fit(X_train, Y_train)

    ypredict = clf.predict(x)
    print("ddddddddddddddddddddddddddddddddd")
    #print(x[0])
    print(ypredict)
    return ypredict

def algo_predict1(input_marks):
    x = np.array([input_marks]).reshape(1, -1)
    root = os.path.dirname(__file__)
    path_df = os.path.join(root, 'recons_dataset/2.csv')
    data = pd.read_csv(path_df)

     
    train, test = train_test_split(data, test_size=0.25)

    X_train = train.drop('CLASS', axis=1)
    Y_train = train['CLASS']

    X_test = test.drop('CLASS', axis=1)
    Y_test = test['CLASS']
    print("ddddddddddddddddddddddddddddddddd")
    print(X_test[0:1])
     
    #clf = RandomForestClassifier()
    clf = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto')

    #clf=linear_model.LogisticRegression(fit_intercept=False)
    #clf = DecisionTreeClassifier(criterion = "entropy", random_state = 1, splitter='best')  
    # Training the classifier
    clf.fit(X_train, Y_train)

    ypredict = clf.predict(x)
    print("ddddddddddddddddddddddddddddddddd")
    #print(x[0])
    print(ypredict)
    return ypredict

def algo_predict2(input_marks):
    x = np.array([input_marks]).reshape(1, -1)
    root = os.path.dirname(__file__)
    path_df = os.path.join(root, 'recons_dataset/3.csv')
    data = pd.read_csv(path_df)

     
    train, test = train_test_split(data, test_size=0.25)

    X_train = train.drop('CLASS', axis=1)
    Y_train = train['CLASS']

    X_test = test.drop('CLASS', axis=1)
    Y_test = test['CLASS']
    print("ddddddddddddddddddddddddddddddddd")
    print(X_test[0:1])
     
    clf = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto')
    #clf = RandomForestClassifier()
    #clf=linear_model.LogisticRegression(fit_intercept=False)
    #clf = DecisionTreeClassifier(criterion = "entropy", random_state = 1, splitter='best')  
    # Training the classifier
    clf.fit(X_train, Y_train)

    ypredict = clf.predict(x)
    print("ddddddddddddddddddddddddddddddddd")
    #print(x[0])
    print(ypredict)
    return ypredict
def algo_predict3(input_marks):
    x = np.array([input_marks]).reshape(1, -1)
    root = os.path.dirname(__file__)
    path_df = os.path.join(root, 'recons_dataset/4.csv')
    data = pd.read_csv(path_df)

     
    train, test = train_test_split(data, test_size=0.25)

    X_train = train.drop('CLASS', axis=1)
    Y_train = train['CLASS']

    X_test = test.drop('CLASS', axis=1)
    Y_test = test['CLASS']
    print("ddddddddddddddddddddddddddddddddd")
    print(X_test[0:1])
     
    #clf = RandomForestClassifier()
    clf = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto')

    #clf=linear_model.LogisticRegression(fit_intercept=False)
    #clf = DecisionTreeClassifier(criterion = "entropy", random_state = 1, splitter='best')  
    # Training the classifier
    clf.fit(X_train, Y_train)

    ypredict = clf.predict(x)
    print("ddddddddddddddddddddddddddddddddd")
    #print(x[0])
    print(ypredict)
    return ypredict
def algo_predict4(input_marks):
    x = np.array([input_marks]).reshape(1, -1)
    root = os.path.dirname(__file__)
    path_df = os.path.join(root, 'recons_dataset/5.csv')
    data = pd.read_csv(path_df)

     
    train, test = train_test_split(data, test_size=0.25)

    X_train = train.drop('CLASS', axis=1)
    Y_train = train['CLASS']

    X_test = test.drop('CLASS', axis=1)
    Y_test = test['CLASS']
    print("ddddddddddddddddddddddddddddddddd")
    print(X_test[0:1])
     
    #clf = RandomForestClassifier()
    clf = KNeighborsClassifier(n_neighbors = 1, weights='uniform', algorithm='auto')

    #clf=linear_model.LogisticRegression(fit_intercept=False)
    #clf = DecisionTreeClassifier(criterion = "entropy", random_state = 1, splitter='best')  
    # Training the classifier
    clf.fit(X_train, Y_train)

    ypredict = clf.predict(x)
    print("ddddddddddddddddddddddddddddddddd")
    #print(x[0])
    print(ypredict)
    return ypredict

def first_subject(input_marks):
    predicted_val=algo_predict(input_marks)
    return predicted_val

def second_subject(input_marks):
    predicted_val=algo_predict1(input_marks)
    return predicted_val

def third_subject(input_marks):
    predicted_val=algo_predict2(input_marks)
    return predicted_val

def fourth_subject(input_marks):
    predicted_val=algo_predict3(input_marks)
    return predicted_val

def fifth_subject(input_marks):
    predicted_val=algo_predict4(input_marks)
    return predicted_val
