from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib
import os
import numpy as np
import pickle
import time
import graphviz
import pandas as pd
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn.metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from model import * 
from sklearn import linear_model
app = Flask(__name__, static_folder='static')

@app.route("/")
def index():
    return render_template('home.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    dbms = int(request.form['dbms'])
    osa = int(request.form['osa'])
    lac = int(request.form['lac'])
    se = int(request.form['se'])
    mea = int(request.form['mea'])
    x = np.array([dbms]).reshape(1, -1)
    final = []
    first=first_subject(dbms)
    second=second_subject(osa)
    third=third_subject(lac)
    fourth=fourth_subject(se)
    fifth=fifth_subject(mea)
    print("dddddddddddddddddd")
    print(final) 
    
    
    return render_template('Result.htm', first=int(first),second=int(second),third=int(third),fourth=int(fourth),fifth=int(fifth))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
