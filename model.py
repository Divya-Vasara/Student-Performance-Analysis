import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
df = pd.read_csv("StudentsPerformance.csv")
categorical_cols = df.select_dtypes(include='object').columns
categorical_cols
Index(['gender', 'race/ethnicity', 'parental level of education', 'lunch',
       'test preparation course'],
      dtype='object')
for i in categorical_cols:
    print(df[i].unique())
df.isnull().sum()
sns.countplot(df['gender'])
count_test = df['test preparation course'].value_counts()
labels = df['test preparation course'].value_counts().index
plt.figure(figsize= (6,6))
plt.pie(count_test,labels=labels,autopct='%1.1f%%')
plt.legend(labels)
plt.show()
df['average_score']=(df['math score']+df['reading score']+df['writing score'])/3
df
sns.scatterplot(x=df['average_score'],y=df['math score'],hue=df['gender'])
sns.scatterplot(x=df['average_score'],y=df['reading score'],hue=df['gender'])
gender = {
    'male':1,
    'female':0
}
race = {
    'group A':0,
    'group B':1,
    'group C':2,
    'group D':3,
    'group E':4
}
df['gender']=df['gender'].map(gender)
df['race/ethnicity']=df['race/ethnicity'].map(race)
dflevel = {
    "bachelor's degree":0,
    'some college':1,
    "master's degree":2,
    "associate's degree":3,
    "high school":4,
    "some high school":5
}
df['parental level of education']=df['parental level of education'].map(level)
dfdf = pd.get_dummies(df,drop_first=True)
dfx = df.drop(columns='average_score').values
y = df['average_score'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x_train,y_train)
RandomForestRegressor()
predictions=model.predict(x_test)
predictions
from sklearn.metrics import r2_score
print(r2_score(predictions,y_test))
0.9972615591014802
pickle.dump(model,open('df.pkl','wb'))
