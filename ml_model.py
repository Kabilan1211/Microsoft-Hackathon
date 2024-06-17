import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,ConfusionMatrixDisplay,precision_score,roc_curve
import seaborn as sns
from sklearn.utils import shuffle
from ydata_profiling import ProfileReport
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

df = pd.read_csv('dataset.csv')
df = shuffle(df,random_state=42)
df.head()

for col in df.columns:

    df[col] = df[col].str.replace('_',' ')
df.head()

df.describe()

null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print(null_checker)

plt.figure(figsize=(10,5))
plt.plot(null_checker.index, null_checker['count'])
plt.xticks(null_checker.index, null_checker.index, rotation=45,
horizontalalignment='right')
plt.title('Before removing Null values')
plt.xlabel('column names')
plt.margins(0.1)
plt.show()

cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)

df = pd.DataFrame(s, columns=df.columns)
df.head()

df = df.fillna(0)
df.head()

df1 = pd.read_csv('Symptom-severity.csv')
df1['Symptom'] = df1['Symptom'].str.replace('_',' ')
df1.head()

df1['Symptom'].unique()

vals = df.values
symptoms = df1['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]

d = pd.DataFrame(vals, columns=cols)
d.head()

d = d.replace('dischromic  patches', 0)
d = d.replace('spotting  urination',0)
df = d.replace('foul smell of urine',0)
df.head(10)

null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print(null_checker)

plt.figure(figsize=(10,5))
plt.plot(null_checker.index, null_checker['count'])
plt.xticks(null_checker.index, null_checker.index, rotation=45,
horizontalalignment='right')
plt.title('After removing Null values')
plt.xlabel('column names')
plt.margins(0.01)
plt.show()

print("Number of symptoms used to identify the disease ",len(df1['Symptom'].unique()))
print("Number of diseases that can be identified ",len(df['Disease'].unique()))

df['Disease'].unique()

data = df.iloc[:,1:].values
labels = df['Disease'].values

x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3,random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,test_size=0.10/(0.30))
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

Model_1 = RandomForestClassifier(n_estimators=100,random_state=42)
Model_2 = SVC(random_state=42)
Model_3 = GaussianNB()

model = BaggingClassifier(base_estimator=Model_1,n_estimators=100,random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

discription = pd.read_csv("symptom_Description.csv")
precaution = pd.read_csv("symptom_precaution.csv")

joblib.dump(model, "model.pkl")