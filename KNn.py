import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.multioutput import MultiOutputClassifier

#%%
df = pd.read_csv("file2.csv")
df.head()

#%%
df['Department'].value_counts()
sns.countplot(x='Department',data=df)

#%%
df['Class'].value_counts()
sns.countplot(x='Class',data=df)

#%%
df['Subclass'].value_counts()
sns.countplot(x='Subclass',data=df)

#%%
# Features & Labels
Xfeatures = df['Title']
ylabels = df[['Department','Class','Subclass']]

#%%
# Split Data
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.2,random_state=7)

#%%
knn = KNeighborsClassifier(n_neighbors=3)
classifier = MultiOutputClassifier(knn)

pipe_knn = Pipeline(steps=[('cv',CountVectorizer()),('knn',classifier)])

#%%
# Fit on Dataset
pipe_knn.fit(x_train,y_train)

#%%
# Accuracy Score
pipe_knn.score(x_test,y_test)

#%%
# Sample Prediction
print(x_test.iloc[0])
print("Actual Prediction:",y_test.iloc[0])

#%%
ex1 = x_test.iloc[0]
ex1
#%%
pipe_knn.predict([ex1])

#%%
# Prediction Prob
# print(pipe_lr.classes_)
# pipe_lr.predict_proba([ex1])
#%%
import pickle

pickle.dump(pipe_knn, open('knn_model.pkl', 'wb'))

#%%
pickled_model = pickle.load(open('Logistic_model.pkl', 'rb'))
pickled_model.predict(x_test)











