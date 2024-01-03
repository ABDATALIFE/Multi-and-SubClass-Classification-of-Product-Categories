import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# model = LogisticRegression()

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
from sklearn.multioutput import MultiOutputClassifier

#%%
# Features & Labels
Xfeatures = df['Title']
ylabels = df[['Department','Class','Subclass']]

#%%
# Split Data
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.2,random_state=7)

#%%
pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),
                          ('lr_multi',MultiOutputClassifier(LogisticRegression(solver='lbfgs', max_iter=1000)))])

#%%
# Fit on Dataset
pipe_lr.fit(x_train,y_train)

#%%
# Accuracy Score
pipe_lr.score(x_test,y_test)

#%%
# Sample Prediction
print(x_test.iloc[0])
print("Actual Prediction:",y_test.iloc[0])

#%%
ex1 = x_test.iloc[0]
ex1
#%%
pipe_lr.predict([ex1])

#%%
# Prediction Prob
# print(pipe_lr.classes_)
# pipe_lr.predict_proba([ex1])
#%%
import pickle

#%%
pickle.dump(pipe_lr, open('Logistic_model.pkl', 'wb'))

#%%
pickled_model = pickle.load(open('Logistic_model.pkl', 'rb'))


#%%
Predictions = pickled_model.predict(Xfeatures)


#%%
Predictions = pd.DataFrame(Predictions)

#%%
MC_Predicted = pd.Series(Predictions[0])
MC_Predicted = MC_Predicted.str.strip(' ')
SC_Predicted = pd.Series(Predictions[1])
SC_Predicted = SC_Predicted.str.strip(' ')
TC_Predicted = pd.Series(Predictions[2])
TC_Predicted = TC_Predicted.str.strip(' ')

#%%
Predicted_Category= pd.Series(MC_Predicted.astype(str) +'-'+SC_Predicted.astype(str)+'-'+TC_Predicted.astype(str))


#%%
df = pd.read_csv('DatasetFinal.csv')
df2 = df[df['Category']=='0']
df_1 = df[df['Category'] == '1']
df3 = df.drop(df[(df['Category']=='0')].index)
df3 = df3.drop(df[(df['Category']=='1')].index)
df3['Category'] = df3['Category'].str.lower()
df3[['MC','SC','TC']] = df3['Category'].str.split("-",expand=True)
df3['MC'] = df3['MC'].str.strip(' ')
df3['SC'] = df3['SC'].str.strip(' ')
df3['TC'] = df3['TC'].str.strip(' ')
df3 = df3.dropna()
del df, df2, df_1

#%%
df3['Category'] = df3['MC'].astype(str)+'-'+df3['SC'].astype(str)+'-'+df3['TC'].astype(str)




#%%

accuracy = []

#%%
for i in range(len(Predicted_Category)):
    
    if (Predicted_Category[i] == df3['Category'][i]) :
        accuracy.append(1)






