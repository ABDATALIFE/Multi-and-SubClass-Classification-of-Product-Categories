import pandas as z
import matplotlib.pyplot as plt

df = z.read_csv("DatasetFinal.csv")
# df.head()
# a = df.Category.unique()
# len(a)

#%%
df[['Department', 'Class','Subclass']] = df.Category.str.split("-", expand = True)
# print(df)
# df

#%%
df.drop(df.columns[[1]], axis=1, inplace=True)

#%%
df = df.dropna()

#%%
df.Department = df.Department.str.capitalize()
df.Class = df.Class.str.capitalize()
df.Subclass = df.Subclass.str.capitalize()

print(len(df.Department.unique()))
print(len(df.Class.unique()))
print(len(df.Subclass.unique()))

#%%
df.to_csv('file2.csv',header = True, index = False)

#%%
dz = z.read_csv("DatasetFinal.csv")
dz= dz[dz['Category'] != '0']
dz['Category'] = dz['Category'].apply(str.lower)

print(dz.Category.unique())
print(len(dz.Category.unique()))

dz.to_csv('file1.csv',header = True, index = False)

