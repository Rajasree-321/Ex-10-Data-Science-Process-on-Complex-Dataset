# Ex-10-Data-Science-Process-on-Complex-Dataset

# Aim: 
To Perform Data Science Process on a complex dataset and save the data to a file. 

# ALGORITHM 
  STEP 1 Read the given Data.
  
  
  STEP 2 Clean the Data Set using Data Cleaning Process.
  
  STEP 3 Apply Feature Generation/Feature Selection Techniques on the data set.
  
  STEP 4 Apply EDA /Data visualization techniques to all the features of the data set.
  
# Code:

import numpy as np 

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("/content/Disease_symptom_and_patient_profile_dataset.csv")

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/96121b9c-2c8e-4c41-bc26-d92a2ffee54d)

df.head()

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/d63b7f1a-7f92-4e1b-a9f7-50bb4eac670d)

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/3ab50acb-49c7-4dba-b24b-eb472c0b5dc3)

df.tail()

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/62e0c72c-96d7-42c1-b734-0986595767c1)

df.isnull().sum()

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/fa9a1d62-bea0-4e44-b972-2e2fabb22f41)

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/d729fc34-e053-41bb-9aab-6014800de827)

# Feature Scaling

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, [0, 2]]

Y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Example dataset
data = [[10, 20, 30],
        [5, 15, 25],
        [3, 6, 9],
        [8, 12, 16]]
        
#Min-max scaling

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

print("Min-max scaled data:")

print(scaled_data)

#Standard scaling

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print("Standard scaled data:")

print(scaled_data)

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/2ad730b5-8a51-43d0-b9fe-5a944eccdde9)

# Data Visualization Methods

#Histogram

np.random.seed(42)

data = np.random.normal(0, 1, 1000)

plt.figure(figsize=(5, 5))

plt.hist(data, bins=30, edgecolor='black')

plt.xlabel('Value')

plt.ylabel('Frequency')

plt.title('Histogram')

plt.show()

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/31dc9cb3-5212-444a-8299-99c914871ed1)

sns.catplot(x='Outcome Variable',y='Age',data=df,kind="swarm")

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/f540584b-69ee-4cfd-a2df-6af7b46a8216)

sns.catplot(x='Difficulty Breathing' , kind='count',data=df , hue = "Outcome Variable")

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/f18b0446-acac-459f-9cd1-053c00329ff5)

sns.catplot(x='Fatigue' , kind='count',data=df , hue = "Outcome Variable")

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/3abe5989-3c1e-41fe-a498-e86f37432b6d)

plt.figure(figsize=(5,5))

sns.barplot(x='Fever',y='Age',data =df);

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/2d58fd5b-5676-487e-95b4-4745006ab9d0)

sns.displot(df['Age'],kde=True)

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/eb5c8a58-c83f-4ed2-9913-f3cbe33acf0e)

df.groupby('Gender').size().plot(kind='pie', autopct='%.2f')

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/f7330a88-c587-4924-99fc-ec380d6fd06e)

sns.catplot(x='Cough' , kind='count',data=df , hue = "Cholesterol Level")

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/ce11a1b5-adff-4975-be54-3c13aa9edb74)

df.groupby('Blood Pressure').size().plot(kind='pie', autopct='%.2f')

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/c91e4cac-b61f-415a-80d9-7d1abf70f089)

sns.catplot(x='Gender' , kind='count',data=df , hue = "Cholesterol Level")

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/ee864155-51c7-4f16-a1e0-8d015a47908c)

#dropping name column

df = df.iloc[:,1:]

df.groupby('Fatigue').size().plot(kind='pie', autopct='%.2f')

![image](https://github.com/Rajasree-321/Ex-10-Data-Science-Process-on-Complex-Dataset/assets/96918911/7c70b579-9149-4e5b-9f4a-9dc7123197b0)

# Result
 Thus the Data Visualization  and Feature Generation/Feature Selection Techniques for the given dataset had been executed successfully


