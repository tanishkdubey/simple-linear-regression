import numpy as np
import pandas as pd 
import sklearn.linear_model as Lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#Code for simple linear regression

    #Reading CSV file so that we can work will it
df = pd.read_csv("data/Salary_dataset.csv")


    #Select model

model = Lm.LinearRegression()

    #defining x,y for the date

x = df[["YearsExperience"]]
y = df["Salary"]

    #spliting data as training and test

x_train ,x_test ,y_train , y_test = train_test_split(x ,y , test_size=0.2, random_state=0)

    #train model
model.fit(x_train , y_train)

y_pred = model.predict(x_test)
 #check how your model perform
print(r2_score(y_test , y_pred))
print(y_pred)