import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import impyute

dataset = pd.read_csv('PlasticMarinePollutionGlobalDataset_Expectation_Maximization.csv')
dataset['3'] = dataset['3'].str.replace(',', '')
dataset['4'] = dataset['4'].str.replace(',', '')
dataset['5'] = dataset['5'].str.replace(',', '')
dataset['8'] = dataset['8'].str.replace(',', '')
dataset['9'] = dataset['9'].str.replace(',', '')
dataset['10'] = dataset['10'].str.replace(',', '')

X = dataset.iloc[:, 3:11]
y = dataset.iloc[:, 1]

X = X.astype(float)

impyute.imputation.cs.em(X, loops=50, inplace=True)

#X.to_csv('Expectation_MAX_Val.csv', sep=',')

X = (X-X.min())/(X.max()-X.min())
y = (y-y.min())/(y.max()-y.min())

X_train, X_eval,y_train,y_eval=train_test_split(X,y,test_size=0.2,random_state=101)

#Creating Feature Columns
feat_cols=[]
for cols in dataset.iloc[:, 3:11]:
    column=tf.feature_column.numeric_column(cols)
    feat_cols.append(column)
    
print(feat_cols)


#The estimator model
model = tf.estimator.DNNRegressor(hidden_units=[6,10,6],feature_columns=feat_cols)

#the input function
input_func = tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=10,num_epochs=10000,shuffle=True)

#Training the Model
model.train(input_fn=input_func,steps=10000)

#Evaluating the model
train_metrics=model.evaluate(input_fn=input_func,steps=10000)


#Now to predict values we do the following
pred_input_func=tf.estimator.inputs.pandas_input_fn(x=X_eval,y=y_eval,batch_size=10,num_epochs=1,shuffle=False)
preds=model.predict(input_fn=pred_input_func)

predictions=list(preds)
final_pred=[]
for pred in predictions:             
    final_pred.append(pred["predictions"])
    
test_metric=model.evaluate(input_fn=pred_input_func,steps=10000)    


