#Model.py
#Build a model to predict the purchase price of a transaction given historical Black Friday transaction data

#Import libraries
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Read in data and replace NAs
allData = pd.read_csv('./train.csv', dtype={'User_ID': str})
allData.fillna(999, inplace=True)

#Create target array and input dataframe
target = np.array(allData.Purchase)
inputData = allData.copy()

#Calculate the Number of Purchases a user has made
inputDataCopy = inputData.copy()
inputDataCopy['User_Purchase_Count'] = 1
inputDataCopy = inputDataCopy.groupby('User_ID', as_index=False)['User_Purchase_Count'].count()
inputData = inputData.merge(inputDataCopy, on='User_ID')

#Calculate the Product ID's average purchase price as a factor
inputDataCopy = inputData.copy()
inputDataCopy = inputDataCopy.groupby('Product_ID', as_index=False)['Purchase'].mean()
inputDataCopy = inputDataCopy.rename(index=str, columns={'Purchase': 'PurchaseAvg'})
inputData = inputData.merge(inputDataCopy, on='Product_ID')

#Drop the target variable from the input data and removed factors that decreased model accuracy
inputData = inputData.drop(['Purchase'], axis=1)

#Reformat categorical variables into strings for encoding
inputData = inputData.applymap(str)
inputData.loc[inputData['Stay_In_Current_City_Years'] == '4+', 'Stay_In_Current_City_Years'] = '4'
inputData = inputData.astype({'Stay_In_Current_City_Years': int, 'User_Purchase_Count': int, 'PurchaseAvg': float})

#Drop factors that were decreasing model accuracy
inputData = inputData.drop(['User_Purchase_Count', 'PurchaseAvg'], axis=1)

#Reformat dataframe into an array
inputData = np.array(inputData)

# Encode categorical variables into integers
cols = [0,1,2,3,4,5,7,8,9,10]
for i in cols:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(inputData[:,i]))
    inputData[:, i] = lbl.transform(inputData[:, i])

#Create training and testing sets
trainData, testData, trainTarget, testTarget = train_test_split(inputData, target, test_size = 0.3, random_state = 17)

#Create boosting model's dataset
xgbTrain = xgb.DMatrix(trainData, label=trainTarget)

#Create parameters for 3 models with varying depth and max iterations
params = {}
depth = [6, 8, 10]
nrounds = [150, 125, 100]

predictions = pd.DataFrame()

#Loop through 3 models to train and determine predictions
for i in range(0, len(depth)):
	params['max_depth'] = depth[i]
	plst = list(params.items())
	model = xgb.train(plst, xgbTrain, nrounds[i])
	modelPredictions = model.predict(xgb.DMatrix(testData))
	predictions['model'+str(i+1)] = modelPredictions

#Average predictions of all 3 models
predictions['average'] = predictions.mean(axis=1)

#Print out predictions, actuals, and Root Mean Squared Error
modelPredictions = predictions["average"]
modelSqError = np.square(abs(modelPredictions - testTarget))
RMSE = np.sqrt(np.mean(modelSqError))

print('\n\nModel Predictions')
print(modelPredictions)
print('\n\nActuals')
print(testTarget)
print('\n\nModel Error')
print(RMSE)

#Root Mean Square Error = 2546






