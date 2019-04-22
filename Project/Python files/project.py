#CAP5627-3

import os
import sys
import numpy as npy
from statistics import mean
from scipy.stats import entropy
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# Function getData() is used to process the data set, and to store the mean, variance, maximum, minimum, entropy values of all the files in Dataframe
def getData(path):
    entireArray = []
    attributeNames = ['Subject', 'Bp_dia_mean', 'Bp_dia_max', 'Bp_dia_min', 'Bp_dia_var', 'Bp_dia_ent', 'Bp_mmhg_mean', 'Bp_mmhg_max', 'Bp_mmhg_min', 'Bp_mmhg_var', 'Bp_mmhg_ent', 'eda_mean', 'eda_max', 'eda_min', 'eda_var', 'eda_ent', 'La_mbp_mean', 'La_mbp_max', 'La_mbp_min', 'La_mbp_var', 'La_mbp_ent', 'La_sys_mean', 'La_sys_max', 'La_sys_min', 'La_sys_var', 'La_sys_ent', 'Pulse_mean', 'Pulse_max', 'Pulse_min', 'Pulse_var', 'Pulse_ent', 'Resp_volt_mean', 'Resp_volt_max', 'Resp_volt_min', 'Resp_volt_var', 'Resp_volt_ent', 'Resp_rate_mean', 'Resp_rate_max', 'Resp_rate_min', 'Resp_rate_var', 'Resp_rate_ent', 'Label']
    for subjects in os.scandir(path):
        for task in os.scandir(subjects.path):
            tValue = 1
            if task.name == 'T8':
                tValue = 0
            mArray = []
            files = os.scandir(task)
            mArray.append(subjects.name)
            for f in files:
                if f.name != 'ECG_mV.txt' and f.name != '.DS_Store':
                    fileRequest = open(f.path, "r")
                    fileData = fileRequest.readlines()
                    dataArray = npy.array(fileData).astype(npy.float)
                    #Calculation of mean, variance, entropy, max, min
                    mean=float("%3.4f" % (npy.mean(dataArray)))
                    maximum=max(dataArray)
                    minimum=min(dataArray)
                    variance=float("%3.4f" % (npy.var(dataArray)))
                    entropy1=float("%3.4f" % (entropyMeasure(dataArray)))
                    mArray.append(mean)
                    mArray.append(maximum)
                    mArray.append(minimum)
                    mArray.append(variance)
                    mArray.append(entropy1)
            mArray.append(tValue)
            entireArray.append(mArray)
    df = pd.DataFrame(entireArray, columns=attributeNames)
    return df
        
        
# Function getAverageData() is used to process the data set, and to store the average value of mean, variance, maximum, minimum, entropy values of all the files in Dataframe
def getAverageData(path):
    attributeNames = ['Subject', 'A1', 'A2', 'A3', 'A4', 'A5', 'Label']
    entireArray = []
    for subjects in os.scandir(path):
        for task in os.scandir(subjects.path):
            tValue = 1
            if task.name == 'T8':
                tValue = 0
            meanArray = []
            maxArray = []
            minArray = []
            varArray = []
            entArray = []
            mArray = []
            files = os.scandir(task)
            for f in files:
                if f.name != 'ECG_mV.txt' and f.name != '.DS_Store':
                    fileRequest = open(f.path, "r")
                    fileData = fileRequest.readlines()
                    dataArray = npy.array(fileData).astype(npy.float)
                    #Calculation of mean, variance, entropy, max, min
                    mean=float("%3.4f" % (npy.mean(dataArray)))
                    maximum=max(dataArray)
                    minimum=min(dataArray)
                    variance=float("%3.4f" % (npy.var(dataArray)))
                    entropy1=float("%3.4f" % (entropyMeasure(dataArray)))
                    meanArray.append(mean)
                    maxArray.append(maximum)
                    minArray.append(minimum)
                    varArray.append(variance)
                    entArray.append(entropy1)
            mArray.append(subjects.name)
            mArray.append(npy.mean(meanArray))
            mArray.append(npy.mean(maxArray))
            mArray.append(npy.mean(minArray))
            mArray.append(npy.mean(varArray))
            mArray.append(npy.mean(entArray))
            mArray.append(tValue)
            entireArray.append(mArray)
    df = pd.DataFrame(entireArray, columns=attributeNames)
    return df


# Function doSvm() is used to do the classification of pain or no pain of the dataset stored in data frame using SVM classifier 
def doSvm(dataSet):
    df = getData(dataSet)
    dfAi = getAverageData(dataSet)
    row_count = round(len(df))
    trainCount = row_count*0.8
    testCount = trainCount
    
    #data splitting for train and test
    dfTrain = df[:int(trainCount)].copy(deep=True)
    dfTest = df[int(testCount):].copy(deep=True)
    dfAiTrain = dfAi[:int(trainCount)].copy(deep=True)
    dfAiTest = dfAi[int(testCount):].copy(deep=True)
    X_train = dfTrain.iloc[:,2:41]
    y_train = dfTrain['Label'] 
    X_test = dfTest.iloc[:,2:41]
    y_test = dfTest['Label'] 
    
    #training our model
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    
    #testing our model
    y_pred = svclassifier.predict(X_test)  
    
    print('---confusion_matrix---')
    print(confusion_matrix(y_test,y_pred))  
    print('---classification_report---')
    print(classification_report(y_test,y_pred))
    print('Accuracy Score:', metrics.accuracy_score(y_test, y_pred))
    print('---Prediction---')
    print(y_pred.round())   
    doCorrelation(dfAiTrain, dfAiTest, y_pred.round())


# Function doRF() is used to do the classification of pain or no pain of the dataset stored in data frame using RANDOM FOREST classifier    
def doRf(dataSet):
    df = getData(dataSet)
    dfAi = getData(dataSet)
    row_count = round(len(df))
    trainCount = row_count*0.8
    testCount = trainCount
    
    #splitting of data for train and test
    dfTrain = df[:int(trainCount)].copy(deep=True)
    dfTest = df[int(testCount):].copy(deep=True)
    dfAiTrain = dfAi[:int(trainCount)].copy(deep=True)
    dfAiTest = dfAi[int(testCount):].copy(deep=True)
    X_train = dfTrain.iloc[:,2:41]
    y_train = dfTrain['Label'] 
    X_test = dfTest.iloc[:,2:41]
    y_test = dfTest['Label']
    
    #training our model
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)  
    regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
    regressor.fit(X_train, y_train)  
    
    #testing our model
    y_pred = regressor.predict(X_test)
    
    print('---confusion_matrix---')
    print(confusion_matrix(y_test,y_pred.round()))
    print('---classification_report---')  
    print(classification_report(y_test,y_pred.round()))
    print('Accuracy Score:', metrics.accuracy_score(y_test, y_pred.round()))
    print('---Prediction---')
    print(y_pred.round())
    doCorrelation(dfAiTrain, dfAiTest, y_pred.round())


# Function doCorrelation() is used to do find the correlation of subjects in the test set with all the subjects in the training set and to do the classification based on the correlation value
def doCorrelation(dfTrain, dfTest, yPrediction):
    y_corr_pred=[]
    y_mis_test=[]
    
    #data splitting for training and testing
    X_train = dfTrain.iloc[:,2:6]
    y_train = dfTrain['Label'] 
    X_test = dfTest.iloc[:,2:6]
    y_test = dfTest['Label']
    
    y_corr=pd.DataFrame(columns=['Label'], index=X_test.index)
    y_predict=pd.DataFrame(columns=['Label'], index=X_test.index)  
    
    #Finding correlation for test set subjects
    for i in range(len(X_test)):
        corr=X_train.iloc[:,2:7].corrwith(X_test.iloc[i,2:7], axis=1)
        ind=npy.argmax(corr)
        y_corr_pred.append(y_train.loc[ind])
    y_corr['Label']=y_corr_pred
    y_predict['Label']=yPrediction
    
    #Classifier vs Correlation Accuracy Score
    print('Classifier vs Correlation Accuracy Score:', metrics.accuracy_score(yPrediction, y_corr_pred))
    
    #Misclassified Correlation vs Ground Truth Accuracy :
    misclassified = y_predict.iloc[npy.where(yPrediction != y_corr_pred)]
    y_misclf_corr=y_corr.loc[misclassified.index]
    y_misclf_test=y_test.loc[misclassified.index]
    print('Misclassified Correlation vs Ground Truth Accuracy :', metrics.accuracy_score(y_misclf_test, y_misclf_corr))     

# Function entropyMeasure is used to find the entropy value for the physilogical values in dataset
def entropyMeasure(labels, base=None):
    value,counts = npy.unique(labels, return_counts=True)
    return entropy(counts, base=base)

# The main() function invokes all other methods based on the command line argument values
def main():
    dataSet = sys.argv[2]
    dataSplit = dataSet.split('/')
    selector = len(dataSplit)
    if sys.argv[1] == 'SVM':
        print(dataSplit[selector-1] + ' ' + 'SVM')
        doSvm(dataSet)
    else:
        print(dataSplit[selector-1] + ' ' + 'RF')
        doRf(dataSet)
        
main()

