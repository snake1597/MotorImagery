import matplotlib.pyplot as plt
import csv, os, sys, copy, pywt
import numpy as np
from loadData import load
from sklearn import preprocessing

def indices(mylist, value):
    return [i for i,x in enumerate(mylist) if x==value]

def showAveragepower():
    path = './bci2003/'
    subject = 'trainData'.split(' ')
    loadData = load(subNames=subject, path=path,isRand=True)
    (x_train, y_train) = loadData.loadTrainDataFromTxt()
    
    d_avg = dict()
    for tp in range(2):
        tmp_index = indices(y_train, tp)
        #print(len(tmp_index)) #show class quantity
        tempList = x_train[tmp_index]
        fixList = np.sum(np.power(tempList, 2), axis=0)
        #fixList = np.sum(tempList,axis=0) #original data
        resultList = np.divide(fixList,tempList.shape[0])
        d_avg[tp] = resultList 
    LH = np.array(d_avg[0])
    RH = np.array(d_avg[1])
    time = np.arange(LH[:,0].size) / 128
    plt.subplot(211)
    plt.plot(time,LH)
    plt.legend(['c3','cZ','c4'], loc='upper left')    
    plt.subplot(212)
    plt.plot(time,RH)
    plt.legend(['c3','cZ','c4'], loc='upper left')    
    plt.show()

def showDWT():
    path = './bci2003'
    subject = 'trainData_DWT'.split(' ')
    loadData = load(subNames=subject, path=path)
    (x_train, y_train) = loadData.loadTrainDataFromTxt()

    d_avg = dict()
    for tp in range(2):
        tmp_index = indices(y_train, tp)
        #print(len(tmp_index)) #show class quantity
        tempList = x_train[tmp_index]
        fixList = np.sum(tempList, axis=0)
        resultList = np.divide(fixList, tempList.shape[0])
        d_avg[tp] = resultList
    plt.subplot(211)
    plt.plot(d_avg[0])
    plt.legend(['c3','cZ','c4'], loc='upper left')    
    plt.subplot(212)
    plt.plot(d_avg[1])
    plt.legend(['c3','cZ','c4'], loc='upper left')    
    plt.show()
    print("end")

if __name__ == "__main__":
    showAveragepower()
    showDWT()
