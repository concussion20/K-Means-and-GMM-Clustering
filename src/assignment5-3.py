#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


# In[3]:


def SSE(clusterAssement):
    return np.sum(clusterAssement[:,1])


# In[4]:


def NMI(A,B):
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)

    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat


# In[5]:


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros([k, n]))
    for j in range(n):
        minj = np.min(dataSet[:,j])
        rangej = float(np.max(dataSet[:,j]) - minj)
        centroids[:,j] = np.mat(minj + rangej * np.random.rand(k, 1))   
    return centroids


# In[6]:


def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]    
    clusterAssement = np.mat(np.zeros([m,2]))
    centroids = randCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJ = distEclud(centroids[j,:], dataSet[i,:])
                if distJ <  minDist:
                    minDist = distJ 
                    minIndex = j 
            if clusterAssement[i,0] != minIndex:
                clusterChanged = True 
            clusterAssement[i,:] = minIndex, minDist ** 2
        
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssement[:,0].A1 == cent)]
            centroids[cent,:] = np.mean(ptsInClust, axis = 0)
    return centroids, clusterAssement 


# In[7]:


def chooseBestK(dataset):
    SSEs = []
    NMIs = []
    for k in range(1, 31):
        centroids, clusterAssement = KMeans(np.mat(dataset.iloc[:,:-1]), k)
        nmi = NMI(clusterAssement[:,0].A1, np.array(dataset.iloc[:,-1]))
        sse = SSE(clusterAssement)
        SSEs.append(sse)
        NMIs.append(nmi)
    
    # find bestK
    sort_index = sorted(range(len(SSEs)), key=lambda k: SSEs[k])
    boundary = 1.2 * SSEs[sort_index[0]]
    bestK = 0
    for i in range(len(sort_index)):
        if SSEs[sort_index[i]] > boundary:
            bestK = min(sort_index[0:i]) + 1
            break
            
    xAxis = list(range(1, 31))
    yAxis = SSEs
    plt.plot(xAxis, yAxis)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.show()
    return bestK, SSEs, NMIs


# In[8]:


# Dermatology dataset
dataset = pd.read_csv('Assignment5/dermatologyData.csv', header = None)
prob_name = 'Dermatology'

bestK1, SSEs1, NMIs1 = chooseBestK(dataset)
print(bestK1)
print(SSEs1)
print(NMIs1)


# In[12]:


# Vowels dataset
dataset = pd.read_csv('Assignment5/vowelsData.csv', header = None)
prob_name = 'Vowels'

bestK2, SSEs2, NMIs2 = chooseBestK(dataset)
print(bestK2)
print(SSEs2)
print(NMIs2)


# In[9]:


# Glass dataset
dataset = pd.read_csv('Assignment5/glassData.csv', header = None)
prob_name = 'Glass'

bestK3, SSEs3, NMIs3 = chooseBestK(dataset)
print(bestK3)
print(SSEs3)
print(NMIs3)


# In[10]:


# Ecoli dataset
dataset = pd.read_csv('Assignment5/ecoliData.csv', header = None)
prob_name = 'Ecoli'

bestK4, SSEs4, NMIs4 = chooseBestK(dataset)
print(bestK4)
print(SSEs4)
print(NMIs4)


# In[13]:


# Yeast dataset
dataset = pd.read_csv('Assignment5/yeastData.csv', header = None)
prob_name = 'Yeast'

bestK5, SSEs5, NMIs5 = chooseBestK(dataset)
print(bestK5)
print(SSEs5)
print(NMIs5)


# In[11]:


# Soybean dataset
dataset = pd.read_csv('Assignment5/soybeanData.csv', header = None)
prob_name = 'Soybean'

bestK6, SSEs6, NMIs6 = chooseBestK(dataset)
print(bestK6)
print(SSEs6)
print(NMIs6)


# In[14]:


bestKTable = {}
datasetNames = ['Dermatology','Vowels','Glass','Ecoli','Yeast','Soybean']
for name in datasetNames:
    bestKTable[name] ={}
bestKs = [bestK1, bestK2, bestK3, bestK4, bestK5, bestK6]
allSSE = [SSEs1, SSEs2, SSEs3, SSEs4, SSEs5, SSEs6]
allNMI = [NMIs1, NMIs2, NMIs3, NMIs4, NMIs5, NMIs6]
for i in range(len(datasetNames)):
    bestKTable[datasetNames[i]]['K'] = bestKs[i]
    bestKTable[datasetNames[i]]['SSE'] = allSSE[i][bestKs[i]-1]
    bestKTable[datasetNames[i]]['NMI'] = allNMI[i][bestKs[i]-1]
df = pd.DataFrame(bestKTable, index = ['K', 'SSE', 'NMI'], columns = datasetNames)
print(df)


# In[15]:


nClassTable = {}
datasetNames = ['Dermatology','Vowels','Glass','Ecoli','Yeast','Soybean']
for name in datasetNames:
    nClassTable[name] ={}
nClass = [6, 11, 6, 5, 9, 15]
allSSE = [SSEs1, SSEs2, SSEs3, SSEs4, SSEs5, SSEs6]
allNMI = [NMIs1, NMIs2, NMIs3, NMIs4, NMIs5, NMIs6]
for i in range(len(datasetNames)):
    nClassTable[datasetNames[i]]['K'] = nClass[i]
    nClassTable[datasetNames[i]]['SSE'] = allSSE[i][nClass[i]-1]
    nClassTable[datasetNames[i]]['NMI'] = allNMI[i][nClass[i]-1]
df = pd.DataFrame(nClassTable, index = ['K', 'SSE', 'NMI'], columns = datasetNames)
print(df)


# In[ ]:




