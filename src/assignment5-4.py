#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import Decimal


# In[34]:


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


# In[35]:


def SSE(dataset, prediction, means):
    sse = np.sum([distEclud(dataset[i], means[prediction[i]]) ** 2 for i in range(len(prediction))])
    return sse


# In[36]:


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


# In[37]:


def init_parameters(dataset, K):
    # use Decimal to improve precision
    weights  = np.random.rand(K)
    weights /= np.sum(weights)        # Normalize
    weights = np.array([Decimal(str(x)) for x in weights])
    
    col = np.shape(dataset)[1]
    means = []
    for i in range(K):
        mean = np.random.rand(col)
        means.append(mean)
    means = [[Decimal(str(x)) for x in y] for y in means]
    means = np.array(means)

    covars  = []
    for i in range(K):
        cov = np.random.rand(col,col)
        cov = [[Decimal(str(x)) for x in y] for y in cov]
        cov = np.array(cov)
        covars.append(cov)
    covars = np.array(covars)
    
    return weights, means, covars

def Gaussian(x,mean,cov):
    dim = np.shape(cov)[0]
    
    # prevent determinant of cov being 0
    cov_eye = cov.astype(np.float64) + np.eye(dim) * 0.001
    covdet = np.linalg.det(cov_eye)
    covinv = np.linalg.pinv(cov_eye)
    
    # use Decimal to improve precision
    covinvDecimal = [[Decimal(str(x)) for x in y] for y in covinv]
    covinvDecimal = np.array(covinvDecimal)
    
    xdiff = (x - mean).reshape((1,dim))
    prob = Decimal(str(1.0/(np.power(np.power(2*np.pi,dim)*np.abs(covdet),0.5))))*            np.exp(Decimal('-0.5')*xdiff.dot(covinvDecimal).dot(xdiff.T))[0][0]
    return prob

def cal_loglikelyhood(m, dataset, K, weights, means, covars):
    loglikelyhood = []
    for n in range(m):
        tmp = np.sum([weights[k]*Gaussian(dataset[n],means[k],covars[k]) for k in range(K)])
        tmp = tmp.ln()
        loglikelyhood.append(tmp)
    loglikelyhood = np.sum(loglikelyhood)
    return loglikelyhood


# In[40]:


def GMM_EM(dataset, K):
    weights, means, covars = init_parameters(dataset, K)
    loglikelyhood = Decimal(1)
    oldloglikelyhood = Decimal(0)
    m,dim = np.shape(dataset)
    # initialize gamma(Znk)
    gammas = [np.zeros(K) for i in range(m)]
    i = 0
    while np.abs(loglikelyhood-oldloglikelyhood) > Decimal('1e-12') and i < 10:
        oldloglikelyhood = loglikelyhood
        # E-step, use Decimal to improve precision
        for n in range(m):
            respons = [Decimal(str(weights[k])) * Gaussian(dataset[n], means[k], covars[k])
                                                for k in range(K)]
            respons = np.array(respons)
            sum_respons = np.sum(respons)
            gammas[n] = respons/sum_respons   # Normalize
        # M-step, use Decimal to improve precision
        for k in range(K):
            # how many samples from m belong to k-th cluster
            nk = np.sum([gammas[n][k] for n in range(m)])
            # update weights
            weights[k] = 1 * nk / m
            # update means
            means[k] = (1/nk) * np.sum([gammas[n][k] * dataset[n] for n in range(m)], axis=0)
            xdiffs = dataset - means[k]
            # update covars
            covars[k] = (1/nk)*np.sum([gammas[n][k]*xdiffs[n].reshape((dim,1)).dot(xdiffs[n].reshape((1,dim))) for n in range(m)],axis=0)
        loglikelyhood = cal_loglikelyhood(m, dataset, K, weights, means, covars)
        i += 1
#         print('loglikelyhood')
#         print(loglikelyhood)
#         print('change')
#         print(np.abs(loglikelyhood-oldloglikelyhood))
    prediction = [np.argmax(gammas[i]) for i in range(m)]
    prediction = np.array(prediction)
    return prediction, means


# In[41]:


def chooseBestK(dataset):
    # use Decimal to improve precision
    data = np.array(dataset.iloc[:,0:-1])
    data = [[Decimal(str(x)) for x in y] for y in data]
    data = np.array(data)

    SSEs = []
    NMIs = []
    for k in range(1, 31):
        prediction, means = GMM_EM(data, k)
        print(prediction)
        sse = SSE(data, prediction, means)
        nmi = NMI(np.array(dataset.iloc[:,-1]), prediction)
        SSEs.append(sse)
        NMIs.append(nmi)
        print(k)
    
    SSEs = [float(x) for x in SSEs]
    
    # find bestK
    sort_index = sorted(range(len(SSEs)), key=lambda k: SSEs[k])
    boundary = 1.2 * SSEs[sort_index[0]]
    bestK_SSE = 0
    for i in range(len(sort_index)):
        if SSEs[sort_index[i]] > boundary:
            bestK_SSE = min(sort_index[0:i]) + 1
            break
            
    sort_index = sorted(range(len(NMIs)), key=lambda k: NMIs[k], reverse=True)
    boundary = 0.8 * NMIs[sort_index[0]]
    bestK_NMI = 0
    for i in range(len(sort_index)):
        if NMIs[sort_index[i]] < boundary:
            bestK_NMI = min(sort_index[0:i]) + 1
            break
            
    xAxis = list(range(1, 31))
    yAxis = SSEs
    plt.plot(xAxis, yAxis)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.show()
    
    xAxis = list(range(1, 31))
    yAxis = NMIs
    plt.plot(xAxis, yAxis)
    plt.xlabel('k')
    plt.ylabel('NMI')
    plt.show()
    
    return bestK_SSE, bestK_NMI, SSEs, NMIs


# In[29]:


# Dermatology dataset
dataset = pd.read_csv('Assignment5/dermatologyData.csv', header = None)
prob_name = 'Dermatology'
bestK_SSE1, bestK_NMI1, SSEs1, NMIs1 = chooseBestK(dataset)
print(bestK_SSE1)
print(bestK_NMI1)
print(SSEs1)
print(NMIs1)


# In[42]:


# Vowels dataset
dataset = pd.read_csv('Assignment5/vowelsData.csv', header = None)
prob_name = 'Vowels'
bestK_SSE2, bestK_NMI2, SSEs2, NMIs2 = chooseBestK(dataset)
print(bestK_SSE2)
print(bestK_NMI2)
print(SSEs2)
print(NMIs2)


# In[28]:


# Glass dataset
dataset = pd.read_csv('Assignment5/glassData.csv', header = None)
prob_name = 'Glass'
bestK_SSE3, bestK_NMI3, SSEs3, NMIs3 = chooseBestK(dataset)
print(bestK_SSE3)
print(bestK_NMI3)
print(SSEs3)
print(NMIs3)


# In[36]:


# Ecoli dataset
dataset = pd.read_csv('Assignment5/ecoliData.csv', header = None)
prob_name = 'Ecoli'
bestK_SSE4, bestK_NMI4, SSEs4, NMIs4 = chooseBestK(dataset)
print(bestK_SSE4)
print(bestK_NMI4)
print(SSEs4)
print(NMIs4)


# In[ ]:


# Yeast dataset
dataset = pd.read_csv('Assignment5/yeastData.csv', header = None)
prob_name = 'Yeast'
bestK_SSE5, bestK_NMI5, SSEs5, NMIs5 = chooseBestK(dataset)
print(bestK_SSE5)
print(bestK_NMI5)
print(SSEs5)
print(NMIs5)


# In[37]:


# Soybean dataset
dataset = pd.read_csv('Assignment5/soybeanData.csv', header = None)
prob_name = 'Soybean'
bestK_SSE6, bestK_NMI6, SSEs6, NMIs6 = chooseBestK(dataset)
print(bestK_SSE6)
print(bestK_NMI6)
print(SSEs6)
print(NMIs6)


# In[ ]:


bestKTable = {}
datasetNames = ['Dermatology','Vowels','Glass','Ecoli','Yeast','Soybean']
for name in datasetNames:
    bestKTable[name] ={}
bestKs = [bestK_SSE1, bestK_SSE2, bestK_SSE3, bestK_SSE4, bestK_SSE5, bestK_SSE6]
allSSE = [SSEs1, SSEs2, SSEs3, SSEs4, SSEs5, SSEs6]
allNMI = [NMIs1, NMIs2, NMIs3, NMIs4, NMIs5, NMIs6]
for i in range(len(datasetNames)):
    bestKTable[datasetNames[i]]['K'] = bestKs[i]
    bestKTable[datasetNames[i]]['SSE'] = allSSE[i][bestKs[i]-1]
    bestKTable[datasetNames[i]]['NMI'] = allNMI[i][bestKs[i]-1]
df = pd.DataFrame(bestKTable, index = ['K', 'SSE', 'NMI'], columns = datasetNames)
print(df)


# In[ ]:


bestKTable = {}
datasetNames = ['Dermatology','Vowels','Glass','Ecoli','Yeast','Soybean']
for name in datasetNames:
    bestKTable[name] ={}
bestKs = [bestK_NMI1, bestK_NMI2, bestK_NMI3, bestK_NMI4, bestK_NMI5, bestK_NMI6]
allSSE = [SSEs1, SSEs2, SSEs3, SSEs4, SSEs5, SSEs6]
allNMI = [NMIs1, NMIs2, NMIs3, NMIs4, NMIs5, NMIs6]
for i in range(len(datasetNames)):
    bestKTable[datasetNames[i]]['K'] = bestKs[i]
    bestKTable[datasetNames[i]]['SSE'] = allSSE[i][bestKs[i]-1]
    bestKTable[datasetNames[i]]['NMI'] = allNMI[i][bestKs[i]-1]
df = pd.DataFrame(bestKTable, index = ['K', 'SSE', 'NMI'], columns = datasetNames)
print(df)


# In[ ]:


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

