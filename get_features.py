# -*- coding: utf-8 -*-
"""
Created on Sat May 18 08:11:21 2024

@author: tbora
"""

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

import tsfel
from python_speech_features import mfcc as psf_mfcc
# from mfcc import MFCC
from scipy import stats


#%%
def get_signals():
    
    walls = ['baker_brics', 'baker_hall', 'office_wall']
    state = ['good', 'bad']
    
    target = []

    for wall in walls:
        for gob in state:
            A = pd.DataFrame([])
            print(str(wall) + ' - ' + str(gob))
            for i in range(1, 101): # number of examples
                for j in range(1, 6): # number of patterns
                    # target.append([wall, gob])
                    pattern = str(j*10)
                    fileName = 'GAcceleratorFrequency'+pattern+'_'+str(i)
                    folder = wall + '/TestData_'+gob+'_wall_'+pattern
                    ls = './' + folder + '/'+fileName+'.txt' 
                    acc_z= []
                    
                    
                    with open(ls, 'r') as f:
                        lines = f.readlines()
                        print('Numero de Linhas: ' + str(len(lines)))
                        
                        if len(lines) > 500:
                            num = 0
                            for line in lines[:-1]:
                                num = num + 1
                                if num>1 and num<=501:
                                   value = line.rstrip().split(',')
                                   acc_z.append(float(value[2])) # 2 for z axis, 1 for y axis, 0 for x axis
                        else:
                            continue

                    A[pattern+'-'+str(i)] = np.array(acc_z)
                    target.append([wall, gob])
                    
            A.to_csv(wall+'_acc_signals_'+gob+'.csv', index=False) 
    pd.DataFrame(target).to_csv('Target.csv', index=False)

#%%
def get_feature_all_signals(domain='stat', fs=100):
    
    walls = ['baker_brics', 'baker_hall', 'office_wall']
    state = ['good', 'bad']
    
    X = []
    target = []

    for wall in walls:
        for gob in state:
            print(str(wall) + ' - ' + str(gob))
            for i in range(1, 101): # number of examples
                for j in range(1, 6): # number of patterns
                    target.append([wall, gob])
                    pattern = str(j*10)
                    fileName = 'GAcceleratorFrequency'+pattern+'_'+str(i)
                    folder = wall + '/TestData_'+gob+'_wall_'+pattern
                    ls = './' + folder + '/'+fileName+'.txt' 
                    acc_z= []
                    aux = []
                    
                    
                    with open(ls, 'r') as f:
                        lines = f.readlines()
                        print('Numero de Linhas: ' + str(len(lines)))
                        num = 0
                        for line in lines[:-1]:
                            num = num + 1
                            if num>1 and num<=500:
                               value = line.rstrip().split(',')
                               acc_z.append(float(value[2])) # 2 for z axis, 1 for y axis, 0 for x axis
                        
                        while(num<500): # caso o sinal seja menor que 5 segundos, ele repete o último valor
                            acc_z.append(acc_z[-1])
                            num = num + 1
                            
                    acc_z = np.array(acc_z) 
                        
                    if domain == 'stat':
                        aux.append( np.mean(abs(acc_z)) ) #MAV
                        aux.append( np.var(acc_z) ) #VAR
                        aux.append( np.sqrt(np.mean(acc_z**2)) ) #RMS
                        aux.append( np.std(acc_z) ) #STD
                        aux.append( np.median(np.abs(acc_z - np.median(acc_z))) ) #MAD
                        aux.append( stats.skew(abs(np.fft.fft(acc_z, acc_z.shape[0]))) ) #skewness freq
                        aux.append( stats.kurtosis(abs(np.fft.fft(acc_z, acc_z.shape[0]))) ) #kurtosis freq
                        aux.append( stats.iqr(acc_z) ) #IQR
                        aux.append( np.mean(acc_z**2) ) #Energy
                        
                        X.append(aux)
                        
                    elif domain == 'mfcc':
                        mfcc_coef = psf_mfcc(np.array(acc_z), fs)
                        X.append( mfcc_coef.reshape((mfcc_coef.shape[0]*mfcc_coef.shape[1],1)).ravel() )
                        
                    elif domain == 'spectral':
                        aux.append(tsfel.feature_extraction.features.spectral_centroid(acc_z, fs=fs))
                        aux.append(tsfel.feature_extraction.features.spectral_decrease(acc_z, fs=fs))
                        aux.append(tsfel.feature_extraction.features.spectral_distance(acc_z, fs=fs))
                        aux.append(tsfel.feature_extraction.features.spectral_entropy(acc_z, fs=fs))
                        aux.append(tsfel.feature_extraction.features.spectral_kurtosis(acc_z, fs=fs))
                        # aux.append(tsfel.feature_extraction.features.spectral_positive_turning(acc_z, fs=fs))
                        aux.append(tsfel.feature_extraction.features.spectral_roll_off(acc_z, fs=fs))
                        # aux.append(tsfel.feature_extraction.features.spectral_roll_on(acc_z, fs=fs))
                        aux.append(tsfel.feature_extraction.features.spectral_skewness(acc_z, fs=fs))
                        # aux.append(tsfel.feature_extraction.features.spectral_slope(acc_z, fs=fs))
                        aux.append(tsfel.feature_extraction.features.spectral_spread(acc_z, fs=fs))
                        # aux.append(tsfel.feature_extraction.features.spectral_variation(acc_z, fs=fs))
                        aux.append(tsfel.feature_extraction.features.max_power_spectrum(acc_z, fs=fs))
                        aux.append(tsfel.feature_extraction.features.max_frequency(acc_z, fs=fs))
                        aux.append(tsfel.feature_extraction.features.median_frequency(acc_z, fs=fs))
                        aux.append(tsfel.feature_extraction.features.power_bandwidth(acc_z, fs=fs))
                    
                        X.append(aux)

    X = pd.DataFrame(X)
    X.to_csv(domain + '_features_all_signals.csv', index=False)
    pd.DataFrame(target).to_csv('Target_all_signals.csv', index=False)

#%%
def vibwall_features(domain='stat', fs=100):
    
    walls = ['baker_brics', 'baker_hall', 'office_wall']
    state = ['good', 'bad']
    
    X = []
    
    for w in walls:
        for s in state:
            
            file_name = w+'_acc_signals_'+s+'.csv'
            data = pd.read_csv(file_name)
         
            for col in range(data.shape[1]):
                
                aux  = []
                
                if domain == 'stat':
                    
                    aux.append( np.mean(abs(data.iloc[:,col])) ) #MAV
                    aux.append( np.var(data.iloc[:,col]) ) #VAR
                    aux.append( np.sqrt(np.mean(data.iloc[:,col]**2)) ) #RMS
                    aux.append( np.std(data.iloc[:,col]) ) #STD
                    aux.append( np.median(np.abs(data.iloc[:,col] - np.median(data.iloc[:,col]))) ) #MAD
                    # aux.append( stats.skew(data.iloc[:,col]) ) #skewness time
                    aux.append( stats.skew(abs(np.fft.fft(data.iloc[:,col], data.iloc[:,col].shape[0]))) ) #skewness freq
                    # aux.append( stats.kurtosis(data[:,col]) ) #kurtosis time
                    aux.append( stats.kurtosis(abs(np.fft.fft(data.iloc[:,col], data.iloc[:,col].shape[0]))) ) #kurtosis freq
                    aux.append( stats.iqr(data.iloc[:,col]) ) #IQR
                    aux.append( np.mean(data.iloc[:,col]**2) ) #Energy
                    
                    X.append(aux)
                    
                elif domain == 'mfcc':
                    
                    mfcc_coef = psf_mfcc(np.array(data.iloc[:,col]), fs)
                    X.append( mfcc_coef.reshape((mfcc_coef.shape[0]*mfcc_coef.shape[1],1)).ravel() )

    X = pd.DataFrame(X)
    X.to_csv(domain + '_vibwall_features.csv', index=False)
            
#%%
def personalized_features(domain='stat', fs = 100):
    
    walls = ['baker_brics', 'baker_hall', 'office_wall']
    state = ['good', 'bad']
    
    X = []
    
    for w in walls:
        for s in state:
            
            file_name = w+'_acc_signals_'+s+'.csv'
            data = pd.read_csv(file_name)
         
            for col in range(data.shape[1]):
                
                aux  = []
                
                # if domain == 'stat':
                #     # Absolute Energy
                #     aux.append( np.sum(data.iloc[:,col]**2) )
                #     # Peak
                #     aux.append( max(abs(data.iloc[:,col])) )
                #     # Mean
                #     aux.append( np.mean(data.iloc[:,col]) )
                #     # Mean Square                
                #     aux.append( np.mean(data.iloc[:,col]**2) )
                #     # Root Mean Square                
                #     aux.append( np.sqrt(np.mean(data.iloc[:,col]**2)) )
                #     # Variance
                #     # aux.append( np.std(data.iloc[:,col])**2 )
                #     # Standard Deviation
                #     aux.append( np.std(data.iloc[:,col]) )
                #     # Skewness
                #     aux.append(tsfel.feature_extraction.features.skewness(data.iloc[:,col]))
                #     # Kurtosis
                #     aux.append(tsfel.feature_extraction.features.kurtosis(data.iloc[:,col]))
                #     # Crest Factor
                #     aux.append(max(abs(data.iloc[:,col]))/np.sqrt(np.mean(data.iloc[:,col]**2)))
                #     # K-Factor
                #     aux.append(max(abs(data.iloc[:,col]))*np.sqrt(np.mean(data.iloc[:,col]**2)))
                #     # Clearance Factor 
                #     aux.append(max(abs(data.iloc[:,col])) / (np.mean(np.sqrt((abs(data.iloc[:,col]))))**2))
                #     # Impulse Factor
                #     aux.append(max(abs(data.iloc[:,col])) / np.mean(abs(data.iloc[:,col])) )
                #     # Shape Factor
                #     aux.append(np.sqrt(np.mean(data.iloc[:,col]**2)) / np.mean(abs(data.iloc[:,col])))
                
                #     X.append(aux)
                    
                # elif domain == 'mfcc':
                #     mfcc_coef = psf_mfcc(np.array(data.iloc[:,col]), fs)
                #     X.append( mfcc_coef.reshape((mfcc_coef.shape[0]*mfcc_coef.shape[1],1)).ravel() )
                
                # el
                if domain == 'spectral':
                    aux.append(tsfel.feature_extraction.features.spectral_centroid(data.iloc[:,col], fs=fs))
                    aux.append(tsfel.feature_extraction.features.spectral_decrease(data.iloc[:,col], fs=fs))
                    aux.append(tsfel.feature_extraction.features.spectral_distance(data.iloc[:,col], fs=fs))
                    aux.append(tsfel.feature_extraction.features.spectral_entropy(data.iloc[:,col], fs=fs))
                    aux.append(tsfel.feature_extraction.features.spectral_kurtosis(data.iloc[:,col], fs=fs))
                    # aux.append(tsfel.feature_extraction.features.spectral_positive_turning(data.iloc[:,col], fs=fs))
                    aux.append(tsfel.feature_extraction.features.spectral_roll_off(data.iloc[:,col], fs=fs))
                    # aux.append(tsfel.feature_extraction.features.spectral_roll_on(data.iloc[:,col], fs=fs))
                    aux.append(tsfel.feature_extraction.features.spectral_skewness(data.iloc[:,col], fs=fs))
                    # aux.append(tsfel.feature_extraction.features.spectral_slope(data.iloc[:,col], fs=fs))
                    aux.append(tsfel.feature_extraction.features.spectral_spread(data.iloc[:,col], fs=fs))
                    # aux.append(tsfel.feature_extraction.features.spectral_variation(data.iloc[:,col], fs=fs))
                    aux.append(tsfel.feature_extraction.features.max_power_spectrum(data.iloc[:,col], fs=fs))
                    aux.append(tsfel.feature_extraction.features.max_frequency(data.iloc[:,col], fs=fs))
                    aux.append(tsfel.feature_extraction.features.median_frequency(data.iloc[:,col], fs=fs))
                    aux.append(tsfel.feature_extraction.features.power_bandwidth(data.iloc[:,col], fs=fs))
                
                    X.append(aux)
                
                # X.append(aux)
                
    X = pd.DataFrame(X)
    X.to_csv(domain+'_personalized_features.csv', index=False)

#%%

get_signals()
# get_feature_all_signals(domain='stat', fs=100)
# get_feature_all_signals(domain='mfcc', fs=100)
# get_feature_all_signals(domain='spectral', fs=100)
vibwall_features(domain='stat', fs=100)
vibwall_features(domain='mfcc', fs=100)
personalized_features(domain='spectral', fs=100)







#%% Rascunho Pra saber tamanho dos menores sinais
# num =0
# signal = []
# for line in lines[1:-1]:
#     num = num + 1
#     value = line.rstrip().split(',')
#     signal.append(float(value[2]))  

# if len(lines) < 500:
#     print('Numero de Linhas: ' + str(len(lines)))
#     pl.plot(signal)
#     pl.show()

#########
# baker_brics - good
# Numero de Linhas: 357
# Numero de Linhas: 151
# Numero de Linhas: 142
# Numero de Linhas: 443
# Numero de Linhas: 105
# baker_brics - bad
# baker_hall - good
# Numero de Linhas: 179
# Numero de Linhas: 289
# Numero de Linhas: 176
# Numero de Linhas: 467
# Numero de Linhas: 43
# baker_hall - bad
# Numero de Linhas: 119
# Numero de Linhas: 179
# Numero de Linhas: 487
# Numero de Linhas: 496
# Numero de Linhas: 492
# Numero de Linhas: 498
# office_wall - good
# Numero de Linhas: 81
# Numero de Linhas: 204
# Numero de Linhas: 183
# Numero de Linhas: 124
# Numero de Linhas: 100
# office_wall - bad
# Numero de Linhas: 137
# Numero de Linhas: 31
# Numero de Linhas: 161
######









