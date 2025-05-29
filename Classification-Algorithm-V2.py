# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:12:59 2024

@author: Tales
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay , classification_report, roc_auc_score, jaccard_score

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from joblib import Parallel, delayed


#%%
def read_data(method, domain, all_signals=True):
    
    if all_signals == True:
        X = pd.read_csv(domain+'_features_all_signals.csv')
        y = pd.read_csv('Target_all_signals.csv')
        
    else:
        X = pd.read_csv(domain+'_'+method+'_features.csv')
        y = pd.read_csv('Target.csv')
    
    return X, y

#%% Define Parameters
encoder = LabelEncoder()
scaler = MinMaxScaler()

method = 'vibwall' #'personalized' , 'vibwall'
domain = 'stat' #'stat' , 'mfcc' , 'spectral
target = 'health' # 'wall', 'health'
all_signals = False
use_wall_info = True

if all_signals:
    root_path = 'Original'
else:
    root_path = 'Proposed'
    
grid_search = True

if grid_search:
    GS = 'optimized'
else:
    GS = 'not_optimized'


N_iter = 30
cv=5

accuracy = np.zeros(( N_iter*cv ))

roc_auc = np.zeros(( N_iter*cv ))

jaccard = np.zeros(( N_iter*cv ))

precision   = {'0' : np.zeros ((  N_iter*cv )),
               '1': np.zeros ((  N_iter*cv )),}

recall      = {'0' : np.zeros ((  N_iter*cv )),
               '1': np.zeros ((  N_iter*cv )),}

F1          = {'0' : np.zeros ((  N_iter*cv )),
               '1': np.zeros ((  N_iter*cv )),}

#%% Read Data
# X, y = read_data(method = method, 
#                   domain = domain, 
#                   all_signals = all_signals)

X1, y = read_data(method = 'vibwall', 
                  domain = 'stat', 
                  all_signals = all_signals)

X2, y = read_data(method = 'personalized', 
                  domain = 'spectral', 
                  all_signals = all_signals)


X = pd.concat([X1, X2], axis=1)


if use_wall_info == True:
    wall_info = pd.DataFrame(y.iloc[:,0])
    X['wall'] = pd.DataFrame(encoder.fit_transform(wall_info.iloc[:,0]))


y = pd.DataFrame(y.iloc[:,1])
y = pd.DataFrame(encoder.fit_transform(y.iloc[:,0]))

X.columns = np.arange(0, X.shape[1])

domain = 'stat_spect'


#%% Algorithm
cont = 0
cm, cr = 0, 0
best_params = []
best_scores = []
feature_importances = []
FI = np.zeros(X.shape[1])

for n in range(N_iter):
    kf = StratifiedKFold(n_splits=cv, random_state=n, shuffle=True)
    kf.get_n_splits(X)

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train = X.iloc[train_index,:]
        y_train = y.iloc[train_index]    
        
        X_test = X.iloc[test_index,:]
        y_test = y.iloc[test_index]   
        
        
        X_train_cv, X_validation_cv, y_train_cv, y_validation_cv = train_test_split(X_train, y_train, 
                                                            test_size=0.25,
                                                            random_state=n, 
                                                            shuffle=True)
        
        X_train = scaler.fit_transform(X_train)  
        X_test = scaler.transform(X_test)
        
        if grid_search:
        
            X_train_cv = scaler.fit_transform(X_train_cv)  
            X_validation_cv = scaler.transform(X_validation_cv)
    
            # model = RandomForestClassifier(random_state=cont)
            # model_name = 'RF'
            # parameters = {'max_depth': [2, 5, 10, 15],
            #               'n_estimators':[10, 50, 100, 200], 
            #                 'criterion': ('gini', 'entropy', 'log_loss'),
            #               }
    
            model = DecisionTreeClassifier(random_state=cont)
            model_name = 'DT'
            parameters = {'criterion': ('gini', 'entropy', 'log_loss'),
                          'splitter': ('best', 'random'),
                          'max_depth': [2, 5, 10, 15],
                          }
    
            # model = KNeighborsClassifier()
            # model_name = 'KNN'
            # parameters = {'n_neighbors': [1, 2, 3, 5, 10, 50, 100],
            #               'weights': ('uniform', 'distance'),}
            
            # model = GradientBoostingClassifier(random_state=cont)
            # model_name = 'GB'
            # parameters ={#'loss': ('log_loss', 'exponential'),
            #                 'learning_rate': [0.05, 0.10, 0.15, 0.20],
            #               'n_estimators' : [10, 50, 100, 200],
            #               #'criterion': ('friedman_mse', 'squared_error'),
            #               'max_depth': [1, 2, 5, 10, 15]
            #             }
            
            
            par_candidates = ParameterGrid(parameters)
            print(f'{len(par_candidates)} candidates')
            
            def fit_model(params):
                clf = model.set_params(**params)
                clf.fit(X_train_cv, y_train_cv)
                score = model.score(X_validation_cv, y_validation_cv)
                return [params, score]
            
            results = Parallel(n_jobs=-1, verbose=10)(delayed(fit_model)(params) for params in par_candidates)
            print(max(results, key=lambda x: x[1]))
            
            best = max(results, key=lambda x: x[1])[0]
            best_params.append(best)
            best_scores.append(max(results, key=lambda x: x[1])[1])
            
            model.set_params(**best)
            
        else:
        
            # model_name = 'RF'
            # model = RandomForestClassifier(max_depth=2, n_estimators=10, random_state=cont)
            model_name = 'DT'
            model = DecisionTreeClassifier(random_state=cont)
            # model_name = 'KNN'
            # model = KNeighborsClassifier(n_neighbors=2)
            # model_name = 'GB'
            # model = DecisionTreeClassifier(random_state=cont)
            # model = KNeighborsClassifier(n_neighbors=2)
            # model = GradientBoostingClassifier(random_state=cont)
        

        model.fit(X_train, y_train.iloc[:,0])
        if model_name != 'KNN':
            feature_importances.append(model.feature_importances_)
            FI += model.feature_importances_
        
        y_pred = model.predict(X_test)

        # Confusion Matrix
        print(confusion_matrix(y_test, y_pred))
        cm += confusion_matrix(y_test, y_pred)
        
        # Classification Report Matrix
        print(classification_report(y_test, y_pred))
        cr = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        
        accuracy[cont]        = cr.loc['accuracy'][0]
        
        precision['0'][cont]  = cr.iloc[0,0]
        precision['1'][cont]  = cr.iloc[1,0]
        
        recall['0'][cont]     = cr.iloc[0,1]
        recall['1'][cont]     = cr.iloc[1,1]
        
        F1['0'][cont]         = cr.iloc[0,2]
        F1['1'][cont]         = cr.iloc[1,2]   
        
        roc_auc[cont] = roc_auc_score(y_test, y_pred)
        jaccard[cont] = jaccard_score(y_test, y_pred)
        
        cont+=1
        
        
print(str(model).split('(')[0])
print('Average Accuracy: ' + str(np.round(100*accuracy.mean(), 2)) 
      + ' (' + str(np.round(100*accuracy.std(),2)) + ') ' ) 

print('ROC_AUC: ' + str(np.round(100*roc_auc.mean(), 2)) 
      + ' (' + str(np.round(100*roc_auc.std(),2)) + ') ' ) 

print('Jaccard: ' + str(np.round(100*jaccard.mean(), 2)) 
      + ' (' + str(np.round(100*jaccard.std(),2)) + ') ' ) 

print('Precision Score: ' + str(np.round(pd.DataFrame(precision).mean().mean()*100,2))
      + '(' + str(np.round(pd.DataFrame(precision).std().mean()*100,2)) + ')')

print('Recall Score: ' + str(np.round(pd.DataFrame(recall).mean().mean()*100,2))
      + '(' + str(np.round(pd.DataFrame(recall).std().mean()*100,2)) + ')')


pd.DataFrame(accuracy).to_json('./Results/' + str(root_path) + '/' + str(domain) + '/accuracy-' + str(model).split('(')[0] + '-' + str(GS) + '-' + str(domain) +'.json' )
pd.DataFrame(roc_auc).to_json('./Results/' + str(root_path) + '/' + str(domain) + '/roc_auc-' + str(model).split('(')[0] + '-' + str(GS) + '-' + str(domain) +'.json' )
pd.DataFrame(jaccard).to_json('./Results/' + str(root_path) + '/' + str(domain) + '/jaccard-' + str(model).split('(')[0] + '-' + str(GS) + '-' + str(domain) +'.json' )
pd.DataFrame(precision).to_json('./Results/' + str(root_path) + '/' + str(domain) + '/precision-' + str(model).split('(')[0] + '-' + str(GS) + '-' + str(domain) +'.json' )
pd.DataFrame(recall).to_json('./Results/' + str(root_path) + '/' + str(domain) +  '/recall-' + str(model).split('(')[0] + '-' + str(GS) + '-' + str(domain) +'.json' )
pd.DataFrame(F1).to_json('./Results/' + str(root_path) + '/' + str(domain) + '/F1-' + str(model).split('(')[0] + '-' + str(GS) + '-' + str(domain) +'.json' )

pd.DataFrame(best_params).to_json('./Results/' + str(root_path) + '/' + str(domain) + '/Best_Params-' + str(model).split('(')[0] + '-' + str(GS) + '-' + str(domain) +'json' )
pd.DataFrame(best_scores).to_json('./Results/' + str(root_path) + '/' + str(domain) + '/Best_Scores-' + str(model).split('(')[0] + '-' + str(GS) + '-' + str(domain) +'json' )


#%%
bp = pd.DataFrame(best_params)

for i in range(bp.shape[1]):
    pl.figure(dpi=500)
    bp.iloc[:,i].value_counts().plot(kind='bar')
    pl.ylabel('Frequency of Selection', fontsize=16)
    pl.xlabel('')
    pl.yticks(fontsize=16)
    pl.xticks(fontsize=16, rotation=0)
#<<<<<<< HEAD
    pl.savefig(fname = './Images/' + str(root_path) + '/' + str(model).split('(')[0] + '-' + str(GS) + '-'+ str(domain) + '_'+ str(i)+ '.pdf',
#=======
    #pl.savefig(fname = './Images/' + str(root_path) + '/' + str(model).split('(')[0] + '-' + str(GS) + '-' + str(domain)+'feature_' + str(i)+'.pdf',
#>>>>>>> 35ee365111dd1f8c094a44d892cca04dc0a72060
                format = "pdf", bbox_inches = "tight")
    pl.show()



#<<<<<<< HEAD
# # #%%
#=======
#%%
#>>>>>>> 35ee365111dd1f8c094a44d892cca04dc0a72060
cm = cm/cm.mean()
cm = 100 * cm / cm.sum(axis=1, keepdims=True)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Damaged'])
fig, ax = pl.subplots(dpi=500, figsize=(5,3))
disp.plot(ax=ax, cmap=pl.cm.Blues, values_format='.2f')
# pl.title( title )
pl.xlabel('Predicted Class')
pl.ylabel('True Class')
#<<<<<<< HEAD
pl.savefig(fname=('./Images/' + str(root_path) + '/CM-stat-spectral-' + str(model).split('(')[0] + '-' + str(GS) + '-' + str(domain) + '.pdf'),
            format="pdf", bbox_inches="tight")
pl.show()

#%%

FI = pd.DataFrame(FI) #importancia acumulada ao longo das 150 iterações
feat_names = ['MAV', 'VAR', 'RMS', 'STD', 'MAD', 
              'Frequency Skewness', 'Frequency Kurtosis', 
              'IQR', 'Energy',
              #
              'Spectral Centroid', 'Spectral Decrease', 'Spectral Distance',
              'Spectral Entropy', 'Spectral Kurtosis', 'Spectral Rolloff', 
              'Spectral Skewness', 'Spectral Spread', 'Max Power Spectrum',
              'Max Frequency', 'Median Frequency', 'Power Bandwith',
              #
              'Constituent Material'
              ]

FI.index = feat_names
FI2 = FI[::-1]

pl.figure(dpi=500)
FI2.plot(kind='barh')
pl.xticks(rotation=0)
pl.legend('')
pl.savefig(fname = './Images/' + str(root_path) + '/Features_Importances_' + str(model).split('(')[0] + '-' + str(GS) + '-'+ str(domain) + '_'+ str(i)+ '.pdf',
            format = "pdf", bbox_inches = "tight")
pl.show()


#%%
# import pylab as pl
# import pandas as pd

# DT = pd.read_csv('FI-DT.csv')
# RF = pd.read_csv('FI-RF.csv')

# a = DT.copy()
# b = RF.copy()

# c = a.sort_values('0')[::-1]
# c = c.merge(b, how='inner', on='Unnamed: 0')
# c.index = c['Unnamed: 0']
# c.drop(columns = ['Unnamed: 0'], inplace=True)
# c.columns = ['Decision Tree', 'Random Forest']


# pl.figure(dpi=500)
# c.plot(kind='bar', width = 0.7, subplots=True)
# # pl.ylabel('Frequency of Seleciom', fontsize=16)
# pl.xlabel('')
# pl.legend('')
# # pl.yticks(fontsize=14)
# pl.xticks(fontsize=14, rotation=90)
#=======
pl.savefig(fname=('./Images/' + str(root_path) + '/CM-' + str(model).split('(')[0] + '-' + str(GS) + '-' + str(domain) + '.pdf'),
            format="pdf", bbox_inches="tight")
pl.show()
#>>>>>>> 35ee365111dd1f8c094a44d892cca04dc0a72060
