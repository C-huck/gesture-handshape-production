# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:11:16 2019

@author: Jack
"""

#Classification
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from featureCoding_fixed4 import featureCoding #Edit filename
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef,confusion_matrix

#stats
import numpy as np
from scipy import stats

#plots
import seaborn as sns
import matplotlib.pyplot as plt

def binomial_cmf(k, n, p):
    c = 0
    for k1 in range(n+1):
        if k1>=k:
            c += binom.pmf(k1, n, p)
    return c

class FeatureSelector():
    """
    ad hoc class so that categorical and numerical features 
    can be preprocessed separately and then joined
    """
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self._feature_names ] 

#import data
l = "handshape-project-02-20-20-main-only-1.csv" #update filename
df = pd.read_csv(l)

#convert handshape codes to component features
lambdafunc = lambda x: pd.Series(featureCoding(x['HS1Code']))
df[['compF','compJ','flexion','nsf_flexion','selfing','aperture_change','nsf_thumb','thumb_flex']] = df.apply(lambdafunc, axis=1)

#Add a feature for presence of 2nd hand
df['HS2Code'] = df['HS2Code'].fillna(0)
df['noHands'] = [0 if x==0 else 1 for x in df['HS2Code'] ]

#remove data from native signer
df = df[df['SignerStatus']==0]

#set up classification
categorical_features = ['apertureChange','noHands','nsf_flexion']
numerical_features = ['compF','compJ','flexion','nsf_flexion','nsf_thumb','thumb_flex']
categorical_pipeline = Pipeline( steps = [ ( 'cat_selector', FeatureSelector(categorical_features) ) ] ) #leaves categorical data untouched
numerical_pipeline = Pipeline( steps = [ ( 'num_selector', FeatureSelector(numerical_features) ), ( 'std_scaler', StandardScaler() ) ] ) #scales numerical data
full_pipeline = FeatureUnion( transformer_list = [ ( 'categorical_pipeline', categorical_pipeline ), ( 'numerical_pipeline', numerical_pipeline ) ] ) 

pipe = Pipeline([
    ('full_pipeline', full_pipeline),
    ('reduce_dim', SelectKBest(f_classif,k=4)),
    ('classify',
                 SVC(C=100, cache_size=200, class_weight='balanced',
                     kernel='linear', max_iter=-1))
])


###Main analysis
X = df; columns = ["All events"]

###Alternating events
#X = df[df['Alternate-1']==1]; columns = ["Alternating events"]

###Events of manipulation/movement
#X = df[df['Alternate-1']==0]
#X = X[(X['Manipulation']==1) | (X['Movement']==1)]; columns = ["Manipulation/Movement events"]

###Events of tool-use/manner
#X = df[df['Alternate-1']==0]
#X = X[(X['Tool']==1) | (X['Manner']==1)]; columns = ["Tool-use/Manner events"]

###Ancillary analyses: effect of event, effect of participant
#Sort by event ID or participant; evaluates whether classification accuracy is affected by particular events/participants
#Trains and tests on disjoint sets of events/participants when uncommented and k-fold splitter shuffle set to False
#X = X.sort_values(by=['EventFilename'])
#X = X.sort_values(by=['Participant'])

y = X['LabelBin']


kf = StratifiedKFold(n_splits=6,shuffle=True,random_state=6) #0 for alts; 3 for manip; 3 for tool; 0 for all
avg = []
coefs = []
raw_acc = []
predictions = []
y_trues = []
f_scores = []
wrong_answers = []
right_answers = []
probas = []
p_vals = []
event_ids = []
participants = []
intercepts = []
for train,test in kf.split(X,y):
    #pipe.fit(X.loc[train],y.loc[train])
    pipe.fit(X.iloc[train],y.iloc[train])
    score = pipe.score(X.iloc[test],y.iloc[test])
    pred = pipe.predict(X.iloc[test])
    actual = y.iloc[test]
    
    event_ids.append(X.iloc[train]['EventFilename'])
    participants.append(X.iloc[train]['Participant'])
    
    selbest = pipe['reduce_dim']
    print(selbest.get_support())
    #print(selbest.scores_)
    f_scores.append(selbest.scores_)
    cl_weights = pipe['classify']
    coef_vec = np.zeros(len(selbest.get_support()))
    #print(np.zeros(len(selbest.get_support)))
    coefs.append(cl_weights.coef_[0])

    intercepts.append(pipe['classify'].intercept_)

    #accu = pipe['accuracy']
    #support_vec = pipe['classify'].n_support_
    #print(support_vec)
    #print(score)
    avg.append(score)
    predictions+=list(pred)
    y_trues+=list(actual)
    
    raw_acc = accuracy_score(pred,actual,normalize=False)
    if list(actual).count(0) > len(actual)/2:
        bbl = list(actual).count(0)/len(actual)
    else:
        bbl = actual.sum()/len(actual)
    p_vals.append(binomial_cmf(raw_acc,len(actual),bbl))
    #print("bbl: ",bbl)

    
    eventNames = X.iloc[test]['EventFilename']
    participantNames = X.iloc[test]['Participant']
    for (name,pname,prediction,label) in zip(eventNames,participantNames,pred,actual):
        if prediction != label:
           wrong_answers.append([name,pname,prediction,label])
        else:
           right_answers.append([name,pname,prediction,label])
    
acc = np.mean(avg)
mcc = matthews_corrcoef(y_trues,predictions)
print("MCC:",round(mcc,4))
print("Acc: {perc} ({num}/{total})".format(perc=round(accuracy_score(y_trues,predictions),4),
                                           num=accuracy_score(y_trues,predictions,normalize=False),
                                           total=len(y_trues)))
print("Std:",round(np.std(avg),4))
print(confusion_matrix(y_trues,predictions,labels=[1,0]))

all_cols = np.array(categorical_features + numerical_features)
all_cols = all_cols[selbest.get_support()]
df_coefs = pd.DataFrame(data=coefs,columns=all_cols)
print(round(df_coefs.mean(axis=0),4))
print("Mean intercept:",round(np.mean(intercepts),4))

###stats and plotting

#calculate blind baseline
blind_bl = mean(y)
if blind_bl < 0.5:
    blind_bl = 1 - blind_bl

#compute p from cumulative mass function
print("p: ",round(binomial_cmf((len(X)*(acc)),len(X),blind_bl),4))

#violin plots of classifier accuracy
plt.rcParams.update({'font.size': 22})
avg = pd.DataFrame(data=avg, columns=columns)
plt.figure(figsize=(6,6))
g = sns.violinplot(data=avg); g.set(ylim=(0,1))
g.axes
plt.axhline(linewidth=4, color='r',y=blind_bl)
plt.show()
