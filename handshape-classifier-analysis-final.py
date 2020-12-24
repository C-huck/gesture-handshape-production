# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:11:16 2019

@author: Jack
"""

from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from featureCoding_fixed4 import featureCoding
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, matthews_corrcoef,confusion_matrix

import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
def binomial_cmf(k, n, p):
    c = 0
    for k1 in range(n+1):
        if k1>=k:
            c += stats.binom.pmf(k1, n, p)
    return c


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h, m#m, m-h, m+h

class FeatureSelector():
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self._feature_names ] 

#l = "handshape-project-10-28-19.csv"
#l = "handshape-project-01-09-20.csv"
#l = "handshape-project-01-28-20_1.csv" #Results reported in conference (ELM, CUNY) abstracts
#l = "handshape-project-02-20-20-main-only.csv"
l = "handshape-project-02-20-20-main-only-1.csv"
df = pd.read_csv(l)

print(set(df['EventFilename']))

df['Flexion'] = [featureCoding(x)[2] for x in df['HS1Code']]
df['NSF_Flexion'] = [featureCoding(x)[3] for x in df['HS1Code']]
df['FingerComplexity'] = [featureCoding(x)[0] for x in df['HS1Code']]
df['JointComplexity'] = [featureCoding(x)[1] for x in df['HS1Code']]
df['apertureChange'] = [featureCoding(x)[5] for x in df['HS1Code']]
df['nsf+thumb'] = [featureCoding(x)[6] for x in df['HS1Code']]
df['thumb_flex'] = [featureCoding(x)[7] for x in df['HS1Code']]

#print(df['apertureChange'].sum())

df['HS2Code'] = df['HS2Code'].fillna(0)
df['noHands'] = [0 if x==0 else 1 for x in df['HS2Code'] ]



#df = df[df['Pilot']==0]
df = df[df['SignerStatus']==0]

df['2H_FingerComplexity'] = [featureCoding(x)[0] if type(x) == str else "None" for x in df['HS2Code'] ]
df['2H_JointComplexity'] = [featureCoding(x)[1] if type(x) == str else "None" for x in df['HS2Code'] ]
df['2H_Flexion'] = [featureCoding(x)[2] if type(x) == str else "None" for x in df['HS2Code'] ]
df['2H_NSF_Flexion'] = [featureCoding(x)[3] if type(x) == str else "None" for x in df['HS2Code'] ]

df['Flex+FingComp'] = df['Flexion']+df['FingerComplexity']


a = ["CM","RVN"]
b = ["HO","NP"]
c = ["CS","IP"]
order = [(0 if x in a else 1 if x in b else 2 if x in c else 3) for x in df['Participant']]
df['order'] = order

#########Stats for 2nd hand
"""
df= df[(df['Tool']==1) | (df['Manner']==1)]

stats.ttest_ind([x for (x,y) in zip(df['2H_JointComplexity'],df['LabelBin']) if type(x)==int and y==1],[x for (x,y) in zip(df['2H_JointComplexity'],df['LabelBin']) if type(x)==int and y==0],equal_var=False)

len([x for (x,y) in zip(df['2H_FingerComplexity'],df['LabelBin']) if type(x)==int and y==1])+len([x for (x,y) in zip(df['2H_FingerComplexity'],df['LabelBin']) if type(x)==int and y==0])-2

np.mean([x for (x,y) in zip(df['2H_FingerComplexity'],df['LabelBin']) if type(x)==int and y==1])
np.mean([x for (x,y) in zip(df['2H_FingerComplexity'],df['LabelBin']) if type(x)==int and y==0])
np.mean([x for (x,y) in zip(df['2H_JointComplexity'],df['LabelBin']) if type(x)==int and y==1])
np.mean([x for (x,y) in zip(df['2H_JointComplexity'],df['LabelBin']) if type(x)==int and y==0])
np.mean([x for (x,y) in zip(df['2H_Flexion'],df['LabelBin']) if type(x)==int and y==1])
np.mean([x for (x,y) in zip(df['2H_Flexion'],df['LabelBin']) if type(x)==int and y==0])
np.mean([x for (x,y) in zip(df['2H_NSF_Flexion'],df['LabelBin']) if type(x)==int and y==1])
np.mean([x for (x,y) in zip(df['2H_NSF_Flexion'],df['LabelBin']) if type(x)==int and y==0])

df['NSF_close'] = [1 if x == 1 else 0 for x in df['NSF_Flexion']]
df['NSF_extend'] = [1 if x == -1 else 0 for x in df['NSF_Flexion']]
"""
###########

#categorical_features = ['apertureChange','noHands','NSF_close','NSF_extend']
categorical_features = ['apertureChange','noHands','NSF_Flexion']
#categorical_features = ['noHands']
numerical_features = ['FingerComplexity','JointComplexity','Flexion']
#numerical_features = ['FingerComplexity','Flexion','NSF_Flexion']
#numerical_features = ['FingerComplexity','JointComplexity','Flexion','NSF_Flexion','Flex+FingComp']
#numerical_features = ['JointComplexity','NSF_Flexion','Flex+FingComp']
categorical_pipeline = Pipeline( steps = [ ( 'cat_selector', FeatureSelector(categorical_features) ) ] )
numerical_pipeline = Pipeline( steps = [ ( 'num_selector', FeatureSelector(numerical_features) ), ( 'std_scaler', StandardScaler() ) ] )
#numerical_pipeline = Pipeline( steps = [ ( 'num_selector', FeatureSelector(numerical_features) ) ] )
full_pipeline = FeatureUnion( transformer_list = [ ( 'categorical_pipeline', categorical_pipeline ), ( 'numerical_pipeline', numerical_pipeline ) ] )

pipe = Pipeline([
    ('full_pipeline', full_pipeline),
    ('reduce_dim', SelectKBest(f_classif,k=4)),
    #('classify', LogisticRegression(random_state=0, solver='liblinear',multi_class='ovr',C=1000,tol=0.001,class_weight='balanced'))
    ('classify',
                 SVC(C=100, cache_size=200, class_weight='balanced', coef0=0.0,
                     decision_function_shape='ovo', degree=3, gamma='auto', #change back to ovo
                     kernel='linear', max_iter=-1, probability=False,
                     random_state=3, shrinking=True, tol=0.001,
                     verbose=False))#,
     #('accuracy',
      #accuracy_score(normalize=False))
])

###FIGURE OUT WHAT THE BELOW DOES####    
"""
X= df[(df['Manner']==1) | (df['LabelBin']==0)];
frames = [ df[(df['Manner']==1)].sample(n=71),df[(df['Movement']==1) & (df['Manner']==0) & (df['LabelBin']==0)].sample(n=71,random_state=5)]
X=pd.concat(frames)
X= df[(df['Manner']==1) & (df['Alternate']==0)]
X2 = df[(df['Alternate']==0) & (df['LabelBin']==0)]
X2 = X2[X2['Manner']==0]
frames = [ df[(df['Manner']==1)].sample(n=71),df[(df['Movement']==1) & (df['Manner']==0) & (df['LabelBin']==0)].sample(n=71,random_state=5)]
X=pd.concat([X,X2])
"""

####Restriction testing
"""
df = df[df['LabelBin']==0]
#X = df[df['EventFilename'].str.contains("cut|picture|put-book|hammer")]
#X = df[df['EventFilename'].str.contains("stick|picture|turn|drop-ball")] #Trans
X = df[df['EventFilename'].str.contains("stick|picture|turn|ball-drop")] #Intrans
y = X['EventFilename']


df[df['EventFilename'].str.contains("ball-") & df['LabelBin']==1][['EventFilename','Flexion']]


#set(df['EventFilename'])

#ball (drop-ball,)
#book

"""

########
#df['Flexion'] = list(df['Flexion'].sample(frac=1,random_state=3))
#df['FingerComplexity'] = list(df['FingerComplexity'].sample(frac=1,random_state=3))
#df['noHands'] = list(df['noHands'].sample(frac=1))

###Sort by event ID
#df = df.sort_values(by="EventFilename")

###Main analysis
#df['Flexion'] = [x-2 for x in df['Flexion']]
#X = df; columns = ["All events"]
#X = df[df['Alternate-1']==1]; columns = ["Alternating events"]
X = df[df['Alternate-1']==0]; columns = ['All but Alts']
#X = df[df['Alternate']==0]
X = X[(X['Manipulation']==1) | (X['Movement']==1)]; columns = ["Manipulation/Movement events"]
#X = X[(X['Tool']==1) | (X['Manner']==1)]; columns = ["Tool-use/Manner events"]

##Ancillary analyses: effect of event, effect of participant
#X = X.sort_values(by=['EventFilename'])
#X = X.sort_values(by=['Participant'])

y = X['LabelBin']
#y['LabelBin'] = [x-1 if x==0 else x for x in y]
#y = X['EventFilename']
#y = X['Manner']

#####
#X['LabelBin'].sum()


kf = StratifiedKFold(n_splits=6,shuffle=True,random_state=6) #0 for alts; 3 for manip; 3 for tool; 0 for all
#kf = StratifiedKFold(n_splits=6,shuffle=False) #0 for alts and manip; 3 for tool; 6 for all
#kf = KFold(n_splits=6,shuffle=False) #0 for alts and manip; 3 for tool; 6 for all
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
    ##FOR LOGREG ONLY
    #proba = np.array(pipe.predict_proba(X.iloc[test]))
    #eventNames = np.array(eventNames).reshape(18,1)
    #participantNames = np.array(participantNames).reshape(18,1)
    #combo = np.append(eventNames,participantNames, 1)
    #combo = np.append(combo,proba, 1)
    #probas.append(combo)
    
acc = np.mean(avg)
#print(acc)
print(y_trues.count(1), len(y_trues))
print(predictions.count(1), len(predictions))
mcc = matthews_corrcoef(y_trues,predictions)
print("MCC:",round(mcc,4))
print("Acc: {perc} ({num}/{total})".format(perc=round(accuracy_score(y_trues,predictions),4),
                                           num=accuracy_score(y_trues,predictions,normalize=False),
                                           total=len(y_trues)))
print("Std:",round(np.std(avg),4))

print(confusion_matrix(y_trues,predictions,labels=[1,0]))
y_trues.count(1)
y_trues.count(0)

#df_coefs = pd.DataFrame(data=coefs,columns=["noHands","Fcomp","Flex","NSF_Flex"])
all_cols = np.array(categorical_features + numerical_features)
all_cols = all_cols[selbest.get_support()]
df_coefs = pd.DataFrame(data=coefs,columns=all_cols)
#df_coefs = pd.DataFrame(data=coefs,columns=["noHands","Fcomp","Flex","NSF_Flex"])
#df_coefs = pd.DataFrame(data=coefs,columns=["noHands","Flex","NSF_Flex"])
#df_coefs = pd.DataFrame(data=coefs,columns=['apertureChange','noHands','NSF_close','NSF_extend','FingerComplexity','JointComplexity','Flexion'])
print(round(df_coefs.mean(axis=0),4))
print("Mean intercept:",round(np.mean(intercepts),4))

#df_means = np.array([df_coefs.mean(axis=0),[0,0.5,0.2,0.9]]).reshape(2,4)
#ax = sns.heatmap(df_means,cmap=sns.diverging_palette(220, 20, n=100),square=True)

df_f_scores = pd.DataFrame(data=f_scores,columns=categorical_features+numerical_features)
df_f_scores.mean(axis=0)

selbest.get_support()
#df_coefs['NSF_Flex']



#for i in range(len(coefs)):
#    print(coefs[i])

#pipe['classify'].class_weight_
############stats and plotting

a = sum(y)
b = len(X) - a
if b > a:
    c = b
else:
    c = a

blind_bl = (c/len(X))

#print(len(X)*acc)

print("p: ",round(binomial_cmf((len(X)*(acc)),len(X),blind_bl),4))

plt.rcParams.update({'font.size': 22})
avg = pd.DataFrame(data=avg, columns=columns)
plt.figure(figsize=(6,6))
g = sns.violinplot(data=avg); g.set(ylim=(0,1))
g.axes
plt.axhline(linewidth=4, color='r',y=blind_bl)
plt.show()

import statsmodels.formula.api as smf
formula_log = "LabelBin ~ FingerComplexity + Flexion + NSF_Flexion + noHands"
md_log = smf.logit(formula_log,data=df[(df['Tool']==1)|(df['Manner']==1)]).fit()
print(md_log.summary())

#import numpy as np
#support_indices = np.cumsum(cl_weights.n_support_)
#cl_weights.dual_coef_[0:support_indices[0]]
#cl_weights.dual_coef_[support_indices[0]:support_indices[1]]
"""

for col in df_coefs.columns:
    print(col,round(mean_confidence_interval(df_coefs[col])[0],2))
    

fileNamesIN = set(df[(df['LabelBin']==0) & (df['noHands']==1)]['EventFilename'])
fileNamesTR = set(df[(df['LabelBin']==1) & (df['noHands']==1)]['EventFilename'])

df_TR_2hand = []
for ii in fileNamesTR:
    df_TR_2hand.append([ii,df[df['EventFilename'] == ii]['noHands'].sum()])
df_IN_2hand = []
for ii in fileNamesIN:
    df_IN_2hand.append([ii,df[df['EventFilename'] == ii]['noHands'].sum()])

sum([x[1] for x in df_TR_2hand])/len(df_TR_2hand)
sum([x[1] for x in df_IN_2hand])/len(df_IN_2hand)

X[X['LabelBin']==1]['noHands'].sum()
X[X['LabelBin']==0]['noHands'].sum()
X['noHands'].sum()
X[X['LabelBin']==1]['apertureChange'].sum()
X[X['LabelBin']==0]['apertureChange'].sum()
np.mean(X[X['LabelBin']==0]['NSF_Flexion'])
np.mean(X[X['LabelBin']==1]['NSF_Flexion'])

mean_confidence_interval(X[X['LabelBin']==0]['NSF_Flexion'])
mean_confidence_interval(X[X['LabelBin']==1]['NSF_Flexion'])


len(df[df['Tool']==1])
df[df['Tool']==1]['noHands'].sum()/len(df[df['Tool']==1])
df[df['Manner']==1]['noHands'].sum()/len(df[df['Manner']==1])
df[df['Tool']==1]['Flexion'].mean()

from collections import Counter
Counter(df[df['apertureChange']==1]['EventFilename'])

Counter(df['NSF_Flexion'])
Counter(df[df['Tool']==1]['NSF_Flexion'])


#X[X['EventFilename'].str.contains("picture")].mean()
#X[X['EventFilename'].str.contains("cut")].mean()
#X[X['EventFilename'].str.contains("put")]['Flexion'].mean()
"""


####Check to make sure every event is properly classified as tool/ manner/ manip/ etc.
"""
eventNames = list(set(df['EventFilename']))
for x in eventNames:
    temp = df[df['EventFilename']==x]
    for y in ['Tool','Manner','Manipulation','Movement']:
        if temp[y].sum() in [0,6]:
            continue
        else:
            print(y,x,temp[y].sum())

x_1 = [x for (x,y) in zip(X['Flexion'],X['LabelBin']) if y ==1]
y_1 = [x for (x,y) in zip(X['FingerComplexity'],X['LabelBin']) if y ==1]
x_2 = [x for (x,y) in zip(X['Flexion'],X['LabelBin']) if y ==0]
y_2 = [x for (x,y) in zip(X['FingerComplexity'],X['LabelBin']) if y ==0]
plt.scatter(x_1,y_1)
plt.scatter(x_2,y_2,color="red")
plt.show()

df['Movement'].sum()/6
df[df['Label']=="TR"]['Alternate'].sum()/6
df[df['Label']=="IN"]['Alternate'].sum()/6

index = ["All verbs","Alternates","Manip./Mvmt.","Tool/Manner"]
columns = ["Aperture","2handed?","F. Comp.","Flexion","NSF Flexion"]
df_means = np.array([[np.nan,1.0961,0.1872,0.4188,np.nan],
[0.4448,1.3052,np.nan,np.nan,-0.3632],
[np.nan,0.7356,0.3203,0.8030,0.1803],
[np.nan,0.4959,0.477,1.1129,0.1217]])#.reshape(4,5)
df_mean = pd.DataFrame(data=df_means,index=index,columns=columns)

ax = sns.heatmap(df_mean,cmap=sns.diverging_palette(220, 20, n=100),square=True,annot=True,linewidths=.5)



####Create correlation plot; plot shows boxplots of individual participant's corr coefs btw each feature and a sem class label
from scipy.stats import pearsonr
indices = categorical_features + numerical_features

part = []
sem_class = []
hs_feat = []
corrs = []
for x in set(df['Participant']):
    df_temp = df[df['Participant']==x]
    for y in ['LabelBin','Tool','Manner','Manipulation','Movement']:
        for z in indices:
            corr = pearsonr(df_temp[y],df_temp[z])[0]
            part.append(x)
            sem_class.append(y)
            hs_feat.append(z)
            corrs.append(corr)
            
f = pd.DataFrame({"Participant" : part,
                  "SemClass" : sem_class,
                  "HsFeat" : hs_feat,
                  "Corr" : corrs
        })

sem_class = []
hs_feat = []
corrs = []
for y in ['LabelBin','Tool','Manner','Manipulation','Movement']:
        for z in indices:
            corr = pearsonr(df[y],df[z])[0]
            sem_class.append(y)
            hs_feat.append(z)
            corrs.append(corr)
    
g = pd.DataFrame({"SemClass" : sem_class,
                  "HsFeat" : hs_feat,
                  "Corr" : corrs
        })


ax = sns.boxplot(x="HsFeat",y="Corr",hue="SemClass",hue_order=["LabelBin","Manipulation","Tool"],data=f)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right" )


ax = sns.pointplot(x="HsFeat",y="Corr",hue="SemClass",hue_order=["LabelBin","Manipulation","Tool"],data=g)


from pyitlib import discrete_random_variable as drv
a = np.array(df[(df['Tool']==1) | (df['Manner']==1)]['Tool'])
a = a.astype(int)
b = np.array(df[(df['Tool']==1) | (df['Manner']==1)]['FingerComplexity'])
b = b.astype(int)
c = np.array([0,0,1,1])
d = np.array([0,0,0,1])
drv.entropy(a,b)



sum([x for (x,y) in zip(df['apertureChange'],df['LabelBin']) if y==1])
sum([x for (x,y) in zip(df['apertureChange'],df['LabelBin']) if y==0])
sum([x for (x,y) in zip(X['apertureChange'],X['LabelBin']) if y==1])
sum([x for (x,y) in zip(X['apertureChange'],X['LabelBin']) if y==0])

probas[2]

featureCoding("BT^(")


df[df['Alternate']==0]['Manner'].sum()
df[df['Alternate']==0]['Tool'].sum()
df[df['Alternate']==0]['Movement'].sum()
df[df['Alternate']==0]['Manipulation'].sum()


df[(df['Alternate']==1) & (df['LabelBin']==1)]['Alternate'].sum()
"""

"""
a = ["CM","RVN"]
b = ["HO","NP"]
c = ["CS","IP"]
order = [(0 if x in a else 1 if x in b else 2 if x in c else 3) for x in df['Participant']]
df['order'] = order

X = df[df['Alternate']==1]

i0 = X[(X['order']==0) & (X['LabelBin']==0)]['Flexion']
t0 = X[(X['order']==0) & (X['LabelBin']==1)]['Flexion']
#np.mean(X[X['order']==0]['FingerComplexity'])

t1= X[(X['order']==1) & (X['LabelBin']==0)]['Flexion']
i1 = X[(X['order']==1) & (X['LabelBin']==1)]['Flexion']
#np.mean(X[X['order']==1]['FingerComplexity'])

t2 = X[(X['order']==2) & (X['LabelBin']==0)]['Flexion']
i2 = X[(X['order']==2) & (X['LabelBin']==1)]['Flexion']
#np.mean(X[X['order']==2]['FingerComplexity'])

stats.ttest_ind(i0,i1)
stats.ttest_ind(i0,i2)
stats.ttest_ind(i1,i2)

stats.ttest_ind(t0,t1)
stats.ttest_ind(t0,t2)
stats.ttest_ind(t1,t2)
"""

#print([round(x,4) for x in avg['All pantomimes']],round(acc,4),round(avg['All pantomimes'].std(),4),round(mcc,4))
#print(p_vals)

#X['Tool'].sum()
#X['Manner'].sum()

#df[(df['Alternate']==1)]['Alternate'].sum()
#df[(df['Alternate']==1) & (df['Manipulation']==1)]['LabelBin'].sum()
#df[(df['Alternate']==1) & (df['Movement']==1)]
#df[(df['Alternate']==1) & (df['Movement']==0) & (df['Manipulation']==0)]['LabelBin'].sum()

#print(list(avg['All pantomimes']))

#a = [0.6666666666666666, 0.6111111111111112, 0.6944444444444444, 0.5277777777777778, 0.5416666666666666, 0.5833333333333334]
#b = [0.625, 0.5972222222222222, 0.6805555555555556, 0.5972222222222222, 0.5138888888888888, 0.8194444444444444]
#stats.ttest_ind(a,b)

"""
set(df[(df['Manipulation'] == 1) & (df['Alternate']==1)]['EventFilename'])
set(df[(df['Manner'] == 1) & (df['Alternate']==1)]['EventFilename'])

df[(df['Alternate']==1)&(df['LabelBin']==1)][numerical_features].mean()
df[(df['Alternate']==1)&(df['LabelBin']==0)][numerical_features].mean()

X = df[df['EventFilename'].str.contains("stick|tower|put-book|book-fall|caps|cream|spin")]
X[(X['Alternate']==1)&(X['LabelBin']==1)][numerical_features+categorical_features].mean()
X[(X['Alternate']==1)&(X['LabelBin']==0)][numerical_features+categorical_features].mean()

stats.ttest_ind(X[(X['Alternate']==1)&(X['LabelBin']==1)]['Flexion'],X[(X['Alternate']==1)&(X['LabelBin']==0)]['Flexion'])
stats.ttest_ind(X[(X['Alternate']==1)&(X['LabelBin']==1)]['FingerComplexity'],X[(X['Alternate']==1)&(X['LabelBin']==0)]['FingerComplexity'])
stats.ttest_ind(X[(X['Alternate']==1)&(X['LabelBin']==1)]['noHands'],X[(X['Alternate']==1)&(X['LabelBin']==0)]['noHands'])


df_2hand = df[df['noHands']==1]

df_2hand['LabelBin'].mean()
len(df_2hand)/len(df)

df[df['noHands']==0]['LabelBin'].mean() 
df[df['noHands']==1]['LabelBin'].mean()

df_2hand[(df_2hand['LabelBin']==1) & (df_2hand['Manipulation']==1)]['2H_Flexion'].mean()
df_2hand[(df_2hand['LabelBin']==0) & (df_2hand['Movement']==1)]['2H_Flexion'].mean()
df_2hand[(df_2hand['LabelBin']==1) & (df_2hand['Manipulation']==1)]['2H_NSF_Flexion'].mean()
df_2hand[(df_2hand['LabelBin']==0) & (df_2hand['Movement']==1)]['2H_NSF_Flexion'].mean()
df_2hand[(df_2hand['LabelBin']==1) & (df_2hand['Manipulation']==1)]['2H_FingerComplexity'].mean()
df_2hand[(df_2hand['LabelBin']==0) & (df_2hand['Movement']==1)]['2H_FingerComplexity'].mean()


df_2hand[df_2hand['LabelBin']==0]['2H_Flexion'].mean()
df_2hand[df_2hand['LabelBin']==1]['2H_NSF_Flexion'].mean()
df_2hand[df_2hand['LabelBin']==0]['2H_NSF_Flexion'].mean()
df_2hand[df_2hand['LabelBin']==1]['2H_NSF_Flexion'].mean()
df_2hand[df_2hand['LabelBin']==0]['2H_FingerComplexity'].mean()
"""

XX = X[['Flexion','NSF_Flexion','FingerComplexity','noHands','apertureChange','JointComplexity']]
XX = pipe['full_pipeline'].fit_transform(XX)
#XX = XX[:,[1,2,4,5]]
XX = XX[:,[1,2,3,5]]

w = np.array(df_coefs.mean(axis=0))

res = np.sum(w * XX,axis=1) + np.mean(intercepts)
res_bin = [1 if x>0 else 0 for x in res]

print("Full model accuracy: ",round(accuracy_score(y,res_bin),4))
print("Full model MCC: ",round(matthews_corrcoef(y,res_bin),4))


df[(df['Tool']==1) | (df['Manner']==1)]['noHands'].mean()
df[(df['Tool']==1) | (df['Manner']==1)]['noHands'].mean()
df[(df['Tool']==1)]['noHands'].mean()
df[(df['Manner']==1)]['noHands'].mean()


df[(df['Manipulation']==1)]['noHands'].mean()
df[(df['Movement']==1)]['noHands'].mean()

df[(df['noHands']==1) & (df['LabelBin']==1)]
df[(df['noHands']==1) & (df['LabelBin']==0)]

#print(set(df['EventFilename']))
