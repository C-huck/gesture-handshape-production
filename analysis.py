#Classification
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from convert_handshape_to_features import featureCoding
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
df = pd.read_csv("handshape-data.csv")

#convert handshape codes to component features
lambdafunc = lambda x: pd.Series(featureCoding(x['HS1Code']))
df[['compF','compJ','flexion','nsf_flexion','selfing','apertureChange','nsf_thumb','thumb_flex']] = df.apply(lambdafunc, axis=1)

#Add a feature for presence of 2nd hand
df['HS2Code'] = df['HS2Code'].fillna(0)
df['noHands'] = [0 if x==0 else 1 for x in df['HS2Code'] ]

#remove data from native signer
df = df[df['SignerStatus']==0]

#set up classification
categorical_features = ['apertureChange','noHands','nsf_flexion']
categorical_pipeline = Pipeline( steps = [ ( 'cat_selector', FeatureSelector(categorical_features) ) ] ) #leaves categorical data untouched

numerical_features = ['compF','compJ','flexion','nsf_thumb','thumb_flex']
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

#Labels
y = X['LabelBin']


kf = StratifiedKFold(n_splits=6,shuffle=True,random_state=6)
avg = []
coefs = []
predictions = []
y_trues = []
p_vals = []
for train,test in kf.split(X,y):
    #Classify, get predictions and ground truth labels
    pipe.fit(X.iloc[train],y.iloc[train])
    pred = pipe.predict(X.iloc[test])
    actual = y.iloc[test]
    
    #Record best predictors for classification
    selbest = pipe['reduce_dim']
    #print(selbest.get_support())
    cl_weights = pipe['classify']
    coefs.append(cl_weights.coef_[0])
    
    #Record predictions and ground truth labels for summary metrics
    predictions+=list(pred)
    y_trues+=list(actual)
    
    #Record accuracy and per-fold significance
    avg.append(accuracy_score(pred,actual)
    blind_bl = mean(actual)
    if blind_bl < 0.5:
        blind_bl = 1 - blind_bl
    p_vals.append(binomial_cmf(accuracy_score(pred,actual,normalize=False),len(actual),bbl))

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

#compute overall significance using cumulative mass function
print("p: ",round(binomial_cmf((len(X)*(acc)),len(X),blind_bl),4))

#violin plots of classifier accuracy
plt.rcParams.update({'font.size': 22})
avg = pd.DataFrame(data=avg, columns=columns)
plt.figure(figsize=(6,6))
g = sns.violinplot(data=avg); g.set(ylim=(0,1))
g.axes
plt.axhline(linewidth=4, color='r',y=blind_bl)
plt.show()
