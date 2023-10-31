from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA

df=pd.read_csv(r'C:\Users\Ariya Rayaneh\Desktop\heart.csv')
pd.set_option('display.max_columns',None)

x=df.drop('output',axis=1)
feature_list=x.columns
print(feature_list)
x=StandardScaler().fit_transform(x)

y=df.output
x_train,x_test , y_train,y_test=train_test_split(x,y,test_size=0.3)


model=LogisticRegression(max_iter=10000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(model.score(x_test,y_test))

def sfm(X,Y):
#Feature Engineering SFM
 sfm = SelectFromModel(model, prefit=True)
 x_new = sfm.transform(X)
 print(np.array(x_new))

def rfe(X,Y):
 rfe = RFE(estimator=model, n_features_to_select=3)
 rfe.fit(X,Y)
 rfe_features = [f for (f, support) in zip(feature_list, rfe.support_) if support]
 print(rfe_features)
 print(rfe.score(X,Y))

#Feature Engineering by SFS
def sfs(X,Y):
 sfs = SequentialFeatureSelector(estimator=model,
           n_features_to_select='auto',
           direction='backward',
           scoring='accuracy',
           cv=10)
# Fit SFS to our features X and outcome y
 sfs.fit(X, Y)
 print(sfs.get_feature_names_out())
 print(sfs.get_support())
 print(sfs._get_param_names())
 print(sfs.get_metadata_routing())

def pca(X, Y, Feature_List):
 pca = PCA()
 X_pca = pca.fit_transform(X)
 component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
 X_pca = pd.DataFrame(X_pca, columns=component_names)
 # Create loadings
 loadings = pd.DataFrame(
     pca.components_.T,  # transpose the matrix of loadings
     columns=component_names,  # so the columns are the principal components
     index=Feature_List,  # and the rows are the original features
 )
 print(X_pca)



 plt.figure(figsize=(8, 6))
 a=np.array(X_pca.iloc[:,0])
 b=np.array(X_pca.iloc[:,1])


 plt.xlabel('First Principal Component')
 plt.ylabel('Second Principal Component')
 plt.title('salam')
 plt.scatter(a, b, c=a, cmap='plasma')
 plt.show()

pca(x,y,feature_list)