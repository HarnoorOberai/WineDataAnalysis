import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)
# Importing and taking a quick look
df=pd.read_csv("winequality-white.csv",sep=";")
#print(df.describe())

#Standardising the data

features=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",]
x=df.loc[:,features].values
y = df.loc[:,['quality']].values
x = StandardScaler().fit_transform(x)
pd.DataFrame(data = x, columns = features).head()

#PCA 2D Projection
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
#print(principalDf.head(5))
#print(df[['quality']].head())
finalDf = pd.concat([principalDf, df[['quality']]], axis = 1)
#print(finalDf.head(5))

print(pca.explained_variance_ratio_)
#[0.29293217 0.14320363] = 43.6% data retention

#Visualize 2D Projection
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue='quality',
    palette=sns.color_palette("muted",7),
    data=finalDf,
    legend="full",
    alpha=0.3
)
plt.show()