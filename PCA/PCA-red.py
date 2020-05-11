import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)
# Importing and taking a quick look
df=pd.read_csv("winequality-red.csv",sep=";")
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
#[0.28173931 0.1750827 ] = 45.6% data retention

#Visualize 2D Projection
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 Component PCA', fontsize = 20)
# qualitys = ['5', '6', '4']
# colors = ['r', 'g', 'b']
# for target, color in zip(qualitys,colors):
#     indicesToKeep = finalDf['quality'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(qualitys)
# ax.grid()
# plt.show()
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue='quality',
    palette=sns.color_palette("hls"),
    data=finalDf,
    legend="full",
    alpha=0.3
)
plt.show()