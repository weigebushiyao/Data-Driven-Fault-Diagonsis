import pandas as pd
from sklearn.decomposition import PCA
data=pd.read_csv('train_data.csv')
print('break1')

"""
pca=PCA(25)
pca.fit(data)
print(pca.components_)                 #返回模型特征向量
print(pca.explained_variance_ratio_)   #方差百分比,选取主成分个数标准
print('break2')
"""


pca=PCA(1)
pca.fit(data)
low_d=pca.transform(data)               #降维处理
pd.DataFrame(low_d).to_csv('train_data_PCA.csv')  #保存结果