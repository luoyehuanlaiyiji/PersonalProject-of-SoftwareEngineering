from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
import warnings

warnings.filterwarnings('ignore')

#导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

array = data.values
X = array[:, 0:8]
Y = array[:, 8]

#卡方检验-特征选定
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
set_printoptions(precision=3)
print(fit.scores_)

#RFE-递归特征消除
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print('特征个数：')
print(fit.n_features_)
print('被选中的数据特征：')
print(fit.support_)
print('特征排名：')
print(fit.ranking_)

#PCA降维
pca = PCA(n_components=3)
fit = pca.fit(X, Y)
print('解释方差： {}'.format(fit.explained_variance_ratio_))
print(fit.components_)

#显示选择后的特征
features = fit.transform(X)
print(features)

#袋装决策树
model = ExtraTreesClassifier()
fit = model.fit(X, Y)
print('特征得分：')
print(fit.feature_importances_)