from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer

#导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

#调整数据尺度
transformer = MinMaxScaler(feature_range=(0, 1)).fit(X)
newX = transformer.transform(X)
#正态化数据
transformer = StandardScaler().fit(X)
newX = transformer.transform(X)
#标准化数据
transformer = Normalizer().fit(X)
newX =transformer.transform(X)
#二值化数据
transformer = Binarizer(threshold=0.0).fit(X)
newX = transformer.transform(X)

#设定数据的打印格式
set_printoptions(precision=3)
print(newX)
