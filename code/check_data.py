from pandas import read_csv
from pandas import set_option

filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
#前10条数据
peek = data.head(10)
print(peek)
#数据行数列数
print('data shape: {}'.format(data.shape))
#数据类型
print(data.dtypes)
#8个方面，count,max,mean,std,25%,50%,75%,min
set_option('display.max_columns', 100)
set_option('precision', 4)
print(data.describe())
#按照类别分组的个数
print(data.groupby('class').size())
#皮尔逊系数，特征之间相关性矩阵
set_option('precision', 2)
print(data.corr(method='pearson'))
#数据离散偏离程度
print(data.skew())



