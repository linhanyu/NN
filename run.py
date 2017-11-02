import pandas as pd
import DNN
import numpy as np
from sklearn.utils import shuffle

data = pd.read_csv('data.txt',header=-1)
data = shuffle(data)

target = pd.get_dummies(data[4])[20:]
train = data.drop(4,axis=1)[20:]
test_target = pd.get_dummies(data[4])[:20]
test_train = data.drop(4,axis=1)[:20]

clf = DNN.dnn_classifier()


dic = {0:'Iris-setosa' ,1:'Iris-versicolor' ,2:'Iris-virginica'}
m = np.vectorize(lambda x:dic[x])

clf.fit(train,target)
res = m(clf.predict(test_train))

print('测试数据')
print(data[:20][4])
print('预测结果')
print(res)
print('准确率CV')
print((res == data[:20][4]).mean())



