from pygks.reg_inn import ISOINNregressor
from numpy import array
from pygks.utils import csv_reader
from sklearn.metrics import mean_squared_error
from numpy import isnan,isinf

r1 = csv_reader('wine_train.csv')
r2 = csv_reader('wine_test.csv')

trainX, trainy = r1.separate_label()
testX, testy = r2.separate_label()
nnReg = ISOINNregressor(K = 10, age_max = 300, nn_lambda = 350, alpha = 10, smooth = None,  del_noise = True)
nnReg.fit(trainX, trainy)

print len(testX[0]),'dimension'
print len(nnReg.nodes),'nodes'
predicty = nnReg.predict(testX)
delets = []
i = 0
for each in predicty:
    if isnan(each) or isinf(each):
        print(i,each,'error after density estimation')
        delets.append(i)
    i += 1

for i in range(len(delets)):
    testy.pop(delets[i]-i)
    predicty.pop(delets[i]-i)

print len(delets),'failed points'
print mean_squared_error(testy,predicty)

