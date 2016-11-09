from pygks.utils import csv_reader
from pygks.reg_gng import GNGregressor
from numpy import array

r = csv_reader('reg_intro.csv')
X,y = r.separate_label()
the_reg = GNGregressor(smooth = -0.5, age_max = 200, nn_lambda = 60)
the_reg.fit(X,y)
test_x = []
draw_x = []
for i in range(100):
    test_x.append(array([i/100.0]))
    draw_x.append(i/100.0)
test_y = the_reg.predict(test_x)
r2 = csv_reader('reg_intro.csv')
testX, testY = r2.separate_label()
from sklearn.metrics import mean_squared_error
print mean_squared_error(testY,the_reg.predict(testX))
import matplotlib.pyplot as plt
fig = plt.figure()
r_draw = csv_reader('reg_intro.csv')
X_raw = r_draw.get_all()
X_draw = []
i = 0
for each in X_raw:
    if i % 3 == 0:
        X_draw.append(X_raw[i])
    i += 1
print X_raw[0]
ax = fig.add_subplot(111)
for i in range(len(X_draw)):
    ax.plot(X_draw[i][0], X_draw[i][1], '.', color = '0.8')
ax.plot(draw_x,test_y,'k-')
plt.show()
the_reg.draw_density()
