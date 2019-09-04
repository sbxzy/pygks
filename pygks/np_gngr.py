from .__future__gks import GKS
from .reg_gng import GNGregressor
from .reg_inn import ISOINNregressor
from numpy import array, zeros, argmin, sum

class npGNGregressor:
    age_max = 200
    nn_lambda = 20
    model = []
    reg = []
    GNG = True
    globus = False

    def __init__(self, age_max_ = 200, nn_lambda_ = 30, GNG = True, globus = False, randomize = False):
        self.age_max = age_max_
        self.nn_lambda = nn_lambda_
        self.GNG = GNG
        self.globus = globus

        if randomize:
            from random import randint as rand
            self.age_max = rand(100,500)
            self.nn_lambda = rand(10,60)

    def fit(self, X, y, sample_weight = None):
##        train = []
##        for i in range(len(y)):
##            tmp = list(X[i])
##            tmp += [y[i]]
##            train.append(array(tmp))
        #print train
        if self.GNG:
            self.model = GNGregressor(age_max = self.age_max,nn_lambda = self.nn_lambda)
        else:
            self.model = ISOINNregressor(age_max = self.age_max,nn_lambda = self.nn_lambda,del_noise = False)
        self.model.fit(X,y)
        points = self.model.nodes
        #print points
        populations = array(self.model.counts)
        if sample_weight != None:
            populations = zeros(len(points))
            for i in range(len(sample_weight)):
                distances = sum((array(train[i] - points))**2,axis = -1)
                winner_index = argmin(distances)
                populations[winner_index] += sample_weight[i]
        populations = array(populations)
        #print populations
        variances = self.model.standard_deviation ** 0.5
        self.reg = GKS(points, populations, variances, 1, self.globus)
        return 0

    def predict(self, X):
        return self.reg.responses(X)
        
    def get_params(self, deep = False):
        return {}
        
    def set_params(self, random_state = False):
        return 0

if __name__ == '__main__':
#    GR = npGNGregressor( GNG=True, globus = True)
#    GR.fit(array([[1,2],[3,2],[4,2],[1,8]]),array([4,3,2,1]), sample_weight = [0.1, 0.3, 0.5, 0.1])
#    print GR.predict(array([[1,2],[3,2],[4,2],[1,8]]))
    from xzyutil.csv_reader import csv_reader
    from sklearn.ensemble import ExtraTreesRegressor as ET

    r1 = csv_reader('/Users/xzy/work/history/NNregression/4train.csv')
    r2 = csv_reader('/Users/xzy/work/history/NNregression/4test.csv')
    
    #data = array([[1,2,3],[2,3,4],[3,4,5],[5,6,7]])
    train_data, y = r1.down_sample_seperate(1)
    test_data, labels = r2.down_sample_seperate(1)

    print(len(train_data))
    
#    from sklearn.ensemble import AdaBoostRegressor as ABR
#    
#    models = ABR(npGNGregressor(age_max_ = 200,nn_lambda_ = 40, GNG = True, globus = True))
    models = npGNGregressor(age_max_ = 200,nn_lambda_ = 60, GNG = True, globus = False)
#    models = ET()
    models.fit(train_data, y)
    
    pred_labels = models.predict(test_data)
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(labels, pred_labels))

