from numpy import array, matrix, diag, exp, inner, nan_to_num
from numpy.core.umath_tests import inner1d
from numpy import argmin

class GKS:
    sv_kernel = None
    setN = None
    Y = None
    percentages = None
    xdim = None
    ydim = None
    globus = True
    def __init__(self, setN, populations, standard_variances, Y_number, globus = False):
        if len(setN[0])!=len(standard_variances):
            print('ill GKS initialization')
        else:
            self.sv_kernel = matrix(diag(array(standard_variances)[:-1*Y_number]**-1.0))
            self.setN = []
            self.Y = []
            for each in setN:
                self.setN.append(each[:-1*Y_number])
                self.Y.append(each[-1*Y_number:])
            self.Y = matrix(self.Y).T
##            print self.Y
            self.percentages = populations / float(sum(populations))
            self.setN = array(self.setN)
            self.xdim = float(len(setN[0]) - Y_number)
            self.ydim = float(Y_number)
            self.globus = globus
            
    def response_1s(self, point):
        dif_vectors = self.setN - point
        dif_and_varianced = array(matrix(dif_vectors)*self.sv_kernel)
##        print self.sv_kernel
##        print matrix(dif_and_varianced) * matrix(dif_vectors.T)
#        nn_index = argmin(dif_and_varianced)
        dif_traces = inner1d(dif_and_varianced , dif_vectors)
        if self.globus:
            #S = len(self.setN)*self.xdim/float(inner(self.percentages, dif_traces))
            if sum(dif_traces) == 0:
                S = float('inf')
            else:
                S = len(self.setN)*self.xdim/float(sum(dif_traces))
        else:
            nn_index = argmin(dif_traces)
#        print dif_traces.shape, dif_and_varianced.shape
#        print nn_index, dif_traces
##        print dif_traces
##        print dif_vectors, 'x',dif_and_varianced, 'x',dif_traces
        #S = self.xdim/float(sum(dif_traces))
            if dif_traces[nn_index] == 0:
                S = float('inf')
            else:
                S = self.xdim/float(dif_traces[nn_index])
        if S < 0.0:
            print('xx''')
            S = 0.0
        weights = exp(-0.5*dif_traces)**S
##        print weights
        results = (self.Y*(matrix(self.percentages * weights).T))/(inner(self.percentages, weights))
##        print results.T
        return array(results.T)[0]

    def responses(self, points):
        results = []
        if self.ydim == 1:
            for each in points:
                results.append(self.response_1s(each)[0])
        else:
            for each in points:
                results.append(self.response_1s(each))
        return results
        
if __name__ == '__main__':
    testgks = GKS([array([1, 2, 2,3]), array([2, 3, 1,5])], array([1, 2]), array([1, 2, 3,1]), 2)
    print(testgks.response_1s(array([1,2])))
    print(testgks.responses([array([1,2]),array([2,0])]))
        
