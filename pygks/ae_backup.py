from numpy import array, matrix, diag, exp, inner, nan_to_num
from numpy.core.umath_tests import inner1d
from numpy import argmin, array

class GKS:
    """Gaussian kernel smoother to transform any clustering method into regression. setN is the list containing numpy arrays which are the weights of clustering centors.
    populations is a list of integers of cluster populations. standard_variances is the list of real
    numbers meaning the standard variances of the dataset along each dimension. smooth is None or real number.
    While set to None, an SSL procedure will be employed. For details, see the responses() method."""
    sv_kernel = None
    setN = None #:Weights of the clustering centers, after instance initialization, it will be a list data structure.
    Y = 1 #:Number of response variables.
    percentages = None #:Distribution of the cluster populations.
    xdim = None #:Dimension of the explanatory variables.
    ydim = None #:Dimension of the response variables.
    __global = True
    smooth = None #:Smooth parameter.
    __S = 0.0
    K = 5 #: Number of clustering centers for smooth parameter calculation.
    
    def __init__(self, setN, populations, standard_variances, Y_number, smooth = None, K = 5):
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
            self.percentages = array(populations) / float(sum(populations))
            self.setN = array(self.setN)
            self.xdim = float(len(setN[0]) - Y_number)
            self.ydim = float(Y_number)
            self.smooth = smooth
            self.K = K
            
            
    def response_1s(self, point):
        dif_vectors = self.setN - point
        dif_and_varianced = array(matrix(dif_vectors)*self.sv_kernel)
        dif_traces = inner1d(dif_and_varianced , dif_vectors)
        weights = exp(-0.5*self.__S*dif_traces)
        results = (self.Y*(matrix(self.percentages * weights).T))/(inner(self.percentages, weights))
        return array(results.T)[0]

    def responses(self, points, prototypes = None):
        """points is a list or array of numpy arrays, and this method returns the regression results
        of the dataset points. If the smooth parameter is initialized as None, the prototypes parameter
        will be required as a list or array of clustering centers in the form of numpy arrays, which is genertated
        by the user chosen clustering method on the same dataset to the one specified by points variable."""
        if self.smooth == None:
            self.K = min(self.K, prototypes)
            accumulated_traces = 0.0
            for point in prototypes:
                dif_vectors = self.setN - point
                dif_and_varianced = array(matrix(dif_vectors)*self.sv_kernel)
                dif_traces = inner1d(dif_and_varianced , dif_vectors)
                nn_index = argmin(dif_traces)
                accumulated_traces += float(dif_traces[nn_index])
                for i in range(self.K - 1):
                    dif_traces[nn_index] = float('inf')
                    nn_index = argmin(dif_traces)
                    accumulated_traces += float(dif_traces[nn_index])
            self.__S = len(self.setN)*self.xdim/accumulated_traces
            if self.__S < 0.0:
                self.__S = 0.0
        else:
            self.__S = len(self.setN)**(-2.0*self.smooth)
        results = []
        if self.ydim == 1:
            for each in points:
                results.append(self.response_1s(each)[0])
        else:
            for each in points:
                results.append(self.response_1s(each))
        return results
        
if __name__ == '__main__':
    testgks = GKS([array([1, 2, 2,3]), array([2, 3, 1,5])], array([1, 2]), array([1, 2, 3,1]), 2, smooth = -0.4)
    print(testgks.response_1s(array([1,2])))
    print(testgks.responses([array([1,2]),array([2,0])]))
        
