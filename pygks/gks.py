from numpy import array, matrix, diag, exp, inner, nan_to_num, pi, sqrt, prod
from numpy.core.umath_tests import inner1d
from numpy import argmin
from math import log

class GKS:
    """Gaussian kernel smoother to transform any clustering method into regression. setN is the list containing numpy arrays which are the weights of clustering centors.
    populations is a list of integers of cluster populations. standard_variances is the list of real
    numbers meaning the standard variances of the dataset along each dimension. smooth is None or real number.
    While set to None, an SSL procedure will be employed. For details, see the responses() method."""
    sv_kernel = None
    setN = None #:Weights of the clustering centers, after instance initialization, it will be a list data structure.
    Y = [] #:Response variables.
    percentages = None #:Distribution of the cluster populations.
    xdim = None #:Dimension of the explanatory variables.
    ydim = None #:Dimension of the response variables.
    __global = True
    smooth = None #:Smooth parameter.
    __S = 1.0
    K = 5 #: Number of clustering centers for smooth parameter calculation.
    percentages_logloss = None

    X_ae = []
    Y_ae = []
    kernel_ae = []
    
    def __init__(self, setN, populations, standard_variances, Y_number, smooth = None, K = 5):
        if len(setN[0])!=len(standard_variances):
            print('ill GKS initialization')
        elif Y_number == 0:
            self.setN = array(setN)
            
            self.percentages = array(populations) / float(sum(populations))
            
            self.xdim = float(len(setN[0]) - Y_number)
            self.ydim = float(Y_number)
            self.smooth = smooth
            self.K = K

            X_len = int(self.xdim)
            for i in range(X_len):
                X_mask = list(range(X_len))
                X_mask.pop(i)
                self.X_ae.append(self.setN[:,X_mask])
                self.Y_ae.append(self.setN[:,i])
                self.kernel_ae.append(matrix(diag(array(standard_variances)[X_mask]**-1.0)))
            self.sv_kernel = matrix(diag(array(standard_variances)**-1.0))
            self.percentages_logloss = self.percentages / (sqrt((2.0*pi)**self.xdim * prod(standard_variances)))
            #print self.percentages_logloss, 'logloss'
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

    def ae_1s_1d(self, X_in, i):
        point = list(X_in)
        point.pop(i)
        point = array(point)
        dif_vectors = self.X_ae[i] - point
        dif_and_varianced = array(matrix(dif_vectors)*self.kernel_ae[i])
        dif_traces = inner1d(dif_and_varianced , dif_vectors)
        weights = exp(-0.5*self.__S*dif_traces)
        results = (self.Y_ae[i]*(matrix(self.percentages * weights).T))/(inner(self.percentages, weights))
        return array(results.T)[0]

    def ae_1s(self, X_in):
        self.__S = len(self.setN)**(-2.0*self.smooth)
        X_construct = []
        for i in range(int(self.xdim)):
            X_construct.append(self.ae_1s_1d(X_in, i))
        return array(X_construct)

    def ae_gradient_1s_1d(self, X_in, i):
        point = list(X_in)
        Y = point.pop(i)
        point = array(point)
        dif_vectors = self.X_ae[i] - point
        dif_and_varianced = array(matrix(dif_vectors)*self.kernel_ae[i])
        dif_traces = inner1d(dif_and_varianced , dif_vectors)
        weights = exp(-0.5*self.__S*dif_traces)

        origin_up = (self.Y_ae[i]*(matrix(self.percentages * weights).T))[0,0]
        origin_down = inner(self.percentages, weights)
        delta_up = (self.Y_ae[i]*(matrix(self.percentages * weights * array(dif_traces)).T))[0,0]
        delta_down = inner(self.percentages, dif_traces * weights)
        #print origin_up,'up'
        #print dif_traces
        #print origin_down
        #print delta_up
        #print delta_down
        results = (self.Y_ae[i]*(matrix(self.percentages * weights).T))/(inner(self.percentages, weights))
        Y_pred = array(results.T)[0]
        #print Y,Y_pred
        gradient = (Y - Y_pred)*(delta_up * origin_down - origin_up * delta_down) / ((origin_down)**2.0)
        #print gradient
        return gradient[0]

    def __recip_kernel(self, Deltas, a):
        return (a) / (a**2*Deltas+pi**2)

    def __recip_kernel_dif(self, Deltas, a):
        return (pi**4-a**4*Deltas**2) / (a**2*Deltas+pi**2)**3
        
    def ae_gradient_1s_y(self, X_in, y, kernel = 'exp'):
        if kernel == 'exp':
            point = array(X_in)
            point = array(point)
            dif_vectors = self.setN - point
            dif_and_varianced = array(matrix(dif_vectors)*self.sv_kernel)
            dif_traces = inner1d(dif_and_varianced , dif_vectors)
            weights = exp(-0.5*self.__S*dif_traces)

            origin_up = (self.Y*(matrix(self.percentages * weights).T))[0,0]
            origin_down = inner(self.percentages, weights)
            delta_up = (self.Y*(matrix(self.percentages * weights * array(dif_traces)).T))[0,0]
            delta_down = inner(self.percentages, dif_traces * weights)
            #print origin_up,'up'
            #print dif_traces
            #print origin_down
            #print delta_up
            #print delta_down
            results = (self.Y*(matrix(self.percentages * weights).T))/(inner(self.percentages, weights))
            Y_pred = array(results.T)[0]
            #print Y,Y_pred
            gradient = (y - Y_pred)*(delta_up * origin_down - origin_up * delta_down) / ((origin_down)**2.0)
            #print gradient
            return gradient[0]
        elif kernel == 'rec':
            point = array(X_in)
            point = array(point)
            dif_vectors = self.setN - point
            dif_and_varianced = array(matrix(dif_vectors)*self.sv_kernel)
            dif_traces = inner1d(dif_and_varianced , dif_vectors)
            weights = self.__recip_kernel(dif_traces, self.__S)
            dif_weights = self.__recip_kernel_dif(dif_traces, self.__S)
            origin_up = (self.Y*(matrix(self.percentages * weights).T))[0,0]
            origin_down = inner(self.percentages, weights)
            D_up = (self.Y*(matrix(self.percentages * dif_weights).T))[0,0]
            origin_down = inner(self.percentages, weights)
            D_down = inner(self.percentages, dif_weights)
            Y_pred = origin_up / origin_down
            gradient = 2*(y - Y_pred)*(D_up * origin_down - origin_up * D_down) / ((origin_down)**2.0)
            return gradient
        else:
            
            print('No kernel specified...')
            return 0

    def ae_gradient_1s(self, X_in):
        X_construct = 0.0
        for i in range(int(self.xdim)):
            X_construct += self.ae_gradient_1s_1d(X_in, i)
        return X_construct / float(self.xdim)
        
    def ae_log_1s(self, point):
        dif_vectors = self.setN - point
        dif_and_varianced = array(matrix(dif_vectors)*self.sv_kernel)
        dif_traces = inner1d(dif_and_varianced , dif_vectors)
        return inner(self.percentages_logloss ,  (self.__S ** (self.xdim)) * exp(-0.5 * (self.__S**2.0) *dif_traces))
        
    def ae_gradient_log_1s(self, point):
        dif_vectors = self.setN - point
        dif_and_varianced = array(matrix(dif_vectors)*self.sv_kernel)
        dif_traces = inner1d(dif_and_varianced , dif_vectors)
        f_s = self.__S ** (self.xdim) * exp(-0.5 * (self.__S**2.0) *dif_traces)
        ugly = (log(self.xdim)-2.0*self.__S*dif_traces)
        f_s_1 = f_s * ugly
        f_s_2 = -2.0 * dif_traces * f_s + ugly**2.0 * f_s
        gradient = inner(self.percentages_logloss, 2.0 * f_s_1* f_s_2)
        return gradient

    def ae_train_S(self, X = None, y=0, step = 1, stop = 0.0001, loss = 'mse', kernel = 'rec'):
        self.__S = len(self.setN)**(-2.0*self.smooth)
        count = 0
        kernel_type = kernel
        if y != 0:
            for i in range(len(X)):
                delta_S = self.ae_gradient_1s_y(X[i], y[i], kernel = kernel_type)
                if (abs(delta_S) < stop):
                    break
                self.__S -= delta_S
                count += 1
        else:
            for each in X:
                if loss == 'mse':
                    delta_S = self.ae_gradient_1s(each) * step
                else:
                    delta_S = self.ae_gradient_log_1s(each) * step
                if (abs(delta_S) < stop):
                    break
                self.__S -= delta_S
                count += 1
            #print self.__S, -0.5*log(self.__S, len(self.setN))
        print('Gradient Decents in', count, 'steps.')
        self.smooth = -0.5*log(self.__S, len(self.setN))
        return -0.5*log(self.__S, len(self.setN))

    def ae_encode_1s(self, point):
        dif_vectors = self.setN - point
        dif_and_varianced = array(matrix(dif_vectors)*self.sv_kernel)
        dif_traces = inner1d(dif_and_varianced , dif_vectors)
        weights = exp(-0.5*self.__S*dif_traces)
        #return self.percentages * weights
        weights /= max(weights)
        return nan_to_num(weights)

    def ae_encode(self, X):
        #self.percentages /= max(self.percentages)
        encoded = []
        for each in X:
            encoded.append(self.ae_encode_1s(each))
        return encoded
    
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
    testgks = GKS([array([1, 2, 2,3]), array([2, 3, 1,5]), array([1, 1, 1,5])], array([1, 2,1]), array([1, 2, 3,1]), 0, smooth = -0.2)
    print(testgks.ae_train_S([array([1,2,2,3]), array([2, 2, 1,4])], loss = 'mse'))
    #print testgks.ae_train_S([array([1,2,2]), array([2, 2, 1])],array([3, 4]),  loss = 'mse')
    #print testgks.ae_encode([array([1, 2, 2]), array([2, 3, 1]), array([1, 1, 1])])
    print(testgks.ae_encode([array([1, 2, 2,3]), array([2, 3, 1,5]), array([1, 1, 1,5])]))
    #print testgks.responses([array([2,3,9])])
        
