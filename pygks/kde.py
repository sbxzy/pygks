from numpy import array, matrix,log
from math import exp
from numpy import identity
from numpy import zeros
from .__gaussian_custom import norm_pdf_multivariate

class density:
    """This is a kernel density estimation framework, but not the complete algorithm.
    The user will have to run a clustering method first to get the clustering centers
     and cluster populations. setN is the list of numpy arrays which are the weights of clustering centers.
        countN is the list containing cluster populations. standard_deviation is the list of standard deviations
            of the dataset along each dimension."""
    setN = []
    countN = array([])
    bandwidth = array([])
    denominator = 0.0
    k_i = array([])

    mu = array([])
    sigma = matrix([])
    validate_position = array([])
    
    def __init__(self,setN,countN,standard_deviation):
        self.validate_position = (standard_deviation != 0.0)
        dimension = sum(self.validate_position)
        # the following is 3 methods for bandwidth calculations
##        band_raw = standard_deviation * (len(setN) ** (-1.0 / (dimension + 4)))
##        band_raw = 1.06 * standard_deviation * (dimension ** (-1.0 / 5.0))
        band_raw = 1.06* standard_deviation * (dimension ** (-1.2))
        #initialize
        tmp_set = []
        for point in setN:
            tmp_set.append(self.cut_array(point,self.validate_position))
        self.setN = tmp_set
        
        self.countN = countN
        self.k_i = countN / float(sum(countN))
        if dimension != len(self.setN[0]):
            input('dimension error')
        self.bandwidth = self.cut_array(band_raw,self.validate_position)
        self.sigma = matrix(identity(dimension))
        self.mu = zeros(dimension)
        decied_bands = self.bandwidth
        very_small = 1.0
        for band in decied_bands:
            very_small = very_small * band
        self.denominator = very_small

    def cut_array(self,x,mask):
        tmp = []
        for i in range(len(list(mask))):
            if mask[i]:
                tmp.append(x[i])
        return array(tmp)
        
    def kernel(self,x): #:Defines a Gaussian kernel as f(x), can define different kernels.
        return norm_pdf_multivariate(x,self.mu,self.sigma)
    
    def estimate(self,x_in): #:Estimate the density of the vector x_in.
        x = self.cut_array(x_in,self.validate_position)
        sum_density = 0.0
        for i in range(len(self.setN)):
            sum_density += self.kernel((x - self.setN[i]) / self.bandwidth) / self.denominator  * self.k_i[i]
        return sum_density
