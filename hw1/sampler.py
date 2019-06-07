import numpy as np
import math
class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):

    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
    
    '''
    #This is using the central limit theorem to approximate univariate normal distribution.
    
    def sample(self):
        xlist = []
        for i in range(12):
            x = np.random.uniform(0,1)
            xlist.append(x)
        xsum = sum(xlist)
        final = xsum - 6
        
        #to generalize this
        final = self.mu + final * self.sigma
        return final
    '''
    
    #This is using Marsaglia polar method.
    def sample(self):
        while 1:
            u = np.random.uniform() * 2 - 1
            v = np.random.uniform() * 2 - 1
            s = u*u + v*v
            if s < 1 and s > 0:
                break
        mul = math.sqrt(-2.0 * math.log(s)/s)
        return self.mu + self.sigma * u * mul

# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.mu = Mu
        self.sigma = Sigma
    
    
    def sample(self):
        A = np.linalg.cholesky(self.sigma)
        i = len(self.mu)
        z = []
        obj = UnivariateNormal(0,1)
        for j in range(i):
            s = obj.sample()
            z.append(s)
        x = self.mu + A.dot(np.asarray(z))
        return x



# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):

    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
        self.ap = ap
        pass

    def sample(self):
        #ap is an array of probablities
        k = len(self.ap)
        apsum = sum(self.ap)

        normAP = []
        for i in self.ap:
            norm = i/apsum
            normAP.append(norm)
        
        cdf = []
        cdf.append(normAP[0])
        for i in range(1,k):
            cdf.append(sum(normAP[0:i+1]))
        print cdf

        x = np.random.uniform(0,1)
        for i in range(len(cdf)):
            if cdf[i] > x:
                return i



# The sample space of this probability model is the union of the sample spaces of
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self,ap,pm):
        self.obj = Categorical(ap)
        self.pm = pm
    
    def sample(self):
        x = self.obj.sample()
        return self.pm[x].sample()













