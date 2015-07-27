import sys
sys.path.append("/usr/local/lib/python2.7/site-packages/")
from sklearn import datasets
import numpy
import numpy.linalg
import numpy.random
from numpy import transpose as tr
import math
import random
from matplotlib import pyplot as plt

def ppca(sample,n_components=2,iter=1000):
    sample=numpy.matrix(sample)
    num_dimention = sample.shape[1]
    x_mean = numpy.average(sample,axis=0)
    latest_weights = weights = numpy.matrix(numpy.random.randn(num_dimention,n_components)*2.0+1.0)
    latent_variables = numpy.random.randn(n_components)
    latest_sigma2 = sigma2 = 10.
    for index in range(iter):
        M = weights.T*weights + float(sigma2)*numpy.matrix(numpy.identity(n_components))
        M_inv = numpy.linalg.inv(M)
        # E step
        mean_latent_variables = [M_inv*weights.T*(x-x_mean).T for x in sample]
        mean_latent_variables2 = [float(sigma2)*M_inv+mean_latent_variables[i]*mean_latent_variables[i].T for i,x in enumerate(sample)]
        
        # M step
        a = numpy.zeros([num_dimention,n_components])
        b = numpy.zeros([n_components,n_components])
        for i,x in enumerate(sample):
            diff = (x-x_mean).T
            a = a + diff*mean_latent_variables[i].T
            b = b + mean_latent_variables2[i]
        weights = a*numpy.linalg.inv(b)
        c = 0.
        for i,x in enumerate(sample):
            diff = (x-x_mean).T
            tr_a = sum(numpy.diag(mean_latent_variables2[i]*weights.T*weights))
            tmp = 2*mean_latent_variables[i].T*weights.T*diff
            norm2 = float(diff.T*diff)
            c += norm2-tmp+tr_a
        sigma2 = c/(sample.shape[0]*sample.shape[1])
        if index % 100 == 0:
            print "weights:\n{0}".format(weights)
        
        epcilon_w = numpy.fabs(latest_weights - weights).max()
        epcilon_s = abs(latest_sigma2 - sigma2)
        if epcilon_w < 1e-5 and epcilon_s < 1e-5 :
            break;
        latest_weights = weights
        latest_sigma2 = sigma2
    return [weights,sigma2,x_mean]

iris = datasets.load_iris()
[weights,sigma2,x_mean] = ppca(iris.data)
m = tr(weights).dot(weights) + float(sigma2) * numpy.eye(weights.shape[1])
m = numpy.linalg.inv(m)
targets = iris.target
for i,x in enumerate(numpy.matrix(iris.data)):
    new_sample = m.dot(tr(weights)).dot((x - x_mean).T)
    plt.scatter(new_sample[0][0],new_sample[1][0],c=['r','g','b'][targets[i]])
plt.show()
