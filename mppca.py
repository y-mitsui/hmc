import sys
sys.path.append("/usr/local/lib/python2.7/site-packages/")
from sklearn import datasets
import numpy
import numpy.linalg
import numpy.random
import math

def ppca(sample,n_components=2,iter=4000):
    sample=numpy.matrix(sample)
    print sample.shape
    num_dimention = sample.shape[1]
    x_mean = numpy.average(sample,axis=0)
    weights = numpy.random.randn(num_dimention,n_components)
    latent_variables = numpy.random.randn(n_components)
    sigma2 = 1.
    for index in range(iter):
        M = numpy.dot(weights.T,weights) + float(sigma2)*numpy.matrix(numpy.identity(n_components))
        M_inv = numpy.linalg.inv(M)
        # E step
        mean_latent_variables = [numpy.dot(numpy.dot(M_inv,weights.T),(x-x_mean).T) for x in sample]
        mean_latent_variables2 = [float(sigma2)*M_inv+numpy.dot(mean_latent_variables[i],mean_latent_variables[i].T) for i,x in enumerate(sample)]
        # M step
        a = numpy.zeros([num_dimention,n_components])
        b = numpy.zeros([n_components,n_components])
        for i,x in enumerate(sample):
            diff = (x-x_mean).T
            a = a + numpy.dot(diff,mean_latent_variables[i].T)
            b = b + mean_latent_variables2[i]
        weights = numpy.dot(a,numpy.linalg.inv(b))

        c = 0.
        for i,x in enumerate(sample):
            diff = (x-x_mean).T
            tr_a = sum(numpy.diag(numpy.dot(numpy.dot(mean_latent_variables2[i],weights.T),weights)))
            tmp = numpy.dot(2*mean_latent_variables[i].T,weights.T)
            norm2 = math.sqrt((numpy.array(x-x_mean)*numpy.array(x-x_mean)).sum())
            c += norm2-numpy.dot(tmp,diff)+tr_a
        sigma2 = c/(sample.shape[0]*sample.shape[1])
        if index % 100 == 0:
            print "weights:\n{0}".format(weights)
    return weights

iris = datasets.load_iris()
weights = ppca(iris.data)
for x in iris.data:
    fa = numpy.dot(iris.data,weights)