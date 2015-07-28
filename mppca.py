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
import gaussian

def ppca(sample,n_components=2,iter=1000):
    sample=numpy.matrix(sample)
    num_dimention = sample.shape[1]
    x_mean = numpy.average(sample,axis=0)
    latest_weights = weights = numpy.matrix(numpy.random.randn(num_dimention,n_components)*2.0+1.0)
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
def gaussian(x,mean,cover):
    context = Gaussian()
    context.setMean(mean)
    context.setCovariance(cover)
    return math.exp(context.lnProb(x))
def mixtureGaussian(x,means,covers,weights):
    weights.append(1.-sum(weights))
    return sum([weight*gaussian(x,mean,cover) for mean,cover,weight in zip(means,covers,weights)])

def mppca(sample,n_components=2,n_gauss=2,iter=1000):
    sample=numpy.matrix(sample)
    num_dimention = sample.shape[1]
    x_mean = numpy.average(sample,axis=0)
    latest_weights = weights = [numpy.matrix(numpy.random.randn(num_dimention,n_components)*2.0+1.0) for _ in range(n_gauss)]
    latest_sigma2 = sigma2 = [10.]*n_gauss
    means = numpy.random.randn(n_gauss,num_dimention)
    gausian_weight = numpy.random.randn(n_gauss-1)

    for index in range(iter):

        for i in range(n_gauss):
            M = weights[i].T*weights[i] + float(sigma2[i])*numpy.matrix(numpy.identity(n_components))
            M_inv=numpy.linalg.inv(M)
            # E step
            mean_latent_variables[i] = [M_inv*weights[i].T*(x-x_mean).T for x in sample]
            mean_latent_variables2[i] = [float(sigma2)*M_inv+mean_latent_variables[i][j]*mean_latent_variables[i][j].T for j,x in enumerate(sample)]
        
        R=[]
        new_gausian_weight=gausian_weight
        new_gausian_weight.append(1.0-gausian_weight.sum())

        mix_gauss = []
        for j,x in enumerate(sample):
            tmp=[]
            for i in range(n_gauss):
                cover=weights[i]*weights[i].T+sigma2*numpy.matrix(numpy.identity(num_dimention))
                tmp.append(new_gausian_weight[i]*gaussian(x,means[i],cover))
            mix_gauss.append(sum(tmp))

        for i in range(n_gauss):
            tmp = []
            for j,x in enumerate(sample):
                cover=weights[i]*weights[i].T+sigma2*numpy.matrix(numpy.identity(num_dimention))
                tmp.append(new_gausian_weight[i]*gaussian(x,means[i],cover)/mix_gauss[j])
            R.append(tmp)

        
        # M step
        gausian_weight = [sum(R[i])/sample.shape[0] for i in range(n_gauss)]
        means = []
        for i in range(n_gauss):
            total=0.
            for j,x in enumerate(sample):
                total += R[i][j]*(x-weights[i]*mean_latent_variables[i])
            means.append(total/sum(R[i]))

        weights = []
        for i in range(n_gauss):
            for j,x in enumerate(sample):
                diff = (x-means[i]).T
                a = a + diff*mean_latent_variables[i].T
                b = b + mean_latent_variables2[i]
            weights.append(a*numpy.linalg.inv(b))
        
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
