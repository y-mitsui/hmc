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

def ppca(sample,n_components=2,iter=100):
    sample=numpy.matrix(sample)
    print sample.shape
    num_dimention = sample.shape[1]
    x_mean = numpy.average(sample,axis=0)
    weights = numpy.random.randn(num_dimention,n_components)
    latent_variables = numpy.random.randn(n_components)
    sigma2 = 10.
    for index in range(iter):
        M = numpy.dot(weights.T,weights) + float(sigma2)*numpy.matrix(numpy.identity(n_components))
        print M
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
    return [weights,sigma2,x_mean]

iris = datasets.load_iris()
#alist = iris.data.tolist()
#blist = iris.target.tolist()
#clist = [0. for _ in range(len(alist))]
#dlist = [0. for _ in range(len(alist))]
#for i in range(len(alist)):
#    idx = random.randint(0,len(alist)-1)
#    clist[idx]=alist[len(alist)-idx]
#    alist[i] = 
#random.shuffle(alist)
#iris.data = numpy.array(alist)
[weights,sigma2,x_mean] = ppca(iris.data)
print sigma2
#sys.exit(1)
m = tr(weights).dot(weights) + float(sigma2) * numpy.eye(weights.shape[1])
m = numpy.linalg.inv(m)
targets = iris.target
for i,x in enumerate(numpy.matrix(iris.data)):
    new_sample = m.dot(tr(weights)).dot((x - x_mean).T)
#    print new_sample 
#    new_sample=numpy.array(numpy.dot(numpy.matrix(x),weights))
    plt.scatter(new_sample[0][0],new_sample[1][0],c=['r','g','b'][targets[i]])
plt.show()
