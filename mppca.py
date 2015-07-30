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
            print diff.T*diff
            print norm2
            sys.exit(1)
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
    left=1./math.sqrt(2.*3.14159)**mean.shape[0]*math.sqrt(numpy.linalg.det(cover))
    right=-1./2.*(x[0]-mean)*numpy.linalg.inv(cover)*(x[0]-mean).T
    return numpy.array(left*numpy.exp(right))[0][0]
def mixtureGaussian(x,means,covers,weights):
    weights.append(1.-sum(weights))
    return sum([weight*gaussian(x,mean,cover) for mean,cover,weight in zip(means,covers,weights)])

def mppca(sample,n_components=2,n_gauss=2,iter=50):
    sample=numpy.matrix(sample)
    num_dimention = sample.shape[1]
    weights = [numpy.matrix(numpy.random.randn(num_dimention,n_components)*2.0+1.0) for _ in range(n_gauss)]
    sigma2 = [10.]*n_gauss
    means = numpy.random.randn(n_gauss,num_dimention)
    stick=1.
    gausian_weight=[]
    for i in range(n_gauss-1):
        tmp=numpy.random.rand() * stick
        stick-=tmp
        gausian_weight.append(tmp)
    gausian_weight.append(1.-sum(gausian_weight))
    gausian_weight=[1./float(n_gauss) for _ in range(n_gauss)]

    for index in range(iter):
        mean_latent_variables = []
        mean_latent_variables2 = []
        for i in range(n_gauss):
            M = weights[i].T*weights[i] + float(sigma2[i])*numpy.matrix(numpy.identity(n_components))
            M_inv=numpy.linalg.inv(M)
            # E step
            mean_latent_variables.append( [M_inv*weights[i].T*(x-means[i]).T for x in sample] )
            mean_latent_variables2.append( [float(sigma2[i])*M_inv+mean_latent_variables[i][j]*mean_latent_variables[i][j].T for j,x in enumerate(sample)] )
        
        R=[]
        
        print gausian_weight
        #if index > 10 :
        #    sys.exit(1)

        cover=[]
        for i in range(n_gauss):
                cover.append(weights[i]*weights[i].T+sigma2[i]*numpy.matrix(numpy.identity(num_dimention)))
        mix_gauss = []
        for j,x in enumerate(sample):
            tmp=[]
            for i in range(n_gauss):
                tmp.append(gausian_weight[i]*gaussian(x,means[i],cover[i]))
            mix_gauss.append(sum(tmp))

        #print "------------------"
        for i in range(n_gauss):
            tmp = []
            for j,x in enumerate(sample):
                tmp.append(gausian_weight[i]*gaussian(x,means[i],cover[i])/mix_gauss[j])
            #print sum(tmp)/sample.shape[0]
            R.append(tmp)

        
        # M step
        gausian_weight = numpy.array([sum(R[i])/sample.shape[0] for i in range(n_gauss)])
        means = []
        for i in range(n_gauss):
            total=numpy.matrix(numpy.zeros([num_dimention,1]))
            for j,x in enumerate(sample):
                total += R[i][j]*(x.T-weights[i]*mean_latent_variables[i][j])
            means.append((total/sum(R[i])).T)
        #print means
        weights = []
        for i in range(n_gauss):
            a = numpy.zeros([num_dimention,n_components])
            b = numpy.zeros([n_components,n_components])
            for j,x in enumerate(sample):
                diff = (x-means[i]).T
                a = a + R[i][j]*diff*mean_latent_variables[i][j].T
                b = b + R[i][j]*mean_latent_variables2[i][j]
            #print b
            weights.append(a*numpy.linalg.inv(b))

        sigma2 = []
        for i in range(n_gauss):
            diffs =[x-means[i] for x in sample]
            tmp = sum([rates*(diff*diff.T)[0,0] for rates,diff in zip(R[i],diffs)])

            tmp2=0.
            for j in range(sample.shape[0]):
                tmp2 += R[i][j]*mean_latent_variables[i][j].T*weights[i].T*diffs[j].T
            tmp2=-2*tmp2

            tmp3=sum([rates*sum(numpy.diag(mean_latent_variable2*weights[i].T*weights[i])) for rates,mean_latent_variable2 in zip(R[i],mean_latent_variables2[i])])
            a=tmp+tmp2+tmp3
            b=sum(R[i])
            #print b
            #if abs(b) < 1e-15 :
            #    b=1e-15 if b > 0. else -1e-15
            #print b
            c=1./(num_dimention*b)
            sigma2.append(c*a[0,0])

        like=0.
        for j,x in enumerate(sample):
            tmp = []
            for i in range(n_gauss):
                cover=weights[i]*weights[i].T+sigma2[i]*numpy.matrix(numpy.identity(num_dimention))
                tmp.append(gausian_weight[i]*gaussian(x,means[i],cover))
            like+=sum(tmp)
        print like
        #print weights
        #if( weights[0][0,0] != val ):
    return [weights,sigma2,means]

iris = datasets.load_iris()
#[weights,sigma2,x_mean] = ppca(iris.data)
#weights=[weights]
#sigma2=[sigma2]
#x_mean=[x_mean]
[weights,sigma2,x_mean] = mppca(iris.data)
print weights
#sys.exit(1)
m = tr(weights[0]).dot(weights[0]) + float(sigma2[0]) * numpy.eye(weights[0].shape[1])
m = numpy.linalg.inv(m)
targets = iris.target
for i,x in enumerate(numpy.matrix(iris.data)):
    new_sample = m.dot(tr(weights[0])).dot((x - x_mean[0]).T)
    plt.scatter(new_sample[0][0],new_sample[1][0],c=['r','g','b'][targets[i]])
plt.show()


sys.exit(1)
m = tr(weights).dot(weights) + float(sigma2) * numpy.eye(weights.shape[1])
m = numpy.linalg.inv(m)
targets = iris.target
for i,x in enumerate(numpy.matrix(iris.data)):
    new_sample = m.dot(tr(weights)).dot((x - x_mean).T)
    plt.scatter(new_sample[0][0],new_sample[1][0],c=['r','g','b'][targets[i]])
plt.show()
