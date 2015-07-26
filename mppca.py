import numpy

def ppca(sample,iter=100):
    num_dimention = sample.shape[1]
    
    weights = numpy.nrand()
    latent_variables = numpy.nrand()
    sigma2 = 1.
    for _ range(iter):
        M = numpy.dot(weights,weights.T) + sigma2*numpy.matrix(numpy.identity(num_dimention))
        a = b = 0.
        for x in sample:
            diff = x-x_mean
            mean_latent_variables = numpy.dot(numpy.dot(M.I,weights.T),diff)
            mean_latent_variables2 = sigma2*M.I+numpy.dot(mean_latent_variables,mean_latent_variables.T)
            a += numpy.dot(diff,mean_latent_variables.T)
            b += mean_latent_variables2
        weights = numpy.dot(a,b.T)
        for x in sample:
            diff = x-x_mean
            mean_latent_variables = numpy.dot(numpy.dot(M.I,weights.T),diff)
            mean_latent_variables2 = sigma2*M.I+numpy.dot(mean_latent_variables,mean_latent_variables.T)
            tr_a=sum(numpy.diag(numpy.dot(numpy.dot(mean_latent_variables2,weights),weights)))
            c += diff*diff-numpy.dot((2*mean_latent_variables).T,weights.T)*diff+tr_a
        sigma2 = c/(sample.shape[0]*sample.shape[1])
    
