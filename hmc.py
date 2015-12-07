#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy
import random
import sys
import math
import time

class HMC:
    """
    ハミルトニアンモンテカルロ法
    """
    def __init__(self, posterior_distribution, posterior_distribution_delta, argument, parameter, range_parameter):
        """
        :param posterior_distribution: callable: 引数にparameterとargumentを受け取り、事後分布の対数尤度を返す関数
        :param posterior_distribution_delta: callable: posterior_distributionの導関数
        :param argument: object: posterior_distribution及びposterior_distribution_deltaに渡す任意の引数
        :param parameter: numpy.ndarray: 初期パラメーター
        :param range_parameter: numpy.ndarray: parameterが取りうる値の範囲。[[最小値1,最大値1],[最小値2,最大値2],...,[最小値N,最大値N]]
        """
        self.argument = argument
        self.posterior_distribution = posterior_distribution
        self.posterior_distribution_delta = posterior_distribution_delta
        self.parameter = parameter
        self.range_parameter = range_parameter
    
    def sampling(self, iter=3000, burn_in=1500, frequency=10, iter_leapfrog=40):
        """
        parameterのマルコフ連鎖をサンプリング
       
        :param iter: int: MCMCの繰り返し回数
        :param burn_in: int: サンプリング前の準備回数
        :param frequency: int: サンプリング頻度
        :param iter_leapfrog: int: リープフロッグ繰り返し数
        :return numpy.ndarray: サンプリングされたパラメーターのマルコフ連鎖
        """
    
        target_accept = 0.8 #目標採択率
        t0 = 10
        gamma = 0.05
        average_H = 0.0
        
        current_parameter = self.parameter
        result = [] #戻り値
        
        step_size  = self.findReasonableEpsilon(self.parameter)
        u = numpy.log(10 * step_size)
        
        log_view_frequency = int(iter / 10)
        
        for i in range(iter):
            momentum = numpy.random.randn(len(self.parameter))
            hamilton = self.posterior_distribution(current_parameter, self.argument) - numpy.sum(momentum**2.0) / 2.0
            current_parameter_condinate = current_parameter
            
            for _ in range(iter_leapfrog):
                try:
                    current_parameter_condinate, momentum = self.leapfrog(current_parameter_condinate,momentum,step_size)
                except Exception, e:
                    print e
                    continue
            
            difference_hamilton = ( (self.posterior_distribution(current_parameter_condinate, self.argument) - numpy.sum(momentum ** 2.0) / 2.0) ) - hamilton

            accept_probability = min(1.,numpy.exp(difference_hamilton))
            if i < burn_in:
                H_t = target_accept - accept_probability
                w = 1. / ( i + t0 )
                average_H = (1 - w) * average_H + w * H_t
                step_size = numpy.exp(u - (numpy.sqrt(i)/gamma)*average_H)
                if i % log_view_frequency == 0:
                    # print dual averaging's parameters
                    print "------ %d ------"%(i)
                    print "average_H:{}".format(average_H)
                    print "step_size:{}".format(step_size)

            if random.random() < numpy.exp(difference_hamilton):
                current_parameter = current_parameter_condinate

            
            if i % log_view_frequency == 0:
                print "====== %d / %d ======"%(i,iter)
                #print "current_parameter:{0}".format(current_parameter)
                print "loglikelihood:{0}".format(self.posterior_distribution(current_parameter,self.argument))
            if i > burn_in and i % frequency == 0:
                result.append(current_parameter)
            
        return result
    
    # step_sizeの初期化
    def findReasonableEpsilon(self, parameter):
        epsilon = 1e-20
        momentum = numpy.random.normal(0., 1., len(parameter))
        hamilton = self.posterior_distribution(parameter, self.argument) - numpy.sum(momentum**2.0) / 2.0
        current_parameter_condinate, momentum = self.leapfrog(parameter, momentum, epsilon)
        difference_hamilton = (self.posterior_distribution(current_parameter_condinate, self.argument) - numpy.sum(momentum ** 2.0) / 2.0)  - hamilton
        accept_probability = numpy.exp(difference_hamilton)

        a = 2 * int( accept_probability > 0.5) - 1

        while accept_probability ** a > 2 **-a :
        
            epsilon = 2. ** a * epsilon
            current_parameter_condinate, momentum = self.leapfrog(current_parameter_condinate, momentum, epsilon)
            difference_hamilton = (self.posterior_distribution(current_parameter_condinate, self.argument) - numpy.sum(momentum ** 2.0) / 2.0) - hamilton
            accept_probability = numpy.exp(difference_hamilton)
        return epsilon
    
    #パラメーターをrange_parameterの値の範囲に収める
    def normalizeRange(self, current_parameter_condinate):
        result = []
        for param,(minParam,maxParam) in zip(current_parameter_condinate,self.range_parameter):
            result.append(max(min(param,maxParam),minParam))
        return numpy.array(result)
        
    # リープフロッグによる経路積分
    def leapfrog(self, current_parameter_condinate, momentum, step_accuracy):
        momentum = momentum + (step_accuracy/2.) * self.posterior_distribution_delta(current_parameter_condinate, self.argument)
        current_parameter_condinate = current_parameter_condinate + step_accuracy * momentum
        #current_parameter_condinate = self.normalizeRange(current_parameter_condinate)
        momentum = momentum + (step_accuracy/2.) * self.posterior_distribution_delta(current_parameter_condinate, self.argument)
    
        return current_parameter_condinate, momentum
        
if __name__ == "__main__":
    ##########     example - 多項分布のパラメーターを階層ベイズ推定    ##########
    import theano
    import theano.tensor as T
    
    def callPosterior(parameter,argument):
        return posterior(parameter)
    
    def callPosteriorDelta(parameter,argument):
        diffence_values = gPosterior(parameter)
        #diffence_values = numpy.hstack(diffence_values)
        #if any(diffence_values != diffence_values):
        #    print "------------------ Catch NaN ------------------------"
        #    print diffence_values
        #    sys.exit(1)
        return diffence_values
    def softmax(param):
        return numpy.exp(param)/numpy.exp(param).sum()

    x = T.dvector('x')
    #u = T.dscalar('u')
    #sigma = T.dscalar('sigma')
    #サンプルを集計したもの
    sample = numpy.array([ 367,86,20,85,142,530,86,3,1,129,17,47,41,27,17,
                1999,834,574,215,302,84,96,15,2,141,32,7,12])
    n = theano.shared(sample, name='n')
    #階層ベイズモデル
    #normalPdfSyntax = - (x.shape[0] / 2.) * T.log( 2. * math.pi * sigma ** 2) - (1./(2*sigma ** 2)) * T.sum((x-u) ** 2)
    posteriorSyntax = T.sum(n * T.log(T.nnet.softmax(x))) 
    posterior = theano.function(inputs=[x], outputs=posteriorSyntax)

    gPosteriorSyntax = T.grad(cost=posteriorSyntax, wrt=x)
    gPosterior = theano.function(inputs=[x], outputs=gPosteriorSyntax)
    t0 = time.time()
    range_parameter = [[-10,10]] * sample.shape[0]
    context = HMC(callPosterior, callPosteriorDelta, None, numpy.random.normal(0, 1, sample.shape[0]), range_parameter)
    estimated_parameter = context.sampling(iter=3000,burn_in=1500,iter_leapfrog=40)
    t1 = time.time()
    print 'Looping times took', t1 - t0, 'seconds'
    print softmax(numpy.average(estimated_parameter,axis=0))
