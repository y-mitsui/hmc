#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <libstandard.h>
#define NUM_SAMPLES 10000
#define NUM_CORPOMENT 2
#define DIMENTION 2
#define NUM_THREAD 3
double gsl_det(gsl_matrix *m);
typedef struct{
    gsl_matrix *invSigma;
    gsl_vector *u;
    double sigmaDetInv;
    double constant;
}mnorm;
typedef struct{
    double *vecTmp;
    gsl_vector **vecX;  
    double tmp[DIMENTION];
}PmcmcBuffer;

PmcmcBuffer *pctx[NUM_THREAD];

mnorm *mnorm_init(double *parameter);
double mnorm_pdf(PmcmcBuffer *pctx,mnorm *ctx,double *x);
//double mnorm_pdf2(PmcmcBuffer *pctx,mnorm *ctx,gsl_vector *x,int idx);
void mnorm_free(mnorm *ctx);

int numParameterConst;
const double PI2=2*M_PI;
double constant;

double xor128(){ 
	static unsigned int x=123456789,y=362436069,z=521288629,w=88675123; 
	unsigned int t; 
	t=(x^(x<<11));
	x=y;
	y=z;
	z=w;
	w=(w^(w>>19))^(t^(t>>8));
	return (double)w/(double)0xFFFFFFFF; 
} 
gsl_vector *rnormMulti(gsl_rng *ctx,const gsl_vector *mean,const gsl_matrix *cover,const int dimention){
	double num;
	int i;
	gsl_vector *rr=gsl_vector_alloc(dimention);
	gsl_vector *tmp2=gsl_vector_alloc(dimention);
	gsl_matrix *tmp=gsl_matrix_clone(cover);
	gsl_linalg_cholesky_decomp(tmp);
	for(i=0;i<dimention;i++){
		num=gsl_ran_gaussian(ctx,1.0);
		gsl_vector_set(rr,i,num);
	}
	gsl_blas_dgemv (CblasNoTrans, 1.0, tmp, rr,0.0,tmp2);
	gsl_blas_daxpy(1.0,mean,tmp2);
	gsl_matrix_free(tmp);
	gsl_vector_free(rr);
	return tmp2;
}
gsl_vector *rgmm(gsl_rng *ctx,double *weight,gsl_vector **mean,gsl_matrix **cover,const int dimention){
	int i;
	double tmp=xor128();
	for(i=0;i<NUM_CORPOMENT;i++){
		tmp-=weight[i];
		if(tmp<=0.0) break;
	}
	gsl_vector *rand=rnormMulti(ctx,mean[i],cover[i],dimention);
	return rand;
}
double gsl_rng_uniform_pos2(){
	double r;
	do{
		r=xor128();
	}while(r==0.0);
	return r;
}
void rnorm(double *result,int n,double mean,double sd){
	double x, y, r2;
	int i;
	for(i=0;i<n;i++){
		do{
			/* choose x,y in uniform square (-1,-1) to (+1,+1) */
			x = -1 + 2 * gsl_rng_uniform_pos2 ();
			y = -1 + 2 * gsl_rng_uniform_pos2 ();

			/* see if it is in the unit circle */
			r2 = x * x + y * y;
		}while (r2 > 1.0 || r2 == 0);

		/* Box-Muller transform */
		result[i]=sd * y * sqrt (-2.0 * log (r2) / r2)+mean;
	}
}

typedef struct{
    double *theta;
    mnorm **model;
    int numSample;
    gsl_vector **sample;
    PmcmcBuffer *pctx;
    int idx;
}gmmArg;

double result[3];
void *gmmThread(void *arg){
    gmmArg *ctx=(gmmArg*)arg;
    double sum;
    int i;
    register double r;
    register int k;
    r=0.0;
    
    register double *theta=ctx->theta;
    mnorm **model=ctx->model;
    gsl_vector **sample=ctx->sample;
    for(k=0;k<ctx->numSample;k++){
	    sum=0.0;
	    for(i=0;i<NUM_CORPOMENT;i++){
		    //sum+=tmp*normLogOrig(sample[k],&parameter[i*numParameterConst+(NUM_CORPOMENT-1)]);
		    sum+=theta[i]*mnorm_pdf(ctx->pctx,model[i],gsl_vector_ptr(sample[k],0));
	    }
	    
	    r+=log(sum);
	}
	result[ctx->idx]=r;
	/*double *ptr=malloc(sizeof(double));
	*ptr=r;
	return ptr;*/
	return NULL;
}
double gmm(gsl_vector **sample,double *parameter){
	double r=0.0;
	double paramSum;
	//double *dsum[NUM_THREAD];
	int i;
	mnorm *model[NUM_CORPOMENT];
	double theta[NUM_CORPOMENT];
	gmmArg arg[NUM_THREAD];
	paramSum=0.0;
	if(parameter[0] > 1.0) parameter[0]=1.0;
	else if(parameter[0] < 0.0) parameter[0]=0.1;
	
	for(i=0;i<11;i++){
	    if(parameter[i] < 0.0) parameter[i]=0.01;
        //printf("parameter[%d]:%lf\n",i,parameter[i]);
    }
	for(i=0;i<NUM_CORPOMENT;i++){
	    model[i]=mnorm_init(&parameter[i*numParameterConst+(NUM_CORPOMENT-1)]);
	    theta[i]=(NUM_CORPOMENT-1==i) ? 1.0-paramSum : parameter[i];
	    paramSum+=parameter[i];
	}
	arg[0].sample=sample;
	arg[1].sample=&sample[NUM_SAMPLES/NUM_THREAD];
	arg[2].sample=&sample[NUM_SAMPLES/NUM_THREAD*2];
	arg[0].pctx=pctx[0];
	arg[1].pctx=pctx[1];
	arg[2].pctx=pctx[2];
	arg[0].idx=0;
	arg[1].idx=1;
	arg[2].idx=2;
	arg[0].numSample=arg[1].numSample=arg[2].numSample=NUM_SAMPLES/NUM_THREAD;
	arg[0].model=arg[1].model=arg[2].model=model;
	arg[0].theta=arg[1].theta=arg[2].theta=theta;
	pthread_t   tid[NUM_THREAD];
	pthread_create(&tid[0],NULL,gmmThread,(void*)&arg[0]);
	pthread_create(&tid[1],NULL,gmmThread,(void*)&arg[1]);
	pthread_create(&tid[2],NULL,gmmThread,(void*)&arg[2]);
	pthread_join(tid[0],NULL);
	pthread_join(tid[1],NULL);
	pthread_join(tid[2],NULL);
	r=result[0]+result[1]+result[2];
	
	
	for(i=0;i<NUM_CORPOMENT;i++){
	    mnorm_free(model[i]);
	}
	return r;
}
inline double min(double x,double y){
    return (x < y) ? x : y;
}
inline double max(double x,double y){
    return (x > y) ? x : y;
}
double differentiate(double (*log_fun)(void *,double *),void *arg,double *param,int targetIndex){
    const double tinyNumber=1.0e-8;
    double tmp=param[targetIndex];
    //printf("tmp:%lf ",tmp);
    param[targetIndex]=tmp-tinyNumber;
    if( targetIndex==0) param[0] = max(min(param[0],1.0),0.0);
    //printf("param[%d]:%lf ",targetIndex,param[targetIndex]);
    double likely1=log_fun(arg,param);
    //printf("likely1:%lf ",likely1);
    param[targetIndex]=tmp+tinyNumber;
    if( targetIndex==0) param[0] = max(min(param[0],1.0),0.0);
    double likely2=log_fun(arg,param);
    //printf("likely2:%lf ",likely2);
    double r=(likely2-likely1)/(2.0*tinyNumber);
    //printf("r:%lf\n",r);
    param[targetIndex]=tmp;
    return r;
}
double squereSum(double *target,int num){
    int i;
    double r=0.0;
    
    for(i=0;i<num;i++){
        r += target[i]*target[i];
    }
    return r;
}

double norm(double *x,double *param){
    double mu=param[0];
    double r=0.0;
    int i;
    for(i=0;i<NUM_SAMPLES;i++){
        r+=log(1.0/sqrt(2*3.141592653589*2.0))+(-((x[i]-mu)*(x[i]-mu))/(2.0*2.0));
    }
    return -r;
}
double *energy_function_delta(void *arg,double *parameter,int numParameter){
    int i;
    const double eps=1.0e-8;
    double *result=malloc(sizeof(double)*numParameter);
    
    for(i=0;i<numParameter;i++){
        double tmp=parameter[i];
        parameter[i] -= eps;
        double row = gmm(arg,parameter);
        parameter[i] += 2.0*eps;
        double high = gmm(arg,parameter);
        result[i]=(high-row)/(2.0*eps);
        parameter[i] = tmp;
    }
    return result;
}
void pmcmc_hamiltonian(double (*energy_function)(void *,double *),void *arg,double *parameter,int numParameter,int mcmc){
    int iter,i,j;
    double *current_parameter = parameter;
    double *momentum=malloc(sizeof(double)*numParameter);
    double *current_parameter_candidate=malloc(sizeof(double)*numParameter);
    int iter_leapfrog=100;
    double leapfrog_accuracy=0.0002;
    
    for (iter = 0; iter < mcmc; ++iter) {
        rnorm(momentum,numParameter,0.0,1.0);
        double hamilton = squereSum(momentum,numParameter) / 2.0 + energy_function(arg,current_parameter);
        memcpy(current_parameter_candidate,current_parameter,sizeof(double)*numParameter);
        for(i=0;i<iter_leapfrog;i++){
            double *diff=energy_function_delta(arg,current_parameter_candidate,numParameter);
            for(j=0;j<numParameter;j++){
                momentum[j] -= leapfrog_accuracy * diff[j] / 2.0;
            }
            free(diff);
            
            for(j=0;j<numParameter;j++){
                current_parameter_candidate[j] += leapfrog_accuracy * momentum[j];
            }
            
            diff=energy_function_delta(arg,current_parameter_candidate,numParameter);

            for(j=0;j<numParameter;j++){
                momentum[j] -= leapfrog_accuracy * diff[j] / 2.0;
            }
            free(diff);
            
        }
        double h2 = squereSum(momentum,numParameter) / 2.0 + energy_function(arg,current_parameter_candidate);
        double differenceHamilton = hamilton - h2;
        
        if (xor128() < exp(differenceHamilton)){
            memcpy(current_parameter,current_parameter_candidate,sizeof(double)*numParameter);
        }
        
        if(iter%1==0){
		    printf("iter:%d \n",iter);
		    for (i = 0; i < numParameter; ++i) {
    		    printf("current_parameter[%d]:%lf\n",i,current_parameter[i]);
    		}
		}
		
    }
}
void pmcmc(double (*log_fun)(void *,double *),void *arg,double *theta,int numTheta,int mcmc){
    int iter,i;
	double *theta_can=malloc(sizeof(double)*numTheta),rnd;
	double userfun_cur;
	double t=0.01;
	for (iter = 0; iter < mcmc; ++iter) {
		for (i = 0; i < numTheta; ++i) {
			rnorm(&rnd,1,0.0,1.0);
			theta_can[i] = theta[i]+rnd*t; //修正予定
			//theta_can[0] = (theta_can[0] > 1.0) ? 1.0 : theta_can[0];
			if(mcmc-iter < 2)
				printf("theta_can[%d]:%lf\n",i,theta_can[i]);
    		
    	}
		double userfun_can = log_fun(arg,theta_can);
		if(iter==0) userfun_cur=userfun_can-1e-10;
		const double ratio = exp(userfun_can - userfun_cur);
		if (xor128() < ratio) {
			for (i = 0; i < numTheta; ++i) {
				theta[i] = theta_can[i];
			}
			userfun_cur = userfun_can;
		}
		if(iter%100==0) printf("iter:%d log:%f\n",iter,userfun_can);
    }
}
PmcmcBuffer* pmcmc_init(/*gsl_vector **sample*/){
   // int i;
    PmcmcBuffer *r=malloc(sizeof(PmcmcBuffer));
    r->vecTmp=malloc(sizeof(double)*DIMENTION);
    /*for(i=0;i<NUM_SAMPLES;i++){
        r->vecX[i]=gsl_vector_clone(sample[i]);
    }*/
    r->tmp[0]=r->tmp[1]=1.0;
    return r;
}
int main(void){
    int i;
    /*double *rnd=malloc(sizeof(double)*NUM_SAMPLES);
    
    rnorm(rnd,NUM_SAMPLES,1000.0,2.0);
    double theta2[1]={20.0};
    pmcmc_hamiltonian((double (*)(void *,double *))norm,rnd,theta2,1,10000);
    exit(1);*/
    
    constant=1.0/pow(sqrt(2*M_PI),(double)DIMENTION);
    gsl_vector **sample=malloc(sizeof(gsl_vector *)*NUM_SAMPLES);
	gsl_rng *ctx = gsl_rng_alloc (gsl_rng_default);
	int dimention=2;
	clock_t start;
    
	gsl_vector **mean=malloc(sizeof(gsl_vector*)*NUM_CORPOMENT);
	gsl_matrix **covar=malloc(sizeof(gsl_matrix*)*NUM_CORPOMENT);

	mean[0]=gsl_vector_alloc(DIMENTION);
	gsl_vector_set(mean[0],0,1.0);
	gsl_vector_set(mean[0],1,6.0);
	covar[0]=gsl_matrix_alloc(DIMENTION,DIMENTION);
	gsl_matrix_set(covar[0],0,0,3.0);
	gsl_matrix_set(covar[0],0,1,0.1);
	gsl_matrix_set(covar[0],1,0,0.1);
	gsl_matrix_set(covar[0],1,1,5.0);
	mean[1]=gsl_vector_alloc(DIMENTION);
	gsl_vector_set(mean[1],0,10.0);
	gsl_vector_set(mean[1],1,20.0);
	covar[1]=gsl_matrix_alloc(DIMENTION,DIMENTION);
	gsl_matrix_set(covar[1],0,0,7.0);
	gsl_matrix_set(covar[1],0,1,0.8);
	gsl_matrix_set(covar[1],1,0,0.8);
	gsl_matrix_set(covar[1],1,1,3.0);
	
	double theta[]={
	    0.9, //weights
	    5.0,5.0,3.0,5.0,0.1, //param1
		5.0,5.0,7.0,3.0,0.8 //param2
	};
	double weight[]={0.5,0.5};
	
    for(i=0;i<NUM_SAMPLES;i++){
        sample[i]=rgmm(ctx,weight,mean,covar,dimention);
		/*for(j=0;j<sample[i]->size;j++){
			printf("%lf ",gsl_vector_get(sample[i],j));
		}
		puts("");*/
		//like1+=gmm2(&likeArg,sample[i],theta);
	}
	numParameterConst=5;
	start=clock();
	pctx[0]=pmcmc_init(/*sample*/);
	pctx[1]=pmcmc_init(/*sample*/);
	pctx[2]=pmcmc_init(/*sample*/);
	pmcmc((double (*)(void *,double *))gmm,sample,theta,sizeof(theta)/sizeof(theta[0]),10000);
	printf("%lf\n",(double)(clock()-start)/(double)CLOCKS_PER_SEC/(double)NUM_THREAD);
	return 1;
}
