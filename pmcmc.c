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
#define NUM_SAMPLES 8000
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
		result[i]=sd * y * sqrt (-2.0 * log (r2) / r2);
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

void pmcmc(double (*log_fun)(void *,double *),void *arg,double *theta,int numTheta,int mcmc){
    int iter,i,j;
	double *theta_can=malloc(sizeof(double)*numTheta),rnd;
	double *arr=malloc(sizeof(double)*numTheta);
	double userfun_cur,pN,dt=0.05;
	double t=0.5,m=1.0,rho=0.05;
	for (iter = 0; iter < mcmc; ++iter) {
	    memcpy(theta_can,theta,sizeof(double)*numTheta);
		for (i = 0; i < numTheta; ++i) {
			rnorm(&rnd,1,0.0,1.0);
			theta_can[i] += rnd;
			
			rnorm(&rnd,1,0.0,1.0);
			pN=rnd;
			
			memcpy(arr,theta_can,sizeof(double)*numTheta);

		    //printf("0: ");
		    //printf("pN:%lf ",pN);
		    pN=pN+differentiate(log_fun,arg,arr,i)*0.5*rho;
		    //printf("pN:%lf\n",pN);
			for(j=0;j<10;j++){
			    arr[i]+=max(min(rho*pN,20),-20);
			    if( i==0) theta_can[0] = max(min(theta_can[0],1.0),0.0);
			    double fact=(i<5-1) ? rho : rho*0.5;
			    //printf("%d: ",j+1);
			    double delta=differentiate(log_fun,arg,arr,i);
			    pN+=fact*delta;
			}
			theta_can[i]=arr[i];
			//printf("theta_can[%d]:%lf\n",i,theta_can[i]);
			if( i==0) theta_can[0] = max(min(theta_can[0],1.0),0.0);
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
		if(iter%100==0){
		    printf("iter:%d log:%f\n",iter,userfun_can);
		    for (i = 0; i < numTheta; ++i) {
    		    printf("theta_can[%d]:%lf\n",i,theta_can[i]);
    		}
		}
    }
}
void pmcmc2(double (*log_fun)(void *,double *),void *arg,double *theta,int numTheta,int mcmc){
    int iter,i;
	double *theta_can=malloc(sizeof(double)*numTheta),rnd;
	double userfun_cur;
	double t=0.01;
	for (iter = 0; iter < mcmc; ++iter) {
		for (i = 0; i < numTheta; ++i) {
			rnorm(&rnd,1,0.0,1.0);
			theta_can[i] = theta[i]+rnd*t; //修正予定
			theta_can[0] = (theta_can[0] > 1.0) ? 1.0 : theta_can[0];
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
	    1.0, //weights
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
