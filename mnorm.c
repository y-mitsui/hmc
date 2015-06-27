#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <cblas.h>
#include <clapack.h>
#include <xmmintrin.h>
#include <immintrin.h>



gsl_vector *gsl_vector_clone(const gsl_vector *src){
	gsl_vector *r=gsl_vector_alloc(src->size);
	gsl_vector_memcpy(r,src);
	return r;
}
gsl_matrix *gsl_matrix_clone(const gsl_matrix *src){
	gsl_matrix *r=gsl_matrix_alloc(src->size1,src->size2);
	gsl_matrix_memcpy(r,src);
	return r;
}
double gsl_det(gsl_matrix *m){
	gsl_permutation * p = gsl_permutation_alloc (m->size1);
	gsl_matrix *lu=gsl_matrix_clone(m);
	int s=0;
	gsl_linalg_LU_decomp (lu,p,&s);           // LU分解
	double n = gsl_linalg_LU_det (lu,s);    // 行列式
	gsl_matrix_free(lu);
	gsl_permutation_free(p);
	
	return n;
}
#define DIMENTION 2
typedef struct{
    gsl_matrix *invSigma;
    double *invSigmaPtr;
    double u[DIMENTION];
    double sigmaDetInv;
    double constant;
    double left;
}mnorm;
typedef struct{
    double *vecTmp;
    gsl_vector **vecX;  
    double tmp[DIMENTION];
}PmcmcBuffer;

mnorm *mnorm_init(double *parameter){
    int i,j;
    mnorm *r=malloc(sizeof(mnorm));
    gsl_matrix *sigma=gsl_matrix_alloc(DIMENTION,DIMENTION);
    	
    //gsl_vector *u=gsl_vector_alloc(DIMENTION);
    
	for(i=0;i<DIMENTION;i++){
	    r->u[i]=parameter[i];
        //gsl_vector_set(u,i,parameter[i]);
    }
    for(i=0;i<DIMENTION;i++){
        //gsl_matrix_set(sigma,i,i,fabs(parameter[i+DIMENTION]));
        gsl_matrix_set(sigma,i,i,parameter[i+DIMENTION]);
    }
    double *pp=&parameter[DIMENTION+DIMENTION];
    for(i=0;i<DIMENTION;i++){
        for(j=i+1;j<DIMENTION;j++){
            double aa=*pp; ////修正予定
            gsl_matrix_set(sigma,i,j,aa);
            gsl_matrix_set(sigma,j,i,aa);
            //printf("aa:%lf\n",aa);
            pp++;
        }
    }
    /* 1/det(sigma) */
    /*printf("gsl_det(sigma):%lf\n",gsl_det(sigma));
    printf("sqrt(gsl_det(sigma)):%lf\n",sqrt(fabs(gsl_det(sigma))));*/
    r->sigmaDetInv=1.0/sqrt(gsl_det(sigma));
    //printf("r->sigmaDetInv:%lf\n",r->sigmaDetInv);
    //printf("r->sigmaDetInv:%lf\n",sqrt(gsl_det(sigma)));
    r->constant=1.0/pow(sqrt(2*M_PI),(double)DIMENTION);
    /*tmp=inv(sigma)*/
    gsl_matrix *tmp=gsl_matrix_clone(sigma);
    gsl_permutation * p = gsl_permutation_alloc (tmp->size1);
    int s=0;
    gsl_linalg_LU_decomp(tmp,p,&s);
    gsl_matrix *invTmp=gsl_matrix_alloc(tmp->size1,tmp->size2);
    gsl_linalg_LU_invert(tmp,p,invTmp);
    gsl_matrix_free(tmp);
    r->invSigma=invTmp;
    gsl_permutation_free(p);
    
    
    
	    gsl_matrix_free(sigma);
	    //gsl_vector_free(u);
	    
	 r->left=r->constant*r->sigmaDetInv ;
	 
	r->invSigmaPtr=gsl_matrix_ptr(r->invSigma,0,0);
    return r;
}
inline void mul(double *vec,double *mat,double *r){
    r[0]=vec[0]*mat[0]+vec[1]*mat[2];
    r[1]=vec[0]*mat[1]+vec[1]*mat[3];
}
double mnorm_pdf(PmcmcBuffer *pctx,mnorm *ctx,double *x){
    pctx->tmp[0]=x[0]-ctx->u[0];
    pctx->tmp[1]=x[1]-ctx->u[1];

    /*pctx->vecTmp[0]=1.0;
    pctx->vecTmp[1]=1.0;*/
    //cblas_dgemv(CblasRowMajor,CblasTrans,ctx->invSigma->size1,ctx->invSigma->size2,1.0,ctx->invSigmaPtr,ctx->invSigma->size1,pctx->tmp,1,0.0,pctx->vecTmp,1);
    mul(pctx->tmp,ctx->invSigmaPtr,pctx->vecTmp);
    
    /*__m128d u2 = {0};
    
        __m128d w = _mm_load_pd(pctx->vecTmp);
        __m128d x2 = _mm_load_pd(pctx->tmp);
 
        x2 = _mm_add_pd(w, x2);
        u2 = _mm_add_pd(u2, x2);

    __attribute__((aligned(32))) double t[2] = {0};
    _mm_store_pd(t, u2);
    double num=t[0] + t[1];*/
    double num=pctx->vecTmp[0]*pctx->tmp[0]+pctx->vecTmp[1]*pctx->tmp[1];
    //double num=cblas_ddot(DIMENTION,pctx->vecTmp,1,pctx->tmp,1);
    //printf("%lf %lf\n",num,num2);
    return ctx->left*exp(-0.5*num);
}
/*double mnorm_pdf2(PmcmcBuffer *pctx,mnorm *ctx,gsl_vector *x,int idx){
    gsl_vector *vecX=gsl_vector_clone(x);
    //gsl_vector *vecTmp=gsl_vector_alloc(DIMENTION);
	//gsl_vector *vecX=pctx->vecX[idx];
	gsl_vector_sub(vecX,ctx->u);
	gsl_blas_dgemv (CblasTrans, 1.0, ctx->invSigma, vecX,0.0,pctx->vecTmp);
    double num;
	gsl_blas_ddot (pctx->vecTmp, vecX,&num);
	gsl_vector_free(vecX);
	//gsl_vector_free(vecTmp);
    double r=ctx->left*exp(-0.5*num);
    return r;
}*/
void mnorm_free(mnorm *ctx){
    gsl_matrix_free(ctx->invSigma);
    //gsl_vector_free(ctx->u);
    free(ctx);
}
