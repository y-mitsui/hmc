#include "libstandard.h"


gsl_matrix *gsl_inverse(gsl_matrix *m){
	int s=0;
	gsl_permutation * p = gsl_permutation_alloc (m->size1);
	gsl_matrix *inv=gsl_matrix_clone(m);
	gsl_matrix *r=gsl_matrix_alloc(inv->size1,inv->size2);
	gsl_linalg_LU_decomp(inv,p,&s);
	gsl_linalg_LU_invert(inv,p,r);
	gsl_matrix_free(inv);
	gsl_permutation_free(p);
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
}/*
void covariance_matrix(double *sample ,int numSample,gsl_vector **means,gsl_matrix **cover){
	gsl_vector *means=gsl_vector_alloc(ctx->dims);
	gsl_matrix *cover=gsl_matrix_alloc(ctx->dims,ctx->dims);
	for(i=0;i<ctx->dims;i++){
		gsl_vector_set(means,i,gsl_stats_mean(&ctx->data[i],ctx->dims,ctx->numData));
	}
	for(i=0;i<ctx->dims;i++){
		gsl_matrix_set(cover,i,i,gsl_stats_variance_m(&ctx->data[i],ctx->dims,ctx->numData,gsl_vector_get(means,i)));
		for(j=i+1;j<ctx->dims;j++){
			gsl_matrix_set(cover,i,j,gsl_stats_covariance_m(&ctx->data[i],ctx->dims,&ctx->data[j],ctx->dims,ctx->numData,gsl_vector_get(means,i),gsl_vector_get(means,j)));
			gsl_matrix_set(cover,j,i,gsl_matrix_get(cover,i,j));
		}
	}
}*/



