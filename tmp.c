double normLogOrig(gsl_vector *x,double *parameter){
	int i,j;
	double r=0.0;
	
    	gsl_matrix *sigma=gsl_matrix_alloc(DIMENTION,DIMENTION);
    	
    	gsl_vector *u=gsl_vector_alloc(DIMENTION);
	    for(i=0;i<DIMENTION;i++){
	    	gsl_vector_set(u,i,parameter[i]);
	    }
	    for(i=0;i<DIMENTION;i++){
	    	gsl_matrix_set(sigma,i,i,fabs(parameter[i+DIMENTION]));
	    }
	    double *pp=&parameter[DIMENTION+DIMENTION];
	    for(i=0;i<DIMENTION;i++){
	    	for(j=i+1;j<DIMENTION;j++){
	    		double aa=*pp; ////修正予定
	    		gsl_matrix_set(sigma,i,j,aa);
	    		gsl_matrix_set(sigma,j,i,aa);
	    		pp++;
	    	}
    	}
    	/* 1/det(sigma) */
    	double sigmaDetInv=1.0/sqrt(gsl_det(sigma));
    	double constant=1.0/pow(sqrt(2*M_PI),(double)DIMENTION);
    	/*tmp=inv(sigma)*/
    	gsl_matrix *tmp=gsl_matrix_clone(sigma);
    	/*puts("tmp:");
    	gsl_matrix_print(tmp);*/
    	//exit(1);
    	
	    /*gsl_linalg_cholesky_decomp(tmp);
	    gsl_linalg_cholesky_invert(tmp);*/
	    
	    gsl_permutation * p = gsl_permutation_alloc (tmp->size1);
	    int s=0;
	    gsl_linalg_LU_decomp(tmp,p,&s);
	    gsl_matrix *invTmp=gsl_matrix_alloc(tmp->size1,tmp->size2);
	    gsl_linalg_LU_invert(tmp,p,invTmp);
	    gsl_matrix_free(tmp);
	    tmp=invTmp;
	    gsl_permutation_free(p);
	    
	    gsl_vector *vecX=gsl_vector_clone(x);
	    gsl_vector *vecTmp=gsl_vector_alloc(DIMENTION);
	
	    gsl_vector_sub(vecX,u);
	    gsl_blas_dgemv (CblasTrans, 1.0, tmp, vecX,0.0,vecTmp);
    	double num;
	    gsl_blas_ddot (vecTmp, vecX,&num);
	    r=constant*sigmaDetInv*exp(-0.5*num);
    	gsl_matrix_free(tmp);
	    gsl_matrix_free(sigma);
	    gsl_vector_free(u);
	    gsl_vector_free(vecX);
	    gsl_vector_free(vecTmp);
	
	return r;
}
