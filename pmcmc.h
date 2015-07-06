

typedef struct{
    gsl_matrix *invSigma;
    double *u;
    double sigmaDetInv;
    double constant;
    double *invSigmaPtr;
    double left;
}mnorm;
typedef struct{
    double *vecTmp;
    gsl_vector **vecX;  
    double *tmp;
}PmcmcBuffer;
typedef struct {
    int numSample;
    gsl_vector **sample;
    int num_corpoment;
}GmmParam;


double gsl_det(gsl_matrix *m);
gsl_matrix *gsl_matrix_clone(const gsl_matrix *src);
gsl_vector *gsl_vector_clone(const gsl_vector *src);
mnorm *mnorm_init(double *parameter,int dimention);
double mnorm_pdf(PmcmcBuffer *pctx,mnorm *ctx,double *x);
//double mnorm_pdf2(PmcmcBuffer *pctx,mnorm *ctx,gsl_vector *x,int idx);
void mnorm_free(mnorm *ctx);