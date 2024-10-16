#define SGEMM sgemm_
extern "C" void SGEMM(char*, char*, int*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*);

const char* sgemm_desc = "Reference sgemm.";

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are MxK, KxN, MxN matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 * This function wraps a call to the BLAS-3 routine sgemm, via the standard FORTRAN interface - hence the reference semantics. */
#include<cblas.h>
 
void custom_sgemm(int M, int K, int N, float* A, float* B, float* C) {
    char TRANSA = 'N';
    char TRANSB = 'N';
    float ALPHA = 1.;
    float BETA = 1.;
    int LDA = M;
    int LDB = K;
    int LDC = M;
    static int b=(openblas_set_num_threads(1),1);
    //static int a=(goto_set_num_threads(1),1);
    //openblas_set_num_threads(1);
    //printf("%d %d\n",openblas_get_num_threads(),openblas_get_num_procs());
    //printf("%s\n",openblas_get_config());
    //SGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
    cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);
}

