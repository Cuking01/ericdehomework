#include "matrix_config.h"
#include "matrix_helper.h"
#include <array>
#include <immintrin.h>
#include <xmmintrin.h>

const char* sgemm_desc = "Simple blocked sgemm.";

void custom_sgemm(int M, int K, int N, float* A, float* B, float* C) {
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++)
            for(int k=0;k<K;k++)
                C[i+j*M]+=A[i+k*M]*B[k+j*K];
}
