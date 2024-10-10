#include <array>
#include <string.h>
#include <immintrin.h>
#include <xmmintrin.h>
//#pragma GCC optimize("O2")

const char* sgemm_desc = "Simple blocked sgemm.";

alignas(64) float mem[2][2000000];

struct Mat
{
    float*p;
    int N,M;
    int n,m;

    inline float& operator[](int x,int y)
    {
        return p[x*M+y];
    }
};

Mat convert_mem_layout_A(const float*s,float*t,int n,int m)
{
    int N=n;
    int M=(((m+15)/16)|1)*16;

    memset(t,0,N*M*4);

    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            t[i*M+j]=s[i+n*j];

    return {t,N,M,n,m};
}

Mat convert_mem_layout_B(const float*s,float*t,int n,int m)
{
    int N=m;
    int M=(((n+15)/16)|1)*16;

    memset(t,0,N*M*4);

    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            t[i+j*M]=s[i+n*j];

    return {t,N,M,m,n};
}

struct ymm
{
    __m256 x;
    operator __m256() const
    {
        return x;
    }

    ymm(const float*p)
    {
        x=_mm256_load_ps(p);
    }

    ymm()
    {
        x=_mm256_setzero_ps();
    }

    void fma(ymm a,ymm b)
    {
        x=_mm256_fmadd_ps(a,b,x);
    }

    void store(float*p) const
    {
        _mm256_store_ps(p,x);
    }

    float sum() const
    {
        alignas(32) float a[8];
        store(a);
        float s=0;
        s=((a[0]+a[1])+(a[2]+a[3]))+((a[4]+a[5])+(a[6]+a[7]));
        return s;
    }
};
void custom_sgemm(int M, int K, int N, float* A, float* B, float* C) {

    Mat a=convert_mem_layout_A(A,mem[0],M,K);
    Mat b=convert_mem_layout_B(B,mem[1],K,N);
    Mat c={C,N,M,N,M};

    // for(int i=0;i<a.n;i++)
    //     for(int j=0;j<b.n;j++)
    //     {
    //         float tmp=0;
    //         for(int k=0;k<a.m;k++)
    //             tmp+=a[i,k]*b[j,k];
    //         c[j,i]=tmp;
    //     }

    int i;
    for(i=0;i+1<a.n;i+=2)
    {
        int j;
        for(j=0;j+3<b.n;j+=4)
        {
            ymm c00,c01,c02,c03;
            ymm c10,c11,c12,c13;

            for(int k=0;k<a.M;k+=8)
            {
                ymm a0(&a[i+0,k]);
                ymm a1(&a[i+1,k]);
                ymm b0(&b[j+0,k]);
                ymm b1(&b[j+1,k]);
                ymm b2(&b[j+2,k]);
                ymm b3(&b[j+3,k]);

                c00.fma(a0,b0);
                c01.fma(a0,b1);
                c02.fma(a0,b2);
                c03.fma(a0,b3);
                c10.fma(a1,b0);
                c11.fma(a1,b1);
                c12.fma(a1,b2);
                c13.fma(a1,b3);
            }

            c[j+0,i+0]+=c00.sum();
            c[j+1,i+0]+=c01.sum();
            c[j+2,i+0]+=c02.sum();
            c[j+3,i+0]+=c03.sum();
            c[j+0,i+1]+=c10.sum();
            c[j+1,i+1]+=c11.sum();
            c[j+2,i+1]+=c12.sum();
            c[j+3,i+1]+=c13.sum();
        }

        for(;j<b.n;j++)
        {
            ymm c00,c10;

            for(int k=0;k<a.M;k+=8)
            {
                ymm a0(&a[i+0,k]);
                ymm a1(&a[i+1,k]);
                ymm b0(&b[j+0,k]);

                c00.fma(a0,b0);
                c10.fma(a1,b0);
            }

            c[j+0,i+0]+=c00.sum();
            c[j+0,i+1]+=c10.sum();
        }
    }

    for(;i<a.n;i++)
    {
        int j;
        for(j=0;j<b.n;j++)
        {
            ymm c00;

            for(int k=0;k<a.M;k+=8)
            {
                ymm a0(&a[i+0,k]);
                ymm b0(&b[j+0,k]);

                c00.fma(a0,b0);
            }
            c[j+0,i+0]+=c00.sum();
        }
    }
}
