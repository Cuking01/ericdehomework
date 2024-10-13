#include <array>
#include <string.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <stdio.h>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <vector>

//#pragma GCC optimize("O2")

const char* sgemm_desc = "Simple blocked sgemm.";

alignas(64) float mem[2][2000016];

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

struct ymm
{
    __m256 x;
    operator __m256() const
    {
        return x;
    }
    operator __m256d() const
    {
        return *(__m256d*)&(this->x);
    }

    ymm(const float*p)
    {
        x=_mm256_load_ps(p);
    }

    ymm(__m256 x):x(x){}
    ymm(__m256d x):x(*(__m256*)&x){}

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

    float sum1() const
    {
        alignas(32) float a[8];
        store(a);
        float s=0;
        s=((a[0]+a[1])+(a[2]+a[3]))+((a[4]+a[5])+(a[6]+a[7]));
        return s;
    }

    float sum2() const
    {
        ymm t=_mm256_permute2f128_ps(x,x,0x11);
        t=_mm256_add_ps(t,x);
        ymm t2=_mm256_movehdup_ps(t);
        t=_mm256_add_ps(t,t2);
        t2=_mm256_permute_ps(t,0xa);
        t=_mm256_add_ps(t,t2);
        return _mm256_cvtss_f32(t);
    }

    float sum() const
    {
        return sum2();
    }

    void print() const
    {
        alignas(32) float a[8];
        store(a);
        for(int i=0;i<8;i++)
            printf("%f ",a[i]);
        puts("");
    }
};

static constexpr int Align_Wide=16;

Mat convert_mem_layout_A(const float*s,float*t,int n,int m)
{
    printf("<%d %d>\n",n,m);
    int N=(n+7)/8*8;
    int M=(((m+Align_Wide-1)/Align_Wide)|1)*Align_Wide;

    for(int j=0;j<m;j++)
    {
        memcpy(t+j*N,s+j*n,n*4);
        memset(t+j*N+n,0,N-n);
    }
    memset(t+m*N,0,(M-m)*N*4);

    for(int i=0;i<N*M;i++)
        printf("%3.0f ",t[i]);
    puts("****");

    printf("<%d %d\n>",n,m);
    printf("<<N%d M%d>>\n",N,M);
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
            printf("%3.0f ",a[i,j]);
        puts("");
    }

    for(int i=0;i<M;i+=8)
    {
        int J;
        if(M==8)J=N-8;
        else J=i*(N/8-1)/(M/8-1);

        for(int j=0;j<=J;j+=8)
        {
            float*p=t+i*N+j;

            ymm a0(p+0*N);
            ymm a1(p+1*N);
            ymm a2(p+2*N);
            ymm a3(p+3*N);
            ymm a4(p+4*N);
            ymm a5(p+5*N);
            ymm a6(p+6*N);
            ymm a7(p+7*N);

            ymm t0=_mm256_permute2f128_ps(a0,a4,0x20);
            ymm t1=_mm256_permute2f128_ps(a1,a5,0x20);
            ymm t2=_mm256_permute2f128_ps(a2,a6,0x20);
            ymm t3=_mm256_permute2f128_ps(a3,a7,0x20);
            ymm t4=_mm256_permute2f128_ps(a0,a4,0x31);
            ymm t5=_mm256_permute2f128_ps(a1,a5,0x31);
            ymm t6=_mm256_permute2f128_ps(a2,a6,0x31);
            ymm t7=_mm256_permute2f128_ps(a3,a7,0x31);
            
            a0=_mm256_unpacklo_pd(t0,t2);
            a1=_mm256_unpacklo_pd(t1,t3);
            a2=_mm256_unpackhi_pd(t0,t2);
            a3=_mm256_unpackhi_pd(t1,t3);
            a4=_mm256_unpacklo_pd(t4,t6);
            a5=_mm256_unpacklo_pd(t5,t7);
            a6=_mm256_unpackhi_pd(t4,t6);
            a7=_mm256_unpackhi_pd(t5,t7);

            a0=_mm256_shuffle_ps(a0,a0,0xd8);
            a1=_mm256_shuffle_ps(a1,a1,0xd8);
            a2=_mm256_shuffle_ps(a2,a2,0xd8);
            a3=_mm256_shuffle_ps(a3,a3,0xd8);
            a4=_mm256_shuffle_ps(a4,a4,0xd8);
            a5=_mm256_shuffle_ps(a5,a5,0xd8);
            a6=_mm256_shuffle_ps(a6,a6,0xd8);
            a7=_mm256_shuffle_ps(a7,a7,0xd8);

            t0=_mm256_unpacklo_ps(a0,a1);
            t1=_mm256_unpackhi_ps(a0,a1);
            t2=_mm256_unpacklo_ps(a2,a3);
            t3=_mm256_unpackhi_ps(a2,a3);
            t4=_mm256_unpacklo_ps(a4,a5);
            t5=_mm256_unpackhi_ps(a4,a5);
            t6=_mm256_unpacklo_ps(a6,a7);
            t7=_mm256_unpackhi_ps(a6,a7);

            a.M=M;

            a0=ymm(&a[j+0,i]);
            a1=ymm(&a[j+1,i]);
            a2=ymm(&a[j+2,i]);
            a3=ymm(&a[j+3,i]);
            a4=ymm(&a[j+4,i]);
            a5=ymm(&a[j+5,i]);
            a6=ymm(&a[j+6,i]);
            a7=ymm(&a[j+7,i]);

            puts("&&&&");
            a0.print();
            a1.print();
            a2.print();

            t0.store(&a[j+0,i]);
            t1.store(&a[j+1,i]);
            t2.store(&a[j+2,i]);
            t3.store(&a[j+3,i]);
            t4.store(&a[j+4,i]);
            t5.store(&a[j+5,i]);
            t6.store(&a[j+6,i]);
            t7.store(&a[j+7,i]);

            t0=_mm256_permute2f128_ps(a0,a4,0x20);
            t1=_mm256_permute2f128_ps(a1,a5,0x20);
            t2=_mm256_permute2f128_ps(a2,a6,0x20);
            t3=_mm256_permute2f128_ps(a3,a7,0x20);
            t4=_mm256_permute2f128_ps(a0,a4,0x31);
            t5=_mm256_permute2f128_ps(a1,a5,0x31);
            t6=_mm256_permute2f128_ps(a2,a6,0x31);
            t7=_mm256_permute2f128_ps(a3,a7,0x31);
            
            a0=_mm256_unpacklo_pd(t0,t2);
            a1=_mm256_unpacklo_pd(t1,t3);
            a2=_mm256_unpackhi_pd(t0,t2);
            a3=_mm256_unpackhi_pd(t1,t3);
            a4=_mm256_unpacklo_pd(t4,t6);
            a5=_mm256_unpacklo_pd(t5,t7);
            a6=_mm256_unpackhi_pd(t4,t6);
            a7=_mm256_unpackhi_pd(t5,t7);

            a0=_mm256_shuffle_ps(a0,a0,0xd8);
            a1=_mm256_shuffle_ps(a1,a1,0xd8);
            a2=_mm256_shuffle_ps(a2,a2,0xd8);
            a3=_mm256_shuffle_ps(a3,a3,0xd8);
            a4=_mm256_shuffle_ps(a4,a4,0xd8);
            a5=_mm256_shuffle_ps(a5,a5,0xd8);
            a6=_mm256_shuffle_ps(a6,a6,0xd8);
            a7=_mm256_shuffle_ps(a7,a7,0xd8);

            t0=_mm256_unpacklo_ps(a0,a1);
            t1=_mm256_unpackhi_ps(a0,a1);
            t2=_mm256_unpacklo_ps(a2,a3);
            t3=_mm256_unpackhi_ps(a2,a3);
            t4=_mm256_unpacklo_ps(a4,a5);
            t5=_mm256_unpackhi_ps(a4,a5);
            t6=_mm256_unpacklo_ps(a6,a7);
            t7=_mm256_unpackhi_ps(a6,a7);

            a.M=N;

            t0.store(&a[i+0,j]);
            t1.store(&a[i+1,j]);
            t2.store(&a[i+2,j]);
            t3.store(&a[i+3,j]);
            t4.store(&a[i+4,j]);
            t5.store(&a[i+5,j]);
            t6.store(&a[i+6,j]);
            t7.store(&a[i+7,j]);

            puts("*******");
            t0.print();
            t1.print();
        }
    }


    return {t,N,M,n,m};
}

Mat convert_mem_layout_B(const float*s,float*t,int n,int m)
{
    int N=m;
    int M=(((n+Align_Wide-1)/Align_Wide)|1)*Align_Wide;

    for(int j=0;j<m;j++)
    {
        memcpy(t+j*M,s+n*j,n*4);
        memset(t+j*M+n,0,(M-n)*4);
    }

    return {t,N,M,m,n};
}


void custom_sgemm_sc(Mat a,Mat b,Mat c)
{
    int i;

    for(i=0;i+3<a.n;i+=4)
    {
        int j;
        for(j=0;j+1<b.n;j+=2)
        {
            ymm c00,c10,c20,c30;
            ymm c01,c11,c21,c31;

            //这里必须预取不然后面读c延迟会特别高。
            _mm_prefetch(&c[j+0,i],_MM_HINT_T0);
            _mm_prefetch(&c[j+1,i],_MM_HINT_T0);

            _mm_prefetch(&b[j+0,16],_MM_HINT_T0);
            _mm_prefetch(&b[j+1,16],_MM_HINT_T0);
            
            for(int k=0;k<a.M;k+=8)
            {
                ymm b0(&b[j+0,k]);
                ymm b1(&b[j+1,k]);
                ymm a0(&a[i+0,k]);
                ymm a1(&a[i+1,k]);
                ymm a2(&a[i+2,k]);
                ymm a3(&a[i+3,k]);

                _mm_prefetch(&b[j+2,k],_MM_HINT_T0);
                _mm_prefetch(&b[j+3,k],_MM_HINT_T0);

                c00.fma(a0,b0);
                c01.fma(a0,b1);
                c10.fma(a1,b0);
                c11.fma(a1,b1);
                c20.fma(a2,b0);
                c21.fma(a2,b1);
                c30.fma(a3,b0);
                c31.fma(a3,b1);
            }


            //对每个ymm内部求和
            // c[j+0,i+0]+=c00.sum();
            // c[j+0,i+1]+=c10.sum();
            // c[j+0,i+2]+=c20.sum();
            // c[j+0,i+3]+=c30.sum();
            // c[j+1,i+0]+=c01.sum();
            // c[j+1,i+1]+=c11.sum();
            // c[j+1,i+2]+=c21.sum();
            // c[j+1,i+3]+=c31.sum();

            //上面注释代码的高效实现：转置再直接求和直接写入：
            ymm t0=_mm256_permute2f128_ps(c00,c01,0x20);
            ymm t1=_mm256_permute2f128_ps(c10,c11,0x20);
            ymm t2=_mm256_permute2f128_ps(c20,c21,0x20);
            ymm t3=_mm256_permute2f128_ps(c30,c31,0x20);
            ymm t4=_mm256_permute2f128_ps(c00,c01,0x31);
            ymm t5=_mm256_permute2f128_ps(c10,c11,0x31);
            ymm t6=_mm256_permute2f128_ps(c20,c21,0x31);
            ymm t7=_mm256_permute2f128_ps(c30,c31,0x31);
            
            c00=_mm256_unpacklo_pd(t0,t2);
            c10=_mm256_unpacklo_pd(t1,t3);
            c20=_mm256_unpackhi_pd(t0,t2);
            c30=_mm256_unpackhi_pd(t1,t3);
            c01=_mm256_unpacklo_pd(t4,t6);
            c11=_mm256_unpacklo_pd(t5,t7);
            c21=_mm256_unpackhi_pd(t4,t6);
            c31=_mm256_unpackhi_pd(t5,t7);

            c00=_mm256_shuffle_ps(c00,c00,0xd8);
            c10=_mm256_shuffle_ps(c10,c10,0xd8);
            c20=_mm256_shuffle_ps(c20,c20,0xd8);
            c30=_mm256_shuffle_ps(c30,c30,0xd8);
            c01=_mm256_shuffle_ps(c01,c01,0xd8);
            c11=_mm256_shuffle_ps(c11,c11,0xd8);
            c21=_mm256_shuffle_ps(c21,c21,0xd8);
            c31=_mm256_shuffle_ps(c31,c31,0xd8);

            t0=_mm256_unpacklo_ps(c00,c10);
            t1=_mm256_unpackhi_ps(c00,c10);
            t2=_mm256_unpacklo_ps(c20,c30);
            t3=_mm256_unpackhi_ps(c20,c30);
            t4=_mm256_unpacklo_ps(c01,c11);
            t5=_mm256_unpackhi_ps(c01,c11);
            t6=_mm256_unpacklo_ps(c21,c31);
            t7=_mm256_unpackhi_ps(c21,c31);


            c00=ymm(&c[j+0,i&(~0x7)]);
            c01=ymm(&c[j+1,i&(~0x7)]);

            t0=_mm256_add_ps(t0,t1);
            t2=_mm256_add_ps(t2,t3);
            t4=_mm256_add_ps(t4,t5);
            t6=_mm256_add_ps(t6,t7);
            t0=_mm256_add_ps(t0,t2);
            t4=_mm256_add_ps(t4,t6);
            t0=_mm256_add_ps(t0,t4);

            if(i&4)
            {
                c00=_mm256_add_ps(c00,_mm256_permute2f128_ps(t0,t0,i&4?0x08:0x80));
                c01=_mm256_add_ps(c01,_mm256_permute2f128_ps(t0,t0,i&4?0x18:0x81));
            }
            else
            {
                c00=_mm256_add_ps(c00,_mm256_permute2f128_ps(t0,t0,i&4?0x08:0x80));
                c01=_mm256_add_ps(c01,_mm256_permute2f128_ps(t0,t0,i&4?0x18:0x81));
            }
            
            _mm256_store_ps(&c[j+0,i&(~7)],c00);
            _mm256_store_ps(&c[j+1,i&(~7)],c01);
        }

        for(;j<b.n;j++)
        {
            ymm c00,c10,c20,c30;

            for(int k=0;k<a.M;k+=8)
            {
                ymm a0(&a[i+0,k]);
                ymm a1(&a[i+1,k]);
                ymm a2(&a[i+2,k]);
                ymm a3(&a[i+3,k]);
                ymm b0(&b[j+0,k]);

                c00.fma(a0,b0);
                c10.fma(a1,b0);
                c20.fma(a2,b0);
                c30.fma(a3,b0);
            }

            c[j+0,i+0]+=c00.sum();
            c[j+0,i+1]+=c10.sum();
            c[j+0,i+2]+=c20.sum();
            c[j+0,i+3]+=c30.sum();
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

void test()
{
    float A[1000];
    int n=8,m=8;
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            A[i+j*n]=i+j*n;
        }
    }

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
            printf("%3.0f ",A[i+j*n]);
        puts("");
    }

    printf("<<%d %d>>\n",n,m);

    Mat a=convert_mem_layout_A(A,mem[0],n,m);

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
            printf("%3.0f ",a[i,j]);
        puts("");
    }
}

void custom_sgemm(int M, int K, int N, float* A, float* B, float* C) {

    test();
    Mat a=convert_mem_layout_A(A,mem[0],M,K);
    Mat b=convert_mem_layout_B(B,mem[1],K,N);
    Mat c={C,N,M,N,M};
    custom_sgemm_sc(a,b,c);
}
