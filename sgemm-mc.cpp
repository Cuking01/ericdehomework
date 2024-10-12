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

void custom_sgemm_sc(Mat a,Mat b,Mat c,int al=0,int ar=-1)
{
    if(ar==-1)ar=a.n;
    int i;

    for(i=al;i+3<ar;i+=4)
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

    for(;i<ar;i++)
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

struct Thread
{
    std::thread thread;
    int l,r;
    bool ready;
    char P[128];
};

struct Work_Threads
{

    std::vector<Thread> threads;
    
    Mat a,b,c;

    std::condition_variable cv_work;
    char P1[128];
    std::condition_variable cv_finish;
    char P2[128];
    std::atomic<int> ready;
    char P3[128];
    std::atomic<int> cnt;
    char P4[128];
    std::mutex mtx;
    char P5[128];
    std::mutex mtx2;

    Work_Threads(int n)
    {
        ready=0;

        auto work_thread=[this](int id)
        {
            auto&me=threads[id];

            while(1)
            {

                {
                    std::unique_lock lock(mtx);
                    cv_work.wait(lock,[this,&me](){return me.ready;});
                    me.ready=false;
                }
                
                custom_sgemm_sc(a,b,c,me.l,me.r);

                cnt--;
                if(cnt==0)cv_finish.notify_one();
            }
        };

        threads.reserve(n);
        for(int i=0;i<n;i++)
            threads.emplace_back(std::thread{work_thread,i},0,0,0);
        
        for(int i=0;i<n;i++)
            threads[i].thread.detach();
    }

    void work(Mat a,Mat b,Mat c)
    {
        {
            std::lock_guard lock(mtx);
            this->a=a;
            this->b=b;
            this->c=c;

            int block_num=(a.n+15)/16;
            int m=block_num/threads.size();
            int r=block_num%threads.size();
            int l=0;
            for(int i=0;i<threads.size();i++)
            {
                threads[i].l=l;
                l+=m*16;
                if(i<r)l+=16;
                threads[i].r=std::min(l,a.n);
            }

            cnt=threads.size();
            for(auto&thread:threads)
                thread.ready=true;
        }

        cv_work.notify_all();

        {
            std::unique_lock lock(mtx2);
            cv_finish.wait(lock,[this](){return cnt==0;});
        }
    }
};

void custom_sgemm_mc(Mat a,Mat b,Mat c)
{
    static Work_Threads work_threads(8);
    work_threads.work(a,b,c);
}

void custom_sgemm(int M, int K, int N, float* A, float* B, float* C) {

    Mat a=convert_mem_layout_A(A,mem[0],M,K);
    Mat b=convert_mem_layout_B(B,mem[1],K,N);
    Mat c={C,N,M,N,M};

    custom_sgemm_mc(a,b,c);
    //custom_sgemm_sc(a,b,c);
}
