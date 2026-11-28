// Scalar K-block packed linear. Independent research oracle, not GGML SIMD.
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
static float half_le(const uint8_t* p) {
    unsigned u=unsigned(p[0])|(unsigned(p[1])<<8), e=(u>>10)&31,m=u&1023;
    if(e==31)return std::numeric_limits<float>::quiet_NaN();
    float x=e?std::ldexp(float(1024+m),int(e)-25):std::ldexp(float(m),-24);
    return u&32768?-x:x;
}
static int sbyte(uint8_t x){return x<128?int(x):int(x)-256;}
static float value(int type,const uint8_t* p,size_t j){
    if(type==4){
        size_t g=j/32;const uint8_t* s=p+4;
        unsigned sc=g<4?s[g]&63:(s[g+4]&15)|((s[g-4]>>2)&48);
        unsigned mn=g<4?s[g+4]&63:(s[g+4]>>4)|((s[g]>>2)&48);
        unsigned q=(p[16+(g/2)*32+j%32]>>(4*(g%2)))&15;
        float d=half_le(p)*float(sc),dm=half_le(p+2)*float(mn);
        return d*float(q)-dm;
    }
    size_t g=j/32, a=j%32;
    unsigned lo=(p[(g/4)*64+(g%2)*32+a]>>(4*((g%4)/2)))&15;
    unsigned hi=(p[128+(g/4)*32+a]>>(2*(g%4)))&3;
    return (half_le(p+208)*float(sbyte(p[192+j/16])))*float(int(lo|(hi<<4))-32);
}
extern "C" int r23_k_linear(int type,const uint8_t* w,size_t wb,const float* x,size_t xn,float* y,size_t yn,size_t rows,size_t cols,size_t batch){
    if((type!=4&&type!=6)||!w||!x||!y||!rows||!cols||!batch||cols%256)return 1;
    size_t bs=type==4?144:210, nb=cols/256;
    if(nb>SIZE_MAX/bs)return 1;
    size_t stride=nb*bs;
    if(rows>SIZE_MAX/stride||batch>SIZE_MAX/cols||batch>SIZE_MAX/rows||wb!=rows*stride||xn!=batch*cols||yn!=batch*rows)return 1;
    for(size_t i=0;i<xn;i++)if(!std::isfinite(x[i]))return 2;
    for(size_t i=0;i<rows*nb;i++){
        const uint8_t* p=w+i*bs;
        if(!std::isfinite(half_le(p+(type==4?0:208)))||(type==4&&!std::isfinite(half_le(p+2))))return 2;
    }
    for(size_t b=0;b<batch;b++)for(size_t r=0;r<rows;r++){
        double sum=0.;
        for(size_t k=0;k<nb;k++){
            const uint8_t* p=w+r*stride+k*bs;
            for(size_t j=0;j<256;j++)sum+=double(value(type,p,j))*double(x[b*cols+k*256+j]);
        }
        if(!std::isfinite(sum)||std::abs(sum)>std::numeric_limits<float>::max())return 3;
        y[b*rows+r]=float(sum);
    }
    return 0;
}
