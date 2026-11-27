"""Direct packed gated FFN + K slab numeric oracle. No real weights or native kernel."""
import argparse,ctypes,hashlib,json,pathlib,numpy as np
import gguf.constants as c
import gguf.quants as q
P=pathlib.Path
PINS={"constants.py":"3e637e339284da81df558c83b9aa4efb9ca3106cf983da15da67766a5d9118db","lazy.py":"dbcf8b7376284ab2d920103657cc52c1c62e93fec3856a08a1ae638394e6a7c1","quants.py":"f896726209ba8c91afd083dd87031c2d39dc4ef6a78e8a6c12763eeb8ed6ff84"}
def need(x,m):
    if not x:raise ValueError(m)
def bind(lib,name):
    f=getattr(ctypes.CDLL(str(lib)),name);p=ctypes.c_void_p;s=ctypes.c_size_t
    f.argtypes=[ctypes.c_int,p,s,p,s,p,s,s,s,s];f.restype=ctypes.c_int;return f
def call(f,code,w,x,rows,cols,wb=None):
    x=np.ascontiguousarray(x,dtype=np.float32);y=np.empty((len(x),rows),np.float32)
    ret=f(code,w.ctypes.data,w.nbytes if wb is None else wb,x.ctypes.data,x.size,y.ctypes.data,y.size,rows,cols,len(x));return ret,y
def silu(x):return x/(1+np.exp(-np.clip(x,-80,80)))
def main():
    p=argparse.ArgumentParser();p.add_argument("--lib",type=P);p.add_argument("--k-lib",type=P);a=p.parse_args()
    for n,h in PINS.items():need(hashlib.sha256((P(q.__file__).parent/n).read_bytes()).hexdigest()==h,"quant source")
    simple=bind(a.lib,"r23_packed_linear");klinear=bind(a.k_lib,"r23_k_linear");rng=np.random.default_rng(2026);out=[]
    for typ,code,f in (("Q4_0",4,simple),("Q8_0",8,simple),("Q4_K",4,klinear),("Q6_K",6,klinear)):
        enum=getattr(c.GGMLQuantizationType,typ);is_k=typ.endswith("_K");errors=[];ratios=[];ffn_errors=[];mixture_errors=[];linear_checks=0
        def pack(rows,cols):
            if not is_k:return np.ascontiguousarray(q.quantize(rng.standard_normal((rows,cols),dtype=np.float32)/np.sqrt(cols),enum))
            bs=144 if code==4 else 210
            z=rng.integers(0,256,(rows,cols//256,bs),dtype=np.uint8)
            # Finite small binary16 scales, independently randomized packed codes.
            if code==4:
                z[:,:,0:2]=np.frombuffer(np.float16(2**-14).tobytes(),np.uint8);z[:,:,2:4]=np.frombuffer(np.float16(2**-14).tobytes(),np.uint8)
            else:z[:,:,-2:]=np.frombuffer(np.float16(2**-17).tobytes(),np.uint8)
            return np.ascontiguousarray(z.reshape(rows,-1))
        def linear(w,x,r,c0):
            nonlocal linear_checks
            ret,y=call(f,code,w,x,r,c0);need(ret==0,"packed linear")
            d=q.dequantize(w,enum).astype(np.float64);oracle=x.astype(np.float64)@d.T
            bound=(np.abs(x.astype(np.float64))@np.abs(d).T)*1e-6+1e-7
            err=np.abs(y-oracle);need(np.all(err<=bound),"linear error bound "+typ)
            linear_checks+=1;errors.append(float(err.max()));ratios.append(float((err/bound).max()))
            return y,d
        for h,m,b in ((256,256,3),(2048,768,1),(2048,768,3)):
            x=rng.standard_normal((b,h),dtype=np.float32)*.5;got=[];ref=[]
            for expert in range(2):
                g,u,d=pack(m,h),pack(m,h),pack(h,m)
                gy,gd=linear(g,x,m,h);uy,ud=linear(u,x,m,h)
                activation=(silu(gy)*uy).astype(np.float32);y,dd=linear(d,activation,h,m)
                xx=x.astype(np.float64);yy=(silu(xx@gd.T)*(xx@ud.T))@dd.T
                rel=float(np.linalg.norm(y-yy)/max(np.linalg.norm(yy),1e-12))
                need(rel<=2e-6,"gated FFN relative L2");ffn_errors.append(rel);got.append(y);ref.append(yy)
            w=np.array([.35,.65],np.float64)
            y=sum(t*v for t,v in zip(got,w));z=sum(t*v for t,v in zip(ref,w))
            rel=float(np.linalg.norm(y-z)/max(np.linalg.norm(z),1e-12));need(rel<=2e-6,"expert mixture relative L2");mixture_errors.append(rel)
        if is_k:
            w=pack(2,256);x=np.ones((1,256),np.float32)
            need(call(f,code,w,x,2,256,wb=w.nbytes-1)[0]==1,"short weights")
            need(call(f,code,w,x[:,:255],2,255)[0]==1,"unaligned")
            x[0,0]=np.nan;need(call(f,code,w,x,2,256)[0]==2,"nonfinite x");x[0,0]=0
            w[0,(0 if code==4 else 208):(2 if code==4 else 210)]=[0,124];need(call(f,code,w,x,2,256)[0]==2,"nonfinite scale")
        out.append(dict(type=typ,source="valid synthetic K blocks" if is_k else "actual quantized synthetic weights",linear_checks=linear_checks,max_linear_abs_error=max(errors),max_linear_error_to_bound=max(ratios),gated_ffn_checks=len(ffn_errors),max_ffn_relative_l2=max(ffn_errors),weighted_two_expert_checks=len(mixture_errors),max_mixture_relative_l2=max(mixture_errors),negative=4 if is_k else 0))
    print(json.dumps(dict(status="PASS_CPU_PACKED_GATED_FFN_K_BLOCKS",results=out,contract="FP32 activation and stage rounding, FP64 scalar accumulation; tolerance, not BF16 bitwise",actual_model="NOT_RUN",native_runtime="NOT_RUN",speed="NOT_MEASURED")))
if __name__=="__main__":main()
