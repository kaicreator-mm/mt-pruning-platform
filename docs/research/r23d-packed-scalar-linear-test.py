"""Pinned GGML quantizer/dequantizer independent oracle for C++ packed linear."""
import argparse,ctypes,hashlib,json,pathlib,numpy as np
import gguf.constants as c
import gguf.quants as q
P=pathlib.Path
PINS={"constants.py":"3e637e339284da81df558c83b9aa4efb9ca3106cf983da15da67766a5d9118db","lazy.py":"dbcf8b7376284ab2d920103657cc52c1c62e93fec3856a08a1ae638394e6a7c1","quants.py":"f896726209ba8c91afd083dd87031c2d39dc4ef6a78e8a6c12763eeb8ed6ff84"}
def need(x,m):
    if not x:raise ValueError(m)
def main():
    a=argparse.ArgumentParser();a.add_argument("--lib",required=True,type=P);args=a.parse_args()
    for n,h in PINS.items():need(hashlib.sha256((P(q.__file__).parent/n).read_bytes()).hexdigest()==h,"quant source")
    lib=ctypes.CDLL(str(args.lib));f=lib.r23_packed_linear
    ptr=ctypes.c_void_p;size=ctypes.c_size_t;f.argtypes=[ctypes.c_int,ptr,size,ptr,size,ptr,size,size,size,size];f.restype=ctypes.c_int
    rng=np.random.default_rng(2026);results=[]
    def run(code,w,x,y,r,k,b,wb=None,xb=None,yb=None):
        return f(code,w.ctypes.data,w.nbytes if wb is None else wb,x.ctypes.data,x.size if xb is None else xb,y.ctypes.data,y.size if yb is None else yb,r,k,b)
    for typ,code in (("Q4_0",4),("Q8_0",8)):
        enum=getattr(c.GGMLQuantizationType,typ);errors=[];ratios=[];checks=0;outputs=0
        for r,k in ((13,32),(64,128),(768,2048),(2048,768)):
            for b in (1,3):
                source=rng.standard_normal((r,k),dtype=np.float32);x=rng.standard_normal((b,k),dtype=np.float32)
                w=np.ascontiguousarray(q.quantize(source,enum));deq=q.dequantize(w,enum).astype(np.float64)
                oracle=x.astype(np.float64)@deq.T;y=np.empty((b,r),dtype=np.float32)
                need(run(code,w,x,y,r,k,b)==0,"kernel status")
                err=np.abs(y.astype(np.float64)-oracle)
                # Scalar double accumulation then float32 rounding. Absolute
                # bound scales with sum of absolute terms, not relative result
                # near zero. Compare independent GGML dequantizer.
                bound=(np.abs(x.astype(np.float64))@np.abs(deq).T)*1e-6+1e-7
                need(np.all(err<=bound),"numeric bound")
                errors.append(float(err.max()));ratios.append(float((err/bound).max()));checks+=1;outputs+=y.size
        # Deliberate signed scale, all nibbles including the two half-blocks,
        # subnormal scale, signed-byte extremes, zero input.
        r,k,b=2,32,2;bs=18 if code==4 else 34
        w=np.zeros((r,bs),dtype=np.uint8);w[0,0:2]=[0,188];w[1,0:2]=[1,0]
        w[:,2:]=np.arange(bs-2,dtype=np.uint8)*17
        x=np.arange(b*k,dtype=np.float32).reshape(b,k)-31;y=np.empty((b,r),dtype=np.float32)
        oracle=x.astype(np.float64)@q.dequantize(w,enum).astype(np.float64).T
        need(run(code,w,x,y,r,k,b)==0 and np.allclose(y,oracle,rtol=1e-6,atol=1e-7),"explicit block oracle")
        need(run(code,w,x,y,r,k,b,wb=w.nbytes-1)==1,"short weights")
        need(run(code,w,x,y,r,31,b)==1,"unaligned columns")
        need(run(code,w,x,y,r,k,b,xb=x.size-1)==1,"short input")
        need(run(code,w,x,y,r,k,b,yb=y.size-1)==1,"short output")
        x[0,0]=np.nan;need(run(code,w,x,y,r,k,b)==2,"nonfinite activation");x[0,0]=0
        w[0,:2]=[0,124];need(run(code,w,x,y,r,k,b)==2,"nonfinite scale")
        results.append(dict(type=typ,random_matrix_checks=checks,output_elements=outputs,max_abs_error=max(errors),max_error_to_bound=max(ratios),explicit_block_oracle=True,negative=6))
    print(json.dumps(dict(status="PASS_CPU_PACKED_SCALAR_LINEAR",results=results,actual_model="NOT_RUN",native_ggml_kernel="NOT_RUN",speed="NOT_MEASURED",scratch="no materialized dequantized matrix in C++ kernel")))
if __name__=="__main__":main()
