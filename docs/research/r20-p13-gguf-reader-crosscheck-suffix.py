# Independent GGUFReader cross-check.
# Execute on authorized V100 with the adjacent slicer loaded into the namespace;
# original job concatenated slicer (without __main__ invocation) and this suffix.

import sys, pathlib, tempfile, json, hashlib, numpy as np
p=pathlib.Path.home()/".cache/mt-pruning/r20-r1"
sys.path.insert(0,str(p/"llama.cpp/gguf-py"))
from gguf import GGUFReader
def compare(path):
    custom=parse(path);other=GGUFReader(path)
    need(custom["start"]==other.data_offset and custom["alignment"]==other.alignment,"reader header")
    need(len(custom["tensors"])==len(other.tensors),"reader tensor count")
    for t,o in zip(custom["tensors"],other.tensors):
        need((t["name"],t["dims"],t["type"],custom["start"]+t["off"],t["bytes"])==(o.name,tuple(o.shape.tolist()),int(o.tensor_type),o.data_offset,o.n_bytes),"reader descriptor "+t["name"])
    need(custom["fields"]["hy_v3.expert_count"][2]==other.get_field("hy_v3.expert_count").contents(),"expert metadata")
    return other
# No access to model payload: compare independent header descriptors only.
real=p.parent/"gguf-r20-r1/k128-f16.gguf"
r=compare(real)
with tempfile.TemporaryDirectory(prefix="mt-p13-reader-") as d:
    q=pathlib.Path(d);src=q/"fixture.gguf";dst=q/"candidate.gguf"
    table=fixture(src);mask={"1":[0,3],"2":[1,2]}
    result=write_new(src,dst,mask,sha_file(src),canon(mask),0,experts=4,layers=3,hidden=3,width=2,top=2)
    s=compare(src);t=compare(dst);count=0
    for row,o in zip(table,t.tensors):
        name,_,_,_,b=row;m=EXPERT.fullmatch(name)
        if m:
            stride=len(b)//4;b=b"".join(b[e*stride:(e+1)*stride] for e in mask[m[1]])
        need(o.data.tobytes()==b,"official reader byte oracle "+name);count+=1
    print(json.dumps({"status":"PASS_READER_CROSSCHECK_SYNTHETIC_AND_REAL_HEADER","real_tensor_descriptors":len(r.tensors),"synthetic_source_descriptors":len(s.tensors),"synthetic_output_descriptors":len(t.tensors),"synthetic_output_byte_oracle":count,"synthetic_output_sha256":result["sha256"],"numpy":np.__version__,"reader_module_sha256":hashlib.sha256((p/"llama.cpp/gguf-py/gguf/gguf_reader.py").read_bytes()).hexdigest(),"real_payload_read":False}))


