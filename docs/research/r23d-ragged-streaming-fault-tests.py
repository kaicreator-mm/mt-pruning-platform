"""Supplemental streaming publication faults and independent source-binding checks."""
import argparse,hashlib,importlib.util,json,pathlib,struct,tempfile,shutil
P=pathlib.Path
PINS={"stream":"2a53ccf64f8dccdc31da096b58d59e91cb8820f8","adapter":"56ceecad1fd2db0d33399840e2231c718c7e6ab0"}
def load(path,name):
    raw=path.read_bytes();assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==PINS[name]
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    a=argparse.ArgumentParser();a.add_argument("--stream",type=P,required=True);a.add_argument("--adapter",type=P,required=True);q=a.parse_args();s=load(q.stream,"stream");r=load(q.adapter,"adapter")
    from mtpruning.testing import tiny_checkpoint
    import torch
    torch.set_num_threads(1)
    def reject(fn):
        try:fn()
        except (ValueError,OSError):return
        raise AssertionError("unexpected acceptance")
    with tempfile.TemporaryDirectory() as d:
        p=P(d);src=tiny_checkpoint(p/"source",p/"none",vocab_size=4096,seed=81,num_experts=8,top_k=2,layers=4)
        mask={"1":[0,2,4],"2":[1,2,3,5,6],"3":list(range(8))}
        dest=p/"pack"
        original=s.copy_tensor;reads=[];tensor_chunks=[]
        class Reader:
            def __init__(self,f):self.f=f;self.count=0
            def seek(self,n):return self.f.seek(n)
            def read(self,n):
                assert 0<n<=s.CHUNK;reads.append(n);self.count+=1;return self.f.read(n)
        def tracked(inp,out,offset,size):
            f=Reader(inp);h=original(f,out,offset,size);tensor_chunks.append(f.count);return h
        s.copy_tensor=tracked
        stage,h=s.stage(r,src,dest,mask,1<<20)
        assert max(tensor_chunks)>=8 and max(reads)==s.CHUNK
        m=s.verify(r,src,stage,h)
        # A source-bound verifier must reject payload corruption even when every
        # untrusted output digest is recomputed.
        bad=p/".bad.unpublished-fixture";shutil.copytree(stage,bad)
        m2=json.loads(s.read(bad/"ragged_manifest.json"))
        key="model.embed_tokens.weight";f=m2["weight_map"][key];desc=s.descriptors(bad/f)[key]
        with (bad/f).open("r+b") as out:
            out.seek(desc["offset"]);x=out.read(1);out.seek(desc["offset"]);out.write(bytes([x[0]^1]))
        with (bad/f).open("rb") as inp:inp.seek(desc["offset"]);b=inp.read(desc["bytes"])
        m2["tensor_sha256"][key]=s.sha(b);m2["shards"][f]=s.fsha(bad/f)
        raw=s.canon(m2)+b"\n";(bad/"ragged_manifest.json").write_bytes(raw)
        s.inspect(r,bad,s.sha(raw)) # structural integrity alone is not provenance
        reject(lambda:s.verify(r,src,bad,s.sha(raw)))
        reject(lambda:s.publish(r,src,bad,p/"bad",s.sha(raw)))
        assert bad.exists() and not (p/"bad").exists()
        # Interrupt after several successful tensor copies. Stage must remain
        # unpublished and lack a completion manifest; source remains intact.
        failed=p/"interrupted";count=0
        def interrupted(inp,out,offset,size):
            nonlocal count
            count+=1
            if count==5:raise OSError("simulated interrupted copy")
            return original(inp,out,offset,size)
        s.copy_tensor=interrupted
        reject(lambda:s.stage(r,src,failed,mask,1<<20))
        pending=list(p.glob(".interrupted.unpublished-*"));assert len(pending)==1 and not failed.exists() and not (pending[0]/"ragged_manifest.json").exists()
        reject(lambda:s.publish(r,src,pending[0],failed,h))
        s.copy_tensor=original
        s.publish(r,src,stage,dest,h);assert dest.exists()
        ck=s.checkpoint_class(r)(dest,h);o=r.eng.Checkpoint(src)
        assert all(torch.equal(ck.tensor(k),o.tensor(v)) for k,v in ck.source_map.items())
        assert all(s.fsha(src/n)==v for n,v in s.strict(s.read(src/"checkpoint_provenance.json"))["files"].items())
        print(json.dumps(dict(status="PASS_CPU_STREAMING_MULTICHUNK_FAULTS",max_copy_request=max(reads),max_chunks_per_tensor=max(tensor_chunks),copy_requests=len(reads),tensor_checks=len(ck.source_map),rehashed_corruption_rejected_by_source_verifier=True,interrupted_copy_stage_preserved=True,real_power_loss="NOT_TESTED",real_model="NOT_RUN")))
if __name__=="__main__":main()
