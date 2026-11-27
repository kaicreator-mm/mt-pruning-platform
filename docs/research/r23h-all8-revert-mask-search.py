"""Create three one-factor all8 ablation masks for targeted fit refinement."""
import hashlib,io,json,pathlib,tarfile
P=pathlib.Path
ARCH="eef19f39837236b569aeaf6f789ff6cfe25674b2acd60dd38a1f507b128dce57"
PLAN="b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
BASE="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
ALL8="1f7402ae0fb540c86ace21914b45ea0c42a113e5f38072a2d3f86795ee8ad70b"
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
root=P(__import__("sys").argv[1]);raw=(root/"remaining4-primary.tgz").read_bytes();assert sha(raw)==ARCH
with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as t:
 files={m.name:t.extractfile(m).read() for m in t.getmembers()}
plan=json.loads(files["screen/plan.json"]);assert sha(files["screen/plan.json"])==PLAN
base=json.loads(files["screen/original.json"])["candidates"]["96"]["kept"]
all8=next(x["kept"] for x in plan["screen"] if x["name"]=="all8")
assert sha(canon(base))==BASE and sha(canon(all8))==ALL8
assert [l for l in all8 if all8[l]!=base[l]]==["10","15","40","42"]
for name,layers in {"all8_revert40":[40],"all8_revert42":[42],"all8_revert40_42":[40,42]}.items():
 mask={l:list(v) for l,v in all8.items()}
 for l in layers:mask[str(l)]=list(base[str(l)])
 assert sha(canon(mask))!=ALL8 and all(len(v)==96 and v==sorted(set(v)) for v in mask.values())
 out=dict(status="R23H_ALL8_SINGLE_LAYER_ABLATION_EXPLORATORY_NOT_ADMITTED",method=name,source_mask_sha256=ALL8,base_mask_sha256=BASE,reverted_layers=layers,mask_sha256=sha(canon(mask)),mask=mask,reason="all8 is the only prior candidate with 11/12 primary fit metrics passing; isolate the L40/L42 local decode drift before full propagation",limits="Mask proposal only; actual prefill and primary fit required. No Fresh or product admission.")
 (root/(name+".json")).write_bytes(canon(out)+b"\n")
 print(name,out["mask_sha256"])
