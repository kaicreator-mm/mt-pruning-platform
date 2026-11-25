"""D17/D18: content-bound receipts and serial materialization planning, not a model runner."""
import hashlib, json, re, sqlite3
from pathlib import Path

FIELDS={'teacher','candidate','corpus','prompt','runtime','protocol','code'}
SPLITS=('calibration','validation','test')
def sha(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def key(identity):
    if not isinstance(identity,dict) or set(identity)!=FIELDS|{'split'} or identity['split'] not in SPLITS or not all(sha(identity[f]) for f in FIELDS):
        raise ValueError('incomplete or invalid experiment identity')
    return hashlib.sha256(canonical(identity).encode()).hexdigest()

def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

class Cache:
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        with self.db:
            self.db.execute('CREATE TABLE IF NOT EXISTS done (key TEXT PRIMARY KEY, payload TEXT NOT NULL, sha TEXT NOT NULL)')
    def put(self,identity,result,artifact):
        if result.get('status')!='COMPLETE' or result.get('split')!=identity.get('split'):
            raise ValueError('only complete experiment receipts are cacheable')
        k=key(identity); p=canonical({'result':result,'artifact_sha256':filehash(artifact)})
        s=hashlib.sha256(p.encode()).hexdigest()
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO done VALUES (?,?,?)',(k,p,s))
            if self.db.execute('SELECT payload,sha FROM done WHERE key=?',(k,)).fetchone()!=(p,s):
                raise ValueError('conflicting immutable result')
    def get(self,identity,artifact):
        row=self.db.execute('SELECT payload,sha FROM done WHERE key=?',(key(identity),)).fetchone()
        if row is None: return None
        p,s=row
        if hashlib.sha256(p.encode()).hexdigest()!=s: raise ValueError('corrupt receipt')
        value=json.loads(p)
        if not Path(artifact).is_file() or filehash(artifact)!=value['artifact_sha256']: raise ValueError('missing or changed artifact')
        return value['result']
    def close(self): self.db.close()

def disk_plan(teacher_bytes,fixed_bytes,expert_bytes,original_counts,top_k,candidates,free_bytes,reserve_bytes=0):
    """Payload planning only. candidates maps name to selected expert count per layer.
    free_bytes excludes the already stored teacher. One physical candidate at a time.
    reserve covers conversion buffers, headers, sidecars and filesystem overhead externally.
    """
    nums=[teacher_bytes,fixed_bytes,free_bytes,reserve_bytes,*expert_bytes]
    if any(type(x) is not int or x<0 for x in nums) or not expert_bytes or any(x==0 for x in expert_bytes): raise ValueError('bad byte inventory')
    if type(top_k) is not int or top_k<1 or len(original_counts)!=len(expert_bytes) or any(type(n) is not int or n<top_k for n in original_counts): raise ValueError('bad original counts')
    if not candidates or teacher_bytes!=fixed_bytes+sum(n*b for n,b in zip(original_counts,expert_bytes)): raise ValueError('inconsistent teacher inventory')
    sizes={}
    for name,counts in candidates.items():
        if not isinstance(name,str) or not name or len(counts)!=len(expert_bytes) or any(type(n) is not int or not top_k<=n<=old for n,old in zip(counts,original_counts)): raise ValueError('bad candidate')
        size=fixed_bytes+sum(n*b for n,b in zip(counts,expert_bytes))
        if size>teacher_bytes: raise ValueError('candidate exceeds teacher payload')
        sizes[name]=size
    peak=max(sizes.values())+reserve_bytes
    return {'candidate_payloads':sizes,'all_candidate_payloads':sum(sizes.values()),'serial_extra_peak_estimate':peak,'serial_fits_estimate':peak<=free_bytes,'teacher_plus_serial_estimate':teacher_bytes+peak,'requires_one_candidate_at_a_time':True}
