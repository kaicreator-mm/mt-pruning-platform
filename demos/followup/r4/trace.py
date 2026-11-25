"""D14-D16: deterministic trace comparison, sample stability and resumable journal."""
import hashlib, json, math, sqlite3, random, statistics

def canon(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(x):
    return hashlib.sha256(canon(x).encode()).hexdigest()

def vector(v):
    if not isinstance(v, list) or not v or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in v) or not 0<sum(v)<math.inf:
        raise ValueError('invalid mass')
    return [x/sum(v) for x in v]

def compare(before, after, mapping):
    """Same input, same token/layer; candidate IDs mapped to teacher IDs."""
    if not before or len(before)!=len(after): raise ValueError('unaligned traces')
    if any(type(x) is not int or x<0 for x in mapping) or len(set(mapping))!=len(mapping): raise ValueError('bad mapping')
    overlaps=[]; changes=0; seen=set()
    for a,b in zip(before,after):
        key=a['key']
        if key!=b['key'] or key in seen: raise ValueError('unaligned/duplicate token')
        seen.add(key); x=a['ids']; y=b['ids']
        if not x or len(x)!=len(y) or len(set(x))!=len(x) or len(set(y))!=len(y) or any(type(i) is not int or i<0 for i in x) or any(type(i) is not int or not 0<=i<len(mapping) for i in y): raise ValueError('bad expert IDs')
        mapped={mapping[i] for i in y}; old=set(x)
        overlaps.append(len(old & mapped)/len(old)); changes+=old!=mapped
    return {'tokens':len(overlaps),'mean_topk_overlap':statistics.mean(overlaps),'changed_fraction':changes/len(overlaps)}

def stability(samples, sizes, keep, seeds):
    """Each row is a sample aggregate; use paired nested permutations per seed."""
    if not samples or not seeds or len(set(seeds))!=len(seeds) or not sizes or sorted(set(sizes))!=sizes or sizes[-1]>=len(samples) or sizes[0]<1: raise ValueError('bad design')
    n=len(samples[0]); rows=[vector(v) for v in samples]
    if not 1<=keep<=n or any(len(v)!=n for v in rows): raise ValueError('bad shape')
    def rank(ids):
        return set(sorted(range(n), key=lambda i:(-sum(rows[j][i] for j in ids),i))[:keep])
    out=[]
    for size in sizes:
        picks=[]; held=[]
        for seed in seeds:
            ids=list(range(len(rows))); random.Random(seed).shuffle(ids)
            chosen=rank(ids[:size]); picks.append(chosen)
            held.append(statistics.mean(sum(rows[j][i] for i in chosen) for j in ids[size:]))
        pairs=[len(a&b)/len(a|b) for j,a in enumerate(picks) for b in picks[j+1:]]
        out.append({'size':size,'mean_jaccard':statistics.mean(pairs) if pairs else None,'min_jaccard':min(pairs) if pairs else None,'mean_heldout_mass':statistics.mean(held)})
    return out

class Journal:
    """One committed sample per (identity,sample_id); exactly-once aggregation by replay."""
    def __init__(self,path,identity):
        if not isinstance(identity,dict) or not identity: raise ValueError('missing identity')
        self.identity=digest(identity); self.db=sqlite3.connect(path)
        try:
            with self.db:
                self.db.execute('CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT NOT NULL)')
                self.db.execute('INSERT OR IGNORE INTO meta VALUES (1,?)',(self.identity,))
                if self.db.execute('SELECT identity FROM meta').fetchone()[0]!=self.identity: raise ValueError('identity mismatch')
                self.db.execute('CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, payload TEXT NOT NULL, sha TEXT NOT NULL)')
        except Exception:
            self.db.close(); raise
    def put(self,sample_id,payload):
        if not isinstance(sample_id,str) or not sample_id: raise ValueError('bad sample id')
        encoded=canon(payload); sha=digest(payload)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO samples VALUES (?,?,?)',(sample_id,encoded,sha))
            if self.db.execute('SELECT payload,sha FROM samples WHERE id=?',(sample_id,)).fetchone()!=(encoded,sha): raise ValueError('conflicting sample')
    def records(self):
        for i,p,s in self.db.execute('SELECT id,payload,sha FROM samples ORDER BY id'):
            value=json.loads(p)
            if digest(value)!=s: raise ValueError('corrupt sample')
            yield i,value
    def close(self): self.db.close()
