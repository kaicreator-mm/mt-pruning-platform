"""D10: author-draft stress prompts, never reference metrics."""
import itertools,json
from pathlib import Path

def expand(data):
    if data.get('status')!='AUTHOR_DRAFT_NOT_REFERENCE': raise ValueError('not draft data')
    out=[]; seen=set()
    for c in data['cases']:
        if c['id'] in seen: raise ValueError('duplicate case')
        seen.add(c['id'])
        for a,b in itertools.permutations(('zh','en','my'),2):
            text=c[a]; req=c['required']
            if not isinstance(text,str) or not text.strip() or any(not isinstance(x,str) or not x or x not in text for x in req) or len(req)!=len(set(req)): raise ValueError('invalid literal contract')
            out.append({'id':c['id']+':'+a+'-'+b,'direction':a+'-'+b,'domain':c['domain'],'source':text,'required_literals':req,'status':'DRAFT_MANUAL_REVIEW_ONLY'})
    return out

def diagnostics(row,output):
    if not isinstance(output,str): raise ValueError('output not text')
    return {'empty':not output.strip(),'literal_count_mismatches':[s for s in row['required_literals'] if output.count(s)!=row['source'].count(s)],'semantic_quality':'NOT_ASSESSED'}
if __name__=='__main__':
    data=json.loads(Path(__file__).with_name('stress_r3.json').read_text(encoding='utf-8')); print(json.dumps(expand(data),ensure_ascii=False,indent=2))
