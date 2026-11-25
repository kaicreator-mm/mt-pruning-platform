"""Prepare/execute a bounded llama.cpp conversion and synthetic throughput benchmark.
No GPU residency verdict is inferred from CLI flags or timing alone.
"""
import argparse, json, subprocess, sys, re, shutil, hashlib
from pathlib import Path
from prep import write, require, digest

def git_head(root):
    return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()

def commands(root,binaries,model,out,quant):
    suffix='.exe' if sys.platform=='win32' else ''
    return [[sys.executable,str(root/'convert_hf_to_gguf.py'),str(model),'--outfile',str(out/'model-bf16.gguf'),'--outtype','bf16'],[str(binaries/('llama-quantize'+suffix)),str(out/'model-bf16.gguf'),str(out/f'model-{quant}.gguf'),quant],[str(binaries/('llama-bench'+suffix)),'-m',str(out/f'model-{quant}.gguf'),'-ngl','99','-p','512,2048','-n','128','-r','3','-o','json']]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--llama',type=Path,required=True); p.add_argument('--bin',type=Path,required=True); p.add_argument('--revision',required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--quant',choices=['Q4_K_M','Q6_K','Q8_0'],required=True); p.add_argument('--execute',action='store_true'); a=p.parse_args()
    root,b,model,out=map(lambda x:x.resolve(),(a.llama,a.bin,a.model,a.out)); require(bool(re.fullmatch('[0-9a-f]{40}',a.revision)),'full revision required'); cmds=commands(root,b,model,out,a.quant)
    if not a.execute: print(json.dumps({'status':'PLAN_ONLY','revision':a.revision,'commands':cmds},ensure_ascii=False,indent=2)); return
    require(git_head(root)==a.revision,'wrong llama.cpp revision'); require(not subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip(),'modified tracked source'); require((model/'config.json').is_file(),'model config absent'); require(not out.exists(),'output exists'); require(model not in out.parents and out!=model,'output inside source model'); require((root/'convert_hf_to_gguf.py').is_file() and all(Path(c[0]).is_file() for c in cmds[1:]),'missing executable'); shards=list(model.glob('*.safetensors')); require(shards,'no safetensors shards'); require(shutil.disk_usage(out.parent).free>=2*sum(x.stat().st_size for x in shards),'insufficient conservative free disk budget'); out.mkdir(); result={'status':'RUNNING','revision':a.revision,'commands':cmds,'model_config_sha256':digest(model/'config.json'),'checkpoint_files':{x.name:digest(x) for x in sorted(shards)},'binary_sha256':{Path(c[0]).name:digest(c[0]) for c in cmds[1:]},'gpu_residency':'NOT_VERIFIED','stages':[]}; write(out/'receipt.json',result)
    for idx,cmd in enumerate(cmds):
        stage={'name':['convert','quantize','bench'][idx]}; result['stages'].append(stage)
        try:
            with (out/f'{idx}.stdout').open('wb') as stdout,(out/f'{idx}.stderr').open('wb') as stderr:
                proc=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=7200,check=False)
            stage['returncode']=proc.returncode; require(proc.returncode==0,'command failed')
            if idx<2:
                target=out/('model-bf16.gguf' if idx==0 else f'model-{a.quant}.gguf'); require(target.is_file() and target.stat().st_size>0,'missing artifact'); stage.update(bytes=target.stat().st_size,sha256=digest(target))
            else:
                rows=json.loads((out/f'{idx}.stdout').read_text(encoding='utf-8')); require(isinstance(rows,list) and rows and all(float(x['avg_ts'])>0 for x in rows),'invalid benchmark output'); stage['rows']=len(rows)
            stage['status']='COMMAND_OK'
        except Exception as e:
            stage.update(status='FAILED',error=str(e)); result['status']='FAILED'; write(out/'receipt.json',result); raise
        write(out/'receipt.json',result)
    result['status']='COMMANDS_OK_RESIDENCY_UNVERIFIED'; write(out/'receipt.json',result)
if __name__=='__main__': main()
