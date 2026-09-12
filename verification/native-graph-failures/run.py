import os, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
env=dict(os.environ)
rt=Path(env['LOCALAPPDATA'])/'Temp/cuda-rt/nvidia'
env['PATH']=os.pathsep.join(str(rt/p) for p in ['cuda_nvrtc/bin','cublas/bin'])+os.pathsep+env['PATH']
env['FERRO_REQUIRE_CUDA']='1'
env.pop('RUST_TEST_THREADS',None)
args=sys.argv[2:]
full=args[:1]==['--full']
cmd=['cargo','test','-j2','-p','ferro-cuda',*args[1:],'--','--nocapture'] if full else ['cargo','test','-j2','-p','ferro-cuda','--lib',*args,'--','--nocapture']
r=subprocess.run(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(out/(sys.argv[1]+'.log')).write_text('COMMAND: '+repr(cmd)+'\n'+r.stdout)
print(r.stdout[-6000:]);sys.exit(r.returncode)
