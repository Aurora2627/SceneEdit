from pathlib import Path
import subprocess,sys,json,datetime
root=Path('D:/career-projects/SceneEdit');setup=Path('D:/career-projects/scenedit-setup-20261022')
result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-v'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8');(setup/'tests.log').write_text(result.stdout,encoding='utf-8');print(result.stdout);print('TEST_EXIT',result.returncode)
if result.returncode:raise SystemExit(result.returncode)
subprocess.run([sys.executable,'-c','from diffsynth.pipelines.wan_video import WanVideoPipeline'],check=True)
import torch
assert torch.cuda.is_available()
frozen=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True);frozen=frozen.replace(str(root/'vendor/DiffSynth-Studio'),'vendor/DiffSynth-Studio').replace(str(root),'.');(root/'requirements-lock-windows.txt').write_text(frozen,encoding='utf-8')
status={'status':'ready','python':sys.executable,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'checked':datetime.datetime.now().astimezone().isoformat()};(setup/'environment-status.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print('ENVIRONMENT_VERIFIED',status)
