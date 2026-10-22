"""Install SceneEdit in its dedicated CUDA environment; leave other environments untouched."""
from pathlib import Path
import datetime,json,os,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[1]
SETUP=Path('D:/career-projects/scenedit-setup-20261022');SETUP.mkdir(exist_ok=True)
expected=Path('D:/career-projects/envs/scenedit/Scripts/python.exe')
assert Path(sys.executable).resolve()==expected.resolve(),'Run with the dedicated scenedit environment'
def save(data):(SETUP/'environment-status.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
status={'started':datetime.datetime.now().astimezone().isoformat(),'status':'installing','python':sys.executable};save(status)
env=os.environ.copy();env['PYTHONUTF8']='1'
def pip(*args):subprocess.run([sys.executable,'-m','pip','--cache-dir','D:/career-projects/cache/pip','--proxy','http://127.0.0.1:7890',*args],env=env,check=True)
try:
 pip('install','--upgrade','pip','setuptools>=77','wheel')
 pip('install','torch==2.6.0','torchvision==0.21.0','--index-url','https://download.pytorch.org/whl/cu124')
 pip('install','transformers==4.53.2','accelerate==1.10.1','peft==0.17.1','bitsandbytes==0.48.1','numpy==2.2.6','opencv-python-headless==4.12.0.88','scipy==1.16.2','pandas<3','pillow','imageio[ffmpeg]','safetensors','einops','modelscope','ftfy','sentencepiece','protobuf')
 pip('install','--no-deps','-e',str(ROOT/'vendor/DiffSynth-Studio'))
 pip('install','--no-deps','-e',str(ROOT))
 import torch
 assert torch.cuda.is_available(),'CUDA is unavailable in SceneEdit'
 subprocess.run([sys.executable,'-c','from diffsynth.pipelines.wan_video import WanVideoPipeline; pass'],check=True)
 frozen=subprocess.check_output([sys.executable,'-m','pip','freeze']).decode('utf-8');(ROOT/'requirements-lock-windows.txt').write_text(frozen.replace(str(ROOT/'vendor/DiffSynth-Studio'), 'vendor/DiffSynth-Studio').replace(str(ROOT),'.'),encoding='utf-8')
 status.update(status='ready',torch=torch.__version__,gpu=torch.cuda.get_device_name(0),finished=datetime.datetime.now().astimezone().isoformat());save(status);print('SCENEDIT_ENV_READY',status['gpu'],flush=True)
except Exception as error:status.update(status='failed',error=str(error));save(status);raise
