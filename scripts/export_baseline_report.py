"""Export verified baseline measurements without redistributing benchmark media."""
import argparse
from pathlib import Path
from scenedit.common import ROOT,read,save,sha,video_info
from scenedit.infer import verify_inputs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    run_root=a.base/'baseline-v1';overall=read(run_root/'status.json');assert overall['status']=='complete' and overall['finished_runs']==4
    rows=read(run_root/'results.json');assert len(rows)==4
    expected={(case,mode) for case in ('car-roundabout','car-turn') for mode in ('with-reference','without-reference')};assert {(r['case'],r['mode']) for r in rows}==expected
    records=[]
    for row in rows:
        case,mode=row['case'],row['mode'];run=run_root/(case+'-'+mode);state=read(run/'status.json');metrics=read(run/'metrics.json');prepared=a.base/case/'prepared-pilot';meta=verify_inputs(prepared)
        assert state['status']=='complete' and state['use_reference']==(mode=='with-reference')
        assert state['profile']==meta['profile'] and meta['profile']['num_frames']==17
        assert state['output_sha256']==sha(run/'edited.mp4') and state['video_info']==video_info(run/'edited.mp4')
        assert metrics['frames']==17 and metrics['mean']==row['metrics']
        for name,digest in metrics['output_png_sha256'].items():assert sha(run/'frames'/name)==digest
        records.append(dict(row,profile=state['profile'],output_sha256=state['output_sha256'],code=state['code'],model_identity_sha256=state['model_identity_sha256'],prepared_sha256=sha(prepared/'prepared.json'),metrics_sha256=sha(run/'metrics.json'),checkpoint_io=state['checkpoint_io']))
    report=a.report;report.mkdir(parents=True,exist_ok=False)
    save(report/'results.json',{'status':'complete','finished':overall['finished'],'records':records,'scope':'Two DAVIS research sequences, one seed, 17 frames per output; diagnostics only; human video review pending.'})
    save(report/'protocol.json',read(run_root/'protocol.json'))
    lines=['# 汽车广告视频背景编辑：初始参考图对比','','实验于 2026-10-22 22:46（北京时间）完成。四组全部生成并通过输入、输出及原始帧 SHA256 校验。','','448×256，17 帧，8 fps，20 步，seed=42；两段素材各比较有/无参考图。未训练 LoRA。','','| 素材 | 参考图 | 保留区域 MAE↓ | 背景变化 | 耗时（秒） | 峰值显存（GiB） |','|---|---|---:|---:|---:|---:|']
    for row in rows:lines.append('| '+row['case']+' | '+row['mode']+' | '+format(row['metrics']['protected_pixel_mae'],'.6f')+' | '+format(row['metrics']['background_pixel_mae'],'.6f')+' | '+format(row['seconds'],'.2f')+' | '+format(row['peak_allocated_gib'],'.3f')+' |')
    lines+=['','MAE 按 RGB 像素差归一化至 0–1。保留区域由编辑掩码的补集定义，也包含缩放填充区域；它不是纯车身语义相似度。背景变化只说明画面发生变化，不代表符合指令。','', '## 结论与可见问题','','参考图效果需逐素材判断，不能用这两个案例、一个种子宣称稳定提升。car-turn 的保留区域 MAE 从 0.112433 降至 0.106875，约降低 4.94%；该数值不等于整体生成质量提升。','','助手已检查两段素材的第 0、8、16 帧对照图：均生成海岸道路背景，车辆大体位置与姿态跟随输入；car-roundabout 尾部及轮廓存在变化，car-turn 车头、格栅和轮胎细节存在较明显改变。抽帧检查不能替代完整视频播放与人工评审；时序稳定性结论待补。','','## 下一步','','优先在开发素材上定位车身细节与边界漂移，固定种子对比控制强度和掩码边界策略；保留 car-turn，不用其反复调参。随后扩大到独立拍摄组和多个种子，再决定是否需要 VACE 控制分支 LoRA。训练需要匹配目标场景的正确监督，不能将海岸提示词与原始城市视频配成错误标签。','','源代码、协议与数值已公开；输入素材及原始输出在 Windows 运行目录中保留，未将数据许可等同于商业使用许可。']
    (report/'README.md').write_text(chr(10).join(lines)+chr(10),encoding='utf-8')
    print('REPORT_VERIFIED',report)
    for row in rows:print(row['case'],row['mode'],row['metrics'],round(row['seconds'],2))
if __name__=='__main__':main()
