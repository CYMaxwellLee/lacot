"""u 編碼想像軌跡。CPU evidence runner; only writes this proposal."""
from pathlib import Path
import subprocess,os,hashlib,json,datetime,time
ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/cymaxwelllee/Projects/lacot')
PY=str(REPO/'.venv/bin/python3');OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
materials=['experiments/scratch_lacot_rollout.py','lacot/model.py','lacot/nf_head.py','lacot/traj_decoder.py','lacot/refine_grad.py','tests_repair/test_model_zero_rounds.py','docs/CANON-u-semantics.md','docs/LaCoT-NF-latent-planning-design.md']
def hashes():return {f:hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in materials}
before=hashes();records=[]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
jobs=[('smoke_wiring',['integration/smoke_wiring.py'],0),
      ('current_mainline',['integration/integration_test.py'],1),
      ('proposal_reference',['integration/integration_test.py','--reference'],0),
      ('b_consistency',['integration/consistency_test.py'],0),
      ('six_mutations',['integration/mutations.py'],0),
      ('b1_baselines',['integration/b1_baselines.py'],0),
      ('r2_zero_rounds',[str(REPO/'tests_repair/test_model_zero_rounds.py')],0),
      ('real_operator_diagnostic',['integration/real_operator_training.py'],1),
      ('m3_independent',['mod-verification/integration/integration_test.py','--reference'],0)]
for name,args,expected in jobs:
    command=[PY,'-B',*args];start=time.perf_counter()
    result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    elapsed=time.perf_counter()-start
    for kind,value in [('stdout',result.stdout),('stderr',result.stderr),('exit',str(result.returncode)+'\n')]:
        (OUT/f'{name}.{kind}.txt').write_text(value)
    record=dict(name=name,command=command,exit=result.returncode,expected=expected,seconds=elapsed)
    records.append(record)
    print(json.dumps(record),flush=True)
    assert result.returncode==expected,(name,result.stdout,result.stderr)
    if name=='current_mainline':
        assert 'FAIL inline-F1-label-isolation' in result.stdout
        assert 'FAIL off-manifold-seed' in result.stdout
        assert 'Traceback' not in result.stderr
    if name=='six_mutations':assert result.stdout.count('CAUGHT ')==6
    if name=='real_operator_diagnostic':
        assert 'FAIL real RefineOperator' in result.stdout and 'Traceback' not in result.stderr
after=hashes();assert before==after
manifest=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    python=PY,device='CPU',gpu_used=False,production_training=False,
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
    materials_before=before,materials_after=after,source_unchanged=True,records=records,
    training=dict(refiner='24 hidden + explicit toy decoder sign feedback',flow='frozen real Flow 1 block hidden8',
                  final_steps_per_seed=600,seeds=[0,1,2],real_operator_diagnostic_steps_per_seed=700,raw_feature_failed_attempts=['try-reference.txt','try-reference-600.txt']))
manifest['proposal_code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py')}
manifest['raw_output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.txt')}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
doc=['# v2 親跑證據\n','u 編碼想像軌跡。以下均親跑、CPU、指定 Python；未修改唯讀 codebase、未用 GPU、未跑真訓練。\n',
     'wiring PASS 是提案 fake 合體；current FAIL 是執行真 inline branch 和真兩 actor 的語意失敗；reference PASS 是小模型構造，三者不互相替代。\n',
     '策略無關 gate 與 b-consistency 分檔。R2 也親跑，不只是列未來命令。8份 production 材料前後 SHA256 相同（manifest.json），不宣稱已hash整個repo。\n']
for record in records:
    name=record['name']
    doc += [f'## {name}\n',f"命令：`{' '.join(record['command'])}`；exit={record['exit']}，{record['seconds']:.2f}s。\n",
            '```text\n'+(OUT/f'{name}.stdout.txt').read_text()+'```\n']
    err=(OUT/f'{name}.stderr.txt').read_text()
    if err:doc+=['stderr（包含 PyTorch nested-tensor 建議，非失敗原因）：\n```text\n'+err+'```\n']
doc += ['## 已知限制及失敗也留存\n',
        'raw latent MLP 的300-step與600-step三seed各有seed1 FAIL，見 evidence/try-reference.txt、try-reference-600.txt。加入toy可解地形的sign feedback後通過；不能據此宣稱現有production RefineOperator已能學會。teacher/decoder/action inverse皆為明示玩具解析世界；真maze qualification仍是worker工作。\n',
        '補跑真 RefineOperator（4維、hidden32、三seed各700步）也有seed0未過品質/模式門檻，real_operator_diagnostic exit=1原樣保留。這是研究診斷FAIL，不是修復PASS；本輪提案不宣稱production已學會。\n',
        'current inline 選現有 CONS=ema，因此 F1 FAIL 即使已有EMA仍發生；F5不是這次頂層FAIL的必要理由。current訓練toy600×3步；reference亦600×3，另M3搬移驗同批設定；flow只抽樣、未訓練。BoN/GD成本明列，沒有宣稱計算預算相同或learned普遍更好。\n',
        'STATUS: DONE（v2設計／考場／證據；未裁案、未合併production）。\nVERIFIED: wiring PASS；current mainline top-level FAIL；reference、six mutations、B1/baselines、EMA、R2、M3獨立harness PASS；真RefineOperator研究診斷FAIL。\n']
(ROOT/'EVIDENCE.md').write_text('\n'.join(doc))
print('PASS evidence complete; 8 read-only source hashes unchanged')
