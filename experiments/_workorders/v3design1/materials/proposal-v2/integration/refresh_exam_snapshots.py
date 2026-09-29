"""u 編碼想像軌跡。Proposal-only snapshot copier; no codebase writes."""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
for name in ('objective','actor','verification','oracle'):
    local=root/f'mod-{name}'/'contracts';local.mkdir(exist_ok=True)
    for file in (root/'contracts').glob('*'):
        if file.is_file():shutil.copy2(file,local/file.name)
local=root/'mod-verification'/'integration';local.mkdir(exist_ok=True)
for file in (root/'integration').glob('*.py'):
    if file.name not in ('refresh_exam_snapshots.py','run_evidence.py','smoke_wiring.py'):
        shutil.copy2(file,local/file.name)
print('PASS refreshed four contract snapshots and independent M3 scientific harness')
