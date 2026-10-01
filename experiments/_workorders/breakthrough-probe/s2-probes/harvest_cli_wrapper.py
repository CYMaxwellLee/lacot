import sys, os, json
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
MIR=SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'
sys.path.insert(0,MIR); sys.dont_write_bytecode=True
os.environ['MUJOCO_GL']='egl'
import harvest
fps={tuple(map(int,k.split(','))):tuple(v) for k,v in json.load(open(SCR+'/n64_fingerprints.json')).items()}
harvest.historical_fingerprints=lambda gate: fps
sys.argv=['harvest.py']+sys.argv[1:]
harvest.main()
