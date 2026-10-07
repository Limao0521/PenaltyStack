"""Adversarial pipe peer for bridge failure tests; not a motion backend."""
import json
import sys
import time
from penalty.motion_contract import JOINT_NAMES, SOURCE_COMMIT

mode = sys.argv[1]
def emit(value):
    print(json.dumps(value), flush=True)
emit({'type':'HELLO','protocol':1,'source_commit':SOURCE_COMMIT,
      'cycle_s':.012,'joint_names':JOINT_NAMES,'motions':['stand']})
init = json.loads(sys.stdin.readline())
emit({'type':'READY','trial':init['trial'],'role':init['role']})
step = json.loads(sys.stdin.readline())
if mode == 'timeout':
    time.sleep(10)
else:
    angles = [0.] * 26
    if mode == 'nonfinite':
        angles[0] = float('nan')
    emit({'type':'OUTPUT','trial':step['trial'],
          'sequence':step['sequence'] + (1 if mode == 'sequence' else 0),
          'angles':angles,'stiffness':[100]*26,'phase':'stand','motion':'stand',
          'busy':False,'kick_completed':False,'odometry':[0,0,0]})
    sys.stdin.readline()
