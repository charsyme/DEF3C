from def3c_learner import DEF_3C
import argparse
import json
from types import SimpleNamespace
import os

parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, default='./json/gefcom_w_1.json')
args = parser.parse_args()

with open(args.json, 'r') as j:
    params = json.loads(j.read(), object_hook=lambda d: SimpleNamespace(**d))
if not os.path.isdir(params.exp.log_dir):
    os.makedirs(params.exp.log_dir, exist_ok=True)

def_3c = DEF_3C(params)
def_3c.fit()

