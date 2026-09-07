"""零差測試的 harness：把【全域】np.random 釘死再跑主檔。
⛔ 不動 repo 一行 code —— ogbench 的 maze add_noise 走全域 np.random
   (locomaze/maze.py:565)，而官方 rollout() 只釘 env.reset(seed=) 與 torch
   ⇒ 不釘它的話同一份 code 跑兩次的 rollout 軌跡就不一樣，零差比對會報假警。"""
import sys
import numpy as np
import runpy

np.random.seed(20260907)
_p = sys.argv[1]
sys.argv = [_p]
runpy.run_path(_p, run_name="__main__")
