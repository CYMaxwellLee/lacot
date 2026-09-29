# v2 親跑證據

u 編碼想像軌跡。以下均親跑、CPU、指定 Python；未修改唯讀 codebase、未用 GPU、未跑真訓練。

wiring PASS 是提案 fake 合體；current FAIL 是執行真 inline branch 和真兩 actor 的語意失敗；reference PASS 是小模型構造，三者不互相替代。

策略無關 gate 與 b-consistency 分檔。R2 也親跑，不只是列未來命令。8份 production 材料前後 SHA256 相同（manifest.json），不宣稱已hash整個repo。

## smoke_wiring

命令：`/home/cymaxwelllee/Projects/lacot/.venv/bin/python3 -B integration/smoke_wiring.py`；exit=0，7.66s。

```text
EXAM_SELF_CHECK (fixtures only, no worker completion)
PASS objective: C1 mixed dtype/autocast/nonfinite; target/noise/R0/1/3/full BPTT/cond
PASS independent mod-objective: identical contract, /tmp relocation, stub fails closed
EXAM_SELF_CHECK (fixtures only, no worker completion)
PASS composition: total, anchor-gradient attribution, clean/sample separation, R2
PASS independent mod-actor: identical contract, /tmp relocation, stub fails closed
EXAM_SELF_CHECK (fixtures only, no worker completion)
PASS oracle: same-plan labels, frozen differentiable decoder, no-grad exposure gauge
PASS independent mod-oracle: identical contract, /tmp relocation, stub fails closed
EXAM_SELF_CHECK (fixtures only, no worker completion)
PASS acceptance schema: identity, bad scaling, mode collapse, exposure failure rejected
CAUGHT total-missing-quality: total includes all weighted terms
CAUGHT wrong-u_target: quality recurrence/target/noise/rounds
CAUGHT ignore-rounds: quality recurrence/target/noise/rounds
CAUGHT zero-noise: quality recurrence/target/noise/rounds
CAUGHT truncated-BPTT: full BPTT gradient
CAUGHT detach-cond: cond gradient
PASS 6/6 mutation kill matrix
PASS independent mod-verification: identical contract, /tmp relocation, stub fails closed
PASS wiring M4 oracle -> M1 objective -> M2 composition -> M3 observed gauge
PASS smoke_wiring: four zero-wait exams; reference wiring only, not production repair
```

## current_mainline

命令：`/home/cymaxwelllee/Projects/lacot/.venv/bin/python3 -B integration/integration_test.py`；exit=1，2.86s。

```text
mode=CURRENT_MAINLINE CPU
inline_source=/home/cymaxwelllee/Projects/lacot/experiments/scratch_lacot_rollout.py lines=(1720, 1734) total_line=1779 sha256=21f4cf306a2987241114fe41339eb830f88249bf3fa565dab35fe69016fa6705
FAIL inline-F1-label-isolation: gradient delta=1.87543e-07
PASS LaCoTActor-F1-label-isolation: gradient delta=0
PASS LaCoTActorState-F1-label-isolation: gradient delta=0
TRAIN {"correction_ratio": 1.2617127769317467, "exposure": {"exposure/anchor_mse": 0.0005146974581293762, "exposure/r0_gap": 0.004986802872736007, "exposure/r0_mse": 0.005501500330865383, "exposure/r1_gap": 0.00557324179681018, "exposure/r1_mse": 0.006087939254939556, "exposure/r2_gap": 0.004558348387945443, "exposure/r2_mse": 0.0050730458460748196, "exposure/r3_gap": 0.003789918322581798, "exposure/r3_mse": 0.004304615780711174, "exposure/valid": true}, "mode_error": [0.0, 0.509765625, 0.396484375, 0.396484375, 0.384765625, 0.369140625, 0.36328125], "quality": [1.3734239339828491, 2.078238010406494, 1.8623281717300415, 1.7328665256500244, 1.625396728515625, 1.5315653085708618, 1.4513981342315674], "seed": 0, "steps": 600}
FAIL off-manifold-seed0: R0/1/3=1.37342/2.07824/1.73287, mode_error=0.39648
PASS exposure-seed0: {'exposure/anchor_mse': 0.0005146974581293762, 'exposure/valid': True, 'exposure/r0_mse': 0.005501500330865383, 'exposure/r0_gap': 0.004986802872736007, 'exposure/r1_mse': 0.006087939254939556, 'exposure/r1_gap': 0.00557324179681018, 'exposure/r2_mse': 0.0050730458460748196, 'exposure/r2_gap': 0.004558348387945443, 'exposure/r3_mse': 0.004304615780711174, 'exposure/r3_gap': 0.003789918322581798}
TRAIN {"correction_ratio": 1.2389223537892433, "exposure": {"exposure/anchor_mse": 0.000569941068533808, "exposure/r0_gap": 0.00539628037950024, "exposure/r0_mse": 0.005966221448034048, "exposure/r1_gap": 0.0059416842996142805, "exposure/r1_mse": 0.0065116253681480885, "exposure/r2_gap": 0.004772765503730625, "exposure/r2_mse": 0.005342706572264433, "exposure/r3_gap": 0.0038999365060590208, "exposure/r3_mse": 0.004469877574592829, "exposure/valid": true}, "mode_error": [0.0, 0.322265625, 0.310546875, 0.314453125, 0.314453125, 0.318359375, 0.31640625], "quality": [1.3692772388458252, 1.9932351112365723, 1.8212519884109497, 1.6964281797409058, 1.591846227645874, 1.5042630434036255, 1.4316984415054321], "seed": 1, "steps": 600}
FAIL off-manifold-seed1: R0/1/3=1.36928/1.99324/1.69643, mode_error=0.31445
PASS exposure-seed1: {'exposure/anchor_mse': 0.000569941068533808, 'exposure/valid': True, 'exposure/r0_mse': 0.005966221448034048, 'exposure/r0_gap': 0.00539628037950024, 'exposure/r1_mse': 0.0065116253681480885, 'exposure/r1_gap': 0.0059416842996142805, 'exposure/r2_mse': 0.005342706572264433, 'exposure/r2_gap': 0.004772765503730625, 'exposure/r3_mse': 0.004469877574592829, 'exposure/r3_gap': 0.0038999365060590208}
TRAIN {"correction_ratio": 1.2299911597962312, "exposure": {"exposure/anchor_mse": 0.0006446379702538252, "exposure/r0_gap": 0.007135323015972972, "exposure/r0_mse": 0.007779960986226797, "exposure/r1_gap": 0.007682471768930554, "exposure/r1_mse": 0.00832710973918438, "exposure/r2_gap": 0.006546041229739785, "exposure/r2_mse": 0.00719067919999361, "exposure/r3_gap": 0.005409172968938947, "exposure/r3_mse": 0.006053810939192772, "exposure/valid": true}, "mode_error": [0.0, 0.537109375, 0.302734375, 0.427734375, 0.30859375, 0.44921875, 0.314453125], "quality": [1.535660743713379, 2.1686463356018066, 2.0197596549987793, 1.888849139213562, 1.7545256614685059, 1.6589395999908447, 1.563921570777893], "seed": 2, "steps": 600}
FAIL off-manifold-seed2: R0/1/3=1.53566/2.16865/1.88885, mode_error=0.42773
PASS exposure-seed2: {'exposure/anchor_mse': 0.0006446379702538252, 'exposure/valid': True, 'exposure/r0_mse': 0.007779960986226797, 'exposure/r0_gap': 0.007135323015972972, 'exposure/r1_mse': 0.00832710973918438, 'exposure/r1_gap': 0.007682471768930554, 'exposure/r2_mse': 0.00719067919999361, 'exposure/r2_gap': 0.006546041229739785, 'exposure/r3_mse': 0.006053810939192772, 'exposure/r3_gap': 0.005409172968938947}
PASS identity-must-fail: identity states measured at R0 cost for every round; correction_ratio=1
FAIL integration: 4 failed checks
```

stderr（包含 PyTorch nested-tensor 建議，非失敗原因）：
```text
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
```

## proposal_reference

命令：`/home/cymaxwelllee/Projects/lacot/.venv/bin/python3 -B integration/integration_test.py --reference`；exit=0，3.37s。

```text
mode=PROPOSAL_REFERENCE CPU
inline_source=/home/cymaxwelllee/Projects/lacot/experiments/scratch_lacot_rollout.py lines=(1720, 1734) total_line=1779 sha256=21f4cf306a2987241114fe41339eb830f88249bf3fa565dab35fe69016fa6705
PASS inline-F1-label-isolation: gradient delta=0
PASS LaCoTActor-F1-label-isolation: gradient delta=0
PASS LaCoTActorState-F1-label-isolation: gradient delta=0
TRAIN {"correction_ratio": 0.0003304013791335517, "exposure": {"exposure/anchor_mse": 2.2737367544323206e-13, "exposure/r0_gap": -7.631211201147847e-15, "exposure/r0_mse": 2.197424642420842e-13, "exposure/r1_gap": 6.175615574477433e-16, "exposure/r1_mse": 2.279912370006798e-13, "exposure/r2_gap": 6.952771691715043e-15, "exposure/r2_mse": 2.343264471349471e-13, "exposure/r3_gap": 6.6405214660392176e-15, "exposure/r3_mse": 2.340141969092713e-13, "exposure/valid": true}, "mode_error": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "quality": [1.3734239339828491, 0.0037182834930717945, 0.0005051918560639024, 0.0004537811619229615, 0.00045439571840688586, 0.00045432604383677244, 0.00045433183549903333], "seed": 0, "steps": 600}
PASS off-manifold-seed0: R0/1/3=1.37342/0.00372/0.00045, mode_error=0.00000
PASS exposure-seed0: {'exposure/anchor_mse': 2.2737367544323206e-13, 'exposure/valid': True, 'exposure/r0_mse': 2.197424642420842e-13, 'exposure/r0_gap': -7.631211201147847e-15, 'exposure/r1_mse': 2.279912370006798e-13, 'exposure/r1_gap': 6.175615574477433e-16, 'exposure/r2_mse': 2.343264471349471e-13, 'exposure/r2_gap': 6.952771691715043e-15, 'exposure/r3_mse': 2.340141969092713e-13, 'exposure/r3_gap': 6.6405214660392176e-15}
TRAIN {"correction_ratio": 0.0002577303821763022, "exposure": {"exposure/anchor_mse": 1.2789769243681803e-13, "exposure/r0_gap": -1.2398746309164044e-14, "exposure/r0_mse": 1.15498946127654e-13, "exposure/r1_gap": -8.673617379884035e-16, "exposure/r1_mse": 1.2703033069882963e-13, "exposure/r2_gap": 0.0, "exposure/r2_mse": 1.2789769243681803e-13, "exposure/r3_gap": 0.0, "exposure/r3_mse": 1.2789769243681803e-13, "exposure/valid": true}, "mode_error": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "quality": [1.3692772388458252, 0.00575296301394701, 0.00037594258901663125, 0.0003529043460730463, 0.00035284398472867906, 0.0003528414526954293, 0.0003528415982145816], "seed": 1, "steps": 600}
PASS off-manifold-seed1: R0/1/3=1.36928/0.00575/0.00035, mode_error=0.00000
PASS exposure-seed1: {'exposure/anchor_mse': 1.2789769243681803e-13, 'exposure/valid': True, 'exposure/r0_mse': 1.15498946127654e-13, 'exposure/r0_gap': -1.2398746309164044e-14, 'exposure/r1_mse': 1.2703033069882963e-13, 'exposure/r1_gap': -8.673617379884035e-16, 'exposure/r2_mse': 1.2789769243681803e-13, 'exposure/r2_gap': 0.0, 'exposure/r3_mse': 1.2789769243681803e-13, 'exposure/r3_gap': 0.0}
TRAIN {"correction_ratio": 0.00019395529477204572, "exposure": {"exposure/anchor_mse": 2.2737367544323206e-13, "exposure/r0_gap": -3.755866060869972e-15, "exposure/r0_mse": 2.2361780938236209e-13, "exposure/r1_gap": -7.632783294297951e-17, "exposure/r1_mse": 2.2729734761028908e-13, "exposure/r2_gap": 0.0, "exposure/r2_mse": 2.2737367544323206e-13, "exposure/r3_gap": 0.0, "exposure/r3_mse": 2.2737367544323206e-13, "exposure/valid": true}, "mode_error": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "quality": [1.535660743713379, 0.007151635363698006, 0.0003347531892359257, 0.00029784953221678734, 0.0002989885106217116, 0.0002988604246638715, 0.0002988722699228674], "seed": 2, "steps": 600}
PASS off-manifold-seed2: R0/1/3=1.53566/0.00715/0.00030, mode_error=0.00000
PASS exposure-seed2: {'exposure/anchor_mse': 2.2737367544323206e-13, 'exposure/valid': True, 'exposure/r0_mse': 2.2361780938236209e-13, 'exposure/r0_gap': -3.755866060869972e-15, 'exposure/r1_mse': 2.2729734761028908e-13, 'exposure/r1_gap': -7.632783294297951e-17, 'exposure/r2_mse': 2.2737367544323206e-13, 'exposure/r2_gap': 0.0, 'exposure/r3_mse': 2.2737367544323206e-13, 'exposure/r3_gap': 0.0}
PASS identity-must-fail: identity states measured at R0 cost for every round; correction_ratio=1
PASS integration: 0 failed checks
```

stderr（包含 PyTorch nested-tensor 建議，非失敗原因）：
```text
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
```

## b_consistency

命令：`/home/cymaxwelllee/Projects/lacot/.venv/bin/python3 -B integration/consistency_test.py`；exit=0，0.80s。

```text
PASS objective: C1 mixed dtype/autocast/nonfinite; target/noise/R0/1/3/full BPTT/cond
PASS composition: total, anchor-gradient attribution, clean/sample separation, R2
PASS F5 real RefineOperator R1 stale EMA target -> nonzero student gradient; teacher detached
NOTE synchronized teacher/student may have zero consistency gradient; quality must teach first
```

## six_mutations

命令：`/home/cymaxwelllee/Projects/lacot/.venv/bin/python3 -B integration/mutations.py`；exit=0，0.79s。

```text
CAUGHT total-missing-quality: total includes all weighted terms
CAUGHT wrong-u_target: quality recurrence/target/noise/rounds
CAUGHT ignore-rounds: quality recurrence/target/noise/rounds
CAUGHT zero-noise: quality recurrence/target/noise/rounds
CAUGHT truncated-BPTT: full BPTT gradient
CAUGHT detach-cond: cond gradient
PASS 6/6 mutation kill matrix
```

## b1_baselines

命令：`/home/cymaxwelllee/Projects/lacot/.venv/bin/python3 -B integration/b1_baselines.py`；exit=0，1.98s。

```text
B1 KD=1024 sigma=0.02 box_probability=2.7689971e-06 logP=-12.7970
B1 KD=1024 sigma=0.05 box_probability=1.7562063e-170 logP=-390.8763
B1 KD=1024 sigma=0.1 box_probability=0 logP=-982.9543
PASS B1 old population optimum correction_ratio=0.999992; identity rejected by same gate
BASELINES [{"compute": {"flow": 1, "refine": 0}, "cost": 1.3734239339828491, "method": "learned-R0", "positive_mode_fraction": 0.509765625, "seconds_excluding_shared_initial_flow": 5.259993486106396e-06}, {"compute": {"flow": 1, "refine": 1}, "cost": 0.0037182834930717945, "method": "learned-R1", "positive_mode_fraction": 0.509765625, "seconds_excluding_shared_initial_flow": 5.403999239206314e-05}, {"compute": {"flow": 1, "refine": 3}, "cost": 0.0004537811619229615, "method": "learned-R3", "positive_mode_fraction": 0.509765625, "seconds_excluding_shared_initial_flow": 0.00012556900037452579}, {"compute": {"decoder_backward": 3, "decoder_forward": 3, "flow": 1}, "cost": 0.02145974710583687, "method": "gradient-3", "positive_mode_fraction": 0.509765625, "seconds_excluding_shared_initial_flow": 0.0003840560093522072}, {"compute": {"decoder_forward": 4, "flow": 4}, "cost": 0.6880272626876831, "method": "best-of-4-same-mode", "positive_mode_fraction": 0.509765625, "seconds_excluding_shared_initial_flow": 0.0009747300064191222}]
NOTE toy oracle exact; BoN mode mask uses toy branch identity. Production BoN uses qualified quality scorer and must also report unrestricted mode coverage.
PASS B1/scaling/baselines probe (no universal learned-vs-selection claim)
```

stderr（包含 PyTorch nested-tensor 建議，非失敗原因）：
```text
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
```

## r2_zero_rounds

命令：`/home/cymaxwelllee/Projects/lacot/.venv/bin/python3 -B /home/cymaxwelllee/Projects/lacot/tests_repair/test_model_zero_rounds.py`；exit=0，0.72s。

```text
```

stderr（包含 PyTorch nested-tensor 建議，非失敗原因）：
```text
.
----------------------------------------------------------------------
Ran 1 test in 0.001s

OK
```

## real_operator_diagnostic

命令：`/home/cymaxwelllee/Projects/lacot/.venv/bin/python3 -B integration/real_operator_training.py`；exit=1，3.60s。

```text
{"seed": 0, "optimizer_steps": 700, "quality": [1.1944732666015625, 0.30660897493362427, 0.2959427535533905, 0.29581576585769653, 0.29581835865974426, 0.2958185076713562, 0.2958185076713562], "mode_error": [0.0, 0.083984375, 0.083984375, 0.083984375, 0.083984375, 0.083984375, 0.083984375]}
{"seed": 1, "optimizer_steps": 700, "quality": [1.1403101682662964, 0.04808411747217178, 0.04695659130811691, 0.04701881483197212, 0.04702075943350792, 0.04702082276344299, 0.04702082648873329], "mode_error": [0.0, 0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875]}
{"seed": 2, "optimizer_steps": 700, "quality": [1.193985939025879, 0.06680600345134735, 0.06151584908366203, 0.0614955760538578, 0.061495739966630936, 0.061495743691921234, 0.061495739966630936], "mode_error": [0.0, 0.015625, 0.015625, 0.015625, 0.015625, 0.015625, 0.015625]}
FAIL real RefineOperator analytic-world off-manifold CPU training; no feedback features added
```

stderr（包含 PyTorch nested-tensor 建議，非失敗原因）：
```text
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
```

## m3_independent

命令：`/home/cymaxwelllee/Projects/lacot/.venv/bin/python3 -B mod-verification/integration/integration_test.py --reference`；exit=0，3.36s。

```text
mode=PROPOSAL_REFERENCE CPU
inline_source=/home/cymaxwelllee/Projects/lacot/experiments/scratch_lacot_rollout.py lines=(1720, 1734) total_line=1779 sha256=21f4cf306a2987241114fe41339eb830f88249bf3fa565dab35fe69016fa6705
PASS inline-F1-label-isolation: gradient delta=0
PASS LaCoTActor-F1-label-isolation: gradient delta=0
PASS LaCoTActorState-F1-label-isolation: gradient delta=0
TRAIN {"correction_ratio": 0.0003304013791335517, "exposure": {"exposure/anchor_mse": 2.2737367544323206e-13, "exposure/r0_gap": -7.631211201147847e-15, "exposure/r0_mse": 2.197424642420842e-13, "exposure/r1_gap": 6.175615574477433e-16, "exposure/r1_mse": 2.279912370006798e-13, "exposure/r2_gap": 6.952771691715043e-15, "exposure/r2_mse": 2.343264471349471e-13, "exposure/r3_gap": 6.6405214660392176e-15, "exposure/r3_mse": 2.340141969092713e-13, "exposure/valid": true}, "mode_error": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "quality": [1.3734239339828491, 0.0037182834930717945, 0.0005051918560639024, 0.0004537811619229615, 0.00045439571840688586, 0.00045432604383677244, 0.00045433183549903333], "seed": 0, "steps": 600}
PASS off-manifold-seed0: R0/1/3=1.37342/0.00372/0.00045, mode_error=0.00000
PASS exposure-seed0: {'exposure/anchor_mse': 2.2737367544323206e-13, 'exposure/valid': True, 'exposure/r0_mse': 2.197424642420842e-13, 'exposure/r0_gap': -7.631211201147847e-15, 'exposure/r1_mse': 2.279912370006798e-13, 'exposure/r1_gap': 6.175615574477433e-16, 'exposure/r2_mse': 2.343264471349471e-13, 'exposure/r2_gap': 6.952771691715043e-15, 'exposure/r3_mse': 2.340141969092713e-13, 'exposure/r3_gap': 6.6405214660392176e-15}
TRAIN {"correction_ratio": 0.0002577303821763022, "exposure": {"exposure/anchor_mse": 1.2789769243681803e-13, "exposure/r0_gap": -1.2398746309164044e-14, "exposure/r0_mse": 1.15498946127654e-13, "exposure/r1_gap": -8.673617379884035e-16, "exposure/r1_mse": 1.2703033069882963e-13, "exposure/r2_gap": 0.0, "exposure/r2_mse": 1.2789769243681803e-13, "exposure/r3_gap": 0.0, "exposure/r3_mse": 1.2789769243681803e-13, "exposure/valid": true}, "mode_error": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "quality": [1.3692772388458252, 0.00575296301394701, 0.00037594258901663125, 0.0003529043460730463, 0.00035284398472867906, 0.0003528414526954293, 0.0003528415982145816], "seed": 1, "steps": 600}
PASS off-manifold-seed1: R0/1/3=1.36928/0.00575/0.00035, mode_error=0.00000
PASS exposure-seed1: {'exposure/anchor_mse': 1.2789769243681803e-13, 'exposure/valid': True, 'exposure/r0_mse': 1.15498946127654e-13, 'exposure/r0_gap': -1.2398746309164044e-14, 'exposure/r1_mse': 1.2703033069882963e-13, 'exposure/r1_gap': -8.673617379884035e-16, 'exposure/r2_mse': 1.2789769243681803e-13, 'exposure/r2_gap': 0.0, 'exposure/r3_mse': 1.2789769243681803e-13, 'exposure/r3_gap': 0.0}
TRAIN {"correction_ratio": 0.00019395529477204572, "exposure": {"exposure/anchor_mse": 2.2737367544323206e-13, "exposure/r0_gap": -3.755866060869972e-15, "exposure/r0_mse": 2.2361780938236209e-13, "exposure/r1_gap": -7.632783294297951e-17, "exposure/r1_mse": 2.2729734761028908e-13, "exposure/r2_gap": 0.0, "exposure/r2_mse": 2.2737367544323206e-13, "exposure/r3_gap": 0.0, "exposure/r3_mse": 2.2737367544323206e-13, "exposure/valid": true}, "mode_error": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "quality": [1.535660743713379, 0.007151635363698006, 0.0003347531892359257, 0.00029784953221678734, 0.0002989885106217116, 0.0002988604246638715, 0.0002988722699228674], "seed": 2, "steps": 600}
PASS off-manifold-seed2: R0/1/3=1.53566/0.00715/0.00030, mode_error=0.00000
PASS exposure-seed2: {'exposure/anchor_mse': 2.2737367544323206e-13, 'exposure/valid': True, 'exposure/r0_mse': 2.2361780938236209e-13, 'exposure/r0_gap': -3.755866060869972e-15, 'exposure/r1_mse': 2.2729734761028908e-13, 'exposure/r1_gap': -7.632783294297951e-17, 'exposure/r2_mse': 2.2737367544323206e-13, 'exposure/r2_gap': 0.0, 'exposure/r3_mse': 2.2737367544323206e-13, 'exposure/r3_gap': 0.0}
PASS identity-must-fail: identity states measured at R0 cost for every round; correction_ratio=1
PASS integration: 0 failed checks
```

stderr（包含 PyTorch nested-tensor 建議，非失敗原因）：
```text
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
```

## 已知限制及失敗也留存

raw latent MLP 的300-step與600-step三seed各有seed1 FAIL，見 evidence/try-reference.txt、try-reference-600.txt。加入toy可解地形的sign feedback後通過；不能據此宣稱現有production RefineOperator已能學會。teacher/decoder/action inverse皆為明示玩具解析世界；真maze qualification仍是worker工作。

補跑真 RefineOperator（4維、hidden32、三seed各700步）也有seed0未過品質/模式門檻，real_operator_diagnostic exit=1原樣保留。這是研究診斷FAIL，不是修復PASS；本輪提案不宣稱production已學會。

current inline 選現有 CONS=ema，因此 F1 FAIL 即使已有EMA仍發生；F5不是這次頂層FAIL的必要理由。current訓練toy600×3步；reference亦600×3，另M3搬移驗同批設定；flow只抽樣、未訓練。BoN/GD成本明列，沒有宣稱計算預算相同或learned普遍更好。

STATUS: DONE（v2設計／考場／證據；未裁案、未合併production）。
VERIFIED: wiring PASS；current mainline top-level FAIL；reference、six mutations、B1/baselines、EMA、R2、M3獨立harness PASS；真RefineOperator研究診斷FAIL。
