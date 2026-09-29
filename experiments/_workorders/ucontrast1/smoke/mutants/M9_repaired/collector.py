"""Instrumentation for the existing rollout; deliberately contains no env loop."""
import hashlib
import json
import math
import os
from pathlib import Path


def digest_array(value):
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


class Collector:
    def __init__(self, arm, sigma, draws, episode_offset=0, stream_salt=0):
        import numpy as np
        import torch
        if arm not in ("A", "B") or draws < 1 or not math.isfinite(sigma) or sigma < 0:
            raise ValueError("Invalid oracle arm/sigma/draw count")
        if arm == "A" and sigma != 0:
            raise ValueError("A must have sigma=0")
        self.np, self.torch = np, torch
        self.arm, self.sigma, self.draws = arm, sigma, draws
        self.episode_offset, self.stream_salt = episode_offset, stream_salt
        self.rows = []

    def reset_seed(self, task, episode):
        return 1000 * task + episode + self.stream_salt

    def begin(self, task, episode, draw, obs, goal):
        self.key = (task, episode)
        self.draw = draw
        # Same historical per-episode torch seed for draw 0. Other draws change
        # only sampler stream; env.reset seed stays fixed across all arms/draws.
        self.base_seed = 7 * task + episode + self.stream_salt
        self.stream_seed = self.base_seed + 1_000_003 * draw
        # B' replans every chunk, replaying draw-0's flow stream each draw.
        # The draw-specific stream remains dedicated to paired action noise.
        self.flow_seed = self.base_seed if self.arm == "B" else self.stream_seed
        self.torch.manual_seed(self.flow_seed)
        # Dedicated generator: action noise cannot advance the flow RNG.
        # Same seed and generator in A/B/all sigma bins (paired common stream).
        self.noise = self.torch.Generator(device="cpu").manual_seed(self.stream_seed)
        self.trace = hashlib.sha256()
        self.u_trace = hashlib.sha256()
        self.first_u = None
        self.u_calls, self.u_samples = 0, 0
        self.initial = digest_array(self.np.asarray(obs))
        self.goal = digest_array(self.np.asarray(goal))
        self.trace.update(bytes.fromhex(self.initial))
        self.trace.update(bytes.fromhex(self.goal))

    def get_u(self, sample):
        u = sample()
        self.u_samples += 1
        uh = digest_array(u.detach().cpu().numpy())
        self.first_u = self.first_u or uh
        self.u_trace.update(bytes.fromhex(uh))
        self.u_calls += 1
        return u

    def action(self, action):
        # Noise is applied to the policy's clipped action immediately before
        # env.step, then clipped again to the SAME [-1,1] action range.
        eps = self.torch.randn(action.shape, generator=self.noise).numpy()
        if self.arm == "B" and self.sigma > 0:
            return self.np.clip(action + self.sigma * eps, -1., 1.).astype(self.np.float32)
        return action

    def step(self, action, obs, reward, terminated, truncated, reached):
        for value in (self.np.asarray(action), self.np.asarray(obs)):
            self.trace.update(bytes.fromhex(digest_array(value)))
        self.trace.update(json.dumps([float(reward), bool(terminated),
                                      bool(truncated), bool(reached)]).encode())

    def end(self, success, steps):
        self.rows.append(dict(task=self.key[0], episode=self.key[1], draw=self.draw,
                              env_seed=self.reset_seed(*self.key),
                              stream_seed=self.stream_seed, flow_seed=self.flow_seed,
                              initial_sha256=self.initial,
                              goal_sha256=self.goal, success=bool(success), steps=steps,
                              first_u_sha256=self.first_u, u_sequence_sha256=self.u_trace.hexdigest(),
                              u_calls=self.u_calls, u_samples=self.u_samples,
                              trajectory_sha256=self.trace.hexdigest()))


def summarize(rows, draws, arm, sigma):
    """R1 reached bits only. Prefix oracle, never preselection/completed."""
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["episode"]), []).append(row)
    if not groups:
        raise ValueError("No completed rollouts")
    tasks = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: r["draw"])
        if [r["draw"] for r in group] != list(range(draws)):
            raise ValueError(f"Missing/duplicate draw: {key}")
        for field in ("env_seed", "initial_sha256", "goal_sha256"):
            if len({r[field] for r in group}) != 1:
                raise ValueError(f"Unpaired resets: {key} {field}")
        if arm == "B":
            if len({r["first_u_sha256"] for r in group}) != 1:
                raise ValueError(f"B first u differs across draws: {key}")
            if any(r["u_samples"] != r["u_calls"] for r in group):
                raise ValueError(f"B skipped chunk replanning: {key}")
            if len({r["flow_seed"] for r in group}) != 1:
                raise ValueError(f"B flow stream differs across draws: {key}")
            if sigma == 0:
                for field in ("trajectory_sha256", "success", "steps", "u_sequence_sha256"):
                    if len({r[field] for r in group}) != 1:
                        raise ValueError(f"Sigma-zero mutant differs: {key} {field}")
        bits = [int(r["success"]) for r in group]
        tasks.append(dict(task=key[0], episode=key[1], success_bits=bits, draws=group))
    n = len(tasks)
    oracle = {str(k): sum(any(t["success_bits"][:k]) for t in tasks) / n
              for k in range(1, draws + 1)}
    if any(oracle[str(k)] > oracle[str(k + 1)] for k in range(1, draws)):
        raise ValueError("oracle@k decreased")
    if arm == "B" and sigma == 0 and oracle[str(draws)] != oracle["1"]:
        raise ValueError("Sigma-zero oracle leak")
    return dict(tasks=tasks, oracle_at_k=oracle, pooled_oracle=oracle[str(draws)],
                per_draw_quality=[sum(t["success_bits"][d] for t in tasks) / n
                                  for d in range(draws)],
                negative_mutant_passed=(arm == "B" and sigma == 0 and draws == 8),
                n_tasks=n, n_draws=len(rows),
                n_success=sum(int(r["success"]) for r in rows),
                n_oracle_success=sum(any(t["success_bits"]) for t in tasks))


def write_result(path, metadata, collector):
    summary = summarize(collector.rows, collector.draws, collector.arm, collector.sigma)
    result = dict(metadata, schema_version=1, arm=collector.arm, sigma=collector.sigma,
                  sigma_units="fraction_of_action_half_range_1", temperature=1.0,
                  draws_per_task=collector.draws, success_definition="R1: any info.success reached",
                  pooling="any draw within each official (task,episode), then mean over episodes",
                  offline_only=True, training_use_prohibited=True,
                  leg_stats=[], leg_stats_reason="flat policy has no subgoal legs",
                  **summary)
    # Completion counts MUST be the final keys in JSON, written only on success.
    counts = {k: result.pop(k) for k in ("n_tasks", "n_draws", "n_success")}
    result.update(counts)
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    return result
