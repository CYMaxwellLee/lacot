"""Offline-only W_phi and paired-window A_omega (DESIGN-v3, oracle 1--5).

No environment imports. Public tensors use raw observation/action units. Paths
are fixed-point interpolated XY, NOT physical timesteps. All loss reductions
are fp32. Metadata is checked at ingestion and again at each fitting boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import weakref
from pathlib import Path
from collections.abc import Mapping
import numpy as np
import torch
from torch import nn

VERSION = "refine-v3/raw-state-interpolated-xy-v1"
HORIZONS = (4, 16, 32)
META_FIELDS = {"source_kind", "dataset_hash", "episode_split", "parent_model_hash",
               "candidate_id", "noise_seed", "iteration", "domain", "horizon",
               "representation_version", "eval_only", "split_seed", "split_hash",
               "partial_chunks_dropped"}
DEFAULT_SPLIT_SEED = 1729
DATA_MANIFEST = Path(__file__).resolve().parents[1] / "experiments/refine_v3/DATA-MANIFEST.json"
FORBIDDEN = {"reward", "rewards", "success", "successes", "oracle_index",
             "oracle_winner", "rescued", "broken", "eval_results"}
RECORD_FIELDS = {"metadata", "state", "actions", "future", "mask", "path",
                 "episode_id", "row", "path_end", "source_proof"}


class _SourceProof:
    def __deepcopy__(self, memo):
        return self


_verified_proofs = weakref.WeakKeyDictionary()


def reject_eval_fields(value):
    if isinstance(value, Mapping):
        bad = set(value) & FORBIDDEN
        if bad:
            raise ValueError(f"evaluation fields forbidden: {sorted(bad)}")
        for v in value.values():
            reject_eval_fields(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            reject_eval_fields(v)


def validate_metadata(meta, *, split=None, source_proof=None):
    reject_eval_fields(meta)
    if not isinstance(meta, Mapping) or set(meta) != META_FIELDS:
        raise ValueError("metadata schema: missing or unknown fields")
    if meta["eval_only"] is not False:
        raise ValueError("eval_only records cannot enter offline fitting")
    if meta["source_kind"] not in {"offline_dataset", "model_generated", "toy_offline"}:
        raise ValueError("source_kind is not whitelisted")
    if meta["episode_split"] not in {"train", "calibration", "test"}:
        raise ValueError("unknown episode_split")
    if split is not None and meta["episode_split"] != split:
        raise ValueError(f"requires {split} split")
    if meta["domain"] not in {"pointmaze", "ant"} or meta["representation_version"] != VERSION:
        raise ValueError("domain/representation version mismatch")
    for k in ("dataset_hash", "parent_model_hash", "split_hash"):
        v = meta[k]
        if k == "parent_model_hash" and v is None and meta["source_kind"] != "model_generated":
            continue
        if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError(f"{k} must be sha256")
    for k in ("noise_seed", "iteration", "horizon", "split_seed", "partial_chunks_dropped"):
        if type(meta[k]) is not int or meta[k] < (1 if k == "horizon" else 0):
            raise ValueError(f"invalid {k}")
    if meta["candidate_id"] is not None and not isinstance(meta["candidate_id"], str):
        raise ValueError("candidate_id must be string or null")
    if meta["source_kind"] == "model_generated" and not meta["candidate_id"]:
        raise ValueError("generated records need candidate_id")
    expected = (meta["dataset_hash"], meta["domain"], meta["split_hash"], meta["episode_split"])
    if meta["source_kind"] == "offline_dataset":
        if not isinstance(source_proof, _SourceProof) or _verified_proofs.get(source_proof) != expected:
            raise ValueError("offline_dataset provenance requires the verified loader")
    elif source_proof is not None:
        raise ValueError("non-offline metadata cannot carry verified provenance")
    return meta


def validate_record(record, *, split=None, physical=False):
    reject_eval_fields(record)
    if set(record) != RECORD_FIELDS:
        raise ValueError("record schema: missing or unknown fields")
    meta = validate_metadata(record["metadata"], split=split, source_proof=record["source_proof"])
    if physical and meta["source_kind"] == "model_generated":
        raise ValueError("W/A physical labels require original offline pairs")
    s, a, f, m, p = (record[k] for k in ("state", "actions", "future", "mask", "path"))
    if any(not isinstance(t, torch.Tensor) or not t.is_floating_point() for t in (s, a, f, p)):
        raise ValueError("floating tensors required")
    if s.ndim != 2 or s.shape[1] < 2 or a.ndim != 3 or a.shape[2] < 1 or f.shape != (len(s), a.shape[1], s.shape[1]):
        raise ValueError("state/action/future shapes")
    if not len(s) or m.dtype != torch.bool or m.shape != a.shape[:2]:
        raise ValueError("nonempty data and boolean physical-step mask required")
    if p.ndim != 3 or p.shape[0] != len(s) or p.shape[2] != 2 or p.shape[1] < 2:
        raise ValueError("path must be [B,T_CAP,2]")
    if a.shape[1] != meta["horizon"] or any(t.device != s.device for t in (a, f, m, p)):
        raise ValueError("horizon/device mismatch")
    if physical and meta["horizon"] != 32:
        raise ValueError("physical records require max-horizon 32 with missing futures masked")
    for k in ("episode_id", "row", "path_end"):
        if record[k].shape != (len(s),) or record[k].dtype != torch.int64:
            raise ValueError(f"invalid {k}")
    if not bool(m[:, :4].all()) or bool((m[:, 1:] & ~m[:, :-1]).any()):
        raise ValueError("mask must contain a complete CHUNK and contiguous physical prefix")
    if bool((m.sum(1) % 4).any()):
        raise ValueError("mask must contain complete action chunks only")
    if bool((record["path_end"] < record["row"]+4).any()):
        raise ValueError("path endpoint must include original CHUNK")
    return record


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class VerifiedArrays:
    observations: np.ndarray
    actions: np.ndarray
    terminals: np.ndarray
    dataset_hash: str
    domain: str
    _token: object


_LOADER_TOKEN = object()
_verified_arrays = weakref.WeakValueDictionary()


def load_offline_npz(path, *, expected_domain=None):
    """Only the repository manifest may grant offline provenance."""
    digest = sha256_file(path)
    manifest = json.loads(DATA_MANIFEST.read_text())["datasets"]
    entry = manifest.get(digest)
    if entry is None:
        raise ValueError("dataset hash is not in the manifest whitelist")
    if Path(path).name != entry["filename"] or (expected_domain is not None and expected_domain != entry["domain"]):
        raise ValueError("dataset filename/domain mismatch with manifest")
    with np.load(path, allow_pickle=False) as z:
        reject_eval_fields(dict.fromkeys(z.files))
        # Original OGBench files can additionally carry raw qpos/qvel/button data.
        allowed = {"observations", "actions", "terminals", "qpos", "qvel", "button_states"}
        if set(z.files) - allowed or not {"observations", "actions", "terminals"} <= set(z.files):
            raise ValueError("NPZ field whitelist violation")
        obs = np.asarray(z["observations"], np.float32)
        act = np.asarray(z["actions"], np.float32)
        term = np.asarray(z["terminals"], bool).reshape(-1)
    validate_arrays(obs, act, term)
    if obs.shape[1] != entry["obs_dim"] or act.shape[1] != entry["act_dim"]:
        raise ValueError("dataset dimensions mismatch with manifest")
    # Immutable byte-backed views prevent mutation between hash verification and record building.
    obs = np.frombuffer(obs.tobytes(), dtype=np.float32).reshape(obs.shape)
    act = np.frombuffer(act.tobytes(), dtype=np.float32).reshape(act.shape)
    term = np.frombuffer(term.tobytes(), dtype=np.bool_).reshape(term.shape)
    verified = VerifiedArrays(obs, act, term, digest, entry["domain"], _LOADER_TOKEN)
    _verified_arrays[id(verified)] = verified
    return verified


def validate_arrays(obs, act, term):
    if obs.ndim != 2 or obs.shape[1] < 2 or act.ndim != 2 or len(act) != len(obs) or term.shape != (len(obs),):
        raise ValueError("invalid offline array shapes")
    if len(obs) == 0 or not term[-1] or np.count_nonzero(term) < 5:
        raise ValueError("need >=5 complete episodes and terminal final row")
    if not np.isfinite(obs).all() or not np.isfinite(act).all():
        raise ValueError("nonfinite offline dataset")


def split_episodes(term, seed=0):
    ends = np.flatnonzero(term)
    starts = np.r_[0, ends[:-1] + 1]
    order = np.random.default_rng(seed).permutation(len(ends))
    n_cal = max(1, len(ends) // 5)
    ids = {"calibration": order[:n_cal], "test": order[n_cal:2*n_cal], "train": order[2*n_cal:]}
    return starts, ends, ids


def split_fingerprint(dataset_hash, split_seed, ids):
    payload = {"dataset_hash": dataset_hash, "split_seed": split_seed,
               "episodes": {split: sorted(int(e) for e in episodes) for split, episodes in sorted(ids.items())}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_records(obs, act=None, term=None, *, domain=None, dataset_hash=None, source_kind="toy_offline",
                  split_seed=DEFAULT_SPLIT_SEED, t_cap=128, chunk=4, max_windows=2048):
    """Split BEFORE windows; retain short W windows with masks, never cross terminal.

    Path endpoint follows make_batch's uniform endpoint plus CHUNK clamp. Its
    row/time is audit-only and is never a model input. A targets actions[:CHUNK].
    """
    if isinstance(obs, VerifiedArrays):
        if (_verified_arrays.get(id(obs)) is not obs or obs._token is not _LOADER_TOKEN or
            act is not None or term is not None):
            raise ValueError("offline provenance requires the verified loader result")
        if domain is not None and domain != obs.domain or dataset_hash is not None and dataset_hash != obs.dataset_hash:
            raise ValueError("caller domain/hash mismatch with manifest")
        domain, dataset_hash, source_kind = obs.domain, obs.dataset_hash, "offline_dataset"
        obs, act, term = obs.observations, obs.actions, obs.terminals
    elif source_kind != "toy_offline":
        raise ValueError("offline_dataset provenance requires the verified loader")
    validate_arrays(obs, act, term)
    if chunk != 4 or t_cap < 2 or max_windows < 1:
        raise ValueError("v3 first batch supports CHUNK=4, T_CAP>=2, positive cap")
    starts, ends, ids = split_episodes(term, split_seed)
    split_hash = split_fingerprint(dataset_hash, split_seed, ids)
    records = {}
    for split, episodes in ids.items():
        rng = np.random.default_rng(split_seed + {"train": 1, "calibration": 2, "test": 3}[split])
        lengths = np.maximum(ends[episodes]-starts[episodes]-chunk+1, 0)
        cumulative = lengths.cumsum()
        total = int(cumulative[-1])
        if not total:
            raise ValueError(f"{split}: no complete CHUNK windows")
        selected = rng.choice(total, min(max_windows, total), replace=False)
        bins = np.searchsorted(cumulative, selected, side="right")
        offsets = np.r_[0, cumulative[:-1]]
        rows = [(int(episodes[b]), int(starts[episodes[b]]+i-offsets[b])) for i, b in zip(selected, bins)]
        s, aa, ff, masks, paths, goals = [], [], [], [], [], []
        partial_chunks_dropped = 0
        for e, r in rows:
            end = int(ends[e]); raw_available = min(32, end-r)
            available = (raw_available//chunk)*chunk
            partial_chunks_dropped += int(raw_available != available)
            a = np.zeros((32, act.shape[1]), np.float32)
            f = np.zeros((32, obs.shape[1]), np.float32)
            m = np.arange(32) < available
            a[:available] = act[r:r+available]
            f[:available] = obs[r+1:r+available+1]
            u = rng.random()
            g = max(int(round(min(r+1, end)*u + end*(1-u))), min(r+chunk, end))
            times = np.linspace(float(r), float(g), t_cap)
            lo = np.floor(times).astype(np.int64); hi = np.minimum(lo+1, g)
            w = (times-lo)[:, None]
            path = (obs[lo, :2]*(1-w) + obs[hi, :2]*w).astype(np.float32)
            s.append(obs[r]); aa.append(a); ff.append(f); masks.append(m); paths.append(path); goals.append(g)
        meta = dict(source_kind=source_kind, dataset_hash=dataset_hash, episode_split=split,
                    parent_model_hash=None, candidate_id=None, noise_seed=split_seed, iteration=0,
                    domain=domain, horizon=32, representation_version=VERSION, eval_only=False,
                    split_seed=split_seed, split_hash=split_hash,
                    partial_chunks_dropped=partial_chunks_dropped)
        proof = _SourceProof() if source_kind == "offline_dataset" else None
        if proof is not None:
            _verified_proofs[proof] = (dataset_hash, domain, split_hash, split)
        records[split] = dict(metadata=meta, source_proof=proof,
                             state=torch.tensor(np.stack(s)), actions=torch.tensor(np.stack(aa)),
                             future=torch.tensor(np.stack(ff)), mask=torch.tensor(np.stack(masks)),
                             path=torch.tensor(np.stack(paths)), episode_id=torch.tensor([e for e, r in rows]),
                             row=torch.tensor([r for e, r in rows]), path_end=torch.tensor(goals))
        validate_record(records[split], split=split, physical=True)
    assert_disjoint(records)
    return records


def assert_disjoint(records):
    seen = set()
    hashes = set()
    for split, rec in records.items():
        validate_record(rec, split=split, physical=True)
        ids = set(rec["episode_id"].tolist())
        if ids & seen:
            raise ValueError("episode overlap across splits")
        seen |= ids
        hashes.add((rec["metadata"]["dataset_hash"], rec["metadata"]["domain"], rec["metadata"]["source_kind"], rec["metadata"]["split_seed"], rec["metadata"]["split_hash"]))
    if len(hashes) != 1:
        raise ValueError("dataset/domain/source mismatch across splits")


def subset(rec, indices):
    return {k: v if k in ("metadata", "source_proof") else v[indices] for k, v in rec.items()}


def subspaces(domain, dim):
    # FIX3 ruler: yaw means quaternion coordinates, NOT an Euler-angle error.
    if domain == "ant":
        if dim != 29:
            raise ValueError("ant requires documented qpos(15)|qvel(14) layout")
        return {"xy": [0, 1], "yaw": list(range(3, 7)), "gait": [2, *range(7, 29)]}
    return {"xy": [0, 1], **({"velocity": list(range(2, dim))} if dim > 2 else {})}


class Normalizer(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("scale", scale.float().clamp_min(1e-4).clone())

    @classmethod
    def fit(cls, values):
        return cls(values.float().mean(0), values.float().std(0, unbiased=False))

    def forward(self, x):
        return (x-self.mean)/self.scale


class MLP(nn.Module):
    def __init__(self, inputs, outputs, hidden):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, outputs))

    def forward(self, x):
        return self.linear(x) + self.net(x)


class ConsequenceEnsemble(nn.Module):
    """Physical CHUNK transition, unrolled and supervised at h=4/16/32."""
    def __init__(self, state_norm, action_norm, *, hidden=64, members=3, obs_only=False):
        super().__init__()
        if members != 3:
            raise ValueError("v3 ensemble size must be 3")
        self.state_norm = state_norm; self.action_norm = action_norm
        self.obs_only = obs_only
        self.dim = len(state_norm.mean); self.adim = len(action_norm.mean); self.chunk = 4
        self.members = nn.ModuleList([MLP(self.dim + (0 if obs_only else 4*self.adim), 4*self.dim, hidden) for _ in range(members)])

    def forward(self, state, actions):
        if state.ndim != 2 or actions.ndim != 3 or actions.shape[0] != len(state) or actions.shape[2] != self.adim:
            raise ValueError("W requires state[B,D], physical actions[B,H,A]")
        if actions.shape[1] not in HORIZONS or state.shape[1] != self.dim or actions.device != state.device:
            raise ValueError("unsupported physical horizon/state/device")
        predictions = []
        for net in self.members:
            current = state
            steps = []
            for t in range(0, actions.shape[1], 4):
                inputs = self.state_norm(current)
                if not self.obs_only:
                    inputs = torch.cat([inputs, self.action_norm(actions[:, t:t+4]).flatten(1)], 1)
                delta = net(inputs).reshape(-1, 4, self.dim)
                predicted = current[:, None] + delta*self.state_norm.scale
                steps.append(predicted)
                current = predicted[:, -1]
            predictions.append(torch.cat(steps, 1))
        return torch.stack(predictions)


class PairedController(nn.Module):
    """A_omega(full initial state, fixed-point path) -> original CHUNK actions.

    No endpoint time, future velocity, path differences, or live-head labels.
    Ant is diagnostic only; services disallow ant latent refinement.
    """
    def __init__(self, state_norm, action_norm, t_cap, *, hidden=64):
        super().__init__()
        self.state_norm = state_norm; self.action_norm = action_norm; self.t_cap = t_cap
        self.adim = len(action_norm.mean)
        self.members = nn.ModuleList([MLP(len(state_norm.mean)+t_cap*2, 4*self.adim, hidden) for _ in range(3)])

    def forward(self, state, path):
        if path.shape != (len(state), self.t_cap, 2) or path.device != state.device:
            raise ValueError("A requires same-window fixed T_CAP path")
        p = (path-self.state_norm.mean[:2])/self.state_norm.scale[:2]
        inputs = torch.cat([self.state_norm(state), p.flatten(1)], 1)
        return torch.stack([net(inputs).reshape(-1, 4, self.adim)*self.action_norm.scale + self.action_norm.mean for net in self.members])


def freeze(module):
    module.requires_grad_(False)
    for parameter in module.parameters():
        parameter.grad = None
    module.eval()
    return module


def reconstruction_loss(pred, rec, norm, spaces):
    err = ((pred.float()-rec["future"].float()[None])/norm.scale).square()
    parts = []
    for h in HORIZONS:
        mask = rec["mask"][:, :h].float()[None, :, :, None]
        for dims in spaces.values():
            parts.append((err[:, :, :h, dims]*mask).sum()/(mask.sum()*len(dims)*len(pred)).clamp_min(1))
    return torch.stack(parts).mean()


def action_witness(prediction, shuffled_prediction, target):
    return float((shuffled_prediction.float()-target.float()).square().mean() -
                 (prediction.float()-target.float()).square().mean())


def train_models(record, *, steps=200, batch_size=64, hidden=64, seed=0, lr=0.003):
    """Fit only TRAIN records. Does not inspect calibration/test or select checkpoints."""
    validate_record(record, split="train", physical=True)
    if record["state"].device.type != "cpu":
        raise ValueError("this first-batch entrypoint is CPU-only; GPU needs F5")
    if type(steps) is not int or steps < 1 or batch_size < 1:
        raise ValueError("positive training budget required")
    torch.manual_seed(seed)
    state_norm = Normalizer.fit(record["state"])
    action_norm = Normalizer.fit(record["actions"][record["mask"]])
    world = ConsequenceEnsemble(state_norm, action_norm, hidden=hidden)
    baseline = ConsequenceEnsemble(state_norm, action_norm, hidden=hidden, obs_only=True)
    controller = PairedController(state_norm, action_norm, record["path"].shape[1], hidden=hidden)
    models = {"world": world, "obs_only": baseline, "controller": controller}
    spaces = subspaces(record["metadata"]["domain"], record["state"].shape[1])
    curves = {}
    # Independent member initializations; shared minibatches. No bootstrap claim.
    for name, model in models.items():
        rng = torch.Generator().manual_seed(seed+{ "world": 11, "obs_only": 12, "controller": 13}[name])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        fixed = subset(record, torch.arange(min(128, len(record["state"]))))
        def loss_on(rec):
            if name == "controller":
                return ((model(rec["state"], rec["path"]).float()-rec["actions"][None, :, :4].float())/action_norm.scale).square().mean()
            return reconstruction_loss(model(rec["state"], rec["actions"]), rec, state_norm, spaces)
        with torch.no_grad():
            initial = float(loss_on(fixed))
        curve = []
        for step in range(steps):
            rec = subset(record, torch.randint(len(record["state"]), (batch_size,), generator=rng))
            loss = loss_on(rec)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            if step == 0 or (step+1) % max(1, steps//5) == 0:
                curve.append({"step": step+1, "loss": float(loss.detach())})
        with torch.no_grad():
            final = float(loss_on(fixed))
        curves[name] = {"initial": initial, "final": final, "curve": curve}
        freeze(model)
    return models, curves


def model_fingerprint(models, metadata, config=None):
    h = hashlib.sha256(json.dumps({"metadata": metadata, "config": config}, sort_keys=True).encode())
    for name, model in sorted(models.items()):
        for key, tensor in sorted(model.state_dict().items()):
            h.update(f"{name}/{key}".encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def toy_arrays(domain, seed=0, episodes=50, length=49):
    """Algebraic offline fixture, not a simulator and not physical ant evidence."""
    rng = np.random.default_rng(seed)
    dim, adim = (4, 2) if domain == "pointmaze" else (29, 8)
    projection = rng.normal(0, 0.18, (adim, dim)).astype(np.float32)
    projection[:, :adim] += np.eye(adim, dtype=np.float32)*0.35
    obs = []; acts = []; terms = []
    for ep in range(episodes):
        state = rng.normal(0, 0.5, dim).astype(np.float32)
        actions = rng.normal(0, 0.65, (length, adim)).astype(np.float32)
        for t in range(length):
            obs.append(state.copy()); acts.append(actions[t]); terms.append(t == length-1)
            state = 0.85*state + actions[t]@projection + rng.normal(0, .012, dim).astype(np.float32)
    return np.stack(obs), np.stack(acts), np.asarray(terms)
