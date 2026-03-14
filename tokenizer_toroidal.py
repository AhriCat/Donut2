# tokenizer_toroidal.py
# Toroidal tokenizer: representations ∈ [0, 1]^D
# Epistemic pole at 0.5 (uncertainty), boundaries 0 ≡ 1 (confidence, identified on S¹)
# Confidence = geodesic distance from 0.5: conf(x) = 2|x - 0.5|
# Assertion hemisphere: [0, 0.5) vs (0.5, 1], with 0 ≡ 1 (wrap)

import json, os, re, hashlib, math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # geometry/repr works without torch; tensor ops will fail gracefully

TOK_VERSION = "0.4.0"

# ── Geometric helpers ──

def circular_distance(a: float, b: float) -> float:
    """Geodesic distance on S¹ ≅ ℝ/ℤ (range [0, 0.5])."""
    d = abs(a - b)
    return min(d, 1.0 - d)

def confidence(x: float) -> float:
    """Epistemic confidence: 0 at pole (0.5), 1 at boundary (0 ≡ 1)."""
    return 2.0 * circular_distance(x, 0.5)

def confidence_vec(v: List[float]) -> float:
    """Mean confidence across all dimensions."""
    return sum(confidence(x) for x in v) / len(v) if v else 0.0

def antipodal(x: float) -> float:
    """
    Geometric antipode on S¹: maximal geodesic distance (0.5).
    Maps pole (0.5) → boundary (0≡1) and vice versa.
    Does NOT preserve confidence — it's a pure geometric operation.
    """
    return (x + 0.5) % 1.0

def negate(x: float) -> float:
    """
    Epistemic negation: reflection through the pole (0.5).
    Preserves confidence, flips hemisphere.
    x=0.3 → 0.7, x=0.1 → 0.9, x=0.5 → 0.5 (pole is fixed).
    
    Note: on S¹ with 0≡1, the maximally confident point is its own
    negation — this is the topological consequence of the identification.
    """
    return 1.0 - x

def hemisphere(x: float) -> int:
    """Which side of the pole: -1 for [0, 0.5), +1 for (0.5, 1], 0 at pole."""
    if abs(x - 0.5) < 1e-12: return 0
    return -1 if x < 0.5 else 1


class TernaryTokenizer:
    """
    Ternary BPE tokenizer with toroidal [0,1]^D representation geometry.

    Epistemic pole:      0.5  (uncertainty)
    Confidence boundary: 0 ≡ 1 (identified on S¹)
    Merge composition:   circular mean of parent representations
    Torus embedding:     (u,v) ∈ [0,1]² → ℝ³ via standard parameterization
    """

    SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    _CORE_SYMBOLS = list(" ") + list(".,;:!?-—'\"()[]{}") 

    UNCERTAINTY_POLE = 0.5
    UNK_RADIUS = 0.075

    def __init__(self, vocab_size: int = 50000,
                 repr_dim: int = 64,
                 repr_seed: int = 0,
                 repr_clamp: str = "sigmoid"):
        self.vocab_size = int(vocab_size)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.SPECIALS)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(self.SPECIALS)}
        self.PAD_ID = self.token_to_id["[PAD]"]
        self.UNK_ID = self.token_to_id["[UNK]"]
        self.BOS_ID = self.token_to_id["[BOS]"]
        self.EOS_ID = self.token_to_id["[EOS]"]

        self.merge_ranks: Dict[Tuple[str, str, str], int] = {}
        self._frozen = False
        self.exhaustiveness: str = 'conservative'

        self.repr_dim = int(repr_dim)
        self.repr_seed = int(repr_seed)
        self.repr_clamp = (repr_clamp or "sigmoid").lower()
        self.parent_map: Dict[str, Tuple[str, str, str]] = {}
        self._repr_cache: Dict[str, List[float]] = {}

        for sym in self._CORE_SYMBOLS:
            if sym not in self.token_to_id:
                nid = len(self.token_to_id)
                self.token_to_id[sym] = nid
                self.id_to_token[nid] = sym

    @property
    def eos_id(self) -> int: return self.EOS_ID
    @property
    def pad_id(self) -> int: return self.PAD_ID
    @property
    def exhaustiveness(self) -> str: return self._exhaustiveness
    @exhaustiveness.setter
    def exhaustiveness(self, value: str):
        if value not in ("conservative", "aggressive"):
            raise ValueError("exhaustiveness must be 'conservative' or 'aggressive'")
        self._exhaustiveness = value

    # ── Training ──

    def train(self, texts: List[str], min_frequency: int = 2,
              max_merges: Optional[int] = None, token_pattern: str = r"\w+|\W"):
        if self._frozen:
            raise RuntimeError("Tokenizer is frozen.")
        chars = set()
        for text in texts:
            chars.update(text)
        for ch in sorted(chars):
            if ch not in self.token_to_id:
                nid = len(self.token_to_id)
                self.token_to_id[ch] = nid
                self.id_to_token[nid] = ch

        vocab: Dict[str, int] = {}
        pat = re.compile(token_pattern)
        for text in texts:
            for word in pat.findall(text):
                if not word: continue
                seq = " ".join(list(word))
                vocab[seq] = vocab.get(seq, 0) + 1

        rank = len(self.merge_ranks)
        merges_done = 0
        while len(self.token_to_id) < self.vocab_size:
            if max_merges is not None and merges_done >= max_merges:
                break
            trip_freq = defaultdict(int)
            for seq, f in vocab.items():
                toks = seq.split()
                if len(toks) < 3: continue
                for i in range(len(toks) - 2):
                    trip_freq[(toks[i], toks[i+1], toks[i+2])] += f
            if not trip_freq: break
            (a, b, c), freq = max(trip_freq.items(), key=lambda kv: kv[1])
            if freq < min_frequency: break
            merged = a + b + c
            if merged not in self.token_to_id:
                nid = len(self.token_to_id)
                self.token_to_id[merged] = nid
                self.id_to_token[nid] = merged
            new_vocab: Dict[str, int] = {}
            for seq, f in vocab.items():
                toks = seq.split()
                i = 0; out = []
                while i < len(toks):
                    if i + 2 < len(toks) and toks[i] == a and toks[i+1] == b and toks[i+2] == c:
                        out.append(merged); i += 3
                    else:
                        out.append(toks[i]); i += 1
                new_vocab[" ".join(out)] = new_vocab.get(" ".join(out), 0) + f
            vocab = new_vocab
            self.merge_ranks[(a, b, c)] = rank
            self.parent_map[merged] = (a, b, c)
            rank += 1
            merges_done += 1
        return merges_done

    def freeze(self): self._frozen = True
    def unfreeze(self): self._frozen = False

    # ── Encode / Decode ──

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        if not self.merge_ranks: return tokens
        for (a, b, c), _ in sorted(self.merge_ranks.items(), key=lambda kv: kv[1]):
            i = 0; out = []; L = len(tokens)
            while i < L:
                if i + 2 < L and tokens[i] == a and tokens[i+1] == b and tokens[i+2] == c:
                    out.append(a + b + c); i += 3
                else:
                    out.append(tokens[i]); i += 1
            tokens = out
        return tokens

    def _encode_list(self, text, add_bos=False, add_eos=False,
                     max_length=None, pad_to_max=False):
        tokens = list(text)
        tokens = self._apply_merges(tokens)
        ids = [self.token_to_id.get(tok, self.UNK_ID) for tok in tokens]
        if add_bos: ids = [self.BOS_ID] + ids
        if add_eos: ids = ids + [self.EOS_ID]
        if max_length is not None:
            ids = ids[:max_length]
            if pad_to_max and len(ids) < max_length:
                ids += [self.PAD_ID] * (max_length - len(ids))
        return ids

    def encode(self, text, *args, add_bos=False, add_eos=False,
               max_length=None, pad_to_max=False, return_torch=False,
               device=None, dtype=None):
        if torch is None and (return_torch or args):
            if args and isinstance(args[0], int):
                # Trainer mode without torch: return plain list
                Lt = int(args[0])
                return self._encode_list(text, add_bos=True, max_length=Lt, pad_to_max=True)
            if return_torch:
                raise RuntimeError("torch not available")

        if torch is not None and len(args) >= 1 and isinstance(args[0], int):
            Lt = int(args[0])
            ids = self._encode_list(text, add_bos=True, max_length=Lt, pad_to_max=True)
            _dtype = dtype or torch.long
            return torch.tensor(ids, dtype=_dtype, device=device)

        ids = self._encode_list(text, add_bos=add_bos, add_eos=add_eos,
                                max_length=max_length, pad_to_max=pad_to_max)
        if return_torch and torch is not None:
            _dtype = dtype or torch.long
            return torch.tensor(ids, dtype=_dtype, device=device)
        return ids

    def encode_ids(self, text: str) -> List[int]:
        return self._encode_list(text)

    def decode(self, ids, skip_specials=True):
        toks = []
        for i in ids:
            tok = self.id_to_token.get(int(i), "[UNK]")
            if skip_specials and tok in self.SPECIALS: continue
            toks.append(tok)
        return "".join(toks)

    def batch_encode(self, texts, **kw):
        return [self._encode_list(t, **kw) for t in texts]

    def batch_decode(self, batch_ids, skip_specials=True):
        return [self.decode(ids, skip_specials=skip_specials) for ids in batch_ids]

    # ── Toroidal Representations [0, 1]^D ──

    def _hash_to_vec(self, token: str) -> List[float]:
        """Deterministic base vector in [0, 1]^D."""
        D = self.repr_dim
        v: List[float] = []
        i = 0
        while len(v) < D:
            h = hashlib.sha256(f"{self.repr_seed}|{i}|{token}".encode("utf-8")).digest()
            for off in range(0, 32, 8):
                u = int.from_bytes(h[off:off+8], "big") / (1 << 64)
                v.append(u)  # already [0, 1] — no remapping
                if len(v) == D: break
            i += 1
        return v

    def _clamp_vec(self, x: List[float]) -> List[float]:
        """Project to [0, 1]. Sigmoid for smooth, clip for hard."""
        if self.repr_clamp == "clip":
            return [max(0.0, min(1.0, t)) for t in x]
        return [1.0 / (1.0 + math.exp(-max(-500, min(500, t)))) for t in x]

    def _circular_mean(self, angles: List[List[float]]) -> List[float]:
        """
        Circular (Fréchet) mean on [0,1]^D treated as (S¹)^D.
        Converts to angular coordinates, averages sin/cos, converts back.
        Respects 0 ≡ 1 identification.
        """
        D = len(angles[0])
        result = []
        for d in range(D):
            sin_sum = 0.0
            cos_sum = 0.0
            for vec in angles:
                theta = 2.0 * math.pi * vec[d]
                sin_sum += math.sin(theta)
                cos_sum += math.cos(theta)
            mean_theta = math.atan2(sin_sum / len(angles), cos_sum / len(angles))
            result.append((mean_theta / (2.0 * math.pi)) % 1.0)
        return result

    def _combine_parents(self, a: str, b: str, c: str) -> List[float]:
        ra = self.get_repr(a)
        rb = self.get_repr(b)
        rc = self.get_repr(c)
        return self._circular_mean([ra, rb, rc])

    # Tokens whose representation is structurally fixed at the pole
    _POLE_TOKENS = {"[PAD]", "[UNK]", ""}

    def get_repr(self, token_or_id) -> List[float]:
        if isinstance(token_or_id, int):
            token = self.id_to_token.get(int(token_or_id), "")
        else:
            token = str(token_or_id)

        if token in self._repr_cache:
            return self._repr_cache[token]

        # Structural assignment: epistemic unknowns live at the pole
        if token in self._POLE_TOKENS:
            vec = [self.UNCERTAINTY_POLE] * self.repr_dim
            self._repr_cache[token] = vec
            return vec

        if token in self.parent_map:
            a, b, c = self.parent_map[token]
            vec = self._combine_parents(a, b, c)
        else:
            vec = self._hash_to_vec(token)
            vec = self._clamp_vec(vec)

        # Epistemic damping: push uncertain dimensions toward pole (0.5)
        vec = np.array(vec)
        dist_from_pole = np.abs(vec - self.UNCERTAINTY_POLE)
        uncertain_mask = dist_from_pole < self.UNK_RADIUS

        vec[uncertain_mask] = self.UNCERTAINTY_POLE + \
            (vec[uncertain_mask] - self.UNCERTAINTY_POLE) * 0.2

        if uncertain_mask.mean() > 0.5:
            vec = np.full_like(vec, self.UNCERTAINTY_POLE)

        vec = vec.tolist()
        self._repr_cache[token] = vec
        return vec

    def confidence_of(self, token_or_id) -> float:
        return confidence_vec(self.get_repr(token_or_id))

    def repr_for_ids(self, ids):
        return [self.get_repr(self.id_to_token.get(int(i), "")) for i in ids]

    def export_repr_matrix(self, order="id"):
        return [self.get_repr(self.id_to_token.get(i, "")) for i in range(len(self.token_to_id))]

    # ── Torus embedding ──

    def torus_embed_pair(self, id_a, id_b, R=1.0, r=0.4, dim=0):
        """
        Embed (token_a, token_b) into ℝ³ via torus parameterization.
        u = repr_a[dim], v = repr_b[dim], both ∈ [0,1]
        """
        u = self.get_repr(id_a)[dim]
        v = self.get_repr(id_b)[dim]
        theta_u = 2.0 * math.pi * u
        theta_v = 2.0 * math.pi * v
        x = (R + r * math.cos(theta_v)) * math.cos(theta_u)
        y = (R + r * math.cos(theta_v)) * math.sin(theta_u)
        z = r * math.sin(theta_v)
        return (x, y, z)

    # ── Save / Load ──

    def to_dict(self):
        merges_sorted = sorted(self.merge_ranks.items(), key=lambda kv: kv[1])
        merges = [{"a": a, "b": b, "c": c, "rank": r} for ((a, b, c), r) in merges_sorted]
        parent_map = [{"merged": k, "a": v[0], "b": v[1], "c": v[2]} for k, v in self.parent_map.items()]
        return {
            "version": TOK_VERSION,
            "type": "ternary-bpe-toroidal",
            "geometry": {
                "space": "[0,1]^D",
                "pole": self.UNCERTAINTY_POLE,
                "unk_radius": self.UNK_RADIUS,
                "identification": "0 ≡ 1 (S¹ per dimension)",
                "confidence": "geodesic_distance_from_pole",
            },
            "vocab_size": self.vocab_size,
            "specials": self.SPECIALS,
            "token_to_id": dict(sorted(self.token_to_id.items(), key=lambda kv: kv[1])),
            "merges": merges,
            "parent_map": parent_map,
            "frozen": self._frozen,
            "repr": {"dim": self.repr_dim, "seed": self.repr_seed, "clamp": self.repr_clamp},
        }

    @classmethod
    def from_dict(cls, obj):
        tp = obj.get("type", "")
        if tp not in ("ternary-bpe", "ternary-bpe-toroidal"):
            raise ValueError(f"Unexpected tokenizer type: {tp}")
        cfg = obj.get("repr", {})
        tok = cls(vocab_size=int(obj.get("vocab_size", 50000)),
                  repr_dim=int(cfg.get("dim", 64)),
                  repr_seed=int(cfg.get("seed", 0)),
                  repr_clamp=str(cfg.get("clamp", "sigmoid")))
        geo = obj.get("geometry", {})
        if "pole" in geo: tok.UNCERTAINTY_POLE = float(geo["pole"])
        if "unk_radius" in geo: tok.UNK_RADIUS = float(geo["unk_radius"])
        tok.token_to_id = {k: int(v) for k, v in obj["token_to_id"].items()}
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok.PAD_ID = tok.token_to_id["[PAD]"]
        tok.UNK_ID = tok.token_to_id["[UNK]"]
        tok.BOS_ID = tok.token_to_id["[BOS]"]
        tok.EOS_ID = tok.token_to_id["[EOS]"]
        tok.merge_ranks = {(m["a"], m["b"], m["c"]): int(m["rank"]) for m in obj.get("merges", [])}
        tok.parent_map = {}
        if "parent_map" in obj:
            for rec in obj["parent_map"]:
                tok.parent_map[rec["merged"]] = (rec["a"], rec["b"], rec["c"])
        else:
            for (a, b, c), _ in sorted(tok.merge_ranks.items(), key=lambda kv: kv[1]):
                tok.parent_map.setdefault(a+b+c, (a, b, c))
        tok._frozen = bool(obj.get("frozen", False))
        tok._repr_cache = {}
        return tok

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def export_merges(self):
        return [k for k, _ in sorted(self.merge_ranks.items(), key=lambda kv: kv[1])]

    def import_merges(self, merges):
        if self._frozen: raise RuntimeError("Frozen.")
        self.merge_ranks.clear(); self.parent_map.clear()
        for rank, (a, b, c) in enumerate(merges):
            merged = a + b + c
            if merged not in self.token_to_id:
                nid = len(self.token_to_id)
                self.token_to_id[merged] = nid
                self.id_to_token[nid] = merged
            self.merge_ranks[(a, b, c)] = rank
            self.parent_map[merged] = (a, b, c)
        self._repr_cache.clear()
