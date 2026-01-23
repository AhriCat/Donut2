# ternary_tokenizer.py
import json, os, re, hashlib, math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch  # NEW: for returning tensors when Lt is passed

TOK_VERSION = "0.3.0"  # bumped: adds representation schema

class TernaryTokenizer:
    """
    Ternary BPE-style tokenizer: merges (a,b,c) -> "abc".
    Adds deterministic, compositional continuous representations ∈ [-1, 1]^D.

    New:
      - Deterministic base reps via hash(seed, token) → U(-1,1)
      - Compositional merged reps via mean(parent reps) then clamp
      - Config: repr_dim, repr_seed, repr_clamp ('tanh'|'clip')
      - Accessors: get_repr(token|id), repr_for_ids(ids), export_repr_matrix()
      - parent_map to retain merge lineage

    Trainer-compatibility:
      - If called as encode(text, Lt:int), returns torch.LongTensor of shape (Lt,)
        with [BOS] at position 0, truncated/padded with [PAD] to Lt.
      - Otherwise, default keyword behavior returns a Python list of ints.
    """

    SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    # minimal core symbol set to ensure spaces/punctuation exist in-vocab
    _CORE_SYMBOLS = list(" ") + list(".,;:!?-—'\"()[]{}")

    def __init__(self, vocab_size: int = 50000,
                 repr_dim: int = 64,
                 repr_seed: int = 0,
                 repr_clamp: str = "tanh"):
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
        # --- NEW: representation config/state ---
        self.repr_dim = int(repr_dim)
        self.repr_seed = int(repr_seed)
        self.repr_clamp = (repr_clamp or "tanh").lower()
        self.parent_map: Dict[str, Tuple[str, str, str]] = {}
        self._repr_cache: Dict[str, List[float]] = {}

        # Ensure core symbols are present so generations have reliable access
        for sym in self._CORE_SYMBOLS:
            if sym not in self.token_to_id:
                nid = len(self.token_to_id)
                self.token_to_id[sym] = nid
                self.id_to_token[nid] = sym

    @property
    def eos_id(self) -> int:
        return self.EOS_ID

    @property
    def exhaustiveness(self) -> str:
        return self._exhaustiveness

    @exhaustiveness.setter
    def exhaustiveness(self, value: str):
        if value not in ("conservative", "aggressive"):
            raise ValueError("exhaustiveness must be 'conservative' or 'aggressive'")
        self._exhaustiveness = value

    @property
    def pad_id(self) -> int:
        return self.PAD_ID


    # ------------------------ training ------------------------

    def train(
        self,
        texts: List[str],
        min_frequency: int = 2,
        max_merges: Optional[int] = None,
        token_pattern: str = r"\w+|\W"
    ):
        if self._frozen:
            raise RuntimeError("Tokenizer is frozen (merges locked). Unfreeze before training more.")

        # 1) Seed with observed characters
        chars = set()
        for text in texts:
            chars.update(text)
        # retain existing ids; add any new chars
        for ch in sorted(chars):
            if ch not in self.token_to_id:
                nid = len(self.token_to_id)
                self.token_to_id[ch] = nid
                self.id_to_token[nid] = ch

        # 2) Initialize vocab as space-separated characters
        vocab: Dict[str, int] = {}
        pat = re.compile(token_pattern)
        for text in texts:
            for word in pat.findall(text):
                if not word: continue
                seq = " ".join(list(word))
                vocab[seq] = vocab.get(seq, 0) + 1

        # 3) Greedy ternary merges
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
                    trip = (toks[i], toks[i + 1], toks[i + 2])
                    trip_freq[trip] += f

            if not trip_freq:
                break

            (a, b, c), freq = max(trip_freq.items(), key=lambda kv: kv[1])
            if freq < min_frequency:
                break

            merged = a + b + c
            if merged not in self.token_to_id:
                nid = len(self.token_to_id)
                self.token_to_id[merged] = nid
                self.id_to_token[nid] = merged

            # Left-to-right non-overlapping replacement
            new_vocab: Dict[str, int] = {}
            for seq, f in vocab.items():
                toks = seq.split()
                i = 0; out = []
                while i < len(toks):
                    if i + 2 < len(toks) and toks[i] == a and toks[i+1] == b and toks[i+2] == c:
                        out.append(merged); i += 3
                    else:
                        out.append(toks[i]); i += 1
                new_seq = " ".join(out)
                new_vocab[new_seq] = new_vocab.get(new_seq, 0) + f
            vocab = new_vocab

            self.merge_ranks[(a, b, c)] = rank
            self.parent_map[merged] = (a, b, c)

            rank += 1
            merges_done += 1

        return merges_done

    def freeze(self): self._frozen = True
    def unfreeze(self): self._frozen = False

    # ------------------------ core encode/decode ------------------------

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        if not self.merge_ranks:
            return tokens
        for (a, b, c), _rk in sorted(self.merge_ranks.items(), key=lambda kv: kv[1]):
            i = 0; out = []; L = len(tokens)
            while i < L:
                if i + 2 < L and tokens[i] == a and tokens[i+1] == b and tokens[i+2] == c:
                    out.append(a + b + c); i += 3
                else:
                    out.append(tokens[i]); i += 1
            tokens = out
        return tokens

    def _encode_list(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        pad_to_max: bool = False,
    ) -> List[int]:
        tokens = list(text)
        tokens = self._apply_merges(tokens)
        ids = [self.token_to_id.get(tok, self.UNK_ID) for tok in tokens]
        if add_bos: ids = [self.BOS_ID] + ids
        if add_eos: ids = ids + [self.EOS_ID]
        if max_length is not None:
            ids = ids[:max_length]
            if pad_to_max and len(ids) < max_length:
                ids = ids + [self.PAD_ID] * (max_length - len(ids))
        return ids

    def encode(
        self,
        text: str,
        *args,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        pad_to_max: bool = False,
        return_torch: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.long,
    ):
        """
        Dual mode:

        (A) Trainer mode (matches your call-sites):
            encode(text, Lt:int)  -> torch.LongTensor (Lt,)
            Behavior: inserts [BOS] at position 0, pads with [PAD] to Lt.

        (B) Legacy/explicit mode:
            encode(text, add_bos=..., add_eos=..., max_length=..., pad_to_max=..., return_torch=False)
            -> list[int] (or torch tensor if return_torch=True)
        """
        # Trainer-style positional Lt
        if len(args) >= 1 and isinstance(args[0], int):
            Lt = int(args[0])
            # Force BOS at start, pad to Lt, return tensor
            ids = self._encode_list(
                text,
                add_bos=True,
                add_eos=False,
                max_length=Lt,
                pad_to_max=True,
            )
            t = torch.tensor(ids, dtype=dtype, device=device) if device is not None else torch.tensor(ids, dtype=dtype)
            return t

        # Otherwise, legacy keyword-driven behavior
        ids = self._encode_list(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            max_length=max_length,
            pad_to_max=pad_to_max,
        )
        if return_torch:
            return torch.tensor(ids, dtype=dtype, device=device) if device is not None else torch.tensor(ids, dtype=dtype)
        return ids

    def encode_tensor(self, text: str, Lt: int, device: Optional[torch.device] = None,
                      dtype: torch.dtype = torch.long) -> torch.Tensor:
        """Explicit helper mirroring trainer-mode behavior."""
        return self.encode(text, Lt, device=device, dtype=dtype)

    def encode_ids(self, text: str) -> List[int]:
        return self._encode_list(text, add_bos=False, add_eos=False)

    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        toks = []
        for i in ids:
            tok = self.id_to_token.get(int(i), "[UNK]")
            if skip_specials and tok in self.SPECIALS:
                continue
            toks.append(tok)
        return "".join(toks)

    # ------------------------ batch helpers ------------------------

    def batch_encode(self, texts: List[str], add_bos: bool = False, add_eos: bool = False,
                     max_length: Optional[int] = None, pad_to_max: bool = False) -> List[List[int]]:
        return [self._encode_list(t, add_bos=add_bos, add_eos=add_eos,
                                  max_length=max_length, pad_to_max=pad_to_max) for t in texts]

    def batch_decode(self, batch_ids: List[List[int]], skip_specials: bool = True) -> List[str]:
        return [self.decode(ids, skip_specials=skip_specials) for ids in batch_ids]

    # ------------------------ representations (NEW) ------------------------

    def _hash_to_vec(self, token: str) -> List[float]:
        """Deterministic base vector in [-1,1] from token text + seed."""
        D = self.repr_dim
        v: List[float] = []
        i = 0
        while len(v) < D:
            h = hashlib.sha256(f"{self.repr_seed}|{i}|{token}".encode("utf-8")).digest()
            for off in range(0, 32, 8):
                u = int.from_bytes(h[off:off+8], "big") / (1 << 64)
                v.append(2.0 * u - 1.0)
                if len(v) == D: break
            i += 1
        return v

    def _clamp_vec(self, x: List[float]) -> List[float]:
        if self.repr_clamp == "clip":
            return [max(-1.0, min(1.0, t)) for t in x]
        return [math.tanh(t) for t in x]

    def _combine_parents(self, a: str, b: str, c: str) -> List[float]:
        ra = self.get_repr(a)
        rb = self.get_repr(b)
        rc = self.get_repr(c)
        D = self.repr_dim
        out = [(ra[i] + rb[i] + rc[i]) / 3.0 for i in range(D)]
        return self._clamp_vec(out)

    def get_repr(self, token_or_id: Any) -> List[float]:
        if isinstance(token_or_id, int):
            token = self.id_to_token.get(int(token_or_id), "")
        else:
            token = str(token_or_id)

        if token in self._repr_cache:
            return self._repr_cache[token]

        if token in self.parent_map:
            a, b, c = self.parent_map[token]
            vec = self._combine_parents(a, b, c)
        else:
            vec = self._hash_to_vec(token)
            vec = self._clamp_vec(vec)
        UNK_CENTER = 0.0   # the zero region
        UNK_WIDTH  = 0.15  # how wide the "unknown" band is in [-1,1]
        UNK_VECTOR = [0.0] * self.repr_dim  # explicit "I don't know" vector

        vec = np.array(vec)
        mask = np.abs(vec - UNK_CENTER) < UNK_WIDTH
        vec[mask] = vec[mask] * 0.2  # damp uncertainty
        if mask.mean() > 0.5:        # if most dims near 0, mark as UNK
            vec = np.array(UNK_VECTOR)
        vec = vec.tolist()

        self._repr_cache[token] = vec
        return vec

    def repr_for_ids(self, ids: List[int]) -> List[List[float]]:
        return [self.get_repr(self.id_to_token.get(int(i), "")) for i in ids]

    def export_repr_matrix(self, order: str = "id") -> List[List[float]]:
        if order != "id":
            raise ValueError("Only order='id' is supported currently.")
        mat: List[List[float]] = []
        for i in range(len(self.token_to_id)):
            tok = self.id_to_token.get(i, "")
            mat.append(self.get_repr(tok))
        return mat

    # ------------------------ save / load ------------------------

    def to_dict(self) -> Dict:
        merges_sorted = sorted(self.merge_ranks.items(), key=lambda kv: kv[1])
        merges = [{"a": a, "b": b, "c": c, "rank": r} for ((a, b, c), r) in merges_sorted]
        parent_map = [{"merged": k, "a": v[0], "b": v[1], "c": v[2]} for k, v in self.parent_map.items()]
        return {
            "version": TOK_VERSION,
            "type": "ternary-bpe",
            "vocab_size": self.vocab_size,
            "specials": self.SPECIALS,
            "token_to_id": dict(sorted(self.token_to_id.items(), key=lambda kv: kv[1])),
            "merges": merges,
            "parent_map": parent_map,
            "frozen": self._frozen,
            "repr": {
                "dim": self.repr_dim,
                "seed": self.repr_seed,
                "clamp": self.repr_clamp,
            },
        }

    @classmethod
    def from_dict(cls, obj: Dict) -> "TernaryTokenizer":
        if obj.get("type") != "ternary-bpe":
            raise ValueError(f"Unexpected tokenizer type: {obj.get('type')}")
        cfg = obj.get("repr", {})
        tok = cls(
            vocab_size=int(obj.get("vocab_size", 50000)),
            repr_dim=int(cfg.get("dim", 64)),
            repr_seed=int(cfg.get("seed", 0)),
            repr_clamp=str(cfg.get("clamp", "tanh")),
        )
        tok.token_to_id = {k: int(v) for k, v in obj["token_to_id"].items()}
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok.PAD_ID = tok.token_to_id["[PAD]"]
        tok.UNK_ID = tok.token_to_id["[UNK]"]
        tok.BOS_ID = tok.token_to_id["[BOS]"]
        tok.EOS_ID = tok.token_to_id["[EOS]"]

        tok.merge_ranks = {}
        for m in obj.get("merges", []):
            trip = (m["a"], m["b"], m["c"])
            tok.merge_ranks[trip] = int(m["rank"])

        tok.parent_map = {}
        if "parent_map" in obj:
            for rec in obj["parent_map"]:
                tok.parent_map[rec["merged"]] = (rec["a"], rec["b"], rec["c"])
        else:
            for (a, b, c), _rk in sorted(tok.merge_ranks.items(), key=lambda kv: kv[1]):
                merged = a + b + c
                tok.parent_map.setdefault(merged, (a, b, c))

        tok._frozen = bool(obj.get("frozen", False))
        tok._repr_cache = {}
        return tok

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "TernaryTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls.from_dict(obj)


    # ------------------------ merges I/O ------------------------

    def export_merges(self) -> List[Tuple[str, str, str]]:
        return [k for (k, _) in sorted(self.merge_ranks.items(), key=lambda kv: kv[1])]

    def import_merges(self, merges: List[Tuple[str, str, str]]):
        if self._frozen:
            raise RuntimeError("Tokenizer is frozen. Unfreeze before importing merges.")
        self.merge_ranks.clear()
        self.parent_map.clear()
        rank = 0
        for (a, b, c) in merges:
            merged = a + b + c
            if merged not in self.token_to_id:
                nid = len(self.token_to_id)
                self.token_to_id[merged] = nid
                self.id_to_token[nid] = merged
            self.merge_ranks[(a, b, c)] = rank
            self.parent_map[merged] = (a, b, c)
            rank += 1
        self._repr_cache.clear()
