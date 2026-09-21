"""
Production-grade Byte Pair Encoding (BPE) Tokenizer Engine.
Supports GPT-4 style regex pre-tokenization and special token handling.
"""
import regex as re
from typing import Dict, List, Tuple, Optional
from .base import get_stats, merge, replace_control_characters

# GPT-4 split pattern for pre-tokenization
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self, pattern: Optional[str] = None):
        self.pattern = pattern if pattern is not None else GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens: Dict[int, str] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = self._build_initial_vocab()

    def _build_initial_vocab(self) -> Dict[int, bytes]:
        """Byte-level foundation: 0-255 mapped to raw individual bytes."""
        return {idx: bytes([idx]) for idx in range(256)}

    def register_special_tokens(self, special_tokens: Dict[str, int]):
        """Register custom control tokens (e.g. {'<|endoftext|>': 100257})."""
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        Train BPE merges on raw input text up to the specified vocabulary size.
        """
        assert vocab_size >= 256, "Vocabulary size must be at least 256 (raw byte baseline)."
        num_merges = vocab_size - 256

        # Step 1: Pre-tokenize string into disjoint chunks using regex
        text_chunks = self.compiled_pattern.findall(text)

        # Step 2: Convert chunks to UTF-8 byte integer sequences
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        # Step 3: Iteratively merge top consecutive pair across all chunks
        self.merges = {}
        self.vocab = self._build_initial_vocab()

        for i in range(num_merges):
            stats: Dict[Tuple[int, int], int] = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            if not stats:
                break

            # Find pair with highest frequency
            best_pair = max(stats, key=stats.get)
            idx = 256 + i

            # Replace pair in all chunks
            ids = [merge(chunk_ids, best_pair, idx) for chunk_ids in ids]

            # Register merge and populate vocab
            self.merges[best_pair] = idx
            self.vocab[idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {idx} ({stats[best_pair]} occurrences)")

    def _encode_chunk(self, byte_ids: List[int]) -> List[int]:
        """Encode a single UTF-8 byte list using learned merges."""
        ids = list(byte_ids)
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the merge with lowest index (earliest learned)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode(self, text: str, allowed_special: str = "none") -> List[int]:
        """
        Tokenize a string into an integer sequence.
        allowed_special: 'all' | 'none'
        """
        special_pattern = ""
        if self.special_tokens and allowed_special == "all":
            # Match any registered special token
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens.keys()) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        tokens = []
        for part in parts:
            if part in self.special_tokens and allowed_special == "all":
                tokens.append(self.special_tokens[part])
            elif part:
                chunks = self.compiled_pattern.findall(part)
                for chunk in chunks:
                    chunk_bytes = list(chunk.encode("utf-8"))
                    tokens.extend(self._encode_chunk(chunk_bytes))
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back into a UTF-8 string."""
        byte_parts = []
        for idx in ids:
            if idx in self.vocab:
                byte_parts.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                byte_parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token ID: {idx}")
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def save(self, prefix: str):
        """Save model merges and vocabulary to disk."""
        model_file = f"{prefix}.model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("bpe-tokenizer-v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for token, idx in self.special_tokens.items():
                f.write(f"{token} {idx}\n")
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} {idx}\n")

    def load(self, model_file: str):
        """Load tokenizer state from a .model file."""
        merges = {}
        special_tokens = {}
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "bpe-tokenizer-v1"
            self.pattern = f.readline().strip()
            self.compiled_pattern = re.compile(self.pattern)
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                line = f.readline().strip().split()
                special_tokens[line[0]] = int(line[1])
            for line in f:
                parts = line.strip().split()
                if parts:
                    merges[(int(parts[0]), int(parts[1]))] = int(parts[2])

        self.merges = merges
        self.register_special_tokens(special_tokens)
        self.vocab = self._build_initial_vocab()
        for (p0, p1), idx in merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
