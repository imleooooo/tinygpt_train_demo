import json


class CharTokenizer:
    """Character-level tokenizer. Maps each unique character to an integer index."""

    def __init__(self):
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self.vocab_size: int = 0

    def build(self, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(chars)
        return self

    def encode(self, text: str, errors: str = "raise") -> list[int]:
        """Encode text to token ids.

        Args:
            text: Input string.
            errors: How to handle characters not in the vocabulary.
                "raise" (default) — raise ValueError listing the unseen chars.
                "ignore" — silently drop unseen chars.
        """
        if errors == "raise":
            unseen = sorted({ch for ch in text if ch not in self.char2idx})
            if unseen:
                chars_repr = ", ".join(repr(c) for c in unseen[:10])
                raise ValueError(
                    f"Prompt contains {len(unseen)} character(s) not in the training vocabulary: "
                    f"{chars_repr}. Use errors='ignore' to drop them."
                )
            return [self.char2idx[ch] for ch in text]
        # errors == "ignore"
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx2char.get(i, "") for i in ids)

    @classmethod
    def from_vocab(cls, char2idx: dict[str, int]) -> "CharTokenizer":
        """Reconstruct a tokenizer from a char2idx mapping (e.g. stored in a checkpoint)."""
        tok = cls()
        tok.char2idx = char2idx
        tok.idx2char = {int(i): ch for ch, i in char2idx.items()}
        tok.vocab_size = len(char2idx)
        return tok

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2idx": self.char2idx}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.char2idx = data["char2idx"]
        tok.idx2char = {int(i): ch for ch, i in tok.char2idx.items()}
        tok.vocab_size = len(tok.char2idx)
        return tok
