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

    def encode(self, text: str) -> list[int]:
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx2char.get(i, "") for i in ids)

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
