import json
import torch


class AprilTokenizer:
    """
    Çok basit sözlük-tabanlı bir tokenizer.
    - Vokabdaki anahtarlar doğrudan string parça (wordpiece) kabul edilir.
    - Boşluk, <pad>, <unk>, <eos> gibi özel token'ların vokabda olması beklenir.
    """
    def __init__(self, vocab_file: str):
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # Güvenli erişim yardımcıları
        self._id_space = self.vocab.get(" ", None)
        self._id_pad = self.vocab.get("<pad>", None)
        self._id_unk = self.vocab.get("<unk>", None)
        self._id_eos = self.vocab.get("<eos>", None)

    def _append_space_if_possible(self, arr: list):
        if self._id_space is not None:
            arr.append(self._id_space)

    def encode(self, text: str) -> torch.Tensor:
        """
        Basit greedy parçalama:
        - Metni boşluklara göre kelimelere ayırır.
        - Kelime içini en uzun eşleşmeyle vokab parçalarına böler.
        - Kelimeler arası ' ' tokenını ekler.
        """
        tokens = []
        for word in text.split():
            i = 0
            L = len(word)
            while i < L:
                found = False
                # en uzun eşleşme
                for j in range(L, i, -1):
                    sub = word[i:j]
                    if sub in self.vocab:
                        tokens.append(self.vocab[sub])
                        i = j
                        found = True
                        break
                if not found:
                    # bilinmeyen karakter/parça
                    if self._id_unk is None:
                        raise KeyError("<unk> vokabda yok.")
                    tokens.append(self._id_unk)
                    i += 1
            # kelimeler arası boşluk
            self._append_space_if_possible(tokens)

        # sonda fazladan boşluk eklenmesin
        if len(tokens) > 0 and not text.endswith(" ") and self._id_space is not None:
            tokens.pop()

        return torch.tensor(tokens, dtype=torch.long)

    def encode_batch(self, texts, context_length: int) -> torch.Tensor:
        out = []
        for t in texts:
            ids = self.encode(t).tolist()
            if len(ids) > context_length:
                ids = ids[:context_length]
            else:
                if self._id_pad is None:
                    raise KeyError("<pad> vokabda yok.")
                ids = ids + [self._id_pad] * (context_length - len(ids))
            out.append(ids)
        return torch.tensor(out, dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return "".join(self.id_to_token.get(int(i), "") for i in ids)

    def tokenize(self, text: str):
        return [self.id_to_token[i] for i in self.encode(text).tolist()]
