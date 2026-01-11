import torch
import torch.nn as nn
import torch.nn.functional as F

from april_decoder_block import AprilDecoderBlock
from april_embedding import AprilEmbedding


class AprilModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        num_layers: int,
        device: str,
    ):
        super().__init__()

        # ----- sakladığımız alanlar -----
        self.device = device
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # ----- model katmanları -----
        # embedding
        self.embedding = AprilEmbedding(
            vocab_size,
            embedding_dim,
            context_length,
            device,
        )

        # transformer decoder blokları
        self.layers = nn.Sequential(
            *[
                AprilDecoderBlock(
                    embedding_dim,
                    num_heads,
                    context_length,
                    device,
                )
                for _ in range(num_layers)
            ]
        )

        # dil modeli çıkışı (LM head)
        # DİKKAT: bias varsayılan True, ve device parametresi senin versiyonunla aynı.
        # Bunu değiştirmiyoruz ki checkpoint load ederken hata çıkmasın.
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] int64 (token id'leri)
        dönüş: [B, T, V] logits
        """
        h = self.embedding(x)   # [B, T, E]
        h = self.layers(h)      # [B, T, E]
        logits = self.lm_head(h)  # [B, T, V]
        return logits

    # --- nucleus/top-k yardımcıları (senin orijinalinden) ---
    @staticmethod
    def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        logits: [V]
        top_p: nucleus threshold (0-1)

        Bu fonksiyon senin eski kodundan aynen alındı;
        sadece yeniden kullanıyoruz.
        """
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # top_p üzerini -inf yap
        remove = cumprobs > top_p
        # ilk token her zaman kalsın diye kaydırma hilesi
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

        # logits orijinal sırasına geri dağılacak
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(0, sorted_idx, sorted_logits)

        return filtered

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 64,
        top_p: float = 1.0,
    ):
        """
        x: [T] LongTensor (tek bir başlangıç dizisi)
        return: tüm token id'leri (list[int]), yani prompt + üretilenler

        CRITICAL FIX:
        - Her adımda modele sadece son self.context_length kadar token veriyoruz.
          Böylece attention mask boyutu ile sequence length hep aynı kalıyor.
          Senin aldığın şu hatayı bu çözüyor:
          RuntimeError: shape '[1, 1, 129, 129]' is invalid ...
        - Checkpoint'i yeniden eğitmeye gerek YOK.
        """

        # x tek sequence olabilir ama biz emin olalım
        if isinstance(x, torch.Tensor):
            tokens = x.to(self.device, dtype=torch.long).view(-1).tolist()
        else:
            # teorik olarak list[int] gelebilir ama bizim run_inference tensor gönderiyor
            tokens = list(x)

        # Eğer prompt zaten context_length'ten uzunsa,
        # modele vereceğimiz ilk context'te sıkıntı olmasın diye kırp.
        # (NOT: tokens listesinin tamamını saklıyoruz, sadece modele beslerken kırpacağız.
        # yani decode ederken promptun tamamı hâlâ elde mevcut olacak.)
        # Burada sadece ilk forward için güvenlik olsun diye TAIL'i tutuyoruz.
        if len(tokens) > self.context_length:
            tokens_tail = tokens[-self.context_length:]
        else:
            tokens_tail = tokens

        # max_new_tokens kadar üret
        for _ in range(max_new_tokens):
            # MODELE VERİLECEK GİRİŞ
            # Her adımda sadece tail (son context_length token) ile ileri geçeceğiz.
            cur_ctx = tokens_tail[-self.context_length:]
            inp = torch.tensor(cur_ctx, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, T_ctx]

            # Modelden logits al (son konumun dağılımı)
            logits = self.forward(inp)[0, -1, :]  # [V]

            # --- sampling sırası seninkinle aynı tutuldu ---
            # 1) top-k
            if top_k and 0 < top_k < logits.numel():
                vals, idxs = torch.topk(logits, k=top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(0, idxs, vals)
                logits = mask

            # 2) top-p
            if 0.0 < top_p < 1.0:
                logits = self._top_p_filtering(logits, top_p)

            # 3) temperature
            if temperature and temperature != 1.0:
                logits = logits / temperature

            # 4) softmax -> örnekleme
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            # global token listesine ekle
            tokens.append(next_id)

            # context tail'e de ekle ki bir sonraki turda modele bunu verelim
            tokens_tail.append(next_id)
            # tail çok uzarsa kırp (BURA ASIL FİX)
            if len(tokens_tail) > self.context_length:
                tokens_tail = tokens_tail[-self.context_length:]

            # <eos> varsa erkenden kes (senin orijinal kodunda vardı, korudum)
            if next_id == 59:
                break

        return tokens
