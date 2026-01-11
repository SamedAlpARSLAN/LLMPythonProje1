# scripts/train_april.py
#
# April modelini eğitir.

import argparse
from pathlib import Path
import sys
import json
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# ---------- import path ayarı ----------
ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIRS = [
    ROOT / "src" / "april",
    ROOT / "src",
    ROOT,
]
for d in CANDIDATE_DIRS:
    if d.exists():
        sys.path.insert(0, str(d))

from april_tokenizer import AprilTokenizer
from april_model import AprilModel
from text_dataset import create_data_loader


def build_dataloader(
    token_ids,
    tokenizer,
    context_length: int,
    batch_size: int,
    stride: int,
    device: str,
):
    """
    token_ids: uzun ID listesi
    tokenizer: AprilTokenizer objesi (pad_id almak için)
    """
    pad_id = tokenizer.vocab["<pad>"]
    return create_data_loader(
        token_ids=token_ids,
        context_length=context_length,
        stride=stride,
        batch_size=batch_size,
        pad_id=pad_id,
        shuffle=True,
        device=device,
        num_workers=0,
    )


def save_checkpoint(
    out_dir: Path,
    model: AprilModel,
    config: dict,
    train_state: dict,
    step: int,
):
    """
    model state + config + train_state tek .pth'ye yaz.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"april_step{step:06d}.pth"

    payload = {
        "model_state": model.state_dict(),
        "config": config,
        "train_state": train_state,
    }

    torch.save(payload, ckpt_path)
    print(f"[save] checkpoint yazıldı -> {ckpt_path}")


def train_one_epoch(
    model: AprilModel,
    dataloader,
    optimizer: AdamW,
    device: str,
    pad_id: int,
    global_step_start: int,
    print_every: int,
    save_every: int,
    out_dir: Path,
    base_config: dict,
    epoch_index: int,
):
    """
    Bir epoch boyunca iterate et.
    Loss hesapla, backward yap.
    Düzenli print et ve düzenli checkpoint kaydet.
    """

    model.train()
    running_loss = 0.0
    global_step = global_step_start

    loop = tqdm(dataloader, desc=f"[epoch {epoch_index}]")
    for xb, yb in loop:
        # xb, yb shape: [B, T]
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        # model output: [B, T, V]
        logits = model(xb)

        # CrossEntropyLoss: flatten
        # logits -> [B*T, V], targets -> [B*T]
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            yb.view(B * T),
            ignore_index=pad_id,
        )

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        global_step += 1

        # tqdm bar üst yazı
        loop.set_postfix(loss=loss_val)

        # ara print
        if global_step % print_every == 0:
            avg_loss = running_loss / print_every
            print(f"[log] step={global_step} avg_loss={avg_loss:.4f}")
            running_loss = 0.0

        # ara checkpoint
        if global_step % save_every == 0:
            train_state = {
                "epoch": epoch_index,
                "global_step": global_step,
                "loss": float(loss_val),
            }
            save_checkpoint(
                out_dir=out_dir,
                model=model,
                config=base_config,
                train_state=train_state,
                step=global_step,
            )

    # epoch sonunda son loss bilgisi geri dön
    final_loss = float(loss_val)
    return global_step, final_loss


def load_tokens_list(tokens_pt_path: Path):
    """
    data_preparation.py'nin ürettiği tokens.pt dosyasını yükler.
    Genelde torch.save(LongTensor) olarak saklıyoruz.
    Burada Python list[int] olarak geri döndürüyoruz.
    """
    obj = torch.load(str(tokens_pt_path), map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return list(obj)


def load_checkpoint_for_resume(ckpt_path: Path, device: str):
    """
    Bizim formatımızdaki bir checkpoint'i oku.
    Dönüş:
      model_state (dict),
      cfg (dict),
      train_state (dict)

    Not: optimizer state kaydetmiyoruz. Devam ederken optimizer'ı sıfırdan kuracağız.
    """
    payload = torch.load(str(ckpt_path), map_location=device)
    model_state = payload["model_state"]
    cfg = payload.get("config", {})
    train_state = payload.get("train_state", {})
    return model_state, cfg, train_state


def main():
    ap = argparse.ArgumentParser()

    # veri / tokenizer
    ap.add_argument("--tokenizer", type=str, required=True,
                    help="model/tokenizer.json yolu")
    ap.add_argument("--tokens_pt", type=str, required=True,
                    help="data/processed/tokens.pt (ID listesi)")

    # model mimarisi
    ap.add_argument("--embedding_dim", type=int, required=True)
    ap.add_argument("--num_heads", type=int, required=True)
    ap.add_argument("--num_layers", type=int, required=True)
    ap.add_argument("--context_length", type=int, required=True)

    # eğitim hiperparametreleri
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--stride", type=int, default=128,
                    help="window kaydırma adımı (varsayılan 128)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--print_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=500)

    # çıktı
    ap.add_argument("--out_dir", type=str, required=True,
                    help="checkpoint klasörü (örn: model/checkpoints_v3)")

    # YENİ: kaldığın yerden devam
    ap.add_argument("--resume_checkpoint", type=str, default=None,
                    help="(opsiyonel) mevcut bir april_stepXXXXX.pth dosyası. "
                         "Verirsen o ağırlıklardan devam ederiz.")

    args = ap.parse_args()

    # cihaz seç
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    # tokenizer yükle
    tok = AprilTokenizer(args.tokenizer)
    vocab_size = tok.vocab_size
    pad_id = tok.vocab["<pad>"]

    print(f"[info] vocab_size = {vocab_size}")
    print(f"[info] pad_id     = {pad_id}")

    # token_ids yükle
    token_ids = load_tokens_list(Path(args.tokens_pt))
    print(f"[info] Toplam token sayısı: {len(token_ids)}")

    # dataloader hazırla
    dataloader = build_dataloader(
        token_ids=token_ids,
        tokenizer=tok,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=args.stride,
        device=device,
    )

    # -------------------------------------------------
    # Model kur / veya checkpoint'ten yükle
    # -------------------------------------------------
    model = AprilModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        context_length=args.context_length,
        num_layers=args.num_layers,
        device=device,
    ).to(device)

    global_step_start = 0
    start_epoch_info = -1
    resume_loss_info = None

    if args.resume_checkpoint is not None:
        ckpt_path = Path(args.resume_checkpoint)
        print(f"[resume] Checkpoint yükleniyor: {ckpt_path}")

        model_state, old_cfg, train_state = load_checkpoint_for_resume(
            ckpt_path, device=device
        )

        # ağırlıkları modele koy
        model.load_state_dict(model_state, strict=True)
        print("[resume] model_state yüklendi ✅")

        # global_step kaldığı yerden devam etsin
        if "global_step" in train_state:
            global_step_start = int(train_state["global_step"])
        if "epoch" in train_state:
            start_epoch_info = int(train_state["epoch"])
        if "loss" in train_state:
            resume_loss_info = float(train_state["loss"])

        print(f"[resume] kaldığın global_step = {global_step_start}")
        if start_epoch_info >= 0:
            print(f"[resume] önceki epoch        = {start_epoch_info}")
        if resume_loss_info is not None:
            print(f"[resume] önceki loss         = {resume_loss_info:.4f}")

        # küçük uyumluluk kontrolü (sadece uyarı amaçlı)
        mismatch_msgs = []
        if "embedding_dim" in old_cfg and old_cfg["embedding_dim"] != args.embedding_dim:
            mismatch_msgs.append("embedding_dim")
        if "num_heads" in old_cfg and old_cfg["num_heads"] != args.num_heads:
            mismatch_msgs.append("num_heads")
        if "num_layers" in old_cfg and old_cfg["num_layers"] != args.num_layers:
            mismatch_msgs.append("num_layers")
        if "context_length" in old_cfg and old_cfg["context_length"] != args.context_length:
            mismatch_msgs.append("context_length")
        if "vocab_size" in old_cfg and old_cfg["vocab_size"] != vocab_size:
            mismatch_msgs.append("vocab_size")

        if mismatch_msgs:
            print("[uyarı][resume] Argümanlarla eski checkpoint config arasında fark var:",
                  ", ".join(mismatch_msgs))
            print("Bu genelde istemediğimiz bir şey. Aynı mimariyle devam ettiğinden emin ol.")

    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # config metadatası (checkpoint içine yazılacak)
    base_config = {
        "vocab_size": vocab_size,
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "context_length": args.context_length,
        "num_layers": args.num_layers,
    }

    out_dir = Path(args.out_dir)

    # eğitim döngüsü
    global_step = global_step_start
    last_loss = resume_loss_info

    for epoch_idx in range(args.epochs):
        # bilgi amaçlı ekrana yaz
        print(f"[epoch] {epoch_idx} başlıyor... (global_step start {global_step})")

        global_step, last_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            pad_id=pad_id,
            global_step_start=global_step,
            print_every=args.print_every,
            save_every=args.save_every,
            out_dir=out_dir,
            base_config=base_config,
            epoch_index=epoch_idx,
        )

    # epochlar bitti -> final checkpoint
    final_train_state = {
        "epoch": args.epochs - 1,
        "global_step": global_step,
        "loss": float(last_loss) if last_loss is not None else None,
    }

    save_checkpoint(
        out_dir=out_dir,
        model=model,
        config=base_config,
        train_state=final_train_state,
        step=global_step,
    )

    print("[done] Eğitim tamamlandı.")


if __name__ == "__main__":
    main()
