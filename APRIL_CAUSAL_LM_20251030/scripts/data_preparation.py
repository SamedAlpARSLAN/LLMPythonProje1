# scripts/data_preparation.py
#
# Bu script eğitim datasını hazırlar.
#
# 1. Bir veya daha fazla PDF/TXT dosyasını okur.
#    (şifreli / açılamayan PDF varsa SKIP eder, crash etmez)
# 2. Her dokümanı temizler (basic_clean).
# 3. Dokümanlar arasına <eos> gibi ayırıcı koyup tek büyük korpus oluşturur.
#    - Tokenizer'da <eos>, </s>, <EOS> varsa onu kullanır,
#      yoksa sadece çift newline koyar.
# 4. AprilTokenizer ile token ID'lere çevirir.
# 5. Ortaya çıkan ID dizisini .pt olarak kaydeder.
# 6. Ayrıca meta.json yazar (num_documents, total_tokens, vocab_size, pad_id, eos_id, source_files).
#
# Kullanım (PowerShell):
#
# (.venv) PS> python scripts\data_preparation.py `
#   --tokenizer model\tokenizer.json `
#   --inputs data\raw `
#   --out_pt data\processed\tokens.pt `
#   --out_meta data\processed\meta.json
#

import argparse
import json
import glob
from pathlib import Path
from typing import List

import torch
from PyPDF2 import PdfReader
import sys
import re

# --------- Yol ayarı: april_* modüllerini import edebilmek için ---------
ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIRS = [
    ROOT / "src" / "april",
    ROOT / "src",
    ROOT,
]
for d in CANDIDATE_DIRS:
    if d.exists():
        sys.path.insert(0, str(d))

from april_tokenizer import AprilTokenizer  # artık görülebilir durumda


# --------- Yardımcı fonksiyonlar ---------

def read_pdf_text(pdf_path: Path) -> str:
    """
    PDF'den metin çıkarır. Sayfaları \n\n ile birleştirir.
    Bu fonksiyon şifreli PDF'te patlayabilir; o yüzden çağıran yerde try/except var.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        raw = p.extract_text() or ""
        pages.append(raw.strip())
    return "\n\n".join(pages).strip()


def read_txt_text(txt_path: Path) -> str:
    """
    TXT dosyasını utf-8 ile oku.
    """
    return txt_path.read_text(encoding="utf-8", errors="ignore")


def expand_inputs(inputs: List[str]) -> List[Path]:
    """
    Kullanıcının verdiği girdileri aç:
      - Tek tek dosya adı
      - Klasör (içindeki .pdf / .txt dosyalarını tara - recursive)
      - wildcard (*.pdf gibi)
    Dönüş: mevcut, benzersiz Path listesi (deterministik sıralı)
    """
    all_paths: List[Path] = []

    for inp in inputs:
        s = str(inp)
        p = Path(inp)

        # wildcard mı?
        if any(ch in s for ch in ["*", "?"]):
            for match in glob.glob(s):
                all_paths.append(Path(match))
            continue

        # klasör mü?
        if p.is_dir():
            for cand in list(p.rglob("*.pdf")) + list(p.rglob("*.txt")):
                all_paths.append(cand)
            continue

        # normal dosya ise direkt ekle
        all_paths.append(p)

    # uniq + var olanlar
    uniq: List[Path] = []
    seen = set()
    for p in all_paths:
        if p.exists() and p not in seen:
            uniq.append(p)
            seen.add(p)

    # deterministik sıraya sok: önce yol ismine göre sırala
    uniq = sorted(uniq, key=lambda x: str(x).lower())
    return uniq


def basic_clean(text: str) -> str:
    """
    PDF/TXT kaynaklı saçma boşlukları normalize eder.
    Türkçe karakterlere dokunmuyoruz.
    """
    # Windows CRLF -> LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # tab -> boşluk
    text = text.replace("\t", " ")

    # birden fazla boşluk -> tek boşluk (satır içi)
    text = re.sub(r"[ ]{2,}", " ", text)

    # 4+ boş satırı -> 2 boş satıra indir
    text = re.sub(r"\n{4,}", "\n\n", text)

    # baş/son trim
    text = text.strip()
    return text


def build_corpus_documents(file_paths: List[Path]) -> List[str]:
    """
    Her dosyayı oku -> temizle -> doküman listesi döndür.
    Her eleman tek bir dokümanı temsil eden string.

    Şifreli / okunamayan PDF varsa SKIP ederiz,
    hata yüzünden script komple çökmez.
    """
    docs: List[str] = []
    skipped: List[str] = []

    for p in file_paths:
        suffix = p.suffix.lower()

        raw = ""
        try:
            if suffix == ".pdf":
                raw = read_pdf_text(p)
            elif suffix == ".txt":
                raw = read_txt_text(p)
            else:
                # desteklenmeyen uzantı -> geç
                continue
        except Exception as e:
            # Örn: şifreli PDF, DRM, bozuk dosya vs.
            skipped.append(f"{p.name} -> {e}")
            continue

        cleaned = basic_clean(raw)

        if cleaned.strip():
            docs.append(cleaned)

    if skipped:
        print("[uyarı] Aşağıdaki dosyalar okunamadı / şifreli, SKIP edildi:")
        for msg in skipped:
            print("   -", msg)

    return docs


def choose_eos_token(tokenizer: AprilTokenizer) -> str | None:
    """
    Tokenizer vocab'ında hangi eos benzeri token varsa onu seç.
    Yoksa None döndür.
    """
    for cand in ["<eos>", "</s>", "<EOS>"]:
        if cand in tokenizer.vocab:
            return cand
    return None


def join_docs_with_eos(docs: List[str], tokenizer: AprilTokenizer) -> str:
    """
    Doküman listesini tek büyük metin haline getir.
    Eğer tokenizer içinde <eos>/<EOS></s> gibi bir özel token varsa
    onu ayraç olarak kullan.
    Yoksa sadece çift newline ile birleştir.
    """
    if not docs:
        return ""

    eos_tok = choose_eos_token(tokenizer)

    if eos_tok is None:
        # fallback: sadece boş satırlarla ayır
        return "\n\n".join(docs)

    # örnek çıktı:
    # "doc1 ... \n <eos> \n doc2 ... \n <eos> \n doc3 ..."
    sep = f"\n {eos_tok} \n"
    return sep.join(docs)


def encode_corpus(tokenizer: AprilTokenizer, corpus_text: str) -> torch.Tensor:
    """
    Büyük korpus string'ini al, AprilTokenizer.encode() ile ID'lere çevir.
    Dönüş: 1D LongTensor.
    """
    if not corpus_text.strip():
        return torch.tensor([], dtype=torch.long)

    ids_tensor = tokenizer.encode(corpus_text)  # bu zaten Tensor dönüyor bizde
    if not isinstance(ids_tensor, torch.Tensor):
        ids_tensor = torch.tensor(ids_tensor, dtype=torch.long)

    return ids_tensor.to(dtype=torch.long)


# --------- main() ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="tokenizer.json yolu (ör: model/tokenizer.json)",
    )
    ap.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Eğitim ham verileri: .pdf, .txt, klasör veya wildcard (*.pdf)",
    )
    ap.add_argument(
        "--out_pt",
        type=str,
        required=True,
        help="Çıkacak token id dizisi (torch.save) ör: data/processed/tokens.pt",
    )
    ap.add_argument(
        "--out_meta",
        type=str,
        required=True,
        help="Çıkacak meta json dosyası ör: data/processed/meta.json",
    )
    args = ap.parse_args()

    tok_path = Path(args.tokenizer)
    out_pt_path = Path(args.out_pt)
    out_meta_path = Path(args.out_meta)

    out_pt_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)

    # tokenizer yükle
    tokenizer = AprilTokenizer(str(tok_path))

    # giriş dosyalarını çöz
    file_paths = expand_inputs(args.inputs)

    if not file_paths:
        raise SystemExit("Hiç dosya bulunamadı. --inputs parametresini kontrol et.")

    # dosyaları oku + temizle (şifreli olanları SKIP)
    docs = build_corpus_documents(file_paths)

    if not docs:
        raise SystemExit("Hiç okunabilir doküman yok. Hepsi boş ya da şifreli olabilir.")

    print(f"[OK] {len(docs)} doküman işlendi.")

    # tek korpus metni oluştur (<eos> benzeri ayraçla)
    corpus_text = join_docs_with_eos(docs, tokenizer)

    # ID'lere çevir
    token_tensor = encode_corpus(tokenizer, corpus_text)
    print(f"[OK] Toplam {token_tensor.numel()} token ID üretildi.")

    # kaydet (.pt)
    torch.save(token_tensor, out_pt_path)
    print(f"[OK] Kaydedildi -> {out_pt_path}")

    # meta bilgisi
    pad_id = tokenizer.vocab.get("<pad>")
    eos_id = None
    eos_tok_candidate = choose_eos_token(tokenizer)
    if eos_tok_candidate is not None:
        eos_id = tokenizer.vocab.get(eos_tok_candidate)

    meta = {
        "num_documents": len(docs),
        "total_tokens": int(token_tensor.numel()),
        "vocab_size": tokenizer.vocab_size,
        "pad_id": pad_id,
        "eos_id": eos_id,
        "source_files": [str(p) for p in file_paths],
    }

    out_meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[OK] Meta -> {out_meta_path}")


if __name__ == "__main__":
    main()
