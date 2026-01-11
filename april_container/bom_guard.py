#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, sys

TEXT_EXTS = {".json",".txt",".model",".vocab",".cfg",".py",".token",".bpe",".md",".ini",".toml",".yaml",".yml"}
FORCE_FILES = {"config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","generation_config.json","merges.txt","vocab.json"}

def is_text_candidate(path: str) -> bool:
    name = os.path.basename(path).lower()
    if name in FORCE_FILES: return True
    return os.path.splitext(path)[1].lower() in TEXT_EXTS

def scrub_bom(path: str, dry: bool):
    with open(path, "rb") as f:
        raw = f.read()
    data = raw
    changed = False

    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
        changed = True
    try:
        txt = data.decode("utf-8")
        if "\ufeff" in txt:
            data = txt.replace("\ufeff","").encode("utf-8")
            changed = True
    except UnicodeDecodeError:
        return "binary", len(raw)

    if not changed:
        return "unchanged", len(raw)

    if dry:
        return "unchanged", len(data)

    try:
        with open(path, "wb") as f:
            f.write(data)
        return "fixed", len(data)
    except OSError as e:
        if getattr(e, "errno", None) == 30:  # EROFS
            return "skipped_ro", len(raw)
        raise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    dry = bool(args.dry or os.environ.get("BOM_GUARD_DRY") == "1")
    total=fixed=skipped=binaries=0

    for r,_,files in os.walk(args.root):
        for fn in files:
            p = os.path.join(r, fn)
            if not is_text_candidate(p): continue
            total += 1
            try:
                st,_ = scrub_bom(p, dry)
            except Exception:
                st = "error"
            if st == "fixed": fixed += 1
            elif st == "skipped_ro": skipped += 1
            elif st == "binary": binaries += 1

    print(f"[bom_guard] scanned={total} fixed={fixed} skipped_ro={skipped} binaries={binaries} dry={dry}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
