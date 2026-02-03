#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

WER_RE = re.compile(r"%WER\s+([0-9]+(?:\.[0-9]+)?)")
CER_RE = re.compile(r"%CER\s+([0-9]+(?:\.[0-9]+)?)")

def parse_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_metric_file(p: Path):
    """Return (wer, cer). None if not present."""
    if not p.is_file():
        return None, None
    text = p.read_text(encoding="utf-8", errors="ignore")
    m_wer = WER_RE.search(text)
    m_cer = CER_RE.search(text)
    wer = float(m_wer.group(1)) if m_wer else None
    cer = float(m_cer.group(1)) if m_cer else None
    return wer, cer

def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_root", help="Folder containing locale subfolders (e.g., .../FT/0/eo)")
    ap.add_argument("--last_locale", default="ia", help="Final model tag: wer_test_after_<last_locale>.txt")
    ap.add_argument("--base_locales", required=True, help='Comma-separated base locales')
    ap.add_argument("--new_locales", required=True, help='Comma-separated new locales')
    ap.add_argument("--out_csv", default=None, help="Output CSV (default: <exp_root>/before_after_<last>.csv)")
    ap.add_argument("--strict", action="store_true", help="Error if required files are missing")
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    last = args.last_locale
    out_csv = Path(args.out_csv) if args.out_csv else exp_root / f"final_after_{last}.csv"

    base_locales = parse_list(args.base_locales)
    new_locales  = parse_list(args.new_locales)

    rows = []
    missing = []

    # ---------- BASE: before + after ----------
    for loc in base_locales:
        loc_dir = exp_root / loc
        before_f = loc_dir / "wer_test_before.txt"
        after_f  = loc_dir / f"wer_test_after_{last}.txt"

        b_wer, b_cer = parse_metric_file(before_f)
        a_wer, a_cer = parse_metric_file(after_f)

        if (b_wer is None and b_cer is None) or (a_wer is None and a_cer is None):
            missing.append((loc, "base", str(before_f), str(after_f)))
            if args.strict:
                continue

        d_wer = (a_wer - b_wer) if (a_wer is not None and b_wer is not None) else None
        rows.append((
            loc, "base",
            b_wer, a_wer, d_wer,
            b_cer, a_cer,
            str(before_f), str(after_f)
        ))

    # ---------- NEW: after only ----------
    for loc in new_locales:
        loc_dir = exp_root / loc
        after_f  = loc_dir / f"wer_test_after_{last}.txt"
        a_wer, a_cer = parse_metric_file(after_f)

        if a_wer is None and a_cer is None:
            missing.append((loc, "new", "", str(after_f)))
            if args.strict:
                continue

        rows.append((
            loc, "new",
            None, a_wer, None,
            None, a_cer,
            "", str(after_f)
        ))

    if args.strict and missing:
        msg = "\n".join([f"- {loc} ({grp}): before={bf} after={af}" for loc, grp, bf, af in missing])
        raise SystemExit(f"Missing files:\n{msg}")

    if not rows:
        raise SystemExit("No rows collected. Check exp_root and locale names.")

    # summary stats
    base_before = [r[2] for r in rows if r[1] == "base"]
    base_after  = [r[3] for r in rows if r[1] == "base"]
    base_delta  = [r[4] for r in rows if r[1] == "base"]
    new_after   = [r[3] for r in rows if r[1] == "new"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "locale", "group",
            "WER_before(%)", "WER_after(%)", "WER_delta(after-before)",
            "CER_before(%)", "CER_after(%)",
            "before_file", "after_file"
        ])

        # base block
        for r in rows:
            if r[1] != "base":
                continue
            loc, grp, bwer, awer, dwer, bcer, acer, bf, af = r
            w.writerow([
                loc, grp,
                "" if bwer is None else f"{bwer:.2f}",
                "" if awer is None else f"{awer:.2f}",
                "" if dwer is None else f"{dwer:.2f}",
                "" if bcer is None else f"{bcer:.2f}",
                "" if acer is None else f"{acer:.2f}",
                bf, af
            ])

        w.writerow([])

        # new block
        for r in rows:
            if r[1] != "new":
                continue
            loc, grp, bwer, awer, dwer, bcer, acer, bf, af = r
            w.writerow([
                loc, grp,
                "", "" if awer is None else f"{awer:.2f}", "",
                "", "" if acer is None else f"{acer:.2f}",
                "", af
            ])

        # summary
        w.writerow([])
        w.writerow(["SUMMARY", "", "", "", "", "", "", "", ""])
        w.writerow(["base_mean_WER_before", "base", f"{mean(base_before):.2f}" if mean(base_before) is not None else "", "", "", "", "", "", ""])
        w.writerow(["base_mean_WER_after",  "base", "", f"{mean(base_after):.2f}" if mean(base_after) is not None else "", "", "", "", "", ""])
        w.writerow(["base_mean_WER_delta",  "base", "", "", f"{mean(base_delta):.2f}" if mean(base_delta) is not None else "", "", "", ""])
        w.writerow(["new_mean_WER_after",   "new",  "", f"{mean(new_after):.2f}" if mean(new_after) is not None else "", "", "", "", ""])

        if missing:
            w.writerow([])
            w.writerow(["MISSING", "", "", "", "", "", "", "", ""])
            for loc, grp, bf, af in missing:
                w.writerow([loc, grp, "", "", "", "", "", bf, af])

    print(f"[OK] wrote: {out_csv}")
    if missing:
        print(f"[WARN] missing files: {len(missing)} (see CSV bottom)")

if __name__ == "__main__":
    main()
