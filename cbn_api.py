#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# original api has at least 7k docs. lmaoo
import argparse, csv, hashlib, json, re, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter, Retry

CBN_API = "https://www.cbn.gov.ng/api/GetAllDocuments"  # site’s master feed
DOCS_HUB = "https://www.cbn.gov.ng/Documents/"          # shows the group taxonomy

DEFAULT_INCLUDE = [
    # Core regulatory terms
    r"\bregulation(s)?\b", r"\bregulatory\b",
    r"\bguideline(s)?\b", r"\bframework\b",
    r"\brule(s)?\b", r"\bpolicy\b", r"\bpolicies\b",
    r"\bcode(s)?\b", r"\bcircular(s)?\b",
    r"\bnotice(s)?\b", r"\bexposure\s*draft\b",
    
    # Compliance and obligations
    r"\bcompliance\b", r"\bobligation(s)?\b",
    r"\brequirement(s)?\b", r"\bstandard(s)?\b",
    r"\bprocedure(s)?\b", r"\bprocess(es)?\b",
    
    # Financial regulation specific terms
    r"\baml\b", r"\bcft\b", r"\bcpf\b",  # Anti-Money Laundering, Counter Financing of Terrorism, Counter Proliferation Financing
    r"\bkyc\b", r"\bknow.*your.*customer\b",
    r"\bcustomer.*due.*diligence\b", r"\bcdd\b",
    r"\bcapital.*adequacy\b", r"\bliquidity\b",
    r"\brisk.*management\b", r"\bgovernance\b",
    
    # Banking supervision terms
    r"\bsupervision\b", r"\bsupervisory\b",
    r"\bprudential\b", r"\benforcement\b",
    r"\bsanction(s)?\b", r"\bpenalt(y|ies)\b",
    
    # Legal and regulatory instruments
    r"\bact\b", r"\blaw\b", r"\bstatute\b",
    r"\bordonnance\b", r"\bdecree\b", r"\bdirective\b",
    r"\bmandate\b", r"\bauthorization\b", r"\blicense\b", r"\blicensing\b",
]
DEFAULT_EXCLUDE = [
    # Press releases, speeches, and communications
    r"\bpress\b", r"\bspeech(es)?\b", r"\bcommuniqu[eé]\b",
    
    # Reports, newsletters, and statistical content
    r"\breport\b", r"\bnewsletter\b", r"\bstatistical\b",
    r"\bbullion\b", r"\bupdate\b", r"\bnews\b",
    
    # Interest rates and monetary policy content
    r"\bdeposit.*lending.*rates?\b", r"\binterest.*rates?\b",
    r"\bweekly.*rates?\b", r"\brates?.*banking.*industry\b",
    r"\bmpc\b", r"\bmonetary.*policy.*committee\b",
    
    # Academic research and econometric terms
    r"\bdsge\b", r"\bardl\b", r"\bvar\b", r"\bcointegration\b",
    r"\basymmetry\b", r"\bthreshold\b", r"\bbayesian\b",
    r"\beconometric\b", r"\beconometrics\b", r"\bregression\b",
    
    # Surveys and business expectations
    r"\bsurvey\b", r"\bbusiness.*expectation\b", r"\bexpectation.*survey\b",
    
    # Food crisis and non-regulatory content
    r"\bfood.*crisis\b", r"\bcrisis\b", r"\bfood.*insecurity\b",
    
    # Balance of payments and statistical reports
    r"\bbalance.*of.*payments\b", r"\bbop\b",
    
    # Advertisement and ICT content
    r"\badvertisement\b", r"\bict.*facilities\b", r"\bupgrade\b",
    
    # Research studies and analysis
    r"\bstudy\b", r"\banalysis\b", r"\bresearch\b", r"\bacademic\b",
    r"\bworking.*paper\b", r"\bdiscussion.*paper\b",
]

PREFER_DEPTS = [
    "financial policy and regulation", "fprd",
    "banking supervision", "bsd",
    "payments", "payment system", "psmd",
    "other financial institutions", "ofi", "ofisd",
    "consumer protection", "ccd",
]

def mk_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "AutoComply-CBN-GroupedPull/1.0",
        "Accept": "application/json, text/plain, */*",
    })
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def looks_pdf(url: str) -> bool:
    return url.lower().endswith(".pdf") or ".pdf?" in url.lower()

def sha256_bytes(b: bytes) -> str:
    import hashlib
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def slugify(text: str, maxlen: int = 80) -> str:
    import html
    t = html.unescape(text or "")
    t = re.sub(r"[^\w\s\-\.]+", "", t)
    t = re.sub(r"\s+", "_", t.strip())
    return (t or "file")[:maxlen]

def parse_date_any(val: Any) -> tuple[Optional[str], Optional[str]]:
    if not val:
        return None, None
    raw = str(val).strip()
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"]
    for f in fmts:
        try:
            return datetime.strptime(raw, f).strftime("%Y-%m-%d"), raw
        except Exception:
            pass
    try:
        return datetime.fromisoformat(raw.replace("Z","+00:00")).date().strftime("%Y-%m-%d"), raw
    except Exception:
        return None, raw

def year_from_url(url: str) -> Optional[int]:
    m = re.search(r"/(19|20)\d{2}/", urlparse(url).path)
    if m:
        return int(m.group(0).strip("/").split("/")[0])
    return None

def compile_res(lst): return [re.compile(r, re.I) for r in lst]
def match_any(pats, text): return any(p.search(text) for p in pats)

def should_keep(title: str, category: str, dept: str, url: str,
                include_re, exclude_re) -> bool:
    blob = " ".join([title, category, dept, url]).lower()
    if match_any(exclude_re, blob):
        return False
    if match_any(include_re, blob):
        return True
    if any(h in blob for h in PREFER_DEPTS):
        return True
    if re.search(r"/FPRD/|/BSD/|/CCD/|/PSMD/|/OFISD/", url, re.I):
        return True
    return False

def download_pdf(session: requests.Session, url: str, out_dir: Path, title: str):
    try:
        r = session.get(url, timeout=60)
        if r.status_code >= 400:
            print(f"[warn] {r.status_code} {url}")
            return None, None, None
        data = r.content
        if not data.startswith(b"%PDF-"):
            print(f"[skip] not a PDF (magic) {url}")
            return None, None, None
    except Exception as e:
        print(f"[error] GET {url} -> {e}")
        return None, None, None

    h = sha256_bytes(data)
    name = slugify(title)
    if not name.lower().endswith(".pdf"): name += ".pdf"
    path = out_dir / name
    i = 1
    while path.exists():
        path = out_dir / f"{path.stem}_{i}{path.suffix}"
        i += 1
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f: f.write(data)
    tmp.replace(path)
    return path, h, len(data)

def main():
    ap = argparse.ArgumentParser(description="CBN grouped pull via API with categories")
    ap.add_argument("--out", type=str, default="regulations_corpus/banking_fintech/CBN")
    ap.add_argument("--sleep", type=float, default=0.7)
    ap.add_argument("--since", type=int, default=2015)
    ap.add_argument("--until", type=int, default=2100)
    ap.add_argument("--csv", type=str, default="cbn_grouped_list.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include", nargs="*", default=DEFAULT_INCLUDE)
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE)
    args = ap.parse_args()

    out_root = Path(args.out)
    manifest = out_root / "manifest.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)

    include_re = compile_res(args.include)
    exclude_re = compile_res(args.exclude)

    s = mk_session()
    print("[fetch] CBN JSON feed")
    resp = s.get(CBN_API, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise SystemExit("unexpected JSON shape from CBN API")

    rows = []
    for d in data:
        # Use actual API field names
        title = (d.get("title") or "").strip()
        link = (d.get("link") or "").strip()
        category = (d.get("keywords") or "").strip()  # Use keywords as category
        department = (d.get("refNo") or "").strip()   # Use refNo to infer department
        pub = d.get("documentDate")
        listed_iso, listed_raw = parse_date_any(pub)

        # Convert relative link to full URL
        if link.startswith("/"):
            url = f"https://www.cbn.gov.ng{link}"
        else:
            url = link

        if not url or not looks_pdf(url):
            continue

        if not should_keep(title, category, department, url, include_re, exclude_re):
            continue

        year_guess = int(listed_iso.split("-")[0]) if listed_iso else year_from_url(url)
        if year_guess is not None and (year_guess < args.since or year_guess > args.until):
            continue

        # map category to a safe folder name (limit length to prevent path issues)
        cat_slug = re.sub(r"[^\w\-]+", "_", category.lower()).strip("_") or "uncategorized"
        cat_slug = cat_slug[:50]  # Limit folder name length
        rows.append({
            "title": title,
            "href": url,
            "listed_date": listed_iso,
            "listed_date_raw": listed_raw,
            "category": category,
            "category_slug": cat_slug,
            "department": department,
            "year_guess": year_guess,
        })

    # de-dupe by href
    rows = list({r["href"]: r for r in rows}.values())
    rows.sort(key=lambda r: (r["listed_date"] or f"{r['year_guess'] or 0}-01-01"), reverse=True)

    # write CSV index
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "title","href","listed_date","listed_date_raw","category","category_slug","department","year_guess"
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {args.csv} with {len(rows)} rows")

    if args.dry_run:
        print("[ok] dry run complete. No downloads.")
        return

    # download into grouped subfolders
    seen_hashes = set()
    downloaded = 0
    for r in rows:
        out_dir = out_root / "pdf" / r["category_slug"]
        path, h, size = download_pdf(s, r["href"], out_dir, r["title"])
        if not path: continue
        if h in seen_hashes:
            print(f"[dedupe] already have {r['href']}")
            continue
        seen_hashes.add(h)

        rec = {
            "industry": "banking_fintech",
            "regulator": "CBN",
            "source_url": r["href"],
            "filename": str(path.relative_to(out_root)),
            "sha256": h,
            "content_length": size,
            "listed_date": r["listed_date"],
            "listed_date_raw": r["listed_date_raw"],
            "category": r["category"],
            "department": r["department"],
            "year_guess": r["year_guess"],
            "docs_hub": DOCS_HUB,
        }
        with open(manifest, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[saved] {path}")
        downloaded += 1
        time.sleep(args.sleep)

    print(f"[done] downloaded {downloaded} PDFs grouped by category")

if __name__ == "__main__":
    main()