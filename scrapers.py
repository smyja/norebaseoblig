import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from urllib import robotparser

# ---------------------------
# Configuration
# ---------------------------

# Starter seed pages grouped by industry -> regulator -> list of URLs
SEEDS = {
    "banking_fintech": {
        "CBN": [
            "https://www.cbn.gov.ng/Documents/",  # docs hub
        ]
    },
    "telecoms": {
        "NCC": [
            "https://www.ncc.gov.ng/page/regulations",
            "https://www.ncc.gov.ng/technical-regulations/",
        ]
    },
    "oil_gas_upstream": {
        "NUPRC": [
            "https://www.nuprc.gov.ng/gazetted-regulations/",
        ]
    },
    "food_drug_health": {
        "NAFDAC": [
            "https://nafdac.gov.ng/regulatory-resources/nafdac-regulations/",
        ]
    },
    # Add more as needed
}

# File matching
PDF_EXT_RE = re.compile(r"\.pdf($|\?)", re.IGNORECASE)

# Clean a string to be a safe filename
def slugify(text: str, maxlen: int = 80) -> str:
    text = unescape(text)
    text = re.sub(r"[^\w\s\-\.]+", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text.strip())
    if not text:
        text = "file"
    return text[:maxlen]

# Create a robust requests Session
def make_session(timeout: int = 30) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    s.headers.update({
        "User-Agent": "reg-scraper/1.0 (+local research use)",
        "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
    })
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.request = _with_timeout(s.request, timeout=timeout)  # type: ignore
    return s

def _with_timeout(func, timeout: int):
    def inner(method, url, **kw):
        kw.setdefault("timeout", timeout)
        return func(method, url, **kw)
    return inner

# Respect robots.txt
def allowed_by_robots(url: str, rp_cache: Dict[str, robotparser.RobotFileParser]) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in rp_cache:
        rp = robotparser.RobotFileParser()
        robots_url = urljoin(base, "/robots.txt")
        try:
            rp.set_url(robots_url)
            rp.read()
        except Exception:
            # If robots fails to load, be conservative and allow
            rp = None  # type: ignore
        rp_cache[base] = rp  # type: ignore
    rp = rp_cache[base]
    if rp is None:
        return True
    return rp.can_fetch("*", url)

# Domain scoping
def same_domain(seed: str, candidate: str) -> bool:
    a = urlparse(seed).netloc.lower()
    b = urlparse(candidate).netloc.lower()
    return a == b

# Extract links from HTML
def extract_links(html: str, base_url: str) -> Tuple[List[str], List[Tuple[str, Optional[str]]]]:
    soup = BeautifulSoup(html, "html.parser")
    page_title = None
    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()
    links = []
    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        abs_url = urljoin(base_url, href)
        abs_url, _ = urldefrag(abs_url)
        # Skip mailto and javascript
        if abs_url.startswith("mailto:") or abs_url.startswith("javascript:"):
            continue
        if PDF_EXT_RE.search(abs_url):
            title = a.get_text(strip=True) or page_title or "document"
            pdf_links.append((abs_url, title))
        else:
            links.append(abs_url)
    return links, pdf_links

# Hash bytes
def sha256_of(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

# Save file safely
def save_binary(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(content)
    tmp.replace(path)

# Write manifest entry
def write_manifest(manifest_path: Path, record: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Load existing hashes so we do not redownload across runs
def load_existing_hashes(manifest_path: Path) -> Set[str]:
    hashes: Set[str] = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    h = rec.get("sha256")
                    if h:
                        hashes.add(h)
                except Exception:
                    continue
    return hashes

def handle_pdf(
    session: requests.Session,
    pdf_url: str,
    anchor_title: str,
    out_dir: Path,
    manifest_path: Path,
    industry: str,
    regulator: str,
    downloaded_hashes: Set[str],
) -> None:
    try:
        r = session.get(pdf_url)
    except Exception as e:
        print(f"[error] GET {pdf_url} -> {e}")
        return

    status = r.status_code
    if status >= 400:
        print(f"[warn] {status} {pdf_url}")
        return

    ctype = r.headers.get("Content-Type", "").lower()
    content = r.content

    # Some servers mislabel content type. Trust magic header if needed.
    if "application/pdf" not in ctype and not PDF_EXT_RE.search(pdf_url):
        if not content.startswith(b"%PDF-"):
            print(f"[skip] Not a PDF by magic header {pdf_url}")
            return

    file_hash = sha256_of(content)
    if file_hash in downloaded_hashes:
        print(f"[dedupe] already have {pdf_url}")
        return
    downloaded_hashes.add(file_hash)

    # Prefer filename from Content-Disposition if present
    cd = r.headers.get("Content-Disposition", "")
    filename_from_cd = None
    m = re.search(r"filename\*=.*?''([^;]+)", cd, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'filename="?([^";]+)"?', cd, flags=re.IGNORECASE)
    if m:
        try:
            filename_from_cd = unescape(m.group(1)).strip()
        except Exception:
            filename_from_cd = None

    # build filename
    url_path_name = Path(urlparse(pdf_url).path).name
    base = filename_from_cd or (slugify(anchor_title) if anchor_title else slugify(url_path_name))
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"
    filename = base

    # avoid collisions with filename length limit
    final_path = out_dir / "pdf" / filename
    i = 1
    max_filename_length = 255  # Most filesystems limit to 255 characters

    while final_path.exists():
        stem = final_path.stem
        suffix = final_path.suffix
        counter_str = f"_{i}"

        # Calculate max stem length to stay under filename limit
        max_stem_length = max_filename_length - len(suffix) - len(counter_str)

        # Truncate stem if necessary
        if len(stem) > max_stem_length:
            truncated_stem = stem[:max_stem_length]
        else:
            truncated_stem = stem

        new_filename = f"{truncated_stem}{counter_str}{suffix}"
        final_path = out_dir / "pdf" / new_filename

        # Safety check. If filename is still too long, use hash-based name
        if len(new_filename) > max_filename_length:
            hash_part = file_hash[:8]  # Use first 8 chars of hash
            final_path = out_dir / "pdf" / f"file_{hash_part}.pdf"
            break

        i += 1

        # Prevent infinite loop
        if i > 10000:
            hash_part = file_hash[:8]
            final_path = out_dir / "pdf" / f"file_{hash_part}.pdf"
            break

    save_binary(final_path, content)
    size = final_path.stat().st_size

    record = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "industry": industry,
        "regulator": regulator,
        "source_url": pdf_url,
        "filename": str(final_path.relative_to(out_dir)),
        "sha256": file_hash,
        "content_length": size,
        "http_status": status,
        "content_type": ctype,
        "title_guess": anchor_title,
    }
    write_manifest(manifest_path, record)
    print(f"[saved] {final_path} ({size} bytes)")

# Crawl one regulator
def crawl_regulator(
    session: requests.Session,
    out_root: Path,
    industry: str,
    regulator: str,
    seed_urls: List[str],
    max_pages: int,
    sleep_s: float,
    respect_robots: bool,
    rp_cache: Dict[str, robotparser.RobotFileParser],
) -> None:
    visited: Set[str] = set()
    queue: deque[str] = deque()
    for u in seed_urls:
        queue.append(u)

    out_dir = out_root / industry / regulator
    manifest = out_dir / "manifest.jsonl"
    downloaded_hashes: Set[str] = load_existing_hashes(manifest)
    seen_pdf_urls: Set[str] = set()

    pages_crawled = 0
    while queue and pages_crawled < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        # stay on the same domain as its seed
        if not any(same_domain(seed, url) for seed in seed_urls):
            continue

        # honor robots only if requested
        if respect_robots and not allowed_by_robots(url, rp_cache):
            print(f"[robots] blocked {url}")
            continue

        # If the URL looks like a PDF by extension, download directly
        if PDF_EXT_RE.search(url):
            pdf_title = Path(urlparse(url).path).name or "document"
            handle_pdf(session, url, pdf_title, out_dir, manifest, industry, regulator, downloaded_hashes)
            time.sleep(sleep_s)
            continue

        # Fetch page
        try:
            r = session.get(url)
        except Exception as e:
            print(f"[error] GET {url} -> {e}")
            continue

        ctype = r.headers.get("Content-Type", "")
        status = r.status_code
        if status >= 400:
            print(f"[warn] {status} {url}")
            continue

        # If this URL actually returned a PDF
        if "application/pdf" in ctype.lower():
            pdf_title = Path(urlparse(url).path).name or "document"
            handle_pdf(session, url, pdf_title, out_dir, manifest, industry, regulator, downloaded_hashes)
            time.sleep(sleep_s)
            continue

        if "html" not in ctype.lower():
            # Not HTML and not PDF; skip
            continue

        pages_crawled += 1
        html = r.text
        links, pdf_links = extract_links(html, url)

        # enqueue internal links
        for link in links:
            if link not in visited and any(same_domain(seed, link) for seed in seed_urls):
                queue.append(link)

        # download PDFs found on this page
        for pdf_url, anchor_title in pdf_links:
            if pdf_url in seen_pdf_urls:
                continue
            if respect_robots and not allowed_by_robots(pdf_url, rp_cache):
                print(f"[robots] blocked {pdf_url}")
                continue
            seen_pdf_urls.add(pdf_url)
            handle_pdf(session, pdf_url, anchor_title, out_dir, manifest, industry, regulator, downloaded_hashes)
            time.sleep(sleep_s)

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Download Nigerian regulations PDFs by industry and regulator")
    ap.add_argument("--out", type=str, default="regulations_corpus", help="Output root directory")
    ap.add_argument("--max-pages", type=int, default=400, help="Max HTML pages to crawl per regulator")
    ap.add_argument("--sleep", type=float, default=0.8, help="Seconds to sleep between requests")
    ap.add_argument("--industry", type=str, action="append", help="Limit to specific industry key. Repeatable")
    ap.add_argument("--add-seed", type=str, nargs=3, metavar=("INDUSTRY", "REGULATOR", "URL"),
                    help="Add a single seed without editing the file")
    ap.add_argument("--respect-robots", action="store_true",
                    help="Opt in to respecting robots.txt checks")
    return ap.parse_args()

def main():
    args = parse_args()
    out_root = Path(args.out)
    seeds = dict(SEEDS)  # copy

    if args.add_seed:
        ind, reg, url = args.add_seed
        seeds.setdefault(ind, {})
        seeds[ind].setdefault(reg, [])
        seeds[ind][reg].append(url)

    if args.industry:
        # filter to requested industries
        seeds = {k: v for k, v in seeds.items() if k in set(args.industry)}

    session = make_session()
    rp_cache: Dict[str, robotparser.RobotFileParser] = {}
    respect_robots = args.respect_robots  # default is False

    for industry, regulators in seeds.items():
        for regulator, urls in regulators.items():
            print(f"\n[run] {industry} -> {regulator}")
            crawl_regulator(
                session=session,
                out_root=out_root,
                industry=industry,
                regulator=regulator,
                seed_urls=urls,
                max_pages=args.max_pages,
                sleep_s=args.sleep,
                respect_robots=respect_robots,
                rp_cache=rp_cache,
            )

    print("\nDone.")
    print(f"Output tree rooted at: {out_root.resolve()}")
    print("Each regulator has a manifest.jsonl with one line per downloaded PDF.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)