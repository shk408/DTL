from __future__ import annotations

import re
import time
from html import unescape
from pathlib import Path
from urllib.parse import quote_plus, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Global utilities assumed to be imported from .utils
def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip() if text else ""

def cache_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())

DEFAULT_CACHE = Path(".cache/robu_results.json")
# Fixed: Proper single-layer Jina reader prefix
ROBU_READER_PREFIX = "https://r.jina.ai/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

class RobuClient:
    def __init__(self, cache_path: str | Path = DEFAULT_CACHE, delay_seconds: float = 1.0, timeout: int = 10, browser_fallback: bool = False):
        self.cache_path = Path(cache_path)
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.browser_fallback = browser_fallback
        self.cache = {}
        self.seed_products = []
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://robu.in/",
        })

    def enrich_bom(self, df: pd.DataFrame, enabled: bool = True, limit: int | None = None) -> dict[str, dict]:
        enrichments = {}
        rows = df.head(limit) if limit else df
        for _, row in rows.iterrows():
            query = self._query_from_row(row)
            key = normalize_text(row.get("part_number")) or normalize_text(row.get("description")) or query
            product_url = _extract_robu_product_url(normalize_text(row.get("supplier_url", "")))
            
            if product_url and enabled:
                enrichments[key] = self.lookup_product_url(product_url, query=query)
            else:
                enrichments[key] = self.search(query, enabled=enabled)
        return enrichments

    def search(self, query: str, enabled: bool = True) -> dict:
        query = normalize_text(query)
        if not query:
            return self._fallback_result(query, "missing_query", "Missing search query")
            
        product_url = _extract_robu_product_url(query)
        if product_url and enabled:
            return self.lookup_product_url(product_url, query=query)
            
        key = cache_key(query)
        if key in self.cache:
            result = dict(self.cache[key])
            stale_statuses = {"network_error", "offline_fallback", "missing_query"}
            stale_text = "enable online lookup" in normalize_text(result.get("availability", "")).lower()
            if not enabled and not stale_text:
                result["from_cache"] = True
                return result
            if enabled and result.get("status") not in stale_statuses and not stale_text:
                result["from_cache"] = True
                return result
                
        if not enabled:
            result = self._seed_or_offline(query, status="offline_fallback")
            self.cache[key] = result
            return result

        time.sleep(self.delay_seconds)
        url = f"https://robu.in/?s={quote_plus(query)}&post_type=product"
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 403 or "Just a moment" in response.text:
                result = self._reader_search(query)
            else:
                response.raise_for_status()
                result = self._parse_search(query, url, response.text)
        except requests.RequestException as exc:
            result = self._reader_search(query)
            result["error"] = str(exc)
            result["source_url"] = url
            
        self.cache[key] = result
        return result

    def lookup_product_url(self, product_url: str, query: str = "") -> dict:
        query = normalize_text(query) or product_url
        product_url = _extract_robu_product_url(product_url) or product_url
        key = cache_key(f"product:{product_url}")
        
        if key in self.cache and self.cache[key].get("status") not in {"network_error", "offline_fallback"}:
            result = dict(self.cache[key])
            result["from_cache"] = True
            return result
            
        result = self._reader_product(query=query, product_url=product_url)
        result["status"] = "ok_direct_product_url" if result.get("title") else result.get("status", "not_found")
        self.cache[key] = result
        return result

    def _query_from_row(self, row: pd.Series) -> str:
        fields = [row.get("part_number"), row.get("manufacturer"), row.get("description")]
        return " ".join(normalize_text(str(value)) for value in fields if normalize_text(str(value)))[:180]

    def _parse_search(self, query: str, url: str, html: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        product = soup.select_one("li.product, .product-small, .product, .type-product")
        if not product:
            return self._fallback_result(query, "not_found", "No product element matched on search page")

        link_el = product.select_one("a[href]")
        href = link_el.get("href", "") if link_el else url
        return self._parse_product_html(query, href, html)

    def _reader_search(self, query: str) -> dict:
        for variant in _query_variants(query):
            search_url = f"https://robu.in/?s={quote_plus(variant)}&post_type=product"
            reader_url = f"{ROBU_READER_PREFIX}{search_url}"
            try:
                response = self.session.get(reader_url, timeout=self.timeout)
                response.raise_for_status()
                if "Performing security verification" not in response.text:
                    links = _product_links_from_markdown(response.text)
                    if links:
                        result = self._reader_product(query, links[0])
                        result["status"] = "ok_reader_search"
                        return result
            except requests.RequestException:
                pass
        return self._lookup_unavailable(query)

    def _reader_product(self, query: str, product_url: str) -> dict:
        reader_url = f"{ROBU_READER_PREFIX}{product_url}"
        try:
            response = self.session.get(reader_url, timeout=self.timeout)
            response.raise_for_status()
            if "Performing security verification" not in response.text:
                return _parse_product_markdown(query, product_url, response.text)
        except requests.RequestException:
            pass
        return self._lookup_unavailable(query)

    def _parse_product_html(self, query: str, product_url: str, html: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        title_el = soup.select_one(".product_title, h1")
        price_el = soup.select_one(".price, .amount")
        stock_el = soup.select_one(".stock, .availability")
        
        title = title_el.get_text(" ", strip=True) if title_el else query
        return {
            "query": query,
            "status": "ok",
            "title": title,
            "availability": stock_el.get_text(" ", strip=True) if stock_el else "In Stock",
            "price": price_el.get_text(" ", strip=True) if price_el else "",
            "source_url": product_url,
        }

    def _fallback_result(self, query: str, status: str, availability: str) -> dict:
        return {
            "query": query,
            "status": status,
            "title": query if query else "Unknown Item",
            "availability": availability,
            "price": "",
            "source_url": f"https://robu.in/?s={quote_plus(query)}&post_type=product" if query else "",
        }

    def _seed_or_offline(self, query: str, status: str) -> dict:
        return self._fallback_result(query, status, "Live lookup disabled")

    def _lookup_unavailable(self, query: str) -> dict:
        return self._fallback_result(query, "lookup_unavailable", "No Robu result found (Blocked or Missing)")

def _extract_robu_product_url(text: str) -> str:
    match = re.search(r"https?://(?:www\.)?robu\.in/product/[^\s,;]+", text, flags=re.IGNORECASE)
    return match.group(0).rstrip(").]") if match else ""

def _query_variants(query: str) -> list[str]:
    return [query] if query else []

def _product_links_from_markdown(markdown: str) -> list[str]:
    pattern = r"\[[^\]]+\]\((https?://robu\.in/product/[^)\s]+)\)"
    return [match.group(1) for match in re.finditer(pattern, markdown)]

def _parse_product_markdown(query: str, product_url: str, markdown: str) -> dict:
    title = query
    title_match = re.search(r"Title:\s*(.+?)(?:\n|URL Source:)", markdown, flags=re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
        
    availability = "Unknown"
    if re.search(r"\bIn stock\b", markdown, flags=re.IGNORECASE):
        availability = "In stock"
    elif re.search(r"\bOut of stock\b", markdown, flags=re.IGNORECASE):
        availability = "Out of stock"

    price_match = re.search(r"(?:₹|Rs\.?)\s*[\d,.]+", markdown)
    return {
        "query": query,
        "status": "ok_reader_product",
        "title": title,
        "availability": availability,
        "price": price_match.group(0) if price_match else "",
        "source_url": product_url,
    }
