"""Robu.in component enrichment with caching and graceful fallback behavior."""

from __future__ import annotations

import re
import time
from html import unescape
from pathlib import Path
from urllib.parse import quote_plus, unquote, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .utils import cache_key, normalize_text, read_json, write_json


DEFAULT_CACHE = Path(".cache/robu_results.json")
SEED_PRODUCTS = Path(__file__).resolve().parent / "data" / "robu_seed_products.json"
ROBU_READER_PREFIX = "https://r.jina.ai/http://r.jina.ai/http://"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


class RobuClient:
    """Small scraper layer for Robu.in search pages.

    The implementation is intentionally defensive: cached results are preferred,
    network errors return structured metadata, and selector failures do not break
    the overall sustainability pipeline.
    """

    def __init__(self, cache_path: str | Path = DEFAULT_CACHE, delay_seconds: float = 1.0, timeout: int = 10, browser_fallback: bool = False):
        self.cache_path = Path(cache_path)
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.browser_fallback = browser_fallback
        self.cache = read_json(self.cache_path, {})
        self.seed_products = read_json(SEED_PRODUCTS, [])
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
        self.save()
        return enrichments

    def search(self, query: str, enabled: bool = True) -> dict:
        query = normalize_text(query)
        if not query:
            return {"query": query, "status": "missing_query", "match_confidence": 0.0}
        product_url = _extract_robu_product_url(query)
        if product_url and enabled:
            return self.lookup_product_url(product_url, query=query)
        key = cache_key(query)
        if key in self.cache:
            result = dict(self.cache[key])
            stale_statuses = {"network_error", "offline_fallback", "missing_query"}
            stale_text = "enable online lookup" in normalize_text(result.get("availability")).lower()
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
                if result.get("status") == "not_found" and self.browser_fallback:
                    browser_result = self._browser_search(query)
                    if browser_result.get("status") not in {"browser_unavailable", "browser_error"}:
                        result = browser_result
        except requests.RequestException as exc:
            result = self._reader_search(query)
            if result.get("status") == "lookup_unavailable" and self.browser_fallback:
                browser_result = self._browser_search(query)
                if browser_result.get("status") not in {"browser_unavailable", "browser_error"}:
                    result = browser_result
            result.setdefault("error", str(exc))
            result.setdefault("source_url", url)
        self.cache[key] = result
        return result

    def lookup_product_url(self, product_url: str, query: str = "") -> dict:
        """Parse a known Robu.in product URL directly."""

        query = normalize_text(query) or product_url
        product_url = _extract_robu_product_url(product_url) or product_url
        key = cache_key(f"product:{product_url}")
        if key in self.cache and self.cache[key].get("status") not in {"network_error", "offline_fallback"}:
            result = dict(self.cache[key])
            result["from_cache"] = True
            return result
        result = self._reader_product(query=query, product_url=product_url)
        if (not result.get("title") or result.get("status") in {"lookup_unavailable", "seed_match"}) and self.browser_fallback:
            browser_result = self._browser_product(query=query, product_url=product_url)
            if browser_result.get("status") not in {"browser_unavailable", "browser_error"}:
                result = browser_result
        result["status"] = "ok_direct_product_url" if result.get("title") else result.get("status", "not_found")
        self.cache[key] = result
        return result

    def save(self) -> None:
        write_json(self.cache_path, self.cache)

    def _query_from_row(self, row: pd.Series) -> str:
        fields = [
            row.get("part_number"),
            row.get("manufacturer"),
            row.get("description"),
            row.get("value"),
            row.get("footprint"),
        ]
        return " ".join(normalize_text(value) for value in fields if normalize_text(value))[:180]

    def _parse_search(self, query: str, url: str, html: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        product = soup.select_one("li.product, .product-small, .product, .type-product")
        if not product:
            return {
                "query": query,
                "status": "not_found",
                "source_url": url,
                "availability": "Unknown",
                "match_confidence": 0.15,
            }

        link_el = product.select_one("a[href]")
        href = link_el.get("href", "") if link_el else url
        return self._parse_product_html(query, href, html)

    def _reader_search(self, query: str) -> dict:
        for variant in _query_variants(query):
            search_url = f"http://robu.in/?s={quote_plus(variant)}&post_type=product"
            reader_url = f"{ROBU_READER_PREFIX}{search_url}"
            try:
                response = self.session.get(reader_url, timeout=self.timeout)
                response.raise_for_status()
                if "Performing security verification" not in response.text:
                    links = _product_links_from_markdown(response.text)
                    if links:
                        result = self._reader_product(query, links[0])
                        result["status"] = "ok_reader_search"
                        result["search_url"] = f"https://robu.in/?s={quote_plus(variant)}&post_type=product"
                        return result
            except requests.RequestException:
                pass

        for product_url in self._web_search_product_links(query):
            result = self._reader_product(query, product_url)
            if result.get("title") and result.get("status") != "offline_fallback":
                result["status"] = "ok_robu_web_search"
                return result

        seed = self._best_seed_match(query)
        if seed:
            result = self._reader_product(query, seed["source_url"], seed)
            result["status"] = result.get("status", "seed_match")
            result["match_confidence"] = max(result.get("match_confidence", 0.0), _seed_confidence(query, seed))
            return result
        return self._lookup_unavailable(query)

    def _web_search_product_links(self, query: str) -> list[str]:
        links: list[str] = []
        for variant in _query_variants(query):
            search_query = f"site:robu.in/product {variant} Robu.in"
            url = f"https://duckduckgo.com/html/?q={quote_plus(search_query)}"
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException:
                continue
            links.extend(_product_links_from_web_html(response.text))
            if links:
                break
        unique = []
        seen = set()
        for link in links:
            normalized = link.split("?")[0].rstrip("/")
            if normalized not in seen:
                unique.append(normalized)
                seen.add(normalized)
        return unique[:5]

    def _reader_product(self, query: str, product_url: str, seed: dict | None = None) -> dict:
        seed = seed or {}
        normalized_url = product_url.replace("https://", "http://")
        reader_url = f"{ROBU_READER_PREFIX}{normalized_url}"
        try:
            response = self.session.get(reader_url, timeout=self.timeout)
            response.raise_for_status()
            if "Performing security verification" not in response.text:
                parsed = _parse_product_markdown(query, product_url, response.text)
                return _merge_seed(seed, parsed)
        except requests.RequestException:
            pass
        fallback = dict(seed) if seed else self._lookup_unavailable(query)
        fallback.update({
            "query": query,
            "status": "seed_match",
            "source_url": product_url,
            "similar_components": _similar_terms(seed.get("title", query)),
        })
        return fallback

    def _browser_search(self, query: str) -> dict:
        if not self.browser_fallback:
            return self._browser_unavailable(query)

        search_url = f"https://robu.in/?s={quote_plus(query)}&post_type=product"
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # pragma: no cover - optional dependency
            return self._browser_error(query, search_url, f"Playwright unavailable: {exc}")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(user_agent=USER_AGENT)
                page.goto(search_url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                html = page.content()
                soup = BeautifulSoup(html, "html.parser")
                link_el = soup.select_one('a[href*="/product/"]')
                product_url = ""
                if link_el and link_el.get("href"):
                    product_url = urljoin("https://robu.in/", link_el.get("href"))
                if product_url:
                    page.goto(product_url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                    html = page.content()
                    browser.close()
                    result = self._parse_product_html(query, product_url, html)
                    result["status"] = "ok_browser_search"
                    result["search_url"] = search_url
                    return result
                browser.close()
                result = self._parse_product_html(query, search_url, html)
                result["status"] = "browser_search_no_product"
                result["search_url"] = search_url
                return result
        except PlaywrightTimeoutError as exc:
            return self._browser_error(query, search_url, f"Browser timeout: {exc}")
        except Exception as exc:  # pragma: no cover - browser runtime issues
            return self._browser_error(query, search_url, f"Browser lookup failed: {exc}")

    def _browser_product(self, query: str, product_url: str) -> dict:
        if not self.browser_fallback:
            return self._browser_unavailable(query)

        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # pragma: no cover - optional dependency
            return self._browser_error(query, product_url, f"Playwright unavailable: {exc}")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(user_agent=USER_AGENT)
                page.goto(product_url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                html = page.content()
                browser.close()
                result = self._parse_product_html(query, product_url, html)
                result["status"] = "ok_browser_product"
                return result
        except PlaywrightTimeoutError as exc:
            return self._browser_error(query, product_url, f"Browser timeout: {exc}")
        except Exception as exc:  # pragma: no cover - browser runtime issues
            return self._browser_error(query, product_url, f"Browser lookup failed: {exc}")

    def _browser_unavailable(self, query: str) -> dict:
        return self._fallback_result(query, "browser_unavailable", "Browser fallback disabled")

    def _browser_error(self, query: str, source_url: str, message: str) -> dict:
        result = self._fallback_result(query, "browser_error", "Browser fallback failed")
        result["error"] = message
        result["source_url"] = source_url
        return result

    def _parse_product_html(self, query: str, product_url: str, html: str, seed: dict | None = None) -> dict:
        seed = seed or {}
        soup = BeautifulSoup(html, "html.parser")
        title_el = soup.select_one(".product_title, .woocommerce-loop-product__title, h1, h2, h3")
        price_el = soup.select_one(".price, .amount")
        stock_el = soup.select_one(".stock, .availability")
        category_el = soup.select_one(".posted_in, .product_cat, .category")
        title = title_el.get_text(" ", strip=True) if title_el else seed.get("title", query)
        source_url = product_url
        metadata = {
            "query": query,
            "status": "ok",
            "title": title or query,
            "category": category_el.get_text(" ", strip=True) if category_el else seed.get("category", _guess_category(title or query)),
            "manufacturer": seed.get("manufacturer", _guess_manufacturer(title or query)),
            "package": seed.get("package", _guess_package(title or query)),
            "datasheet_link": _find_datasheet(soup) or seed.get("datasheet_link", ""),
            "availability": stock_el.get_text(" ", strip=True) if stock_el else seed.get("availability", "Available status not shown"),
            "price": price_el.get_text(" ", strip=True) if price_el else seed.get("price", ""),
            "similar_components": _similar_terms(title or query),
            "source_url": source_url,
            "match_confidence": max(seed.get("match_confidence", 0.0), _match_confidence(query, title)),
        }
        return _merge_seed(seed, metadata)

    def _best_seed_match(self, query: str) -> dict | None:

        query_tokens = set(_tokenize(query))
        best_score = 0.0
        best = None
        for product in self.seed_products:
            keywords = set(_tokenize(" ".join(product.get("keywords", [])) + " " + product.get("title", "")))
            if not keywords:
                continue
            overlap = len(query_tokens & keywords)
            score = overlap / max(len(query_tokens), 1)
            if score > best_score:
                best_score = score
                best = product
        return best if best_score >= 0.18 else None

    def _seed_or_offline(self, query: str, status: str) -> dict:
        seed = self._best_seed_match(query)
        if seed:
            result = dict(seed)
            result.update({
                "query": query,
                "status": status,
                "similar_components": _similar_terms(seed.get("title", query)),
                "match_confidence": _seed_confidence(query, seed),
            })
            return result
        return self._offline_disabled(query)

    def _fallback_result(self, query: str, status: str, availability: str) -> dict:
        source_url = f"https://robu.in/?s={quote_plus(query)}&post_type=product"
        return {
            "query": query,
            "status": status,
            "title": query,
            "category": _guess_category(query),
            "manufacturer": _guess_manufacturer(query),
            "package": _guess_package(query),
            "datasheet_link": "",
            "availability": availability,
            "price": "",
            "similar_components": _similar_terms(query),
            "source_url": source_url,
            "match_confidence": 0.0,
        }

    def _offline_disabled(self, query: str) -> dict:
        return self._fallback_result(query, "offline_fallback", "Live lookup disabled")

    def _lookup_unavailable(self, query: str) -> dict:
        return self._fallback_result(
            query,
            "lookup_unavailable",
            "No Robu result found or Robu blocked lookup",
        )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", text.lower())


def _extract_robu_product_url(text: str) -> str:
    match = re.search(r"https?://(?:www\.|stg\.)?robu\.in/product/[^\s,;]+", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(0).rstrip(").]")


def _query_variants(query: str) -> list[str]:
    tokens = _tokenize(query)
    variants = [query]
    part_like = [token for token in tokens if any(char.isdigit() for char in token) and len(token) >= 4]
    if part_like:
        variants.append(" ".join(part_like[:3]))
        variants.append(" ".join(part_like[:3]) + " through hole")
        variants.append(" ".join(part_like[:3]) + " THT")
    useful = [
        token
        for token in tokens
        if token not in {"generic", "active", "unknown", "rohs", "component", "module"}
    ]
    if useful:
        variants.append(" ".join(useful[:6]))
    compact = query.replace("-", " ").replace("_", " ")
    variants.append(compact)

    lower_query = query.lower()
    tht_terms = {"tht", "through", "hole", "through-hole", "dip", "axial", "radial", "terminal", "header", "connector"}
    if any(term in lower_query for term in tht_terms):
        base = " ".join(useful[:6]) if useful else compact
        variants.extend([
            f"{base} through hole",
            f"{base} THT",
            f"{base} DIP",
            f"{base} axial radial",
        ])
    if any(term in lower_query for term in ["terminal", "screw", "block"]):
        variants.append(f"{compact} screw terminal block")
    if any(term in lower_query for term in ["header", "berg", "pin"]):
        variants.append(f"{compact} berg strip header")
    if any(term in lower_query for term in ["jst", "xh", "ph"]):
        variants.append(f"{compact} JST connector")

    deduped = []
    seen = set()
    for variant in variants:
        clean = " ".join(variant.split()).strip()
        if clean and clean.lower() not in seen:
            deduped.append(clean[:140])
            seen.add(clean.lower())
    return deduped


def _seed_confidence(query: str, seed: dict) -> float:
    query_tokens = set(_tokenize(query))
    seed_tokens = set(_tokenize(" ".join(seed.get("keywords", [])) + " " + seed.get("title", "")))
    if not query_tokens:
        return 0.2
    return round(max(0.35, min(0.88, len(query_tokens & seed_tokens) / len(query_tokens))), 2)


def _merge_seed(seed: dict, parsed: dict) -> dict:
    merged = dict(seed)
    for key, value in parsed.items():
        if value in ("", None, []):
            continue
        if key == "availability" and value == "Unknown" and seed.get("availability"):
            continue
        if key == "title" and str(value).startswith(("Published Time:", "Warning:")) and seed.get("title"):
            continue
        merged[key] = value
    return merged


def _product_links_from_markdown(markdown: str) -> list[str]:
    links = []
    pattern = r"\[[^\]]+\]\((https?://(?:stg\.)?robu\.in/product/[^)\s]+)\)"
    for match in re.finditer(pattern, markdown):
        links.append(match.group(1))
    links.extend(_product_links_from_web_html(markdown))
    return links


def _product_links_from_web_html(html_text: str) -> list[str]:
    links = []
    decoded_html = unescape(html_text)
    for encoded in re.findall(r"uddg=([^&\"']+)", decoded_html):
        decoded = unquote(encoded)
        if re.match(r"https?://(?:stg\.)?robu\.in/product/", decoded):
            links.append(decoded)
    direct_pattern = r"https?://(?:stg\.)?robu\.in/product/[a-zA-Z0-9\-_%/]+"
    links.extend(re.findall(direct_pattern, decoded_html))
    return links


def _parse_product_markdown(query: str, product_url: str, markdown: str) -> dict:
    title = ""
    title_match = re.search(r"Title:\s*(.+?)(?:\n|URL Source:)", markdown, flags=re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
    for line in markdown.splitlines():
        cleaned = line.strip("# ").strip()
        if title:
            break
        if cleaned and not cleaned.startswith((
            "Title:",
            "URL Source:",
            "Published Time:",
            "Markdown Content:",
            "Warning:",
        )):
            title = cleaned
            break

    availability = "Unknown"
    if re.search(r"\bIn stock\b", markdown, flags=re.IGNORECASE):
        availability = "In stock"
    elif re.search(r"\bOut of stock\b", markdown, flags=re.IGNORECASE):
        availability = "Out of stock"

    price_match = re.search(r"(?:₹|Rs\.?)\s*[\d,.]+", markdown)
    sku_match = re.search(r"SKU:\s*([A-Z0-9\-]+)", markdown, flags=re.IGNORECASE)
    category_match = re.search(r"Category:\s*([^\n|]+)", markdown, flags=re.IGNORECASE)
    brand_match = re.search(r"(?:Brand|Manufacturer):\s*\|?\s*([^\n|]+)", markdown, flags=re.IGNORECASE)
    package_match = re.search(r"(?:Case/Package|Package|Mounting Type):\s*\|?\s*([^\n|]+)", markdown, flags=re.IGNORECASE)
    datasheet_match = re.search(r"(https?://[^\s)]+\.pdf)", markdown, flags=re.IGNORECASE)

    return {
        "query": query,
        "status": "ok_reader_product",
        "title": title or query,
        "category": category_match.group(1).strip() if category_match else _guess_category(title or query),
        "manufacturer": brand_match.group(1).strip() if brand_match else _guess_manufacturer(title or query),
        "package": package_match.group(1).strip() if package_match else _guess_package(title or query),
        "datasheet_link": datasheet_match.group(1) if datasheet_match else "",
        "availability": availability,
        "price": price_match.group(0).replace("₹", "Rs") if price_match else "",
        "sku": sku_match.group(1) if sku_match else "",
        "similar_components": _similar_terms(title or query),
        "source_url": product_url.replace("http://", "https://"),
        "match_confidence": _match_confidence(query, title),
    }


def _guess_category(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ["terminal block", "screw terminal"]):
        return "Terminal connector"
    if any(term in lowered for term in ["header", "berg strip", "pin header", "jst"]):
        return "Header connector"
    if any(term in lowered for term in ["resistor", "ohm"]):
        return "Passive resistor"
    if any(term in lowered for term in ["capacitor", "uf", "nf", "pf"]):
        return "Passive capacitor"
    if any(term in lowered for term in ["connector", "header", "terminal"]):
        return "Connector"
    if any(term in lowered for term in ["battery", "cell", "lipo"]):
        return "Battery"
    if any(term in lowered for term in ["arduino", "esp", "mcu", "ic", "module"]):
        return "IC or module"
    return "Electronic component"


def _guess_package(text: str) -> str:
    lowered = text.lower()
    if "through-hole" in lowered or "through hole" in lowered or "tht" in lowered:
        return "THT"
    if "terminal block" in lowered or "screw terminal" in lowered:
        return "Terminal block"
    if "pin header" in lowered or "berg strip" in lowered:
        return "Pin header"
    if "axial" in lowered:
        return "Axial"
    if "radial" in lowered:
        return "Radial"
    match = re.search(
        r"\b(0201|0402|0603|0805|1206|sot-?\d+|soic-?\d+|qfn-?\d+|bga|dip-?\d*|to-?\d+|jst-?[a-z0-9]+)\b",
        text,
        re.IGNORECASE,
    )
    return match.group(0).upper() if match else ""


def _guess_manufacturer(text: str) -> str:
    known = ["Texas Instruments", "STMicroelectronics", "Microchip", "Arduino", "Espressif", "Vishay", "Murata", "Samsung"]
    lowered = text.lower()
    for manufacturer in known:
        if manufacturer.lower() in lowered:
            return manufacturer
    return ""


def _find_datasheet(product) -> str:
    for link in product.select("a[href]"):
        href = link.get("href", "")
        label = link.get_text(" ", strip=True).lower()
        if "datasheet" in label or href.lower().endswith(".pdf"):
            return href
    return ""


def _similar_terms(text: str) -> list[str]:
    category = _guess_category(text)
    if category in {"Terminal connector", "Header connector"}:
        return ["through-hole connector", "standard pitch header", "screw terminal block"]
    if category == "Battery":
        return ["socketed battery holder", "standard replaceable cell", "RoHS-compliant holder"]
    if category == "Connector":
        return ["through-hole connector", "locking recyclable connector", "standard pitch header"]
    if category == "Passive resistor":
        return ["RoHS 0805 resistor", "larger 1206 serviceable resistor"]
    if category == "Passive capacitor":
        return ["RoHS MLCC", "aluminum electrolytic with clear polarity marking"]
    return ["RoHS equivalent", "halogen-free equivalent", "widely available module"]


def _match_confidence(query: str, title: str) -> float:
    if not title:
        return 0.2
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    title_terms = set(re.findall(r"[a-z0-9]+", title.lower()))
    if not query_terms:
        return 0.2
    overlap = len(query_terms & title_terms) / len(query_terms)
    return round(max(0.2, min(0.95, overlap)), 2)
