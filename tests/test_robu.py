import pandas as pd

from pcb_sustainability.ingestion import load_bom
from pcb_sustainability.robu import RobuClient, _guess_category, _guess_package, _product_links_from_web_html, _query_variants


def test_robu_offline_seed_match_returns_real_robu_url(tmp_path):
    client = RobuClient(cache_path=tmp_path / "cache.json", delay_seconds=0)
    result = client.search("ESP32-WROOM-32 module", enabled=False)
    assert result["status"] == "offline_fallback"
    assert "robu.in" in result["source_url"]
    assert result["match_confidence"] >= 0.35


def test_robu_unknown_component_gracefully_falls_back(tmp_path):
    client = RobuClient(cache_path=tmp_path / "cache.json", delay_seconds=0)
    result = client.search("very unusual custom asic xyz123", enabled=False)
    assert result["status"] == "offline_fallback"
    assert result["availability"] == "Live lookup disabled"


def test_bom_can_carry_supplier_url_column():
    df = load_bom(pd.io.common.StringIO(
        "part_number,description,supplier url\n"
        "ABC123,Demo part,https://robu.in/product/demo-product/\n"
    ))
    assert df.loc[0, "supplier_url"] == "https://robu.in/product/demo-product/"


def test_web_html_product_link_extraction_handles_search_redirects():
    html = (
        '<a class="result__a" href="/l/?kh=-1&uddg='
        'https%3A%2F%2Frobu.in%2Fproduct%2Fstm32f103c8t6-blue-pill%2F">'
        "STM32F103C8T6</a>"
    )
    links = _product_links_from_web_html(html)
    assert links == ["https://robu.in/product/stm32f103c8t6-blue-pill/"]


def test_through_hole_terms_are_used_for_lookup_variants():
    variants = _query_variants("2 pin screw terminal block through hole connector")
    joined = " | ".join(variants).lower()
    assert "through hole" in joined
    assert "tht" in joined
    assert "screw terminal block" in joined


def test_through_hole_package_and_category_detection():
    assert _guess_package("DIP-8 through hole IC") == "THT"
    assert _guess_package("5mm radial electrolytic capacitor") == "Radial"
    assert _guess_package("2 pin screw terminal block") == "Terminal block"
    assert _guess_category("2 pin screw terminal block") == "Terminal connector"
