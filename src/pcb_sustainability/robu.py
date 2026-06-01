def _query_from_row(self, row: pd.Series) -> str:
        # 1. Normalize all keys to lowercase to bypass case-sensitivity issues
        row_dict = {str(k).lower().strip(): v for k, v in row.items()}
        
        ref = str(row_dict.get("reference", "")).strip()
        val = str(row_dict.get("value", "")).strip()
        fp = str(row_dict.get("footprint", "")).lower()
        
        # Start with the core component value
        query_parts = [val]
        
        # 2. Apply smart text conversion based on KiCad footprints/references
        if ref.startswith("R") and not ("rv" in ref.lower() or "potentiometer" in fp):
            if val.isdigit():
                query_parts = [val + " ohm"]
            query_parts.append("resistor")
            
        elif "potentiometer" in fp or ref.startswith("RV"):
            query_parts = [val, "potentiometer"]
            if "3296w" in fp:
                query_parts.append("3296W")
                
        elif ref.startswith("C") or "capacitor" in fp:
            if val.lower().endswith("u"):
                query_parts = [val + "f"]  # Convert 1u to 1uf
            query_parts.append("capacitor")
            
        elif "led" in val.lower() or "led" in fp:
            if "3.0mm" in fp or "3mm" in fp:
                query_parts = ["3mm LED"]
            elif "5.0mm" in fp or "5mm" in fp:
                query_parts = ["5mm LED"]
            else:
                query_parts.append("LED")
                
        elif ref.startswith("J") or "connector" in fp or "pinheader" in fp:
            pins = ""
            if "01x02" in val or "1x02" in val:
                pins = "2 pin"
            elif "01x03" in val or "1x03" in val:
                pins = "3 pin"
            
            if "2.54mm" in fp:
                query_parts = [pins, "2.54mm berg strip pin header"] if pins else [val, "2.54mm header"]
            else:
                query_parts.append("connector")
                
        elif ref.startswith("U") or "package_dip" in fp:
            if "dip-8" in fp:
                query_parts.append("DIP-8 IC")
                
        # Return the cleaned string for the web scraper search loop
        return " ".join([p for p in query_parts if p]).strip()

    def enrich_bom(self, df: pd.DataFrame, enabled: bool = True, limit: int | None = None) -> dict[str, dict]:
        enrichments = {}
        rows = df.head(limit) if limit else df
        for _, row in rows.iterrows():
            row_dict = {str(k).lower().strip(): v for k, v in row.items()}
            query = self._query_from_row(row)
            
            # Use a robust key strategy so we can accurately match entries back to the DataFrame
            key = (normalize_text(str(row_dict.get("part_number") or row_dict.get("mpn"))) or 
                   normalize_text(str(row_dict.get("value"))) or 
                   normalize_text(str(row_dict.get("reference"))) or 
                   query)
                   
            product_url = _extract_robu_product_url(normalize_text(str(row_dict.get("supplier_url", ""))))
            if product_url and enabled:
                enrichments[key] = self.lookup_product_url(product_url, query=query)
            else:
                enrichments[key] = self.search(query, enabled=enabled)
        self.save()
        return enrichments
