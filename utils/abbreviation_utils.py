
abbreviation_map = {
    "AS": "Administrative Sanction",
    "TS": "Technical Sanction",
    "GO": "Government Order",
    "TO": "Treasury Officer",
    "AE": "Assistant Engineer",
    "EE": "Executive Engineer"
}

def expand_abbreviations(query: str, abbr_map: dict = abbreviation_map) -> str:
    words = query.split()
    expanded = [abbr_map.get(word.upper().strip("?:,."), word) for word in words]
    return " ".join(expanded)
