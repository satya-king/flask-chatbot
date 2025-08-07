
abbreviation_map = {
    "AS": "Administrative Sanction",
    "TS": "Technical Sanction",
    "GO": "Government Order",
    "TO": "Treasury Officer",
    "AE": "Assistant Engineer",
    "EE": "Executive Engineer",
    "SE": "Superintending Engineer",
    "CE": "Chief Engineer",
    "DE": "Divisional Engineer",
    "EE": "Executive Engineer",
    "DPR": "Detailed Project Report",
    "RFP": "Request for Proposal",
    "NIT": "Notice Inviting Tender",
    "DDO":" Drawing and Disbursing Officer",
    "HOA": "Head of Account",
    "PWD": "Public Works Department",
    "APCFSS": "Andhra Pradesh Centre for Financial Systems and Services",
    "NIDHI": "National Initiative for Digital Infrastructure",
    "AP": "Andhra Pradesh"
}

def expand_abbreviations(query: str, abbr_map: dict = abbreviation_map) -> str:
    words = query.split()
    expanded = [abbr_map.get(word.upper().strip("?:,."), word) for word in words]
    return " ".join(expanded)
