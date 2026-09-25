"""Text normalisation for business names and addresses.

Every stage (blocking, features, model) compares records through the fields built
here, so the same rules apply to train and test and to every country. Nothing in
this module is specific to US or India: country only selects extra lookup tables
where they exist, and unknown countries (e.g. France) fall back to the generic rules.
"""

import re

from anyascii import anyascii

# Legal forms and filler words that carry little identity. Covers US, India and
# French forms; removed to build the "core" name.
LEGAL_WORDS = {
    "inc", "incorporated", "llc", "l", "c", "ltd", "limited", "private", "pvt", "pvt.",
    "corp", "corporation", "co", "company", "llp", "plc", "lp", "pllc", "pc", "pa",
    "the", "and", "of", "&",
    "sarl", "sas", "sasu", "eurl", "ei", "sa", "sci", "snc", "scop", "cie", "et",
    "com", "www", "in", "net", "org",
}

# Address abbreviations -> canonical long form (applied token by token).
ADDRESS_ABBREV = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "dr": "drive", "ln": "lane", "ct": "court", "ter": "terrace", "terr": "terrace",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "hwy": "highway",
    "pl": "place", "sq": "square", "cir": "circle", "pkwy": "parkway", "trl": "trail",
    "mt": "mount", "ft": "fort", "apt": "apartment", "ste": "suite", "fl": "floor",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "r": "rue", "crs": "cours", "imp": "impasse", "che": "chemin", "rte": "route",
    "nr": "near", "opp": "opposite", "bldg": "building", "sec": "sector",
    "hno": "house", "h": "house",
}

# Tokens that add nothing to an address comparison.
ADDRESS_NOISE = {"no", "number", "unit", "po", "box", "null", "none", "na", "#"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}

# Indian state names -> the short code used in the data ("DL", "MH", ...), so
# "Delhi", "DL" and "दिल्ली" compare equal. Native-script entries are the exact
# anyascii renderings measured on train (the 16 most frequent native-script states).
INDIA_STATES = {
    "delhi": "dl", "new delhi": "dl", "nct of delhi": "dl", "dilli": "dl",
    "maharashtra": "mh", "mharastr": "mh",
    "tamil nadu": "tn", "tamilnadu": "tn", "tmilnatu": "tn",
    "karnataka": "ka", "krnatk": "ka",
    "uttar pradesh": "up", "uttr prdes": "up",
    "gujarat": "gj", "gujrat": "gj",
    "west bengal": "wb", "pscimbng": "wb",
    "telangana": "tg", "telmgan": "tg",
    "haryana": "hr", "hriyana": "hr",
    "rajasthan": "rj", "rajsthan": "rj",
    "kerala": "kl", "kerlm": "kl",
    "bihar": "br",
    "madhya pradesh": "mp", "mdhy prdes": "mp",
    "andhra pradesh": "ap", "amdhrprdes": "ap",
    "punjab": "pb", "pmjab": "pb",
    "odisha": "od", "orissa": "od", "od isa": "od",
    "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk",
    "himachal pradesh": "hp", "goa": "ga", "chandigarh": "ch",
    "jammu and kashmir": "jk", "puducherry": "py", "pondicherry": "py",
}

_PAREN_ID = re.compile(r"\((?:id|ref|no)[^)]*\)", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WEBSITE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|in|net|org|co|fr|biz|info)\b")
_DIGITS = re.compile(r"\d+")


def to_ascii_lower(text):
    """Transliterate any script to ASCII (Devanagari, Tamil, accents...) and lower-case it."""
    if not text:
        return ""
    return anyascii(text).lower()


def tokens(text):
    """Split ASCII-lowered text into alphanumeric tokens."""
    return [t for t in _NON_ALNUM.split(text) if t]


def normalize_name(name):
    """Return (norm, core, compact) versions of a business name.

    norm    : all tokens, transliterated and lower-cased
    core    : norm minus legal forms / filler words (falls back to norm if empty)
    compact : core with no separators and 0->o, so 'www.0ne-logistics.com',
              'One Logistics Pvt Ltd' and 'onelogistics' can meet
    """
    text = _PAREN_ID.sub(" ", name or "")
    low = to_ascii_lower(text).strip()
    site = _WEBSITE.match(low)
    if site:
        low = site.group(1).replace("-", " ")
    toks = tokens(low.replace("&", " and "))
    norm = " ".join(toks)
    core_toks = [t for t in toks if t not in LEGAL_WORDS]
    core = " ".join(core_toks) if core_toks else norm
    compact = core.replace(" ", "").replace("0", "o")
    return norm, core, compact


def _replace_states(text, country):
    """Map full state names to short codes for the countries we have tables for."""
    table = US_STATES if country == "US" else INDIA_STATES if country == "India" else None
    if not table:
        return text
    for full, code in table.items():
        if full in text:
            text = re.sub(rf"\b{re.escape(full)}\b", code, text)
    return text


def normalize_address(address, country):
    """Return (norm, numbers) for an address.

    norm    : transliterated, abbreviations expanded, state names mapped to codes,
              numbers stripped of leading zeros ('B-00200' -> 'b 200'), noise dropped
    numbers : space-joined sorted set of the numbers in the address
    """
    low = to_ascii_lower(address or "")
    low = _replace_states(" ".join(tokens(low)), country)
    out = []
    for t in low.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = ADDRESS_ABBREV.get(t, t)
            if t in ADDRESS_NOISE:
                continue
        out.append(t)
    nums = sorted({(d.lstrip("0") or "0") for d in _DIGITS.findall(low)})
    return " ".join(out), " ".join(nums)
