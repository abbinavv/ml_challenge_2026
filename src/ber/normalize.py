"""Text normalisation for business names and addresses.

Every stage (blocking, features, model) compares records through the fields built
here, so the same rules apply to train and test and to every country. Nothing in
this module is specific to US or India: country only selects extra lookup tables
where they exist, and unknown countries (e.g. France) fall back to the generic rules.

Each rule below targets a difference measured on known true matches
(src/audit_normalization.py), e.g. "St"/"Saint", "trading as" names, OCR-style
character swaps ("5ervices", "lnc"), duplicated words and honorifics.
"""

import json
import os
import re

from anyascii import anyascii
from rapidfuzz import fuzz

# Word dictionary for names written in Indian scripts, learned from training pairs
# by src/learn_translit.py (e.g. "praivet" -> "private"). Optional: if the file is
# absent, native-script names fall back to plain transliteration.
_TRANSLIT_PATH = os.environ.get("BER_TRANSLIT", os.path.join(os.environ.get("BER_CACHE", "cache"), "translit.json"))
TRANSLIT = json.load(open(_TRANSLIT_PATH, encoding="utf-8")) if os.path.exists(_TRANSLIT_PATH) else {}

# Sound-alike map for transliterated words never seen in training (learn_translit.py).
_SKEL_PATH = os.path.join(os.environ.get("BER_CACHE", "cache"), "skelmap.json")
_SK = json.load(open(_SKEL_PATH, encoding="utf-8")) if os.path.exists(_SKEL_PATH) else {}
SKELMAP, LATIN_VOCAB = _SK.get("map", {}), set(_SK.get("latin_vocab", []))

# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

# Legal forms and connector words; removed to build the "core" name.
LEGAL_WORDS = {
    "inc", "incorporated", "llc", "l", "c", "ltd", "limited", "private", "pvt",
    "corp", "corporation", "co", "company", "llp", "plc", "lp", "pllc", "pc", "pa",
    "lcsw", "pty", "gmbh", "the", "and", "of",
    "sarl", "sas", "sasu", "eurl", "ei", "sa", "sci", "snc", "scop", "cie", "et", "fils",
    "com", "www", "in", "net", "org",
}

# Honorifics / titles that sources add in front of names ("Mr Ram Logistics",
# "Smt Ananda Traders", "M/S ..."); removed everywhere in the name.
HONORIFICS = {"mr", "mrs", "ms", "smt", "dr", "messrs", "ms", "shrimati", "kumari"}

# Religious/respect prefixes common in Indian business names: kept as ONE
# canonical word, because they can be part of the real name ("Sri Balaji ...").
SRI_VARIANTS = {"sri": "sri", "shri": "sri", "shree": "sri", "sree": "sri", "shreee": "sri"}

# Generic business descriptors that sources add or drop freely ("Superior
# Hospitality" vs "Superior Hospitality Services Corp"). Removed only for the
# "key" name, which is used as an extra, more forgiving comparison.
DESCRIPTORS = {
    "center", "centre", "services", "service", "partners", "partner", "group",
    "associates", "solutions", "enterprises", "enterprise", "trading", "traders",
    "holdings", "international", "global", "industries", "industry", "technologies",
    "technology", "tech", "systems", "system", "sys", "consultants", "consulting",
    "management", "ventures", "labs", "lab", "business", "worldwide", "products",
    "brothers", "bros", "sons", "india", "usa", "america", "sri", "a", "s", "m",
}

# "X trading as Y": the business is Y (measured: the S1 name is the part after).
_ALIAS = re.compile(
    r"\b(?:trading as|t/a|doing business as|d/b/a|dba|formerly known as|formerly|"
    r"f/k/a|fka|also known as|a/k/a|aka|nee|known as)\b", re.IGNORECASE)
_PAREN_ID = re.compile(r"\((?:id|ref|no)[^)]*\)|#\s*\d+", re.IGNORECASE)
_WEBSITE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|in|net|org|co|fr|biz|info)\b")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")
# Characters that sources swap for look-alike digits ("5ervices", "INIMITA8LE", "c0m").
_OCR = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "6": "g", "8": "b"})

# Name variants that sources use interchangeably (Indian spellings).
NAME_VARIANTS = {"lakshmi": "laxmi", "laxmee": "laxmi", "jai": "jay", "jaya": "jay",
                 "centre": "center", "organisation": "organization"}

# Marker words that never belong to the business name itself.
MARKER_WORDS = {"dba", "nee", "aka", "fka", "tradingas", "formerly"}


def fold(t):
    """Fold letters that sources confuse, identically on both sides:
    'l' and 'i' ('famiiy', 'heaith', 'iimited'), 'rn' and 'm' ('intemational')."""
    return t.replace("rn", "m").replace("l", "i")


def to_ascii_lower(text):
    """Transliterate any script to ASCII (Devanagari, Tamil, accents...) and lower-case it."""
    if not text:
        return ""
    return anyascii(text).lower()


def tokens(text):
    """Split ASCII-lowered text into alphanumeric tokens."""
    return [t for t in _NON_ALNUM.split(text) if t]


_VOWELS = str.maketrans("", "", "aeiouy")
_SOUND = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("kh", "k"), ("gh", "g"),
          ("bh", "b"), ("dh", "d"), ("w", "v"), ("q", "k"), ("c", "k"), ("z", "j"), ("x", "ks"),
          ("g", "j"), ("m", "n")]


def skeleton(word):
    """Consonant skeleton of a word, for sound-alike matching across spellings and
    transliterations: 'stores'/'stors' -> 'strs', 'electronics'/'ilektroniks' ->
    'lktrnks'. All vowels are dropped (including the first letter), sounds that
    transliteration swaps are merged (m/n, g/j, ph/f, w/v, c/k), repeats collapse."""
    w = word
    for a, b in _SOUND:
        w = w.replace(a, b)
    out = ""
    for ch in w.translate(_VOWELS):
        if ch != out[-1:]:
            out += ch
    return out


_SKEL_CACHE = {}


def sound_alike(word):
    """Latin word for an unseen transliterated word: among Latin words with the same
    skeleton, the one spelled most like it (rapidfuzz ratio >= 58), else the word."""
    if word in _SKEL_CACHE:
        return _SKEL_CACHE[word]
    best, score = word, 58.0
    if len(word) <= 4:                      # too short to map safely ('stil', 'jnrl')
        _SKEL_CACHE[word] = word
        return word
    for cand in SKELMAP.get(skeleton(word), ()):
        if len(cand) < 4 or not any(v in cand for v in "aeiou"):   # real words only
            continue
        if not 0.7 <= len(cand) / len(word) <= 1.6:                 # comparable length ('pharmesi' != 'farms')
            continue
        r = fuzz.ratio(word, cand)
        if r >= score:
            best, score = cand, r
    _SKEL_CACHE[word] = best
    return best


def _fix_token(t):
    """Undo OCR-style swaps inside a word: digits inside mostly-letter words
    ('5ervices' -> 'services') and a leading 'l' read for 'I' ('lnc' -> 'inc')."""
    letters = sum(ch.isalpha() for ch in t)
    digits = len(t) - letters
    if digits and letters >= 2 and digits <= 2 and letters > digits:
        t = t.translate(_OCR)
    if t.startswith("ln") and len(t) >= 3:
        t = "i" + t[1:]
    return t


def _dedupe(toks):
    """Drop repeated words, keeping first occurrence ('Hermosillo Hermosillo Gas')."""
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _name_tokens(text, native):
    """Clean one name segment into tokens."""
    low = to_ascii_lower(text)
    site = _WEBSITE.search(low)                # before dots are removed
    if site:                                   # www.baycenter.com -> baycenter
        low = low[:site.start()] + " " + site.group(1).replace("-", " ") + " " + low[site.end():]
    low = re.sub(r"['’`.]", "", low)          # P.L.L.C. -> pllc, Ernesta's -> ernestas
    low = re.sub(r"\bm\s*/\s*s\b", " ", low)  # M/S (Messrs)
    low = low.replace("&", " and ").replace("/", " ")
    toks = [_fix_token(t) for t in tokens(low)]
    if native:
        out = []
        for t in toks:
            m = TRANSLIT.get(t) or TRANSLIT.get(fold(t))       # learned from training pairs
            if m:
                out.append(m)
            elif SKELMAP and not (t in LATIN_VOCAB or t.isdigit() or len(t) < 3):
                out.append(sound_alike(t))                     # unseen word: map by sound
            else:
                out.append(t)
        toks = out
    toks = [NAME_VARIANTS.get(t, SRI_VARIANTS.get(t, t)) for t in toks
            if t not in HONORIFICS and t not in MARKER_WORDS]
    return _dedupe([fold(t) for t in toks])


def normalize_name(name):
    """Return (norm, core, compact, key, alt) versions of a business name.

    norm    : all tokens, transliterated, cleaned, de-duplicated
    core    : norm minus legal forms (falls back to norm if empty)
    compact : core with no separators ('onelogistics'), meets website-style names
    key     : core minus generic descriptors ('superior hospitality')
    alt     : the other half of a 'X trading as Y' / 'X | website' name, else ''
    """
    raw = _PAREN_ID.sub(" ", name or "")
    native = any(ord(ch) > 0x24F for ch in raw)
    main, alt = raw, ""
    if " | " in main:                          # 'Turcios, Cheney and | www.turciosc.com'
        main, alt = main.split(" | ", 1)
    main = re.sub(r"(?<=\b[a-zA-Z])\.(?=[a-zA-Z]\b)", "", main).replace(".", " ")  # D.B.A. -> DBA
    m = _ALIAS.search(main)
    if m:                                      # 'Quoumbracalox trading as Hermosillo Gas'
        before, after = main[:m.start()], main[m.end():]
        if after.strip():
            main, alt = after, before
    toks = _name_tokens(main, native)
    norm = " ".join(toks)
    core_toks = [t for t in toks if t not in LEGAL_WORDS]
    core = " ".join(core_toks) if core_toks else norm
    key_toks = [t for t in core_toks if t not in DESCRIPTORS]
    # Removing generic words must not leave a tiny, collision-prone name
    # ('SI Consultant' and 'Si Group' -> 'si'): keep the core name instead.
    if not key_toks or (len(key_toks) == 1 and len(key_toks[0]) < 6):
        key_toks = core_toks
    key = " ".join(key_toks) if key_toks else core
    compact = core.replace(" ", "")
    alt_toks = [t for t in _name_tokens(alt, native) if t not in LEGAL_WORDS] if alt else []
    return norm, core, compact, key, " ".join(alt_toks)


LEGAL_WORDS = {fold(w) for w in LEGAL_WORDS}
HONORIFICS = {fold(w) for w in HONORIFICS} | HONORIFICS
DESCRIPTORS = {fold(w) for w in DESCRIPTORS}

# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------

# Every spelling -> ONE short canonical form. Short forms avoid collisions such as
# "St Louis" -> "Street Louis": both "street" and "saint" become "st".
ADDRESS_CANON = {
    "street": "st", "str": "st", "saint": "st",
    "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr", "lane": "ln",
    "court": "ct", "terrace": "ter", "terr": "ter", "boulevard": "blvd", "bd": "blvd",
    "bld": "blvd", "highway": "hwy", "place": "pl", "square": "sq", "circle": "cir",
    "parkway": "pkwy", "trail": "trl", "mount": "mt", "fort": "ft", "township": "twp",
    "apartment": "apt", "appt": "apt", "suite": "ste", "sector": "sec", "building": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "near": "nr", "opposite": "opp", "opp": "opp",
    "rue": "r", "cours": "crs", "impasse": "imp", "chemin": "che", "route": "rte",
    "allee": "all", "quai": "qu", "centre": "ctr", "center": "ctr",
    # renamed Indian cities / common variants
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "gurgaon": "gurugram", "poona": "pune", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "mysore": "mysuru", "baroda": "vadodara", "belgaum": "belagavi",
    "mangalore": "mangaluru", "allahabad": "prayagraj", "simla": "shimla",
    "benares": "varanasi", "banaras": "varanasi", "pondicherry": "puducherry",
}

# Formatting words that carry no location identity (numbers next to them are kept).
ADDRESS_NOISE = {
    "no", "number", "unit", "po", "box", "pmb", "null", "none", "na", "nil",
    "door", "house", "hno", "hn", "h", "plot", "block", "blk", "flat", "shop", "floor", "fl",
    "cdp", "county", "city", "district", "dist", "region", "off", "at", "post", "via",
    "ground", "gf", "premises", "twp", "suburban", "urban", "of", "incorporated",
    "rdc", "rez", "chaussee",
}

ORDINAL_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
                 "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10"}

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
    "kerala": "kl", "keralam": "kl", "kerlm": "kl",
    "bihar": "br",
    "madhya pradesh": "mp", "mdhy prdes": "mp",
    "andhra pradesh": "ap", "amdhrprdes": "ap",
    "punjab": "pb", "pmjab": "pb",
    "odisha": "od", "orissa": "od", "od isa": "od", "odisa": "od",
    "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk",
    "himachal pradesh": "hp", "goa": "ga", "chandigarh": "ch",
    "jammu and kashmir": "jk", "puducherry": "py",
}

# French regions and the departements that appear in the data -> one region code,
# so 'Hauts-de-France' and 'Nord' (or 'Nouvelle-Aquitaine' and 'Gironde') agree.
FRANCE_REGIONS = {
    "hauts de france": "hdf", "nord": "hdf", "pas de calais": "hdf",
    "nouvelle aquitaine": "naq", "gironde": "naq",
    "pays de la loire": "pdl", "loire atlantique": "pdl",
}

# Generic street-type words: dropped for the address "key", which keeps only the
# identifying words (street name, locality, city) and numbers.
ADDRESS_GENERIC = {
    "st", "rd", "ave", "dr", "ln", "ct", "ter", "blvd", "hwy", "pl", "sq", "cir", "pkwy",
    "trl", "way", "apt", "ste", "sec", "bldg", "n", "s", "e", "w", "nr", "opp", "r", "crs",
    "imp", "che", "rte", "all", "qu", "main", "nagar", "colony", "phase", "town", "village",
    "park", "road", "marg", "area", "layout", "extension", "ext", "cross", "industrial", "estate",
}

_ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
_GLUED = re.compile(r"^([a-z]{1,4})(\d+)$")


def _replace_states(text, country):
    """Map full state names to short codes for the countries we have tables for."""
    table = {"US": US_STATES, "India": INDIA_STATES, "France": FRANCE_REGIONS}.get(country)
    if not table:
        return text
    for full, code in sorted(table.items(), key=lambda kv: -len(kv[0])):  # "west virginia" before "virginia"
        if full in text:
            text = re.sub(rf"\b{re.escape(full)}\b", code, text)
    return text


def normalize_address(address, country):
    """Return (norm, numbers, key) for an address.

    norm    : transliterated, every spelling mapped to one short canonical form,
              state names -> codes, ordinals '2nd' -> '2', leading zeros removed
              ('B-00200' -> 'b 200'), formatting words dropped
    numbers : space-joined sorted set of the numbers in the address
    key     : norm minus generic street-type words (identifying words + numbers)
    """
    low = to_ascii_lower(address or "")
    low = re.sub(r"['’`]", "", low)
    low = low.replace(".", " ")                 # 'No.301' -> 'no 301' (deleting dots glued numbers to words)
    low = _replace_states(" ".join(tokens(low)), country)
    out = []
    toks = []
    for t in low.split():                       # split 'no301' / 'g02' -> 'no 301' / 'g 2'
        m = _GLUED.match(t)
        toks += [m.group(1), m.group(2)] if m and not _ORDINAL.match(t) else [t]
    for t in toks:
        m = _ORDINAL.match(t)
        if m:
            t = m.group(1)
        t = ORDINAL_WORDS.get(t, t)
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = ADDRESS_CANON.get(t, t)
            if t in ADDRESS_NOISE:
                continue
        out.append(t)
    out = _dedupe(out)
    nums = sorted({t for t in out if t.isdigit()} |
                  {(d.lstrip("0") or "0") for d in _DIGITS.findall(" ".join(out))})
    key = [t for t in out if t not in ADDRESS_GENERIC]
    return " ".join(out), " ".join(nums), " ".join(key)
