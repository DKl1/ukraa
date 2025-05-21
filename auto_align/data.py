import os
import csv
import logging
import docx2txt

from lxml import etree
from collections import OrderedDict
from pdfminer.high_level import extract_text as pdf_extract
from bs4 import BeautifulSoup

import nltk
logger = logging.getLogger(__name__)


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return open(path, "r", encoding="utf-8").read()
    elif ext == ".pdf":
        return pdf_extract(path)
    elif ext == ".docx":
        return docx2txt.process(path)
    elif ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            return "\n".join(row[0] for row in reader if row)
    elif ext in {".html", ".htm"}:
        html = open(path, "r", encoding="utf-8").read()
        return BeautifulSoup(html, "lxml").get_text(separator="\n")
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def preprocess(sentences, min_len=3, max_symbol_ratio=0.5):
    """
    - drop sent < min_len
    - drop if non-alnum ratio > max_symbol_ratio
    - dedup
    """
    stats = {"short": 0, "noisy": 0}
    filtered = []
    for s in sentences:
        s = s.strip()
        if len(s) < min_len:
            stats["short"] += 1
            continue
        symbols = sum(1 for c in s if not c.isalnum() and not c.isspace())
        if symbols / len(s) > max_symbol_ratio:
            stats["noisy"] += 1
            continue
        filtered.append(s)
    deduped = list(OrderedDict.fromkeys(filtered))
    stats["deduped"] = len(filtered) - len(deduped)
    logger.info(
        f"Preprocess: removed {stats['short']} short, "
        f"{stats['noisy']} noisy, "
        f"{stats['deduped']} duplicates"
    )
    return deduped


def load_and_preprocess(path: str):
    raw = extract_text(path)
    sents = nltk.tokenize.sent_tokenize(raw)
    return preprocess(sents)


def parse_tmx(path: str, src_code: str, tgt_code: str):

    XML_NS = "http://www.w3.org/XML/1998/namespace"
    tree = etree.parse(path)
    root = tree.getroot()

    src_sents, tgt_sents = [], []
    for tu in root.findall(".//{*}tu"):
        src_text = None
        tgt_text = None

        for tuv in tu.findall("{*}tuv"):
            lang = tuv.get(f"{{{XML_NS}}}lang")
            seg = tuv.find("{*}seg")
            if seg is None or seg.text is None:
                continue

            text = seg.text.strip()
            if lang == src_code:
                src_text = text
            elif lang == tgt_code:
                tgt_text = text

        # Only keep if we have both sides
        if src_text and tgt_text:
            src_sents.append(src_text)
            tgt_sents.append(tgt_text)

    logging.getLogger(__name__).info(f"Parsed {len(src_sents)} units from TMX")
    return src_sents, tgt_sents
