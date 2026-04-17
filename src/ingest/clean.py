"""
clean.py — strip boilerplate from Camden FOI Document Text and fix encoding.

Each raw record follows this structure:
    [header]     Date / Ref / Dear Requester / Thank you for your request...
                 "We have dealt with this under the Freedom of Information Act 2000."
    [body]       The actual substantive response  ← we want this
    [address]    London Borough of Camden / Judd Street...  (appears inline)
    [footer]     "Further Information We do not give our consent..."
                 Your Rights / ICO details / Yours sincerely...
"""

import re
import unicodedata

# Matches the entire header up to and including the statutory FOIA/EIR declaration.
# Non-greedy .*? with DOTALL so it stops at the first matching sentence.
_HEADER_RE = re.compile(
    r"^.*?We have dealt with this under the "
    r"(?:Freedom of Information Act 2000|Environmental Information Regulations)"
    r"[^.]*\.",
    re.DOTALL | re.IGNORECASE,
)

# The Camden address block that appears inline mid-document (PDF artefact).
_ADDRESS_RE = re.compile(
    r"London Borough of Camden\s+"
    r"Information and Records Management\s+"
    r"Judd Street\s+London[.,\s]+WC1H\s+9JE\s+"
    r"e-?mail:\s*foi@camden\.gov\.uk",
    re.IGNORECASE,
)

# Footer boilerplate — everything from this phrase to the end of the document.
_FOOTER_RE = re.compile(
    r"Further Information\s+We do not give our consent.*$",
    re.DOTALL | re.IGNORECASE,
)


def _fix_encoding(text: str) -> str:
    """Normalise unicode (smart quotes, dashes, etc.) and remove control characters."""
    text = unicodedata.normalize("NFKC", text)
    # Strip C0/C1 control characters, keeping tab (\x09) and newline (\x0a)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def clean(text: str) -> str:
    """Return the substantive response body with all boilerplate removed."""
    text = _fix_encoding(text)
    text = _HEADER_RE.sub("", text, count=1).strip()
    text = _ADDRESS_RE.sub("", text)
    text = _FOOTER_RE.sub("", text).strip()
    # Collapse runs of blank lines and trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
