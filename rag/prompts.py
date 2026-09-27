from __future__ import annotations

import config

ADVICE_REFUSAL = (
    "I can share public facts about this scheme, but I can't give investment advice "
    "or recommendations. Please consult a SEBI-registered investment advisor for "
    "guidance."
)

PERFORMANCE_REFUSAL = (
    "I don't share return or performance figures. Please check the official factsheet "
    "for audited performance."
)

PII_REFUSAL = (
    "I can't help with personal account details. Please remove any account numbers, "
    "phone numbers, or email addresses from your question."
)

OUT_OF_CORPUS = (
    "I couldn't find that in my sources. Please check the official scheme page for the "
    "most current information."
)

DISCLAIMER = "Facts-only. No investment advice."

EDUCATION_LINK = "https://www.investor.gov.in/"


def factsheet_link(scheme: str | None = None) -> str:
    if scheme and scheme in config.SCHEME_URLS:
        return config.SCHEME_URLS[scheme]
    return config.HDFC_MF_HOME
