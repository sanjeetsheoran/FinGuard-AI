"""
FinGuard AI - Feature Configuration

Defines feature groups for the IEEE-CIS fraud detection dataset.

This module only defines feature roles.
No data is loaded and no preprocessing is performed here.
"""

# ============================================================
# Target
# ============================================================

TARGET_COLUMN = "isFraud"


# ============================================================
# Identifiers
# ============================================================

IDENTIFIER_COLUMNS = [
    "TransactionID",
]


# ============================================================
# Transaction Core Features
# ============================================================

TRANSACTION_NUMERIC_FEATURES = [
    "TransactionDT",
    "TransactionAmt",
]


TRANSACTION_CATEGORICAL_FEATURES = [
    "ProductCD",
]


# ============================================================
# Card Features
# ============================================================

CARD_CATEGORICAL_FEATURES = [
    "card4",
    "card6",
]


CARD_NUMERIC_FEATURES = [
    "card1",
    "card2",
    "card3",
    "card5",
]


# ============================================================
# Address / Distance Features
# ============================================================

ADDRESS_NUMERIC_FEATURES = [
    "addr1",
    "addr2",
    "dist1",
    "dist2",
]


# ============================================================
# Email Features
# ============================================================

EMAIL_CATEGORICAL_FEATURES = [
    "P_emaildomain",
    "R_emaildomain",
]


# ============================================================
# Match / Verification Features
# ============================================================

MATCH_CATEGORICAL_FEATURES = [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]


# ============================================================
# Identity / Device Features
# ============================================================

IDENTITY_PREFIXES = [
    "id_",
]


# ============================================================
# Vesta Numerical Features
# ============================================================

VESTA_NUMERIC_PREFIXES = [
    "C",
    "D",
    "V",
]


# ============================================================
# Explicit Interaction Features
# ============================================================

INTERACTION_FEATURES = [
    "ProductCD_card6",
    "ProductCD_card4",
    "ProductCD_P_emaildomain",
    "ProductCD_R_emaildomain",
]


# ============================================================
# Derived Features
# ============================================================

DERIVED_NUMERIC_FEATURES = [
    "TransactionHour",
    "TransactionDay",
    "TransactionAmtLog",
]


DERIVED_BOOLEAN_FEATURES = [
    "identity_available",
]


# ============================================================
# Helper Functions
# ============================================================

def is_identity_feature(column_name: str) -> bool:
    """Return True if a column belongs to the identity feature family."""
    return any(
        column_name.startswith(prefix)
        for prefix in IDENTITY_PREFIXES
    )


def is_vesta_numeric_feature(column_name: str) -> bool:
    """Return True for C/D/V feature families."""
    return any(
        column_name.startswith(prefix)
        for prefix in VESTA_NUMERIC_PREFIXES
    )


def is_target(column_name: str) -> bool:
    """Return True if the column is the target."""
    return column_name == TARGET_COLUMN


def is_identifier(column_name: str) -> bool:
    """Return True if the column is an identifier."""
    return column_name in IDENTIFIER_COLUMNS