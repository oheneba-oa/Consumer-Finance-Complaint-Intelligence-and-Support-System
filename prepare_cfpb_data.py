from pathlib import Path
from collections import Counter, defaultdict
import re

import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------
RAW_FILE = Path(
    "data/raw/complaints.csv"
)

PROCESSED_DIR = Path(
    "data/processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
TARGET_YEAR = 2025

CHUNK_SIZE = 100_000

# Roughly 2% deterministic sample
SAMPLE_MODULUS = 50


# --------------------------------------------------
# REQUIRED CANONICAL COLUMNS
# --------------------------------------------------
REQUIRED_COLUMNS = [
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company_public_response",
    "company",
    "state",
    "tags",
    "submitted_via",
    "date_sent_to_company",
    "company_response_to_consumer",
    "timely_response",
    "complaint_id"
]


# --------------------------------------------------
# NORMALISE COLUMN NAME
# --------------------------------------------------
def column_key(name):
    """
    Convert raw column names into a simplified form
    so minor spacing or punctuation differences do
    not break the script.
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(name).lower()
    )


# --------------------------------------------------
# MAP RAW COLUMN KEYS TO CLEAN COLUMN NAMES
# --------------------------------------------------
COLUMN_KEY_MAP = {
    "datereceived":
        "date_received",

    "product":
        "product",

    "subproduct":
        "sub_product",

    "issue":
        "issue",

    "subissue":
        "sub_issue",

    "companypublicresponse":
        "company_public_response",

    "company":
        "company",

    "state":
        "state",

    "tags":
        "tags",

    "submittedvia":
        "submitted_via",

    "datesenttocompany":
        "date_sent_to_company",

    "companyresponsetoconsumer":
        "company_response_to_consumer",

    "timelyresponse":
        "timely_response",

    "complaintid":
        "complaint_id"
}


# --------------------------------------------------
# INSPECT REAL CSV HEADER
# --------------------------------------------------
print(
    "Inspecting CFPB CSV columns..."
)

header = pd.read_csv(
    RAW_FILE,
    nrows=0
)


# --------------------------------------------------
# CREATE RENAME LOOKUP
# --------------------------------------------------
rename_lookup = {}

for original_column in header.columns:

    key = column_key(
        original_column
    )

    if key in COLUMN_KEY_MAP:

        rename_lookup[
            original_column
        ] = COLUMN_KEY_MAP[key]


# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------
detected_clean_columns = set(
    rename_lookup.values()
)

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in detected_clean_columns
]


if missing_columns:

    print()
    print(
        "ERROR: Some required columns "
        "could not be identified."
    )

    print(
        "Missing columns:"
    )

    for column in missing_columns:

        print(
            f"- {column}"
        )

    print()
    print(
        "Raw CSV columns detected:"
    )

    for column in header.columns:

        print(
            f"- {repr(column)}"
        )

    raise SystemExit(
        "Column validation failed."
    )


# --------------------------------------------------
# SELECT ONLY REQUIRED RAW COLUMNS
# --------------------------------------------------
raw_columns_to_use = list(
    rename_lookup.keys()
)


print()
print(
    "Required columns detected successfully."
)

print()


# --------------------------------------------------
# AGGREGATION COUNTERS
# --------------------------------------------------
product_counts = Counter()

issue_counts = Counter()

company_counts = Counter()

response_counts = Counter()

submission_channel_counts = Counter()

state_counts = Counter()

month_counts = Counter()


# --------------------------------------------------
# PRODUCT-LEVEL RESPONSE STATISTICS
# --------------------------------------------------
product_timely_total = defaultdict(
    int
)

product_timely_yes = defaultdict(
    int
)


# --------------------------------------------------
# OVERALL COUNTERS
# --------------------------------------------------
total_2025_complaints = 0

total_timely_known = 0

total_timely_yes = 0


# --------------------------------------------------
# SAMPLE STORAGE
# --------------------------------------------------
sample_chunks = []


# --------------------------------------------------
# SAFE VALUE COUNTS
# --------------------------------------------------
def safe_value_counts(series):
    """
    Return counts while excluding missing values.
    """

    cleaned = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    cleaned = cleaned[
        cleaned != ""
    ]

    return Counter(
        cleaned
        .value_counts()
        .to_dict()
    )


# --------------------------------------------------
# PROCESS RAW FILE IN CHUNKS
# --------------------------------------------------
print(
    "Starting CFPB data preparation..."
)

print()


chunk_number = 0


for chunk in pd.read_csv(

    RAW_FILE,

    usecols=raw_columns_to_use,

    chunksize=CHUNK_SIZE,

    low_memory=False
):

    chunk_number += 1

    print(
        f"Processing chunk "
        f"{chunk_number}..."
    )


    # ----------------------------------------------
    # RENAME TO CLEAN COLUMN NAMES
    # ----------------------------------------------
    chunk = chunk.rename(
        columns=rename_lookup
    )


    # ----------------------------------------------
    # PARSE DATE RECEIVED
    # ----------------------------------------------
    chunk[
        "date_received"
    ] = pd.to_datetime(

        chunk[
            "date_received"
        ],

        errors="coerce"
    )


    # ----------------------------------------------
    # FILTER TO TARGET YEAR
    # ----------------------------------------------
    chunk_2025 = chunk[

        chunk[
            "date_received"
        ].dt.year == TARGET_YEAR

    ].copy()


    if chunk_2025.empty:

        continue


    # ----------------------------------------------
    # PARSE DATE SENT TO COMPANY
    # ----------------------------------------------
    chunk_2025[
        "date_sent_to_company"
    ] = pd.to_datetime(

        chunk_2025[
            "date_sent_to_company"
        ],

        errors="coerce"
    )


    # ----------------------------------------------
    # CLEAN STRING COLUMNS
    # ----------------------------------------------
    string_columns = [
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "company_public_response",
        "company",
        "state",
        "tags",
        "submitted_via",
        "company_response_to_consumer",
        "timely_response"
    ]


    for column in string_columns:

        chunk_2025[
            column
        ] = (

            chunk_2025[
                column
            ]
            .astype("string")
            .str.strip()
        )


    # ----------------------------------------------
    # ADD MONTH
    # ----------------------------------------------
    chunk_2025[
        "month"
    ] = (

        chunk_2025[
            "date_received"
        ]
        .dt.to_period("M")
        .astype(str)
    )


    # ----------------------------------------------
    # UPDATE TOTAL
    # ----------------------------------------------
    total_2025_complaints += len(
        chunk_2025
    )


    # ----------------------------------------------
    # AGGREGATIONS
    # ----------------------------------------------
    product_counts.update(
        safe_value_counts(
            chunk_2025[
                "product"
            ]
        )
    )


    issue_counts.update(
        safe_value_counts(
            chunk_2025[
                "issue"
            ]
        )
    )


    company_counts.update(
        safe_value_counts(
            chunk_2025[
                "company"
            ]
        )
    )


    response_counts.update(
        safe_value_counts(
            chunk_2025[
                "company_response_to_consumer"
            ]
        )
    )


    submission_channel_counts.update(
        safe_value_counts(
            chunk_2025[
                "submitted_via"
            ]
        )
    )


    state_counts.update(
        safe_value_counts(
            chunk_2025[
                "state"
            ]
        )
    )


    month_counts.update(
        safe_value_counts(
            chunk_2025[
                "month"
            ]
        )
    )


    # ----------------------------------------------
    # TIMELY RESPONSE
    # ----------------------------------------------
    timely_clean = (

        chunk_2025[
            "timely_response"
        ]
        .str.lower()
    )


    known_timely_mask = (
        timely_clean
        .isin(
            [
                "yes",
                "no"
            ]
        )
    )


    yes_timely_mask = (
        timely_clean
        .eq("yes")
    )


    total_timely_known += int(
        known_timely_mask.sum()
    )


    total_timely_yes += int(
        yes_timely_mask.sum()
    )


    # ----------------------------------------------
    # TIMELY RESPONSE BY PRODUCT
    # ----------------------------------------------
    for product, group in (

        chunk_2025
        .groupby(
            "product",
            dropna=True
        )
    ):

        product_timely = (

            group[
                "timely_response"
            ]
            .str.lower()
        )


        known_mask = (

            product_timely
            .isin(
                [
                    "yes",
                    "no"
                ]
            )
        )


        yes_mask = (
            product_timely
            .eq("yes")
        )


        product_timely_total[
            str(product)
        ] += int(
            known_mask.sum()
        )


        product_timely_yes[
            str(product)
        ] += int(
            yes_mask.sum()
        )


    # ----------------------------------------------
    # DETERMINISTIC SAMPLE
    # ----------------------------------------------
    complaint_ids = pd.to_numeric(

        chunk_2025[
            "complaint_id"
        ],

        errors="coerce"
    )


    sample_mask = (

        complaint_ids.notna()

        & (
            complaint_ids
            .fillna(0)
            .astype("int64")
            % SAMPLE_MODULUS
            == 0
        )
    )


    sampled_rows = (

        chunk_2025[
            sample_mask
        ]
        .copy()
    )


    if not sampled_rows.empty:

        sample_chunks.append(
            sampled_rows
        )


# --------------------------------------------------
# COMBINE SAMPLE
# --------------------------------------------------
print()

print(
    "Combining sampled complaint records..."
)


if sample_chunks:

    sample_df = pd.concat(
        sample_chunks,
        ignore_index=True
    )

else:

    sample_df = pd.DataFrame()


# --------------------------------------------------
# SAVE SAMPLE
# --------------------------------------------------
sample_df.to_csv(

    PROCESSED_DIR
    / "cfpb_2025_sample.csv",

    index=False
)


# --------------------------------------------------
# PRODUCT COUNTS
# --------------------------------------------------
product_df = pd.DataFrame(

    product_counts.items(),

    columns=[
        "product",
        "complaint_count"
    ]
)


product_df = (
    product_df
    .sort_values(
        "complaint_count",
        ascending=False
    )
)


product_df[
    "percentage_of_complaints"
] = (

    product_df[
        "complaint_count"
    ]

    / total_2025_complaints

    * 100
)


product_df.to_csv(

    PROCESSED_DIR
    / "complaints_by_product.csv",

    index=False
)


# --------------------------------------------------
# ISSUE COUNTS
# --------------------------------------------------
issue_df = pd.DataFrame(

    issue_counts.items(),

    columns=[
        "issue",
        "complaint_count"
    ]
)


issue_df = (
    issue_df
    .sort_values(
        "complaint_count",
        ascending=False
    )
)


issue_df.to_csv(

    PROCESSED_DIR
    / "complaints_by_issue.csv",

    index=False
)


# --------------------------------------------------
# COMPANY COUNTS
# --------------------------------------------------
company_df = pd.DataFrame(

    company_counts.items(),

    columns=[
        "company",
        "complaint_count"
    ]
)


company_df = (
    company_df
    .sort_values(
        "complaint_count",
        ascending=False
    )
)


company_df.to_csv(

    PROCESSED_DIR
    / "complaints_by_company.csv",

    index=False
)


# --------------------------------------------------
# RESPONSE COUNTS
# --------------------------------------------------
response_df = pd.DataFrame(

    response_counts.items(),

    columns=[
        "company_response",
        "complaint_count"
    ]
)


response_df = (
    response_df
    .sort_values(
        "complaint_count",
        ascending=False
    )
)


response_df.to_csv(

    PROCESSED_DIR
    / "complaints_by_response.csv",

    index=False
)


# --------------------------------------------------
# CHANNEL COUNTS
# --------------------------------------------------
channel_df = pd.DataFrame(

    submission_channel_counts.items(),

    columns=[
        "submitted_via",
        "complaint_count"
    ]
)


channel_df = (
    channel_df
    .sort_values(
        "complaint_count",
        ascending=False
    )
)


channel_df.to_csv(

    PROCESSED_DIR
    / "complaints_by_channel.csv",

    index=False
)


# --------------------------------------------------
# STATE COUNTS
# --------------------------------------------------
state_df = pd.DataFrame(

    state_counts.items(),

    columns=[
        "state",
        "complaint_count"
    ]
)


state_df = (
    state_df
    .sort_values(
        "complaint_count",
        ascending=False
    )
)


state_df.to_csv(

    PROCESSED_DIR
    / "complaints_by_state.csv",

    index=False
)


# --------------------------------------------------
# MONTH COUNTS
# --------------------------------------------------
month_df = pd.DataFrame(

    month_counts.items(),

    columns=[
        "month",
        "complaint_count"
    ]
)


month_df = (
    month_df
    .sort_values(
        "month"
    )
)


month_df.to_csv(

    PROCESSED_DIR
    / "complaints_by_month.csv",

    index=False
)


# --------------------------------------------------
# TIMELY RESPONSE BY PRODUCT
# --------------------------------------------------
product_response_rows = []


for product in product_counts:

    known_total = (
        product_timely_total[
            product
        ]
    )

    yes_total = (
        product_timely_yes[
            product
        ]
    )


    if known_total > 0:

        timely_rate = (
            yes_total
            / known_total
            * 100
        )

    else:

        timely_rate = None


    product_response_rows.append(
        {
            "product":
                product,

            "known_response_count":
                known_total,

            "timely_response_count":
                yes_total,

            "timely_response_rate":
                timely_rate
        }
    )


product_response_df = pd.DataFrame(
    product_response_rows
)


product_response_df = (
    product_response_df
    .sort_values(
        "known_response_count",
        ascending=False
    )
)


product_response_df.to_csv(

    PROCESSED_DIR
    / "timely_response_by_product.csv",

    index=False
)


# --------------------------------------------------
# OVERALL SUMMARY
# --------------------------------------------------
if total_timely_known > 0:

    timely_response_rate = (

        total_timely_yes
        / total_timely_known
        * 100
    )

else:

    timely_response_rate = None


summary_df = pd.DataFrame(
    [
        {
            "year":
                TARGET_YEAR,

            "total_complaints":
                total_2025_complaints,

            "sample_records":
                len(sample_df),

            "unique_products":
                len(product_counts),

            "unique_issues":
                len(issue_counts),

            "unique_companies":
                len(company_counts),

            "known_timely_responses":
                total_timely_known,

            "timely_responses":
                total_timely_yes,

            "timely_response_rate":
                timely_response_rate
        }
    ]
)


summary_df.to_csv(

    PROCESSED_DIR
    / "summary_metrics.csv",

    index=False
)


# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------
print()

print(
    "CFPB data preparation completed."
)

print(
    f"Total 2025 complaints: "
    f"{total_2025_complaints:,}"
)

print(
    f"Sample records saved: "
    f"{len(sample_df):,}"
)


if timely_response_rate is not None:

    print(
        f"Timely response rate: "
        f"{timely_response_rate:.2f}%"
    )


print()

print(
    "Processed files saved to:"
)

print(
    PROCESSED_DIR.resolve()
)