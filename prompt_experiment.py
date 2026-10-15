from pathlib import Path
from itertools import combinations

import pandas as pd
import chromadb

from ollama import chat
from pydantic import BaseModel, ValidationError
from typing import Literal
from sentence_transformers import SentenceTransformer


# ==================================================
# SETTINGS
# ==================================================
MODEL_NAME = "llama3.2"

COLLECTION_NAME = "customer_service_knowledge"

CHROMA_PATH = "data/chroma_db"

NUMBER_OF_RESULTS = 4

RELEVANCE_THRESHOLD = 0.65

RUNS_PER_CASE = 3

TEMPERATURE = 0.2


# ==================================================
# OUTPUT DIRECTORY
# ==================================================
OUTPUT_DIR = Path(
    "evaluation/prompt_experiments"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# STRUCTURED OUTPUT SCHEMA
# ==================================================
class ComplaintAssessment(BaseModel):

    product: Literal[
        "Credit reporting or other personal consumer reports",
        "Debt collection",
        "Credit card",
        "Money transfer, virtual currency, or money service",
        "Checking or savings account",
        "Mortgage",
        "Student loan",
        "Vehicle loan or lease",
        "Payday loan, title loan, personal loan, or advance loan",
        "Prepaid card",
        "Debt or credit management",
        "Uncertain"
    ]

    summary: str

    recommended_action: str

    evidence_available: bool

    human_review_required: bool


# ==================================================
# PROMPT VARIANT A
# BASIC DOMAIN PROMPT
# ==================================================
PROMPT_A = """
You are an AI assistant that helps consumers
understand financial complaints.

Use the CFPB information supplied to you when
answering the user.

Classify the complaint into the most appropriate
financial product category.

Provide:

- product
- summary
- recommended action
- whether evidence is available
- whether human review is required

Do not invent information.
"""


# ==================================================
# PROMPT VARIANT B
# STRICT GROUNDING PROMPT
# ==================================================
PROMPT_B = """
You are an AI Consumer Complaint Support Assistant.

Your answers must be grounded in the CFPB context
supplied to you.

Do not invent CFPB rules, deadlines, procedures,
statistics, complaint outcomes or company actions.

If the retrieved CFPB information does not support
a claim, do not present that claim as fact.

Classify the complaint into the most appropriate
CFPB financial product category.

If there is not enough information to identify a
product confidently, use "Uncertain".

Provide:

- the financial product
- a concise summary of the consumer's issue
- an action supported by the supplied CFPB context
- whether relevant supporting evidence is available
- whether human review is required

If relevant evidence is unavailable, clearly state
the limitation and recommend human review.

Do not provide legal advice.
"""


# ==================================================
# PROMPT VARIANT C
# EVIDENCE-AWARE / PRODUCTION PROMPT
# ==================================================
PROMPT_C = """
You are an AI Consumer Complaint Intelligence and
Support Assistant working with CFPB complaint
information.

Your job is to interpret the user's request while
remaining grounded in the supplied CFPB evidence.

Follow these rules:

1. Use the retrieved CFPB context as the primary
   evidence source.

2. Never invent CFPB policies, procedures,
   deadlines, legal requirements, statistics,
   company responses or complaint outcomes.

3. Separate what the consumer has told you from
   what the CFPB evidence supports.

4. If the evidence is incomplete, irrelevant or
   insufficient, say so rather than filling the gap
   using assumptions.

5. Classify the complaint using the CFPB financial
   product categories provided by the output schema.

6. Use "Uncertain" when the available information
   does not support a confident product
   classification.

7. Keep the summary factual and concise.

8. Make the recommended action practical but do not
   present legal, financial or regulatory advice as
   a guaranteed outcome.

9. Require human review when the situation cannot
   be safely or adequately resolved from the
   available evidence.

10. Avoid unnecessary verbosity.

Return:

- product
- summary
- recommended action
- evidence availability
- human review requirement
"""


# ==================================================
# PROMPT DICTIONARY
# ==================================================
PROMPT_VARIANTS = {
    "A_Basic": PROMPT_A,
    "B_Strict_Grounding": PROMPT_B,
    "C_Evidence_Aware": PROMPT_C
}


# ==================================================
# TEST CASES
# ==================================================
TEST_CASES = [
    {
        "case_id": "TC01",
        "scenario": "Response deadline",
        "question":
            "How long does a company have to provide "
            "a timely response to a CFPB complaint?",

        "expected_product":
            "Uncertain",

        "expected_concepts": [
            "15 calendar days",
            "60 calendar days"
        ]
    },

    {
        "case_id": "TC02",
        "scenario": "Incorrect credit report",
        "question":
            "My credit report has an account that "
            "does not belong to me and the incorrect "
            "information is lowering my score. "
            "What should I do?",

        "expected_product":
            "Credit reporting or other personal consumer reports",

        "expected_concepts": [
            "credit"
        ]
    },

    {
        "case_id": "TC03",
        "scenario": "Unrecognized debt",
        "question":
            "A debt collector keeps contacting me "
            "about a debt I do not recognize. "
            "What kind of complaint is this and "
            "what should I do?",

        "expected_product":
            "Debt collection",

        "expected_concepts": [
            "debt"
        ]
    },

    {
        "case_id": "TC04",
        "scenario": "Frozen checking account",
        "question":
            "My bank froze my checking account after "
            "suspicious activity and I cannot access "
            "my funds. How should this complaint be "
            "classified and what can I do next?",

        "expected_product":
            "Checking or savings account",

        "expected_concepts": [
            "checking"
        ]
    },

    {
        "case_id": "TC05",
        "scenario": "Out-of-domain complaint",
        "question":
            "My landlord has not returned my "
            "apartment security deposit. Can the "
            "CFPB help me recover it?",

        "expected_product":
            "Uncertain",

        "expected_concepts": []
    }
]


# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================
print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# CONNECT TO CHROMADB
# ==================================================
print(
    "Connecting to CFPB knowledge base..."
)

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ==================================================
# RAG SEARCH
# ==================================================
def search_knowledge_base(query):

    query_embedding = (
        embedding_model
        .encode(query)
        .tolist()
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],

        n_results=NUMBER_OF_RESULTS,

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = (
        results["documents"][0]
    )

    metadatas = (
        results["metadatas"][0]
    )

    distances = (
        results["distances"][0]
    )

    context_parts = []

    sources = []


    for (
        document,
        metadata,
        distance
    ) in zip(
        documents,
        metadatas,
        distances
    ):

        if (
            distance
            > RELEVANCE_THRESHOLD
        ):
            continue


        similarity = (
            1 - distance
        )


        source = metadata.get(
            "source",
            "Unknown source"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        chunk = metadata.get(
            "chunk",
            "Unknown"
        )


        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk}\n"
            f"Similarity: {similarity:.3f}\n"
            f"{document}"
        )


        sources.append(
            f"{source} | "
            f"Page {page} | "
            f"Chunk {chunk}"
        )


    return (
        "\n\n".join(
            context_parts
        ),
        sources
    )


# ==================================================
# RUN ONE PROMPT
# ==================================================
def run_prompt(
    prompt,
    question,
    context
):

    if context:

        evidence_message = (
            "Retrieved CFPB evidence:\n\n"
            + context
        )

    else:

        evidence_message = (
            "No sufficiently relevant CFPB "
            "evidence was retrieved for this "
            "request."
        )


    response = chat(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": prompt
            },

            {
                "role": "system",
                "content": evidence_message
            },

            {
                "role": "user",
                "content": question
            }
        ],

        format=(
            ComplaintAssessment
            .model_json_schema()
        ),

        options={
            "temperature":
                TEMPERATURE
        }
    )


    return (
        ComplaintAssessment
        .model_validate_json(
            response.message.content
        )
    )


# ==================================================
# KEY CONCEPT COVERAGE
# ==================================================
def calculate_concept_coverage(
    output_text,
    expected_concepts
):

    if not expected_concepts:

        return None


    output_lower = (
        output_text.lower()
    )


    matches = 0


    for concept in expected_concepts:

        if (
            concept.lower()
            in output_lower
        ):
            matches += 1


    return (
        matches
        / len(expected_concepts)
    )


# ==================================================
# WORD COUNT
# ==================================================
def count_words(text):

    return len(
        text.split()
    )


# ==================================================
# CONSISTENCY SIMILARITY
# ==================================================
def calculate_similarity(
    texts
):

    if len(texts) < 2:

        return None


    embeddings = (
        embedding_model
        .encode(
            texts,
            normalize_embeddings=True
        )
    )


    similarities = []


    for first, second in combinations(
        range(len(texts)),
        2
    ):

        score = float(
            embeddings[first]
            @ embeddings[second]
        )

        similarities.append(
            score
        )


    if not similarities:

        return None


    return (
        sum(similarities)
        / len(similarities)
    )


# ==================================================
# SAVE PROMPT VARIANTS
# ==================================================
prompt_document = (
    "# Prompt Experiment Variants\n\n"
)


for (
    variant_name,
    prompt
) in PROMPT_VARIANTS.items():

    prompt_document += (
        f"## {variant_name}\n\n"
        f"```text\n"
        f"{prompt.strip()}\n"
        f"```\n\n"
    )


(
    OUTPUT_DIR
    / "prompt_variants.md"
).write_text(
    prompt_document,
    encoding="utf-8"
)


# ==================================================
# RUN EXPERIMENT
# ==================================================
results = []


print()

print(
    "Starting prompt experiment..."
)

print(
    f"Prompt variants: "
    f"{len(PROMPT_VARIANTS)}"
)

print(
    f"Test cases: "
    f"{len(TEST_CASES)}"
)

print(
    f"Runs per case: "
    f"{RUNS_PER_CASE}"
)

print()


for test_case in TEST_CASES:

    case_id = (
        test_case[
            "case_id"
        ]
    )

    question = (
        test_case[
            "question"
        ]
    )

    expected_product = (
        test_case[
            "expected_product"
        ]
    )

    expected_concepts = (
        test_case[
            "expected_concepts"
        ]
    )


    print(
        f"Retrieving evidence for "
        f"{case_id}..."
    )


    # Retrieve once so every prompt receives
    # exactly the same evidence.
    context, sources = (
        search_knowledge_base(
            question
        )
    )


    actual_evidence_available = bool(
        context
    )


    for (
        variant_name,
        prompt
    ) in PROMPT_VARIANTS.items():

        for run_number in range(
            1,
            RUNS_PER_CASE + 1
        ):

            print(
                f"Running "
                f"{variant_name} | "
                f"{case_id} | "
                f"Run {run_number}"
            )


            try:

                assessment = run_prompt(
                    prompt,
                    question,
                    context
                )


                combined_output = (
                    f"{assessment.product} "
                    f"{assessment.summary} "
                    f"{assessment.recommended_action}"
                )


                concept_coverage = (
                    calculate_concept_coverage(
                        combined_output,
                        expected_concepts
                    )
                )


                product_match = (
                    assessment.product
                    == expected_product
                )


                evidence_flag_match = (
                    assessment.evidence_available
                    == actual_evidence_available
                )


                results.append(
                    {
                        "variant":
                            variant_name,

                        "case_id":
                            case_id,

                        "scenario":
                            test_case[
                                "scenario"
                            ],

                        "run":
                            run_number,

                        "question":
                            question,

                        "expected_product":
                            expected_product,

                        "retrieved_evidence":
                            actual_evidence_available,

                        "sources":
                            " || ".join(
                                sources
                            ),

                        "valid_output":
                            True,

                        "predicted_product":
                            assessment.product,

                        "product_match":
                            product_match,

                        "evidence_available":
                            assessment.evidence_available,

                        "evidence_flag_match":
                            evidence_flag_match,

                        "human_review_required":
                            assessment.human_review_required,

                        "summary":
                            assessment.summary,

                        "recommended_action":
                            assessment.recommended_action,

                        "word_count":
                            count_words(
                                combined_output
                            ),

                        "key_concept_coverage":
                            concept_coverage,

                        "combined_output":
                            combined_output,

                        "error":
                            ""
                    }
                )


            except (
                ValidationError,
                Exception
            ) as error:

                results.append(
                    {
                        "variant":
                            variant_name,

                        "case_id":
                            case_id,

                        "scenario":
                            test_case[
                                "scenario"
                            ],

                        "run":
                            run_number,

                        "question":
                            question,

                        "expected_product":
                            expected_product,

                        "retrieved_evidence":
                            actual_evidence_available,

                        "sources":
                            " || ".join(
                                sources
                            ),

                        "valid_output":
                            False,

                        "predicted_product":
                            "",

                        "product_match":
                            False,

                        "evidence_available":
                            None,

                        "evidence_flag_match":
                            False,

                        "human_review_required":
                            None,

                        "summary":
                            "",

                        "recommended_action":
                            "",

                        "word_count":
                            0,

                        "key_concept_coverage":
                            None,

                        "combined_output":
                            "",

                        "error":
                            str(error)
                    }
                )


# ==================================================
# SAVE RAW RESULTS
# ==================================================
results_df = pd.DataFrame(
    results
)


raw_results_path = (
    OUTPUT_DIR
    / "prompt_experiment_results.csv"
)


results_df.to_csv(
    raw_results_path,
    index=False
)


# ==================================================
# BUILD CASE-LEVEL SUMMARY
# ==================================================
case_summary_rows = []


for (
    variant_name,
    case_id
), group in results_df.groupby(
    [
        "variant",
        "case_id"
    ]
):

    valid_group = group[
        group[
            "valid_output"
        ]
        == True
    ]


    if not valid_group.empty:

        texts = (
            valid_group[
                "combined_output"
            ]
            .tolist()
        )


        consistency_similarity = (
            calculate_similarity(
                texts
            )
        )


        structured_validity_rate = (
            len(valid_group)
            / len(group)
        )


        product_accuracy_rate = (
            group[
                "product_match"
            ]
            .astype(float)
            .mean()
        )


        evidence_flag_accuracy = (
            group[
                "evidence_flag_match"
            ]
            .astype(float)
            .mean()
        )


        average_word_count = (
            valid_group[
                "word_count"
            ]
            .mean()
        )


        concept_values = (
            valid_group[
                "key_concept_coverage"
            ]
            .dropna()
        )


        if not concept_values.empty:

            concept_coverage = (
                concept_values.mean()
            )

        else:

            concept_coverage = None


        product_consistency = (
            valid_group[
                "predicted_product"
            ]
            .value_counts(
                normalize=True
            )
            .iloc[0]
        )


        human_review_consistency = (
            valid_group[
                "human_review_required"
            ]
            .value_counts(
                normalize=True
            )
            .iloc[0]
        )


    else:

        consistency_similarity = None

        structured_validity_rate = 0

        product_accuracy_rate = 0

        evidence_flag_accuracy = 0

        average_word_count = 0

        concept_coverage = None

        product_consistency = 0

        human_review_consistency = 0


    case_summary_rows.append(
        {
            "variant":
                variant_name,

            "case_id":
                case_id,

            "scenario":
                group[
                    "scenario"
                ].iloc[0],

            "structured_validity_rate":
                structured_validity_rate,

            "product_accuracy_rate":
                product_accuracy_rate,

            "product_consistency":
                product_consistency,

            "response_similarity":
                consistency_similarity,

            "evidence_flag_accuracy":
                evidence_flag_accuracy,

            "key_concept_coverage":
                concept_coverage,

            "average_word_count":
                average_word_count,

            "human_review_consistency":
                human_review_consistency
        }
    )


case_summary_df = pd.DataFrame(
    case_summary_rows
)


case_summary_path = (
    OUTPUT_DIR
    / "prompt_experiment_case_summary.csv"
)


case_summary_df.to_csv(
    case_summary_path,
    index=False
)


# ==================================================
# BUILD OVERALL VARIANT SUMMARY
# ==================================================
overall_rows = []


for variant_name in (
    PROMPT_VARIANTS.keys()
):

    variant_cases = (
        case_summary_df[
            case_summary_df[
                "variant"
            ]
            == variant_name
        ]
    )


    overall_rows.append(
        {
            "variant":
                variant_name,

            "structured_validity_rate":
                variant_cases[
                    "structured_validity_rate"
                ].mean(),

            "product_accuracy_rate":
                variant_cases[
                    "product_accuracy_rate"
                ].mean(),

            "product_consistency":
                variant_cases[
                    "product_consistency"
                ].mean(),

            "response_similarity":
                variant_cases[
                    "response_similarity"
                ].mean(),

            "evidence_flag_accuracy":
                variant_cases[
                    "evidence_flag_accuracy"
                ].mean(),

            "key_concept_coverage":
                variant_cases[
                    "key_concept_coverage"
                ].mean(),

            "average_word_count":
                variant_cases[
                    "average_word_count"
                ].mean(),

            "human_review_consistency":
                variant_cases[
                    "human_review_consistency"
                ].mean()
        }
    )


overall_df = pd.DataFrame(
    overall_rows
)


overall_path = (
    OUTPUT_DIR
    / "prompt_experiment_overall_summary.csv"
)


overall_df.to_csv(
    overall_path,
    index=False
)


# ==================================================
# MANUAL REVIEW TEMPLATE
# ==================================================
manual_review_df = (
    case_summary_df[
        [
            "variant",
            "case_id",
            "scenario"
        ]
    ]
    .copy()
)


manual_review_df[
    "accuracy_score_1_5"
] = ""


manual_review_df[
    "structure_score_1_5"
] = ""


manual_review_df[
    "hallucination_risk_1_5"
] = ""


manual_review_df[
    "tone_score_1_5"
] = ""


manual_review_df[
    "usefulness_score_1_5"
] = ""


manual_review_df[
    "review_notes"
] = ""


manual_review_path = (
    OUTPUT_DIR
    / "prompt_manual_scoring.csv"
)


manual_review_df.to_csv(
    manual_review_path,
    index=False
)


# ==================================================
# FINISH
# ==================================================
print()

print(
    "Prompt experiment completed."
)

print()

print(
    "Overall results:"
)

print(
    overall_df.to_string(
        index=False
    )
)

print()

print(
    "Files created:"
)

print(
    raw_results_path
)

print(
    case_summary_path
)

print(
    overall_path
)

print(
    manual_review_path
)

print(
    OUTPUT_DIR
    / "prompt_variants.md"
)