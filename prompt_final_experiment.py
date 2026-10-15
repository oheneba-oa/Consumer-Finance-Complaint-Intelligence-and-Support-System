from pathlib import Path
from itertools import combinations

import pandas as pd
import chromadb

from ollama import chat
from pydantic import BaseModel
from typing import Literal
from sentence_transformers import SentenceTransformer


# ==================================================
# SETTINGS
# ==================================================
MODEL_NAME = "llama3.2"

COLLECTION_NAME = "customer_service_knowledge"

CHROMA_PATH = "data/chroma_db"

NUMBER_OF_RESULTS = 4

RUNS_PER_CASE = 3

TEMPERATURE = 0.2

MAX_TOP_DISTANCE = 0.50

MAX_CHUNK_DISTANCE = 0.52


# ==================================================
# OUTPUT DIRECTORY
# ==================================================
OUTPUT_DIR = Path(
    "evaluation/prompt_final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# STRUCTURED OUTPUT
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
# ORIGINAL VARIANT C
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
# MINIMALLY REFINED VARIANT E
# ==================================================
PROMPT_E = """
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

5. Classify a user's complaint using the CFPB
   financial product categories provided by the
   output schema.

6. Use "Uncertain" when the available information
   does not support a confident financial product
   classification.

7. IMPORTANT: If the user is asking a general
   question about the CFPB complaint process,
   response deadlines, complaint handling or CFPB
   procedures and has not identified a specific
   financial product or service, set product to
   "Uncertain".

   Do not infer a financial product merely because
   the retrieved CFPB context mentions one.

8. Keep the summary factual and concise.

9. Make the recommended action practical but do not
   present legal, financial or regulatory advice as
   a guaranteed outcome.

10. Require human review when the situation cannot
    be safely or adequately resolved from the
    available evidence.

11. Avoid unnecessary verbosity.

Return:

- product
- summary
- recommended action
- evidence availability
- human review requirement
"""


PROMPTS = {
    "C_Original": PROMPT_C,
    "E_Minimal_Refinement": PROMPT_E
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
        ],
        "expected_evidence": True,
        "expected_human_review": False
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
        ],
        "expected_evidence": True,
        "expected_human_review": False
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
        ],
        "expected_evidence": True,
        "expected_human_review": False
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
        ],
        "expected_evidence": True,
        "expected_human_review": False
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
        "expected_concepts": [],
        "expected_evidence": False,
        "expected_human_review": True
    }
]


# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# CONNECT TO CHROMA
# ==================================================
print("Connecting to CFPB knowledge base...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ==================================================
# RETRIEVAL
# ==================================================
def search_knowledge_base(query):

    embedding = (
        embedding_model
        .encode(query)
        .tolist()
    )

    results = collection.query(
        query_embeddings=[
            embedding
        ],
        n_results=NUMBER_OF_RESULTS,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not distances:

        return "", [], None

    best_distance = distances[0]

    if best_distance > MAX_TOP_DISTANCE:

        return "", [], best_distance

    context_parts = []
    sources = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        if distance > MAX_CHUNK_DISTANCE:
            continue

        similarity = 1 - distance

        source = metadata.get(
            "source",
            "Unknown"
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

    if not context_parts:

        return "", [], best_distance

    return (
        "\n\n".join(
            context_parts
        ),
        sources,
        best_distance
    )


# ==================================================
# RUN PROMPT
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
            "temperature": TEMPERATURE
        }
    )

    return (
        ComplaintAssessment
        .model_validate_json(
            response.message.content
        )
    )


# ==================================================
# HELPERS
# ==================================================
def calculate_concept_coverage(
    text,
    expected_concepts
):

    if not expected_concepts:

        return None

    lower_text = text.lower()

    matches = sum(
        1
        for concept in expected_concepts
        if concept.lower() in lower_text
    )

    return (
        matches
        / len(expected_concepts)
    )


def calculate_similarity(texts):

    if len(texts) < 2:

        return None

    embeddings = (
        embedding_model
        .encode(
            texts,
            normalize_embeddings=True
        )
    )

    scores = []

    for first, second in combinations(
        range(len(texts)),
        2
    ):

        scores.append(
            float(
                embeddings[first]
                @ embeddings[second]
            )
        )

    return (
        sum(scores)
        / len(scores)
    )


# ==================================================
# RUN EXPERIMENT
# ==================================================
results = []

print()
print("Starting final prompt experiment...")
print()


for test_case in TEST_CASES:

    question = (
        test_case["question"]
    )

    context, sources, best_distance = (
        search_knowledge_base(
            question
        )
    )

    evidence_available = bool(
        context
    )

    print(
        f"{test_case['case_id']} | "
        f"distance={best_distance:.4f} | "
        f"evidence={evidence_available}"
    )

    for prompt_name, prompt in PROMPTS.items():

        for run_number in range(
            1,
            RUNS_PER_CASE + 1
        ):

            assessment = run_prompt(
                prompt,
                question,
                context
            )

            # Deterministic application control
            # decides whether retrieval evidence
            # genuinely passed the relevance gate.
            assessment.evidence_available = (
                evidence_available
            )

            if not evidence_available:

                assessment.human_review_required = (
                    True
                )

            combined = (
                f"{assessment.product} "
                f"{assessment.summary} "
                f"{assessment.recommended_action}"
            )

            results.append(
                {
                    "prompt":
                        prompt_name,

                    "case_id":
                        test_case[
                            "case_id"
                        ],

                    "scenario":
                        test_case[
                            "scenario"
                        ],

                    "run":
                        run_number,

                    "best_distance":
                        best_distance,

                    "retrieval_accepted":
                        evidence_available,

                    "expected_product":
                        test_case[
                            "expected_product"
                        ],

                    "predicted_product":
                        assessment.product,

                    "product_correct":
                        assessment.product
                        == test_case[
                            "expected_product"
                        ],

                    "evidence_correct":
                        assessment.evidence_available
                        == test_case[
                            "expected_evidence"
                        ],

                    "human_review_correct":
                        assessment.human_review_required
                        == test_case[
                            "expected_human_review"
                        ],

                    "concept_coverage":
                        calculate_concept_coverage(
                            combined,
                            test_case[
                                "expected_concepts"
                            ]
                        ),

                    "word_count":
                        len(
                            combined.split()
                        ),

                    "summary":
                        assessment.summary,

                    "recommended_action":
                        assessment.recommended_action,

                    "combined_output":
                        combined,

                    "sources":
                        " || ".join(
                            sources
                        )
                }
            )


# ==================================================
# RAW RESULTS
# ==================================================
results_df = pd.DataFrame(
    results
)

results_path = (
    OUTPUT_DIR
    / "final_prompt_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ==================================================
# CASE SUMMARY
# ==================================================
case_rows = []


for (
    prompt_name,
    case_id
), group in results_df.groupby(
    [
        "prompt",
        "case_id"
    ]
):

    texts = (
        group[
            "combined_output"
        ]
        .tolist()
    )

    concepts = (
        group[
            "concept_coverage"
        ]
        .dropna()
    )

    case_rows.append(
        {
            "prompt":
                prompt_name,

            "case_id":
                case_id,

            "scenario":
                group[
                    "scenario"
                ].iloc[0],

            "product_accuracy":
                group[
                    "product_correct"
                ].mean(),

            "evidence_accuracy":
                group[
                    "evidence_correct"
                ].mean(),

            "human_review_accuracy":
                group[
                    "human_review_correct"
                ].mean(),

            "concept_coverage":
                (
                    concepts.mean()
                    if not concepts.empty
                    else None
                ),

            "response_similarity":
                calculate_similarity(
                    texts
                ),

            "average_word_count":
                group[
                    "word_count"
                ].mean()
        }
    )


case_df = pd.DataFrame(
    case_rows
)

case_path = (
    OUTPUT_DIR
    / "final_prompt_case_summary.csv"
)

case_df.to_csv(
    case_path,
    index=False
)


# ==================================================
# OVERALL SUMMARY
# ==================================================
overall_rows = []


for prompt_name, group in case_df.groupby(
    "prompt"
):

    overall_rows.append(
        {
            "prompt":
                prompt_name,

            "product_accuracy":
                group[
                    "product_accuracy"
                ].mean(),

            "evidence_accuracy":
                group[
                    "evidence_accuracy"
                ].mean(),

            "human_review_accuracy":
                group[
                    "human_review_accuracy"
                ].mean(),

            "concept_coverage":
                group[
                    "concept_coverage"
                ].mean(),

            "response_similarity":
                group[
                    "response_similarity"
                ].mean(),

            "average_word_count":
                group[
                    "average_word_count"
                ].mean()
        }
    )


overall_df = pd.DataFrame(
    overall_rows
)

overall_path = (
    OUTPUT_DIR
    / "final_prompt_overall_summary.csv"
)

overall_df.to_csv(
    overall_path,
    index=False
)


# ==================================================
# COMPLETE
# ==================================================
print()
print("Final prompt experiment completed.")
print()

print(
    overall_df.to_string(
        index=False
    )
)

print()
print("Files created:")
print(results_path)
print(case_path)
print(overall_path)