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

# Query-level gate:
# if even the best retrieved chunk is weaker than
# this, treat the query as having no relevant evidence.
MAX_TOP_DISTANCE = 0.50

# Once a query passes the gate, only include chunks
# that meet this individual relevance threshold.
MAX_CHUNK_DISTANCE = 0.52


# ==================================================
# OUTPUT DIRECTORY
# ==================================================
OUTPUT_DIR = Path(
    "evaluation/prompt_refinement"
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
# ORIGINAL BEST PROMPT
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
# REVISED PRODUCTION CANDIDATE
# ==================================================
PROMPT_D = """
You are an AI Consumer Complaint Intelligence and
Support Assistant working with CFPB complaint
information.

Your task is to interpret consumer financial
complaints and answer CFPB complaint-process
questions using only the supplied evidence.

Follow these rules carefully:

1. Use the retrieved CFPB context as the primary
   evidence source.

2. Never invent CFPB policies, procedures,
   deadlines, legal requirements, statistics,
   company responses, jurisdiction or complaint
   outcomes.

3. Separate information supplied by the user from
   information supported by the CFPB evidence.

4. If the supplied evidence is absent, irrelevant,
   incomplete or insufficient, clearly say that the
   available CFPB evidence does not support a
   specific answer. Do not fill the gap with general
   assumptions.

5. Classify a financial product only when the user
   has described or clearly identified a specific
   financial product or service.

6. If the user asks a general question about the
   CFPB complaint process without identifying a
   specific financial product, set product to
   "Uncertain". Do not infer a product category from
   the retrieved context.

7. If the request falls outside the financial
   products represented by the CFPB complaint
   categories, set product to "Uncertain".

8. Keep the summary factual, concise and limited to
   the user's issue and the supplied evidence.

9. The recommended action must be supported by the
   supplied evidence. If no relevant CFPB evidence
   is available, do not invent another regulator,
   agency, legal remedy or procedure. State that the
   current knowledge base cannot support a specific
   recommendation and recommend appropriate human
   review.

10. Set evidence_available to true only when the
    supplied CFPB context directly supports the
    response.

11. Set human_review_required to true whenever
    relevant evidence is unavailable or the request
    cannot be handled confidently from the supplied
    CFPB information.

12. Do not provide legal advice or guarantee an
    outcome.

13. Avoid unnecessary verbosity.

Return:

- product
- summary
- recommended action
- evidence availability
- human review requirement
"""


PROMPTS = {
    "C_Original_Evidence_Aware": PROMPT_C,
    "D_Revised_Production": PROMPT_D
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
# LOAD MODELS / DATABASE
# ==================================================
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Connecting to CFPB knowledge base...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ==================================================
# IMPROVED RETRIEVAL
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

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not distances:
        return "", [], None

    best_distance = distances[0]

    # Query-level rejection
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
            f"Chunk {chunk} | "
            f"Distance {distance:.4f}"
        )

    if not context_parts:

        return "", [], best_distance

    return (
        "\n\n".join(context_parts),
        sources,
        best_distance
    )


# ==================================================
# RUN MODEL
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
            "No sufficiently relevant CFPB evidence "
            "was retrieved for this request."
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
def concept_coverage(
    text,
    expected_concepts
):

    if not expected_concepts:
        return None

    text = text.lower()

    matches = sum(
        1
        for concept in expected_concepts
        if concept.lower() in text
    )

    return (
        matches
        / len(expected_concepts)
    )


def response_similarity(texts):

    if len(texts) < 2:
        return None

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
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
print("Starting refinement experiment...")
print()


for test_case in TEST_CASES:

    question = test_case["question"]

    context, sources, best_distance = (
        search_knowledge_base(
            question
        )
    )

    actual_evidence = bool(
        context
    )

    print(
        f"{test_case['case_id']} | "
        f"best distance = "
        f"{best_distance:.4f} | "
        f"evidence = {actual_evidence}"
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

            # Application-level evidence status
            # overrides model self-report.
            assessment.evidence_available = (
                actual_evidence
            )

            if not actual_evidence:

                assessment.human_review_required = (
                    True
                )

            combined_output = (
                f"{assessment.product} "
                f"{assessment.summary} "
                f"{assessment.recommended_action}"
            )

            results.append(
                {
                    "prompt":
                        prompt_name,

                    "case_id":
                        test_case["case_id"],

                    "scenario":
                        test_case["scenario"],

                    "run":
                        run_number,

                    "best_distance":
                        best_distance,

                    "retrieval_accepted":
                        actual_evidence,

                    "predicted_product":
                        assessment.product,

                    "expected_product":
                        test_case[
                            "expected_product"
                        ],

                    "product_correct":
                        assessment.product
                        == test_case[
                            "expected_product"
                        ],

                    "evidence_available":
                        assessment.evidence_available,

                    "expected_evidence":
                        test_case[
                            "expected_evidence"
                        ],

                    "evidence_correct":
                        assessment.evidence_available
                        == test_case[
                            "expected_evidence"
                        ],

                    "human_review_required":
                        assessment.human_review_required,

                    "expected_human_review":
                        test_case[
                            "expected_human_review"
                        ],

                    "human_review_correct":
                        assessment.human_review_required
                        == test_case[
                            "expected_human_review"
                        ],

                    "concept_coverage":
                        concept_coverage(
                            combined_output,
                            test_case[
                                "expected_concepts"
                            ]
                        ),

                    "word_count":
                        len(
                            combined_output.split()
                        ),

                    "summary":
                        assessment.summary,

                    "recommended_action":
                        assessment.recommended_action,

                    "sources":
                        " || ".join(
                            sources
                        ),

                    "combined_output":
                        combined_output
                }
            )


# ==================================================
# SAVE RAW RESULTS
# ==================================================
results_df = pd.DataFrame(
    results
)

results_path = (
    OUTPUT_DIR
    / "refinement_results.csv"
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
                response_similarity(
                    texts
                ),

            "average_word_count":
                group[
                    "word_count"
                ].mean(),

            "best_distance":
                group[
                    "best_distance"
                ].iloc[0],

            "retrieval_accepted":
                group[
                    "retrieval_accepted"
                ].iloc[0]
        }
    )


case_df = pd.DataFrame(
    case_rows
)

case_path = (
    OUTPUT_DIR
    / "refinement_case_summary.csv"
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
    / "refinement_overall_summary.csv"
)

overall_df.to_csv(
    overall_path,
    index=False
)


# ==================================================
# FINISH
# ==================================================
print()
print("Refinement experiment completed.")
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