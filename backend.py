import re

from functools import lru_cache

from pathlib import Path

from typing import Literal


import chromadb

import pandas as pd


from ollama import chat

from pydantic import BaseModel

from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

PROCESSED_DIR = DATA_DIR / "processed"

CHROMA_PATH = DATA_DIR / "chroma_db"

MODEL_NAME = "llama3.2"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTION_NAME = "customer_service_knowledge"

NUMBER_OF_RESULTS = 4

MAX_TOP_DISTANCE = 0.50

MAX_CHUNK_DISTANCE = 0.52


# ============================================================
# DATA PATHS
# ============================================================

SUMMARY_PATH = (
    PROCESSED_DIR
    / "summary_metrics.csv"
)

PRODUCT_PATH = (
    PROCESSED_DIR
    / "complaints_by_product.csv"
)

ISSUE_PATH = (
    PROCESSED_DIR
    / "complaints_by_issue.csv"
)

COMPANY_PATH = (
    PROCESSED_DIR
    / "complaints_by_company.csv"
)

RESPONSE_PATH = (
    PROCESSED_DIR
    / "complaints_by_response.csv"
)

CHANNEL_PATH = (
    PROCESSED_DIR
    / "complaints_by_channel.csv"
)

STATE_PATH = (
    PROCESSED_DIR
    / "complaints_by_state.csv"
)

MONTH_PATH = (
    PROCESSED_DIR
    / "complaints_by_month.csv"
)

TIMELY_PRODUCT_PATH = (
    PROCESSED_DIR
    / "timely_response_by_product.csv"
)


# ============================================================
# STRUCTURED OUTPUT MODELS
# ============================================================

class RouteDecision(BaseModel):

    route: Literal[
        "complaint_support",
        "data_analysis",
    ]

    reason: str


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
        "Uncertain",
    ]

    summary: str

    recommended_action: str

    evidence_available: bool

    human_review_required: bool


# ============================================================
# ROUTER PROMPT
# ============================================================

ROUTER_PROMPT = """
You are the request router for an AI consumer
complaint intelligence application.

Choose complaint_support when the user:

- describes a financial consumer complaint
- asks what they should do about a complaint
- asks about the CFPB complaint process
- asks about company response deadlines
- asks about complaint handling procedures
- asks for guidance based on CFPB documents

Choose data_analysis when the user asks for numerical
or statistical analysis of the 2025 CFPB Consumer
Complaint Database, including:

- complaint counts
- percentages
- complaint rates
- products
- issues
- companies
- company response statistics
- submission channels
- states
- monthly trends
- comparisons

Do not answer the question.

Only decide which application capability
should handle the request.
"""


# ============================================================
# SUPPORT PROMPT
# ============================================================

SUPPORT_PROMPT = """
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


# ============================================================
# GROUNDING CONSTRAINT
# ============================================================

GROUNDING_CONSTRAINT = """
Important grounding rules:

Do not turn a neutral consumer description into a
legal conclusion.

Do not describe conduct as harassment, abusive,
illegal, unlawful, fraud, fraudulent, a legal
violation or criminal conduct unless the consumer
explicitly described it that way.

Do not introduce another regulator, lawyer, legal
service, government agency or outside organisation
unless the retrieved evidence directly supports it.

Do not promise or imply that:

- an account will be unfrozen
- money will be recovered
- a debt will be cancelled
- information will be removed
- an account will be restored
- the consumer will win the complaint

If the CFPB evidence only describes how similar
complaints were handled, do not present a
guaranteed remedy.
"""


# ============================================================
# ANALYTICS EXPLANATION PROMPT
# ============================================================

ANALYSIS_PROMPT = """
You are a consumer-complaint data analyst.

You will receive:

1. A user question.
2. A result already calculated using deterministic
   Python/Pandas code from the processed 2025 CFPB
   Consumer Complaint Database.

Explain the calculated result clearly and concisely.

Do not recalculate the result.

Do not alter the supplied values.

Do not invent additional statistics.

Do not imply that every complaint received by the
CFPB is necessarily represented in the public
Consumer Complaint Database.

Base your explanation only on the calculated result
provided to you.
"""


# ============================================================
# RESOURCE LOADING
# ============================================================

@lru_cache(maxsize=1)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


@lru_cache(maxsize=1)
def load_collection():

    client = (
        chromadb.PersistentClient(
            path=str(
                CHROMA_PATH
            )
        )
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


@lru_cache(maxsize=1)
def load_cfpb_data():

    return {

        "summary":
            pd.read_csv(
                SUMMARY_PATH
            ),

        "products":
            pd.read_csv(
                PRODUCT_PATH
            ),

        "issues":
            pd.read_csv(
                ISSUE_PATH
            ),

        "companies":
            pd.read_csv(
                COMPANY_PATH
            ),

        "responses":
            pd.read_csv(
                RESPONSE_PATH
            ),

        "channels":
            pd.read_csv(
                CHANNEL_PATH
            ),

        "states":
            pd.read_csv(
                STATE_PATH
            ),

        "months":
            pd.read_csv(
                MONTH_PATH
            ),

        "timely_by_product":
            pd.read_csv(
                TIMELY_PRODUCT_PATH
            ),
    }


# ============================================================
# TEXT MATCHING
# ============================================================

def contains_phrase(
    text,
    phrase,
):

    pattern = (
        r"(?<!\w)"
        + re.escape(
            phrase.lower()
        ).replace(
            r"\ ",
            r"\s+"
        )
        + r"(?!\w)"
    )

    return bool(
        re.search(
            pattern,
            text.lower(),
        )
    )


# ============================================================
# PRODUCT ALIASES
# ============================================================

PRODUCT_ALIASES = {

    "Credit reporting or other personal consumer reports": [
        "credit report",
        "credit reporting",
        "consumer report",
        "credit bureau",
        "credit score",
    ],

    "Debt collection": [
        "debt collector",
        "debt collection",
        "collection agency",
    ],

    "Credit card": [
        "credit card",
    ],

    "Money transfer, virtual currency, or money service": [
        "money transfer",
        "virtual currency",
        "cryptocurrency",
        "crypto",
        "money service",
    ],

    "Checking or savings account": [
        "checking account",
        "savings account",
        "bank account",
    ],

    "Mortgage": [
        "mortgage",
    ],

    "Student loan": [
        "student loan",
    ],

    "Vehicle loan or lease": [
        "vehicle loan",
        "car loan",
        "auto loan",
        "vehicle lease",
        "car lease",
    ],

    "Payday loan, title loan, personal loan, or advance loan": [
        "payday loan",
        "title loan",
        "personal loan",
        "advance loan",
    ],

    "Prepaid card": [
        "prepaid card",
    ],

    "Debt or credit management": [
        "debt management",
        "credit management",
        "credit counseling",
        "credit counselling",
    ],
}


# ============================================================
# PRODUCT DETECTION
# ============================================================

def find_products_in_question(
    question,
):

    matches = []

    for (
        product,
        aliases,
    ) in PRODUCT_ALIASES.items():

        if any(
            contains_phrase(
                question,
                alias,
            )
            for alias in aliases
        ):

            matches.append(
                product
            )

    return matches


def find_product_in_question(
    question,
):

    matches = (
        find_products_in_question(
            question
        )
    )

    if len(matches) == 1:

        return matches[0]

    return None


# ============================================================
# MULTI-TURN CONVERSATION
# ============================================================

def is_contextual_followup(
    question,
):

    q = (
        question
        .lower()
        .strip()
        .rstrip("?.!")
    )

    followup_phrases = {
        "what should i do next",
        "what do i do next",
        "what can i do next",
        "what should i do",
        "what can i do",
        "what now",
        "what are my next steps",
        "what is the next step",
        "what's the next step",
        "how do i proceed",
        "tell me more",
        "can you explain further",
        "what does that mean",
        "and then",
    }

    return (
        q
        in followup_phrases
    )


def get_previous_user_message(
    messages,
    current_question,
):

    if not messages:

        return None

    user_messages = [

        str(
            message.get(
                "content",
                "",
            )
        ).strip()

        for message in messages

        if (
            message.get(
                "role"
            )
            == "user"

            and str(
                message.get(
                    "content",
                    "",
                )
            ).strip()
        )
    ]

    if not user_messages:

        return None

    current_clean = (
        current_question.strip()
    )

    if (
        user_messages[-1]
        == current_clean
    ):

        if (
            len(
                user_messages
            )
            >= 2
        ):

            return (
                user_messages[-2]
            )

        return None

    return user_messages[-1]


def build_contextual_question(
    question,
    conversation_history=None,
):

    if conversation_history is None:

        conversation_history = []

    if not is_contextual_followup(
        question
    ):

        return question

    previous_user_message = (
        get_previous_user_message(
            conversation_history,
            question,
        )
    )

    if not previous_user_message:

        return question

    return (
        f"Previous user message: "
        f"{previous_user_message}\n"
        f"Current follow-up: "
        f"{question}"
    )


# ============================================================
# ANALYTICS INTENT
# ============================================================

def has_analysis_intent(
    question,
):

    terms = [
        "percentage",
        "percent",
        "rate",
        "how many",
        "number of",
        "total",
        "count",
        "average",
        "mean",
        "median",
        "trend",
        "monthly",
        "most common",
        "highest",
        "lowest",
        "top",
        "breakdown",
        "distribution",
        "statistics",
        "compare",
        "comparison",
    ]

    return any(
        contains_phrase(
            question,
            term,
        )
        for term in terms
    )


def is_clear_data_analysis_question(
    question,
):

    statistical_terms = [
        "percentage",
        "percent",
        "how many",
        "number of",
        "total",
        "count",
        "average",
        "mean",
        "median",
        "trend",
        "monthly",
        "most",
        "most common",
        "highest",
        "lowest",
        "top",
        "breakdown",
        "distribution",
        "statistics",
        "compare",
        "comparison",
        "response rate",
        "timely response rate",
    ]

    scope_terms = [
        "cfpb",
        "dataset",
        "data",
        "complaint",
        "complaints",
        "financial product",
        "product",
        "products",
        "issue",
        "issues",
        "company",
        "companies",
        "response",
        "responses",
        "submission channel",
        "channel",
        "channels",
        "state",
        "states",
        "month",
        "months",
        "2025",
    ]

    has_statistical_term = any(
        contains_phrase(
            question,
            term,
        )
        for term in statistical_terms
    )

    has_dataset_scope = any(
        contains_phrase(
            question,
            term,
        )
        for term in scope_terms
    )

    return (
        has_statistical_term
        and has_dataset_scope
    )


# ============================================================
# PROCESS / DEADLINE DETECTION
# ============================================================

def is_response_deadline_question(
    question,
):

    deadline_terms = [
        "how long does a company",
        "how long do companies",
        "how long to respond",
        "response deadline",
        "response time",
        "when does a company respond",
        "how many days does a company",
        "how many days do companies",
    ]

    return any(
        contains_phrase(
            question,
            term,
        )
        for term in deadline_terms
    )


def is_general_process_question(
    question,
):

    if is_response_deadline_question(
        question
    ):

        return True

    process_terms = [
        "cfpb complaint process",
        "complaint process",
        "complaint handling",
        "handling complaints",
        "complaint procedure",
        "complaint procedures",
        "what happens after i submit",
        "what happens after submitting",
        "after a complaint is submitted",
        "after submitting a complaint",
    ]

    return any(
        contains_phrase(
            question,
            term,
        )
        for term in process_terms
    )


# ============================================================
# CLEAR SUPPORT REQUEST DETECTION
# ============================================================

def is_clear_complaint_support_question(
    question,
):

    if find_products_in_question(
        question
    ):

        return True

    support_terms = [
        "what should i do",
        "what can i do",
        "what do i do",
        "what should i do next",
        "what can i do next",
        "what kind of complaint",
        "classify",
        "classified",
        "file a complaint",
        "submit a complaint",
        "cfpb help",
        "help me",
        "recover it",
        "refund me",
        "dispute",
        "debt collector",
        "frozen",
        "froze",
        "cannot access",
        "can't access",
        "do not recognize",
        "don't recognize",
    ]

    return any(
        contains_phrase(
            question,
            term,
        )
        for term in support_terms
    )


def is_short_or_unstructured_input(
    question,
):

    words = re.findall(
        r"[A-Za-z]+",
        question,
    )

    return (
        len(words)
        <= 4
    )


# ============================================================
# DETERMINISTIC ROUTING
# ============================================================

def deterministic_route_override(
    user_input,
):

    if is_general_process_question(
        user_input
    ):

        return RouteDecision(
            route="complaint_support",

            reason=(
                "Deterministic routing safeguard: "
                "CFPB complaint-process or "
                "response-deadline request."
            ),
        )

    if is_clear_data_analysis_question(
        user_input
    ):

        return RouteDecision(
            route="data_analysis",

            reason=(
                "Deterministic routing safeguard: "
                "clear statistical or aggregate "
                "CFPB data request."
            ),
        )

    if is_clear_complaint_support_question(
        user_input
    ):

        return RouteDecision(
            route="complaint_support",

            reason=(
                "Deterministic routing safeguard: "
                "clear consumer complaint or "
                "guidance request."
            ),
        )

    if is_short_or_unstructured_input(
        user_input
    ):

        return RouteDecision(
            route="complaint_support",

            reason=(
                "Deterministic safe fallback for "
                "short or insufficiently structured "
                "input."
            ),
        )

    return None


# ============================================================
# REQUEST ROUTER
# ============================================================

def route_request(
    user_input,
    conversation_history=None,
):

    if conversation_history is None:

        conversation_history = []

    routing_input = (
        build_contextual_question(
            user_input,
            conversation_history,
        )
    )

    override = (
        deterministic_route_override(
            routing_input
        )
    )

    if override is not None:

        return override

    response = chat(
        model=MODEL_NAME,

        messages=[
            {
                "role":
                    "system",

                "content":
                    ROUTER_PROMPT,
            },

            {
                "role":
                    "user",

                "content":
                    routing_input,
            },
        ],

        format=(
            RouteDecision
            .model_json_schema()
        ),
    )

    return (
        RouteDecision
        .model_validate_json(
            response.message.content
        )
    )


# ============================================================
# RETRIEVAL QUERY
# ============================================================

def build_retrieval_query(
    question,
):

    products = (
        find_products_in_question(
            question
        )
    )

    if len(products) == 1:

        return (
            f"Official CFPB guidance and complaint "
            f"information for the financial product "
            f"category '{products[0]}'. "
            f"Consumer issue: {question}"
        )

    if is_general_process_question(
        question
    ):

        return (
            "Official CFPB complaint process, "
            "company response deadlines, response "
            "timeframes, complaint handling and "
            f"company response requirements. "
            f"Question: {question}"
        )

    return question


# ============================================================
# KNOWLEDGE RETRIEVAL
# ============================================================

def search_knowledge_base(
    query,
):

    embedding_model = (
        load_embedding_model()
    )

    collection = (
        load_collection()
    )

    retrieval_query = (
        build_retrieval_query(
            query
        )
    )

    embedding = (
        embedding_model
        .encode(
            retrieval_query
        )
        .tolist()
    )

    results = collection.query(

        query_embeddings=[
            embedding
        ],

        n_results=
            NUMBER_OF_RESULTS,

        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        results[
            "documents"
        ][0]
    )

    metadatas = (
        results[
            "metadatas"
        ][0]
    )

    distances = (
        results[
            "distances"
        ][0]
    )

    if not distances:

        return (
            "",
            [],
            None,
            retrieval_query,
        )

    best_distance = (
        distances[0]
    )

    if (
        best_distance
        > MAX_TOP_DISTANCE
    ):

        return (
            "",
            [],
            best_distance,
            retrieval_query,
        )

    context_parts = []

    sources = []

    for (
        document,
        metadata,
        distance,
    ) in zip(
        documents,
        metadatas,
        distances,
    ):

        if (
            distance
            > MAX_CHUNK_DISTANCE
        ):

            continue

        similarity = (
            1 - distance
        )

        source = metadata.get(
            "source",
            "Unknown source",
        )

        file_type = metadata.get(
            "file_type",
            "unknown",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        chunk = metadata.get(
            "chunk",
            "Unknown",
        )

        if file_type == "pdf":

            source_label = (
                f"{source} - "
                f"Page {page}, "
                f"Chunk {chunk}"
            )

        else:

            source_label = (
                f"{source} - "
                f"Chunk {chunk}"
            )

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk}\n"
            f"Similarity: {similarity:.3f}\n"
            f"{document}"
        )

        sources.append(
            source_label
        )

    if not context_parts:

        return (
            "",
            [],
            best_distance,
            retrieval_query,
        )

    return (
        "\n\n".join(
            context_parts
        ),

        sources,

        best_distance,

        retrieval_query,
    )


# ============================================================
# DETERMINISTIC DEADLINE RESPONSE
# ============================================================

def build_response_deadline_assessment():

    return ComplaintAssessment(

        product=
            "Uncertain",

        summary=(
            "Companies generally provide a response "
            "within 15 calendar days after receiving "
            "a CFPB complaint. If a complaint cannot "
            "be closed within 15 calendar days, the "
            "company may provide an interim explanation "
            "and then provide a final response within "
            "60 calendar days."
        ),

        recommended_action=(
            "Use the CFPB complaint process to monitor "
            "the company's response. If the complaint "
            "is not closed within the initial 15-day "
            "period, the company may provide an interim "
            "response while a final response remains "
            "pending."
        ),

        evidence_available=
            True,

        human_review_required=
            False,
    )


# ============================================================
# NO-EVIDENCE RESPONSE
# ============================================================

def build_no_evidence_assessment(
    question,
):

    product = (
        find_product_in_question(
            question
        )
    )

    if product:

        summary = (
            f"The request appears to concern "
            f"'{product}', but the current CFPB "
            f"knowledge base did not return "
            f"sufficiently relevant evidence."
        )

    else:

        product = (
            "Uncertain"
        )

        summary = (
            "The current CFPB knowledge base did "
            "not return sufficiently relevant "
            "evidence to support a specific answer."
        )

    return ComplaintAssessment(

        product=
            product,

        summary=
            summary,

        recommended_action=(
            "A specific recommendation cannot be "
            "provided from the available CFPB "
            "evidence. Human review is recommended."
        ),

        evidence_available=
            False,

        human_review_required=
            True,
    )


# ============================================================
# SAFETY CHECKS
# ============================================================

EXTERNAL_REFERENCE_TERMS = [
    "local attorney",
    "consumer attorney",
    "state attorney general",
    "attorney general",
    "federal trade commission",
    "national consumer law center",
    "local consumer protection agency",
    "legal aid",
    "legal counsel",
    "lawyer",
    "attorney",
    "ftc",
    "nclc",
]


UNSAFE_RECOMMENDATION_PHRASES = [
    "will remove",
    "will delete",
    "will refund",
    "will recover",
    "will cancel",
    "will unfreeze",
    "will restore",
    "guaranteed",
    "guarantee that",
    "request unfreezing",
    "request the account be unfrozen",
    "request that the account be unfrozen",
    "unfreezing of the account",
    "restore access to the account",
    "unfreeze",
    "unfrozen",
]


ESCALATION_TERMS = [
    "harassment",
    "harassing",
    "abusive",
    "illegal",
    "unlawful",
    "violation",
    "violating",
    "fraud",
    "fraudulent",
    "criminal",
    "scam",
]


def contains_unsupported_external_reference(
    recommendation,
    context,
):

    recommendation_lower = (
        recommendation.lower()
    )

    context_lower = (
        context.lower()
    )

    return any(
        term in recommendation_lower
        and term not in context_lower

        for term
        in EXTERNAL_REFERENCE_TERMS
    )


def contains_unsupported_outcome(
    recommendation,
):

    recommendation_lower = (
        recommendation.lower()
    )

    return any(
        phrase in recommendation_lower

        for phrase
        in UNSAFE_RECOMMENDATION_PHRASES
    )


def contains_unstated_escalation(
    summary,
    question,
):

    summary_lower = (
        summary.lower()
    )

    question_lower = (
        question.lower()
    )

    return any(
        term in summary_lower
        and term not in question_lower

        for term
        in ESCALATION_TERMS
    )


# ============================================================
# SAFE SUMMARIES
# ============================================================

def build_safe_summary(
    question,
    product,
):

    q = (
        question.lower()
    )

    if (
        product
        == "Debt collection"

        and "debt" in q
    ):

        return (
            "You report that a debt collector is "
            "contacting you about a debt you do "
            "not recognize."
        )

    if (
        product
        == "Credit reporting or other personal consumer reports"
    ):

        return (
            "You report an issue involving "
            "information appearing on your "
            "credit report."
        )

    if (
        product
        == "Checking or savings account"
    ):

        return (
            "You report an issue affecting access "
            "to your checking or savings account."
        )

    if (
        product
        != "Uncertain"
    ):

        return (
            f"The request concerns "
            f"'{product}'."
        )

    return (
        "The request does not contain enough "
        "information for a confident product "
        "classification."
    )


# ============================================================
# GROUNDED FALLBACK ACTION
# ============================================================

def build_grounded_fallback_action(
    question,
):

    product = (
        find_product_in_question(
            question
        )
    )

    if is_general_process_question(
        question
    ):

        return (
            "Use the retrieved CFPB complaint-process "
            "guidance to understand the applicable "
            "company response timeframe and complaint "
            "handling process."
        )

    if (
        product
        == "Debt collection"
    ):

        return (
            "Use the retrieved CFPB debt-collection "
            "guidance to review or request validation "
            "information about the debt and keep "
            "records of communications."
        )

    if (
        product
        == "Credit reporting or other personal consumer reports"
    ):

        return (
            "Use the retrieved CFPB guidance to "
            "dispute inaccurate information with "
            "the relevant credit reporting company "
            "and, where applicable, the company that "
            "supplied the information."
        )

    if (
        product
        == "Checking or savings account"
    ):

        return (
            "Contact the bank to understand why "
            "the account is restricted and what "
            "steps are available to address the issue."
        )

    return (
        "Use the retrieved CFPB evidence as "
        "informational guidance. The available "
        "evidence does not support a guaranteed "
        "outcome."
    )


# ============================================================
# APPLY DETERMINISTIC SAFEGUARDS
# ============================================================

def apply_deterministic_safeguards(
    assessment,
    question,
    context,
):

    assessment.evidence_available = (
        True
    )

    products = (
        find_products_in_question(
            question
        )
    )

    if len(products) == 1:

        assessment.product = (
            products[0]
        )

    elif len(products) > 1:

        assessment.product = (
            "Uncertain"
        )

    elif is_general_process_question(
        question
    ):

        assessment.product = (
            "Uncertain"
        )


    if len(products) > 1:

        assessment.human_review_required = (
            True
        )

    elif (
        assessment.product
        == "Uncertain"

        and not is_general_process_question(
            question
        )
    ):

        assessment.human_review_required = (
            True
        )

    else:

        assessment.human_review_required = (
            False
        )


    # --------------------------------------------------------
    # UNRECOGNIZED DEBT RELIABILITY SAFEGUARD
    # --------------------------------------------------------

    # The LLM sometimes produced a generic recommendation
    # to file a complaint instead of consistently mentioning
    # debt validation. For an unrecognized debt, use the
    # deterministic grounded debt-collection action.

    if (
        assessment.product
        == "Debt collection"

        and (
            contains_phrase(
                question,
                "do not recognize",
            )

            or contains_phrase(
                question,
                "don't recognize",
            )

            or contains_phrase(
                question,
                "not my debt",
            )

            or contains_phrase(
                question,
                "not mine",
            )
        )
    ):

        assessment.recommended_action = (
            build_grounded_fallback_action(
                question
            )
        )


    if contains_unstated_escalation(
        assessment.summary,
        question,
    ):

        assessment.summary = (
            build_safe_summary(
                question,
                assessment.product,
            )
        )


    if (
        contains_unsupported_external_reference(
            assessment.recommended_action,
            context,
        )

        or contains_unsupported_outcome(
            assessment.recommended_action
        )
    ):

        assessment.recommended_action = (
            build_grounded_fallback_action(
                question
            )
        )


    return assessment


# ============================================================
# CFPB DATA ANALYSIS
# ============================================================

def analyse_cfpb_data(
    question,
    data=None,
):

    if data is None:

        data = (
            load_cfpb_data()
        )


    q = (
        question.lower()
    )


    summary = (
        data[
            "summary"
        ]
        .iloc[0]
    )


    # --------------------------------------------------------
    # TOTAL COMPLAINTS
    # --------------------------------------------------------

    if (
        (
            contains_phrase(
                q,
                "how many",
            )

            or contains_phrase(
                q,
                "total",
            )

            or contains_phrase(
                q,
                "number of",
            )
        )

        and (
            contains_phrase(
                q,
                "complaint",
            )

            or contains_phrase(
                q,
                "complaints",
            )
        )

        and not contains_phrase(
            q,
            "company",
        )

        and not contains_phrase(
            q,
            "issue",
        )

        and not contains_phrase(
            q,
            "product",
        )

        and not contains_phrase(
            q,
            "month",
        )

        and not contains_phrase(
            q,
            "monthly",
        )

        and not contains_phrase(
            q,
            "trend",
        )
    ):

        total = int(
            summary[
                "total_complaints"
            ]
        )

        return {

            "supported":
                True,

            "result":
                (
                    f"The processed 2025 CFPB "
                    f"Consumer Complaint Database "
                    f"contains {total:,} published "
                    f"complaints."
                ),

            "table":
                None,
        }


    # --------------------------------------------------------
    # TIMELY RESPONSE RATE
    # --------------------------------------------------------

    if (
        contains_phrase(
            q,
            "timely",
        )

        and contains_phrase(
            q,
            "response",
        )

        and (
            contains_phrase(
                q,
                "percentage",
            )

            or contains_phrase(
                q,
                "percent",
            )

            or contains_phrase(
                q,
                "rate",
            )
        )

        and not contains_phrase(
            q,
            "product",
        )
    ):

        rate = float(
            summary[
                "timely_response_rate"
            ]
        )

        count = int(
            summary[
                "timely_responses"
            ]
        )

        known = int(
            summary[
                "known_timely_responses"
            ]
        )

        return {

            "supported":
                True,

            "result":
                (
                    f"{count:,} of {known:,} "
                    f"published 2025 complaints "
                    f"with known response status "
                    f"were marked as receiving a "
                    f"timely response. This equals "
                    f"{rate:.2f}%."
                ),

            "table":
                None,
        }


    # --------------------------------------------------------
    # MOST COMMON PRODUCT
    # --------------------------------------------------------

    if (
        (
            contains_phrase(
                q,
                "most",
            )

            or contains_phrase(
                q,
                "highest",
            )

            or contains_phrase(
                q,
                "top",
            )
        )

        and (
            contains_phrase(
                q,
                "product",
            )

            or contains_phrase(
                q,
                "financial product",
            )
        )
    ):

        table = (
            data[
                "products"
            ]
            .head(10)
            .copy()
        )

        top = (
            table.iloc[0]
        )

        return {

            "supported":
                True,

            "result":
                (
                    f"The most common financial "
                    f"product in the processed "
                    f"2025 complaint data is "
                    f"'{top['product']}', with "
                    f"{int(top['complaint_count']):,} "
                    f"complaints "
                    f"({top['percentage_of_complaints']:.2f}%)."
                ),

            "table":
                table,
        }


    # --------------------------------------------------------
    # PRODUCT BREAKDOWN
    # --------------------------------------------------------

    if (
        contains_phrase(
            q,
            "product",
        )

        and (
            contains_phrase(
                q,
                "breakdown",
            )

            or contains_phrase(
                q,
                "distribution",
            )

            or contains_phrase(
                q,
                "all products",
            )
        )
    ):

        return {

            "supported":
                True,

            "result":
                (
                    "The table below shows complaint "
                    "volume and percentage by CFPB "
                    "financial product."
                ),

            "table":
                data[
                    "products"
                ].copy(),
        }


    detected_product = (
        find_product_in_question(
            question
        )
    )


    # --------------------------------------------------------
    # SPECIFIC PRODUCT COUNT
    # --------------------------------------------------------

    if (
        detected_product

        and (
            contains_phrase(
                q,
                "how many",
            )

            or contains_phrase(
                q,
                "number",
            )

            or contains_phrase(
                q,
                "count",
            )

            or contains_phrase(
                q,
                "complaints",
            )
        )
    ):

        matches = (
            data[
                "products"
            ][
                data[
                    "products"
                ][
                    "product"
                ]
                == detected_product
            ]
        )

        if not matches.empty:

            row = (
                matches.iloc[0]
            )

            return {

                "supported":
                    True,

                "result":
                    (
                        f"'{detected_product}' "
                        f"accounts for "
                        f"{int(row['complaint_count']):,} "
                        f"published complaints in "
                        f"the processed 2025 data, "
                        f"representing "
                        f"{row['percentage_of_complaints']:.2f}% "
                        f"of complaints."
                    ),

                "table":
                    matches,
            }


    # --------------------------------------------------------
    # TOP ISSUES
    # --------------------------------------------------------

    if (
        (
            contains_phrase(
                q,
                "issue",
            )

            or contains_phrase(
                q,
                "problem",
            )
        )

        and (
            contains_phrase(
                q,
                "most common",
            )

            or contains_phrase(
                q,
                "top",
            )

            or contains_phrase(
                q,
                "highest",
            )
        )
    ):

        table = (
            data[
                "issues"
            ]
            .head(10)
            .copy()
        )

        top = (
            table.iloc[0]
        )

        return {

            "supported":
                True,

            "result":
                (
                    f"The most common issue in "
                    f"the processed 2025 data is "
                    f"'{top['issue']}', with "
                    f"{int(top['complaint_count']):,} "
                    f"complaints."
                ),

            "table":
                table,
        }


    # --------------------------------------------------------
    # TOP COMPANIES
    # --------------------------------------------------------

    if (
        (
            contains_phrase(
                q,
                "company",
            )

            or contains_phrase(
                q,
                "companies",
            )
        )

        and (
            contains_phrase(
                q,
                "most complaints",
            )

            or contains_phrase(
                q,
                "top",
            )

            or contains_phrase(
                q,
                "highest",
            )
        )
    ):

        table = (
            data[
                "companies"
            ]
            .head(10)
            .copy()
        )

        top = (
            table.iloc[0]
        )

        return {

            "supported":
                True,

            "result":
                (
                    f"The company with the highest "
                    f"number of published complaints "
                    f"in the processed 2025 data is "
                    f"'{top['company']}', with "
                    f"{int(top['complaint_count']):,} "
                    f"complaints."
                ),

            "table":
                table,
        }


    # --------------------------------------------------------
    # RESPONSE BREAKDOWN
    # --------------------------------------------------------

    if (
        contains_phrase(
            q,
            "response",
        )

        and (
            contains_phrase(
                q,
                "category",
            )

            or contains_phrase(
                q,
                "breakdown",
            )

            or contains_phrase(
                q,
                "type",
            )
        )

        and not contains_phrase(
            q,
            "timely",
        )
    ):

        return {

            "supported":
                True,

            "result":
                (
                    "The table below shows how "
                    "companies responded to published "
                    "2025 complaints."
                ),

            "table":
                data[
                    "responses"
                ].copy(),
        }


    # --------------------------------------------------------
    # SUBMISSION CHANNEL
    # --------------------------------------------------------

    if (
        (
            contains_phrase(
                q,
                "channel",
            )

            or contains_phrase(
                q,
                "submitted via",
            )

            or contains_phrase(
                q,
                "submission",
            )
        )

        and (
            contains_phrase(
                q,
                "most",
            )

            or contains_phrase(
                q,
                "top",
            )

            or contains_phrase(
                q,
                "breakdown",
            )

            or contains_phrase(
                q,
                "common",
            )
        )
    ):

        table = (
            data[
                "channels"
            ]
            .copy()
        )

        top = (
            table.iloc[0]
        )

        return {

            "supported":
                True,

            "result":
                (
                    f"The most common complaint "
                    f"submission channel is "
                    f"'{top['submitted_via']}', "
                    f"with "
                    f"{int(top['complaint_count']):,} "
                    f"complaints."
                ),

            "table":
                table,
        }


    # --------------------------------------------------------
    # STATE COUNTS
    # --------------------------------------------------------

    if (
        contains_phrase(
            q,
            "state",
        )

        and (
            contains_phrase(
                q,
                "most",
            )

            or contains_phrase(
                q,
                "highest",
            )

            or contains_phrase(
                q,
                "top",
            )
        )
    ):

        table = (
            data[
                "states"
            ]
            .head(10)
            .copy()
        )

        top = (
            table.iloc[0]
        )

        return {

            "supported":
                True,

            "result":
                (
                    f"The state with the highest "
                    f"raw number of published "
                    f"complaints in the processed "
                    f"2025 data is "
                    f"'{top['state']}', with "
                    f"{int(top['complaint_count']):,} "
                    f"complaints."
                ),

            "table":
                table,
        }


    # --------------------------------------------------------
    # MONTHLY TREND
    # --------------------------------------------------------

    if (
        contains_phrase(
            q,
            "month",
        )

        or contains_phrase(
            q,
            "monthly",
        )

        or contains_phrase(
            q,
            "trend",
        )
    ):

        table = (
            data[
                "months"
            ]
            .copy()
        )

        highest = (
            table
            .sort_values(
                "complaint_count",
                ascending=False,
            )
            .iloc[0]
        )

        return {

            "supported":
                True,

            "result":
                (
                    f"The month with the highest "
                    f"number of published complaints "
                    f"in the processed 2025 data was "
                    f"{highest['month']}, with "
                    f"{int(highest['complaint_count']):,} "
                    f"complaints."
                ),

            "table":
                table,
        }


    # --------------------------------------------------------
    # TIMELY RESPONSE BY PRODUCT
    # --------------------------------------------------------

    if (
        detected_product

        and contains_phrase(
            q,
            "timely",
        )

        and contains_phrase(
            q,
            "response",
        )
    ):

        table = (
            data[
                "timely_by_product"
            ]
        )

        match = (
            table[
                table[
                    "product"
                ]
                == detected_product
            ]
        )

        if not match.empty:

            row = (
                match.iloc[0]
            )

            return {

                "supported":
                    True,

                "result":
                    (
                        f"For "
                        f"'{detected_product}', "
                        f"{int(row['timely_response_count']):,} "
                        f"of "
                        f"{int(row['known_response_count']):,} "
                        f"complaints with known "
                        f"response status were marked "
                        f"timely, giving a rate of "
                        f"{row['timely_response_rate']:.2f}%."
                    ),

                "table":
                    match,
            }


    # --------------------------------------------------------
    # UNSUPPORTED CALCULATION
    # --------------------------------------------------------

    return {

        "supported":
            False,

        "result":
            (
                "The current deterministic CFPB "
                "analysis tool does not yet support "
                "that calculation."
            ),

        "table":
            None,
    }


# ============================================================
# SUPPORT PIPELINE
# ============================================================

def run_support_pipeline(
    question,
    previous_messages=None,
):

    if previous_messages is None:

        previous_messages = []


    effective_question = (
        build_contextual_question(
            question,
            previous_messages,
        )
    )


    (
        context,
        sources,
        best_distance,
        retrieval_query,
    ) = search_knowledge_base(
        effective_question
    )


    if not context:

        assessment = (
            build_no_evidence_assessment(
                effective_question
            )
        )


    elif is_response_deadline_question(
        effective_question
    ):

        assessment = (
            build_response_deadline_assessment()
        )


    else:

        conversation = [

            {
                "role":
                    "system",

                "content":
                    SUPPORT_PROMPT,
            },

            {
                "role":
                    "system",

                "content":
                    GROUNDING_CONSTRAINT,
            },

            {
                "role":
                    "system",

                "content":
                    (
                        "Retrieved CFPB evidence:\n\n"
                        + context
                    ),
            },

        ] + previous_messages + [

            {
                "role":
                    "user",

                "content":
                    question,
            },

        ]


        response = chat(
            model=MODEL_NAME,

            messages=
                conversation,

            format=(
                ComplaintAssessment
                .model_json_schema()
            ),
        )


        assessment = (
            ComplaintAssessment
            .model_validate_json(
                response.message.content
            )
        )


        assessment = (
            apply_deterministic_safeguards(
                assessment,
                effective_question,
                context,
            )
        )


    return {

        "assessment":
            assessment,

        "sources":
            sources,

        "best_distance":
            best_distance,

        "retrieval_query":
            retrieval_query,

        "effective_question":
            effective_question,

        "context":
            context,
    }


# ============================================================
# FORMAT SUPPORT RESPONSE
# ============================================================

def assessment_to_text(
    assessment,
):

    return (

        f"Product: "
        f"{assessment.product}\n\n"

        f"Summary: "
        f"{assessment.summary}\n\n"

        f"Recommended Action: "
        f"{assessment.recommended_action}\n\n"

        f"Evidence Available: "
        f"{assessment.evidence_available}\n\n"

        f"Human Review Required: "
        f"{assessment.human_review_required}"
    )


# ============================================================
# ANALYTICS EXPLANATION
# ============================================================

def explain_analysis_result(
    question,
    calculated_result,
):

    response = chat(
        model=MODEL_NAME,

        messages=[

            {
                "role":
                    "system",

                "content":
                    ANALYSIS_PROMPT,
            },

            {
                "role":
                    "user",

                "content":
                    (
                        f"Question:\n"
                        f"{question}\n\n"
                        f"Calculated result:\n"
                        f"{calculated_result}"
                    ),
            },
        ],
    )


    return (
        response.message.content
    )


# ============================================================
# STREAMING ANALYTICS EXPLANATION
# ============================================================

def stream_analysis_explanation(
    question,
    calculated_result,
):

    response_stream = chat(
        model=MODEL_NAME,

        messages=[

            {
                "role":
                    "system",

                "content":
                    ANALYSIS_PROMPT,
            },

            {
                "role":
                    "user",

                "content":
                    (
                        f"Question:\n"
                        f"{question}\n\n"
                        f"Calculated result:\n"
                        f"{calculated_result}"
                    ),
            },
        ],

        stream=True,
    )


    for chunk in response_stream:

        token = (
            chunk.message.content
        )

        if token:

            yield token