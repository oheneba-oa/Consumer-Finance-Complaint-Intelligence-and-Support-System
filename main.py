import re
from typing import Literal

import chromadb
import pandas as pd
import streamlit as st

from ollama import chat
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer


# ============================================================
# APPLICATION SETTINGS
# ============================================================

MODEL_NAME = "llama3.2"

COLLECTION_NAME = "customer_service_knowledge"
CHROMA_PATH = "data/chroma_db"

NUMBER_OF_RESULTS = 4

MAX_TOP_DISTANCE = 0.50
MAX_CHUNK_DISTANCE = 0.52


# ============================================================
# CFPB DATA PATHS
# ============================================================

SUMMARY_PATH = "data/processed/summary_metrics.csv"
PRODUCT_PATH = "data/processed/complaints_by_product.csv"
ISSUE_PATH = "data/processed/complaints_by_issue.csv"
COMPANY_PATH = "data/processed/complaints_by_company.csv"
RESPONSE_PATH = "data/processed/complaints_by_response.csv"
CHANNEL_PATH = "data/processed/complaints_by_channel.csv"
STATE_PATH = "data/processed/complaints_by_state.csv"
MONTH_PATH = "data/processed/complaints_by_month.csv"
TIMELY_PRODUCT_PATH = "data/processed/timely_response_by_product.csv"


# ============================================================
# STRUCTURED OUTPUT MODELS
# ============================================================

class RouteDecision(BaseModel):
    route: Literal[
        "complaint_support",
        "data_analysis"
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
        "Uncertain"
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

For example, do not describe conduct as:

- harassment
- abusive
- illegal
- unlawful
- fraud
- fraudulent
- a legal violation
- criminal conduct

unless the consumer explicitly described it that way.

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
complaints were handled, say so rather than
presenting a guaranteed remedy.
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
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# LOAD CHROMADB
# ============================================================

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


collection = load_collection()


# ============================================================
# LOAD CFPB DATA
# ============================================================

@st.cache_data
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
            )
    }


cfpb_data = load_cfpb_data()


# ============================================================
# TEXT MATCHING HELPER
# ============================================================

def contains_phrase(
    text,
    phrase
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
            text.lower()
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
        "credit score"
    ],

    "Debt collection": [
        "debt collector",
        "debt collection",
        "collection agency"
    ],

    "Credit card": [
        "credit card"
    ],

    "Money transfer, virtual currency, or money service": [
        "money transfer",
        "virtual currency",
        "cryptocurrency",
        "crypto",
        "money service"
    ],

    "Checking or savings account": [
        "checking account",
        "savings account",
        "bank account"
    ],

    "Mortgage": [
        "mortgage"
    ],

    "Student loan": [
        "student loan"
    ],

    "Vehicle loan or lease": [
        "vehicle loan",
        "car loan",
        "auto loan",
        "vehicle lease",
        "car lease"
    ],

    "Payday loan, title loan, personal loan, or advance loan": [
        "payday loan",
        "title loan",
        "personal loan",
        "advance loan"
    ],

    "Prepaid card": [
        "prepaid card"
    ],

    "Debt or credit management": [
        "debt management",
        "credit management",
        "credit counseling",
        "credit counselling"
    ]
}


# ============================================================
# PRODUCT DETECTION
# ============================================================

def find_products_in_question(
    question
):

    matches = []

    for (
        product,
        aliases
    ) in PRODUCT_ALIASES.items():

        if any(
            contains_phrase(
                question,
                alias
            )
            for alias in aliases
        ):

            matches.append(
                product
            )

    return matches


def find_product_in_question(
    question
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
# MULTI-TURN CONTEXT
# ============================================================

def is_contextual_followup(
    question
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
        "and then"
    }

    return q in followup_phrases


def get_previous_user_message(
    messages,
    current_question
):

    if not messages:
        return None

    user_messages = [

        str(
            message.get(
                "content",
                ""
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
                    ""
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

        if len(
            user_messages
        ) >= 2:

            return (
                user_messages[-2]
            )

        return None

    return user_messages[-1]


def build_contextual_question(
    question,
    conversation_history=None
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
            question
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
    question
):

    analysis_terms = [
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
        "comparison"
    ]

    return any(
        contains_phrase(
            question,
            term
        )
        for term in analysis_terms
    )


def is_clear_data_analysis_question(
    question
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
        "timely response rate"
    ]

    dataset_scope_terms = [
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
        "channels",
        "state",
        "states",
        "month",
        "months",
        "2025"
    ]

    has_statistical_term = any(
        contains_phrase(
            question,
            term
        )
        for term in statistical_terms
    )

    has_dataset_scope = any(
        contains_phrase(
            question,
            term
        )
        for term in dataset_scope_terms
    )

    return (
        has_statistical_term
        and has_dataset_scope
    )


# ============================================================
# PROCESS / DEADLINE DETECTION
# ============================================================

def is_response_deadline_question(
    question
):

    deadline_terms = [
        "how long does a company",
        "how long do companies",
        "how long to respond",
        "response deadline",
        "response time",
        "when does a company respond",
        "how many days does a company",
        "how many days do companies"
    ]

    return any(
        contains_phrase(
            question,
            term
        )
        for term in deadline_terms
    )


def is_general_process_question(
    question
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
        "after submitting a complaint"
    ]

    return any(
        contains_phrase(
            question,
            term
        )
        for term in process_terms
    )


# ============================================================
# CLEAR SUPPORT REQUEST DETECTION
# ============================================================

def is_clear_complaint_support_question(
    question
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
        "don't recognize"
    ]

    return any(
        contains_phrase(
            question,
            term
        )
        for term in support_terms
    )


def is_short_or_unstructured_input(
    question
):

    words = re.findall(
        r"[A-Za-z]+",
        question
    )

    return len(words) <= 4


# ============================================================
# DETERMINISTIC ROUTING
# ============================================================

def deterministic_route_override(
    user_input
):

    # 1. CFPB process/deadline questions
    if is_general_process_question(
        user_input
    ):

        return RouteDecision(
            route="complaint_support",

            reason=(
                "Deterministic routing safeguard: "
                "CFPB complaint-process or "
                "response-deadline request."
            )
        )

    # 2. Clear statistical CFPB requests
    if is_clear_data_analysis_question(
        user_input
    ):

        return RouteDecision(
            route="data_analysis",

            reason=(
                "Deterministic routing safeguard: "
                "clear statistical or aggregate "
                "CFPB data request."
            )
        )

    # 3. Clear complaint/support requests
    if is_clear_complaint_support_question(
        user_input
    ):

        return RouteDecision(
            route="complaint_support",

            reason=(
                "Deterministic routing safeguard: "
                "clear consumer complaint or "
                "guidance request."
            )
        )

    # 4. Very short / invalid / unclear input
    if is_short_or_unstructured_input(
        user_input
    ):

        return RouteDecision(
            route="complaint_support",

            reason=(
                "Deterministic safe fallback for "
                "short or insufficiently structured "
                "input."
            )
        )

    return None


# ============================================================
# ROUTER
# ============================================================

def route_request(
    user_input,
    conversation_history=None
):

    if conversation_history is None:
        conversation_history = []

    routing_input = (
        build_contextual_question(
            user_input,
            conversation_history
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
                    ROUTER_PROMPT
            },

            {
                "role":
                    "user",

                "content":
                    routing_input
            }
        ],

        format=(
            RouteDecision
            .model_json_schema()
        )
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
    question
):

    detected_products = (
        find_products_in_question(
            question
        )
    )

    if len(
        detected_products
    ) == 1:

        product = (
            detected_products[0]
        )

        return (
            f"Official CFPB guidance and complaint "
            f"information for the financial product "
            f"category '{product}'. "
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
# RAG SEARCH
# ============================================================

def search_knowledge_base(
    query
):

    retrieval_query = (
        build_retrieval_query(
            query
        )
    )

    query_embedding = (
        embedding_model
        .encode(
            retrieval_query
        )
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

    if not distances:

        return (
            "",
            [],
            None,
            retrieval_query
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
            retrieval_query
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
            > MAX_CHUNK_DISTANCE
        ):
            continue

        similarity = (
            1 - distance
        )

        source = metadata.get(
            "source",
            "Unknown source"
        )

        file_type = metadata.get(
            "file_type",
            "unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        chunk = metadata.get(
            "chunk",
            "Unknown"
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
            retrieval_query
        )

    return (
        "\n\n".join(
            context_parts
        ),
        sources,
        best_distance,
        retrieval_query
    )


# ============================================================
# DETERMINISTIC DEADLINE RESPONSE
# ============================================================

def build_response_deadline_assessment():

    return ComplaintAssessment(
        product="Uncertain",

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

        evidence_available=True,

        human_review_required=False
    )


# ============================================================
# NO-EVIDENCE RESPONSE
# ============================================================

def build_no_evidence_assessment(
    user_input
):

    detected_product = (
        find_product_in_question(
            user_input
        )
    )

    if detected_product:

        product = detected_product

        summary = (
            f"The request appears to concern "
            f"'{detected_product}', but the current "
            f"CFPB knowledge base did not return "
            f"sufficiently relevant evidence to "
            f"support a specific answer."
        )

    else:

        product = "Uncertain"

        summary = (
            "The current CFPB knowledge base did "
            "not return sufficiently relevant "
            "evidence to support a specific answer "
            "to this request."
        )

    return ComplaintAssessment(
        product=product,

        summary=summary,

        recommended_action=(
            "A specific recommendation cannot be "
            "provided from the available CFPB "
            "evidence. Human review is recommended."
        ),

        evidence_available=False,

        human_review_required=True
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
    "nclc"
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
    "unfrozen"
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
    "scam"
]


def contains_unsupported_external_reference(
    recommendation,
    context
):

    recommendation_lower = (
        recommendation.lower()
    )

    context_lower = (
        context.lower()
    )

    for term in (
        EXTERNAL_REFERENCE_TERMS
    ):

        if (
            term in recommendation_lower
            and term not in context_lower
        ):

            return True

    return False


def contains_unsupported_outcome(
    recommendation
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
    user_input
):

    summary_lower = (
        summary.lower()
    )

    user_lower = (
        user_input.lower()
    )

    for term in (
        ESCALATION_TERMS
    ):

        if (
            term in summary_lower
            and term not in user_lower
        ):

            return True

    return False


# ============================================================
# SAFE SUMMARY
# ============================================================

def build_safe_summary(
    question,
    product
):

    q = question.lower()

    if (
        product
        == "Debt collection"
        and "debt" in q
        and (
            "do not recognize" in q
            or "don't recognize" in q
            or "not recognize" in q
        )
    ):

        return (
            "You report that a debt collector is "
            "contacting you about a debt you do "
            "not recognize."
        )

    if (
        product
        == "Credit reporting or other personal consumer reports"
        and "credit report" in q
    ):

        return (
            "You report that your credit report "
            "contains information you believe is "
            "incorrect and that the issue is "
            "affecting your credit score."
        )

    if (
        product
        == "Checking or savings account"
        and (
            "frozen" in q
            or "freeze" in q
        )
    ):

        return (
            "You report that your checking or "
            "savings account was restricted and "
            "that you cannot currently access "
            "your funds."
        )

    if product != "Uncertain":

        return (
            f"The request concerns "
            f"'{product}'."
        )

    return (
        "The request does not contain enough "
        "information to assign a specific CFPB "
        "financial product confidently."
    )


# ============================================================
# GROUNDED FALLBACK ACTION
# ============================================================

def build_grounded_fallback_action(
    question
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

    if product == "Debt collection":

        return (
            "Use the retrieved CFPB debt-collection "
            "guidance to review or request validation "
            "information about the debt and keep "
            "records of the collector's communications. "
            "The available evidence does not establish "
            "that unlawful conduct occurred."
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
            "supplied the information. The evidence "
            "does not guarantee a particular outcome."
        )

    if (
        product
        == "Checking or savings account"
    ):

        return (
            "Contact the bank to understand why the "
            "account is restricted and what steps "
            "are available to address the issue. "
            "Use the retrieved CFPB guidance as "
            "informational support; restoration of "
            "account access is not guaranteed."
        )

    return (
        "Use the retrieved CFPB evidence as "
        "informational guidance. The available "
        "evidence does not support a guaranteed "
        "specific outcome."
    )


# ============================================================
# APPLY SAFEGUARDS
# ============================================================

def apply_deterministic_safeguards(
    assessment,
    question,
    context
):

    assessment.evidence_available = True

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

    if contains_unstated_escalation(
        assessment.summary,
        question
    ):

        assessment.summary = (
            build_safe_summary(
                question,
                assessment.product
            )
        )

    unsupported_reference = (
        contains_unsupported_external_reference(
            assessment.recommended_action,
            context
        )
    )

    unsupported_outcome = (
        contains_unsupported_outcome(
            assessment.recommended_action
        )
    )

    if (
        unsupported_reference
        or unsupported_outcome
    ):

        assessment.recommended_action = (
            build_grounded_fallback_action(
                question
            )
        )

    return assessment


# ============================================================
# DATA ANALYSIS
# ============================================================

def analyse_cfpb_data(
    question,
    data
):

    q = question.lower()

    summary = (
        data["summary"]
        .iloc[0]
    )

    # TOTAL COMPLAINTS
    if (
        (
            contains_phrase(
                q,
                "how many"
            )
            or contains_phrase(
                q,
                "total"
            )
            or contains_phrase(
                q,
                "number of"
            )
        )
        and (
            contains_phrase(
                q,
                "complaint"
            )
            or contains_phrase(
                q,
                "complaints"
            )
        )
        and not contains_phrase(
            q,
            "company"
        )
        and not contains_phrase(
            q,
            "issue"
        )
        and not contains_phrase(
            q,
            "product"
        )
        and not contains_phrase(
            q,
            "month"
        )
        and not contains_phrase(
            q,
            "monthly"
        )
        and not contains_phrase(
            q,
            "trend"
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
                None
        }

    # TIMELY RESPONSE RATE
    if (
        contains_phrase(
            q,
            "timely"
        )
        and contains_phrase(
            q,
            "response"
        )
        and (
            contains_phrase(
                q,
                "percent"
            )
            or contains_phrase(
                q,
                "percentage"
            )
            or contains_phrase(
                q,
                "rate"
            )
        )
        and not contains_phrase(
            q,
            "product"
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
                None
        }

    # TOP PRODUCT
    if (
        (
            contains_phrase(
                q,
                "most"
            )
            or contains_phrase(
                q,
                "highest"
            )
            or contains_phrase(
                q,
                "top"
            )
        )
        and (
            contains_phrase(
                q,
                "product"
            )
            or contains_phrase(
                q,
                "financial product"
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
                table
        }

    # PRODUCT BREAKDOWN
    if (
        contains_phrase(
            q,
            "product"
        )
        and (
            contains_phrase(
                q,
                "breakdown"
            )
            or contains_phrase(
                q,
                "distribution"
            )
            or contains_phrase(
                q,
                "all products"
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
                ].copy()
        }

    detected_product = (
        find_product_in_question(
            question
        )
    )

    # SPECIFIC PRODUCT COUNT
    if (
        detected_product
        and (
            contains_phrase(
                q,
                "how many"
            )
            or contains_phrase(
                q,
                "number"
            )
            or contains_phrase(
                q,
                "count"
            )
            or contains_phrase(
                q,
                "complaints"
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
                    matches
            }

    # TOP ISSUES
    if (
        (
            contains_phrase(
                q,
                "issue"
            )
            or contains_phrase(
                q,
                "problem"
            )
        )
        and (
            contains_phrase(
                q,
                "most common"
            )
            or contains_phrase(
                q,
                "top"
            )
            or contains_phrase(
                q,
                "highest"
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
                table
        }

    # TOP COMPANIES
    if (
        (
            contains_phrase(
                q,
                "company"
            )
            or contains_phrase(
                q,
                "companies"
            )
        )
        and (
            contains_phrase(
                q,
                "most complaints"
            )
            or contains_phrase(
                q,
                "top"
            )
            or contains_phrase(
                q,
                "highest"
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
                table
        }

    # RESPONSE BREAKDOWN
    if (
        contains_phrase(
            q,
            "response"
        )
        and (
            contains_phrase(
                q,
                "category"
            )
            or contains_phrase(
                q,
                "breakdown"
            )
            or contains_phrase(
                q,
                "type"
            )
        )
        and not contains_phrase(
            q,
            "timely"
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
                ].copy()
        }

    # CHANNEL
    if (
        (
            contains_phrase(
                q,
                "channel"
            )
            or contains_phrase(
                q,
                "submitted via"
            )
            or contains_phrase(
                q,
                "submission"
            )
        )
        and (
            contains_phrase(
                q,
                "most"
            )
            or contains_phrase(
                q,
                "top"
            )
            or contains_phrase(
                q,
                "breakdown"
            )
            or contains_phrase(
                q,
                "common"
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
                table
        }

    # STATE
    if (
        contains_phrase(
            q,
            "state"
        )
        and (
            contains_phrase(
                q,
                "most"
            )
            or contains_phrase(
                q,
                "highest"
            )
            or contains_phrase(
                q,
                "top"
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
                table
        }

    # MONTHLY TREND
    if (
        contains_phrase(
            q,
            "month"
        )
        or contains_phrase(
            q,
            "monthly"
        )
        or contains_phrase(
            q,
            "trend"
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
                ascending=False
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
                table
        }

    # TIMELY RESPONSE BY PRODUCT
    if (
        detected_product
        and contains_phrase(
            q,
            "timely"
        )
        and contains_phrase(
            q,
            "response"
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
                    match
            }

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
            None
    }


# ============================================================
# EXPLAIN ANALYSIS
# ============================================================

def explain_analysis_result(
    question,
    calculated_result
):

    response = chat(
        model=MODEL_NAME,

        messages=[
            {
                "role":
                    "system",

                "content":
                    ANALYSIS_PROMPT
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
                    )
            }
        ]
    )

    return response.message.content


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# PAGE
# ============================================================

st.title(
    "AI Consumer Complaint Intelligence "
    "& Support Assistant"
)

st.write(
    "Ask about the 2025 CFPB Consumer "
    "Complaint Database, the CFPB complaint "
    "process, or describe a financial "
    "consumer complaint."
)

st.caption(
    "The application automatically selects "
    "between CFPB knowledge retrieval and "
    "deterministic complaint-data analysis."
)


# ============================================================
# DATASET SUMMARY
# ============================================================

summary = (
    cfpb_data[
        "summary"
    ]
    .iloc[0]
)


with st.expander(
    "About the 2025 CFPB Dataset"
):

    st.write(
        f"Published complaints analysed: "
        f"{int(summary['total_complaints']):,}"
    )

    st.write(
        f"Financial product categories: "
        f"{int(summary['unique_products'])}"
    )

    st.write(
        f"Unique complaint issues: "
        f"{int(summary['unique_issues'])}"
    )

    st.write(
        f"Companies represented: "
        f"{int(summary['unique_companies']):,}"
    )

    st.write(
        f"Timely response rate: "
        f"{summary['timely_response_rate']:.2f}%"
    )

    st.write(
        f"Record-level sample available: "
        f"{int(summary['sample_records']):,}"
    )

    st.caption(
        "The analytical data are derived "
        "from complaints published in the "
        "CFPB Consumer Complaint Database."
    )


# ============================================================
# DISPLAY HISTORY
# ============================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message[
            "role"
        ]
    ):

        st.write(
            message[
                "content"
            ]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask about CFPB complaints or "
    "describe a consumer issue..."
)


# ============================================================
# PROCESS REQUEST
# ============================================================

if user_input:

    st.session_state.messages.append(
        {
            "role":
                "user",

            "content":
                user_input
        }
    )

    with st.chat_message(
        "user"
    ):

        st.write(
            user_input
        )

    try:

        route = route_request(
            user_input,
            st.session_state.messages
        )

        # ====================================================
        # DATA ANALYSIS
        # ====================================================

        if (
            route.route
            == "data_analysis"
        ):

            analysis = (
                analyse_cfpb_data(
                    user_input,
                    cfpb_data
                )
            )

            if analysis[
                "supported"
            ]:

                calculated_result = (
                    analysis[
                        "result"
                    ]
                )

                explanation = (
                    explain_analysis_result(
                        user_input,
                        calculated_result
                    )
                )

                stored_response = (
                    f"Calculated Result:\n\n"
                    f"{calculated_result}\n\n"
                    f"Explanation:\n\n"
                    f"{explanation}"
                )

                st.session_state.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            stored_response
                    }
                )

                with st.chat_message(
                    "assistant"
                ):

                    st.markdown(
                        "**Calculated Result**"
                    )

                    st.write(
                        calculated_result
                    )

                    if (
                        analysis[
                            "table"
                        ]
                        is not None
                    ):

                        st.dataframe(
                            analysis[
                                "table"
                            ],

                            use_container_width=True
                        )

                    st.markdown(
                        "**AI Explanation**"
                    )

                    st.write(
                        explanation
                    )

                    st.caption(
                        "The numerical result was "
                        "calculated from the processed "
                        "CFPB data using Python/Pandas. "
                        "The language model only "
                        "explains the deterministic "
                        "result."
                    )

                    with st.expander(
                        "View Tool Decision"
                    ):

                        st.write(
                            "Tool: CFPB Data Analysis"
                        )

                        st.write(
                            f"Reason: "
                            f"{route.reason}"
                        )

            else:

                response_text = (
                    analysis[
                        "result"
                    ]
                    + "\n\n"
                    + "Supported examples include "
                    + "complaint totals, financial "
                    + "products, complaint issues, "
                    + "companies, company responses, "
                    + "submission channels, states, "
                    + "monthly trends and timely "
                    + "response rates."
                )

                st.session_state.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            response_text
                    }
                )

                with st.chat_message(
                    "assistant"
                ):

                    st.warning(
                        response_text
                    )

        # ====================================================
        # COMPLAINT SUPPORT / RAG
        # ====================================================

        else:

            effective_question = (
                build_contextual_question(
                    user_input,
                    st.session_state.messages
                )
            )

            (
                context,
                sources,
                best_distance,
                retrieval_query
            ) = search_knowledge_base(
                effective_question
            )

            evidence_available = bool(
                context
            )

            if not evidence_available:

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

                retrieval_message = (
                    "Retrieved CFPB evidence:\n\n"
                    + context
                )

                messages_for_model = [

                    {
                        "role":
                            "system",

                        "content":
                            SUPPORT_PROMPT
                    },

                    {
                        "role":
                            "system",

                        "content":
                            GROUNDING_CONSTRAINT
                    },

                    {
                        "role":
                            "system",

                        "content":
                            retrieval_message
                    }

                ] + st.session_state.messages

                response = chat(
                    model=MODEL_NAME,

                    messages=
                        messages_for_model,

                    format=(
                        ComplaintAssessment
                        .model_json_schema()
                    )
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
                        context
                    )
                )

            assessment_text = (
                f"Product: "
                f"{assessment.product}\n\n"

                f"Summary: "
                f"{assessment.summary}\n\n"

                f"Recommended Action: "
                f"{assessment.recommended_action}\n\n"

                f"Supporting Evidence Available: "
                f"{assessment.evidence_available}\n\n"

                f"Human Review Required: "
                f"{assessment.human_review_required}"
            )

            st.session_state.messages.append(
                {
                    "role":
                        "assistant",

                    "content":
                        assessment_text
                }
            )

            with st.chat_message(
                "assistant"
            ):

                st.write(
                    assessment_text
                )

                if sources:

                    st.markdown(
                        "**Retrieved CFPB Sources**"
                    )

                    for source in sources:

                        st.write(
                            f"- {source}"
                        )

                    with st.expander(
                        "View Retrieved Context"
                    ):

                        st.write(
                            context
                        )

                else:

                    st.warning(
                        "No sufficiently relevant "
                        "CFPB evidence was retrieved. "
                        "The application did not allow "
                        "the language model to generate "
                        "a substantive answer and "
                        "flagged the request for "
                        "human review."
                    )

                with st.expander(
                    "View Structured Output"
                ):

                    st.json(
                        assessment.model_dump()
                    )

                with st.expander(
                    "View Tool Decision"
                ):

                    st.write(
                        "Tool: CFPB Knowledge "
                        "Retrieval / RAG"
                    )

                    st.write(
                        f"Reason: "
                        f"{route.reason}"
                    )

                    if (
                        effective_question
                        != user_input
                    ):

                        st.write(
                            "Conversation context "
                            "applied: Yes"
                        )

                        st.code(
                            effective_question
                        )

                    st.write(
                        "Retrieval query:"
                    )

                    st.code(
                        retrieval_query
                    )

                    if (
                        best_distance
                        is not None
                    ):

                        st.write(
                            "Best retrieval distance: "
                            f"{best_distance:.4f}"
                        )

                        st.write(
                            "Query acceptance "
                            "threshold: "
                            f"{MAX_TOP_DISTANCE:.2f}"
                        )

                        st.write(
                            "Chunk acceptance "
                            "threshold: "
                            f"{MAX_CHUNK_DISTANCE:.2f}"
                        )

    except ValidationError as error:

        st.error(
            "The language model returned a "
            "response that failed structured "
            "output validation."
        )

        with st.expander(
            "Technical details"
        ):

            st.write(
                error
            )

    except Exception as error:

        st.error(
            "An error occurred while "
            "processing the request."
        )

        with st.expander(
            "Technical details"
        ):

            st.write(
                error
            )


# ============================================================
# RESET CONVERSATION
# ============================================================

if st.button(
    "Start New Conversation"
):

    st.session_state.messages = []

    st.rerun()