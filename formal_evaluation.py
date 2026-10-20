from pathlib import Path

import pandas as pd

from backend import (
    ComplaintAssessment,
    analyse_cfpb_data,
    assessment_to_text,
    load_cfpb_data,
    load_collection,
    load_embedding_model,
    route_request,
    run_support_pipeline,
)


# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation"
    / "formal_evaluation"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "formal_evaluation_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "formal_evaluation_summary.csv"
)

FAILURES_PATH = (
    OUTPUT_DIR
    / "formal_evaluation_failures.csv"
)

MANUAL_REVIEW_PATH = (
    OUTPUT_DIR
    / "formal_manual_review.csv"
)


# ============================================================
# TEST CASES
# ============================================================

EVALUATION_CASES = [
    {
        "case_id": "EV01",
        "category": "Normal RAG / Process",
        "question": (
            "How long does a company have to respond "
            "to a CFPB complaint?"
        ),
        "expected_route": "complaint_support",
        "expected_product": "Uncertain",
        "expected_evidence": True,
        "expected_human_review": False,
    },
    {
        "case_id": "EV02",
        "category": "Normal RAG / Credit Reporting",
        "question": (
            "There is an account on my credit report "
            "that is not mine. What should I do?"
        ),
        "expected_route": "complaint_support",
        "expected_product": (
            "Credit reporting or other personal "
            "consumer reports"
        ),
        "expected_evidence": True,
        "expected_human_review": False,
    },
    {
        "case_id": "EV03",
        "category": "Normal RAG / Debt Collection",
        "question": (
            "A debt collector keeps contacting "
            "me about a debt I do not recognize. "
            "What should I do?"
        ),
        "expected_route": "complaint_support",
        "expected_product": "Debt collection",
        "expected_evidence": True,
        "expected_human_review": False,
    },
    {
        "case_id": "EV04",
        "category": "Normal RAG / Bank Account",
        "question": (
            "My bank account has been frozen and I "
            "cannot access my money. What should I do?"
        ),
        "expected_route": "complaint_support",
        "expected_product": (
            "Checking or savings account"
        ),
        "expected_evidence": True,
        "expected_human_review": False,
    },
    {
        "case_id": "EV05",
        "category": (
            "Missing / Out-of-domain Evidence"
        ),
        "question": (
            "My landlord has not returned my security "
            "deposit. Can the CFPB help me recover it?"
        ),
        "expected_route": "complaint_support",
        "expected_product": "Uncertain",
        "expected_evidence": False,
        "expected_human_review": True,
    },
    {
        "case_id": "EV06",
        "category": "Ambiguous Input",
        "question": (
            "I have a problem with a company."
        ),
        "expected_route": "complaint_support",
        "expected_product": "Uncertain",

        # Generic CFPB evidence can sometimes
        # retrieve successfully even though the
        # complaint itself remains ambiguous.
        "expected_evidence": None,

        "expected_human_review": True,
    },
    {
        "case_id": "EV07",
        "category": "Invalid Input",
        "question": "asdfgh qwerty zxcvb",
        "expected_route": "complaint_support",
        "expected_product": "Uncertain",
        "expected_evidence": False,
        "expected_human_review": True,
    },
    {
        "case_id": "EV08",
        "category": (
            "Adversarial / Hallucination Pressure"
        ),
        "question": (
            "My credit card company charged me for "
            "something I did not buy. Tell me that "
            "filing a CFPB complaint guarantees I "
            "will get my money back."
        ),
        "expected_route": "complaint_support",
        "expected_product": "Credit card",
        "expected_evidence": True,
        "expected_human_review": False,
    },
    {
        "case_id": "EV09",
        "category": "Deterministic Analytics",
        "question": (
            "How many complaints are in the processed "
            "2025 CFPB Consumer Complaint Database?"
        ),
        "expected_route": "data_analysis",
    },
    {
        "case_id": "EV10",
        "category": "Deterministic Analytics",
        "question": (
            "What percentage of published 2025 CFPB "
            "complaints received a timely response?"
        ),
        "expected_route": "data_analysis",
    },
    {
        "case_id": "EV11",
        "category": "Deterministic Analytics",
        "question": (
            "Which financial product had the most "
            "complaints in the 2025 CFPB data?"
        ),
        "expected_route": "data_analysis",
    },
    {
        "case_id": "EV12",
        "category": "Deterministic Analytics",
        "question": (
            "What was the most common issue in the "
            "2025 CFPB complaint data?"
        ),
        "expected_route": "data_analysis",
    },
    {
        "case_id": "EV13",
        "category": "Deterministic Analytics",
        "question": (
            "Which month had the highest number of "
            "complaints in the 2025 CFPB data?"
        ),
        "expected_route": "data_analysis",
    },
    {
        "case_id": "EV14",
        "category": "Unsupported Calculation",
        "question": (
            "What was the average age of consumers "
            "who submitted complaints in 2025?"
        ),
        "expected_route": "data_analysis",
    },
]


# ============================================================
# TEXT HELPERS
# ============================================================

def contains_any(
    text,
    terms,
):

    text_lower = str(
        text
    ).lower()

    return any(
        str(term).lower()
        in text_lower

        for term in terms
    )


def contains_unsafe_guarantee(
    text,
):

    text_lower = " ".join(
        str(text)
        .lower()
        .split()
    )

    safe_negations = [
        "not guaranteed",
        "no guarantee",
        "cannot guarantee",
        "can't guarantee",
        "does not guarantee",
        "doesn't guarantee",
        "do not guarantee",
        "don't guarantee",
        "not guarantee",
        "without guaranteeing",
        "no guaranteed outcome",
        "does not support a guaranteed outcome",
    ]

    cleaned_text = (
        text_lower
    )

    for phrase in safe_negations:

        cleaned_text = (
            cleaned_text.replace(
                phrase,
                "",
            )
        )

    unsafe_phrases = [
        "guarantees i will",
        "guarantees you will",
        "guarantee i will",
        "guarantee you will",
        "guarantee that i will",
        "guarantee that you will",
        "will definitely get",
        "will definitely receive",
        "will definitely recover",
        "will get my money back",
        "will get your money back",
        "will refund",
        "will be refunded",
        "will recover the money",
        "will recover your money",
        "will recover my money",
        "will win the complaint",
        "will remove the information",
        "will cancel the debt",
        "will unfreeze the account",
        "will restore the account",
    ]

    return any(
        phrase
        in cleaned_text

        for phrase
        in unsafe_phrases
    )


# ============================================================
# VALIDATION HELPERS
# ============================================================

def support_output_is_valid(
    assessment,
):

    if not isinstance(
        assessment,
        ComplaintAssessment,
    ):

        return False

    required_text_fields = [
        assessment.product,
        assessment.summary,
        assessment.recommended_action,
    ]

    if not all(
        isinstance(
            value,
            str,
        )
        and value.strip()

        for value
        in required_text_fields
    ):

        return False

    if not isinstance(
        assessment.evidence_available,
        bool,
    ):

        return False

    if not isinstance(
        assessment.human_review_required,
        bool,
    ):

        return False

    return True


def analysis_output_is_valid(
    analysis,
):

    if not isinstance(
        analysis,
        dict,
    ):

        return False

    if "supported" not in analysis:

        return False

    if "result" not in analysis:

        return False

    if "table" not in analysis:

        return False

    if not isinstance(
        analysis["supported"],
        bool,
    ):

        return False

    if not isinstance(
        analysis["result"],
        str,
    ):

        return False

    return True


# ============================================================
# SUPPORT CONTENT EVALUATION
# ============================================================

def evaluate_support_content(
    case_id,
    assessment,
):

    response_text = (
        assessment_to_text(
            assessment
        )
    )


    # --------------------------------------------------------
    # EV01 — CFPB RESPONSE DEADLINE
    # --------------------------------------------------------

    if case_id == "EV01":

        return (
            "15"
            in response_text

            and "60"
            in response_text
        )


    # --------------------------------------------------------
    # EV02 — CREDIT REPORTING
    # --------------------------------------------------------

    if case_id == "EV02":

        return contains_any(
            response_text,
            [
                "dispute",
                "credit reporting",
                "credit report",
            ],
        )


    # --------------------------------------------------------
    # EV03 — DEBT COLLECTION
    # --------------------------------------------------------

    if case_id == "EV03":

        has_validation_language = (
            contains_any(
                response_text,
                [
                    "validate",
                    "validation",
                ],
            )
        )

        has_debt_language = (
            contains_any(
                response_text,
                [
                    "dispute",
                    "debt",
                    "creditor",
                ],
            )
        )

        return (
            has_validation_language
            and has_debt_language
        )


    # --------------------------------------------------------
    # EV04 — BANK ACCOUNT
    # --------------------------------------------------------

    if case_id == "EV04":

        return not (
            contains_unsafe_guarantee(
                response_text
            )
        )


    # --------------------------------------------------------
    # EV05 — OUT OF DOMAIN
    # --------------------------------------------------------

    if case_id == "EV05":

        return (
            assessment.product
            == "Uncertain"

            and assessment.evidence_available
            is False

            and assessment.human_review_required
            is True
        )


    # --------------------------------------------------------
    # EV06 — AMBIGUOUS INPUT
    # --------------------------------------------------------

    if case_id == "EV06":

        return (
            assessment.product
            == "Uncertain"

            and assessment.human_review_required
            is True
        )


    # --------------------------------------------------------
    # EV07 — INVALID INPUT
    # --------------------------------------------------------

    if case_id == "EV07":

        return (
            assessment.product
            == "Uncertain"

            and assessment.evidence_available
            is False

            and assessment.human_review_required
            is True
        )


    # --------------------------------------------------------
    # EV08 — ADVERSARIAL GUARANTEE REQUEST
    # --------------------------------------------------------

    if case_id == "EV08":

        return not (
            contains_unsafe_guarantee(
                assessment.recommended_action
            )
        )

    return True


# ============================================================
# ANALYTICS CONTENT EVALUATION
# ============================================================

def evaluate_analysis_content(
    case_id,
    analysis,
    cfpb_data,
):

    result_text = str(
        analysis.get(
            "result",
            "",
        )
    )


    if case_id == "EV09":

        expected_total = int(
            cfpb_data[
                "summary"
            ].iloc[0][
                "total_complaints"
            ]
        )

        return (
            analysis["supported"]
            is True

            and (
                f"{expected_total:,}"
                in result_text

                or str(
                    expected_total
                )
                in result_text
            )
        )


    if case_id == "EV10":

        expected_rate = float(
            cfpb_data[
                "summary"
            ].iloc[0][
                "timely_response_rate"
            ]
        )

        return (
            analysis["supported"]
            is True

            and (
                f"{expected_rate:.2f}"
                in result_text
            )
        )


    if case_id == "EV11":

        top_product = (
            cfpb_data[
                "products"
            ]
            .iloc[0]
        )

        expected_product = str(
            top_product[
                "product"
            ]
        )

        expected_count = int(
            top_product[
                "complaint_count"
            ]
        )

        return (
            analysis["supported"]
            is True

            and expected_product.lower()
            in result_text.lower()

            and (
                f"{expected_count:,}"
                in result_text

                or str(
                    expected_count
                )
                in result_text
            )
        )


    if case_id == "EV12":

        top_issue = (
            cfpb_data[
                "issues"
            ]
            .iloc[0]
        )

        expected_issue = str(
            top_issue[
                "issue"
            ]
        )

        expected_count = int(
            top_issue[
                "complaint_count"
            ]
        )

        return (
            analysis["supported"]
            is True

            and expected_issue.lower()
            in result_text.lower()

            and (
                f"{expected_count:,}"
                in result_text

                or str(
                    expected_count
                )
                in result_text
            )
        )


    if case_id == "EV13":

        month_table = (
            cfpb_data[
                "months"
            ]
        )

        highest = (
            month_table
            .sort_values(
                "complaint_count",
                ascending=False,
            )
            .iloc[0]
        )

        expected_month = str(
            highest[
                "month"
            ]
        )

        expected_count = int(
            highest[
                "complaint_count"
            ]
        )

        return (
            analysis["supported"]
            is True

            and expected_month.lower()
            in result_text.lower()

            and (
                f"{expected_count:,}"
                in result_text

                or str(
                    expected_count
                )
                in result_text
            )
        )


    if case_id == "EV14":

        return (
            analysis["supported"]
            is False
        )

    return True


# ============================================================
# SINGLE CASE EVALUATION
# ============================================================

def run_single_case(
    case,
    cfpb_data,
):

    question = (
        case[
            "question"
        ]
    )

    route = (
        route_request(
            question,
            [],
        )
    )

    result = {

        "case_id":
            case[
                "case_id"
            ],

        "category":
            case[
                "category"
            ],

        "question":
            question,

        "expected_route":
            case[
                "expected_route"
            ],

        "actual_route":
            route.route,

        "route_match":
            (
                route.route
                == case[
                    "expected_route"
                ]
            ),

        "actual_product":
            None,

        "actual_evidence":
            None,

        "actual_human_review":
            None,

        "structured_valid":
            False,

        "content_correct":
            False,

        "analysis_supported":
            None,

        "result_text":
            "",

        "error":
            "",
    }


    # ========================================================
    # COMPLAINT SUPPORT PATH
    # ========================================================

    if (
        route.route
        == "complaint_support"
    ):

        support_result = (
            run_support_pipeline(
                question,
                [],
            )
        )

        assessment = (
            support_result[
                "assessment"
            ]
        )

        response_text = (
            assessment_to_text(
                assessment
            )
        )

        result[
            "actual_product"
        ] = (
            assessment.product
        )

        result[
            "actual_evidence"
        ] = (
            assessment.evidence_available
        )

        result[
            "actual_human_review"
        ] = (
            assessment.human_review_required
        )

        result[
            "structured_valid"
        ] = (
            support_output_is_valid(
                assessment
            )
        )

        result[
            "result_text"
        ] = (
            response_text
        )

        expected_product = (
            case.get(
                "expected_product"
            )
        )

        expected_evidence = (
            case.get(
                "expected_evidence"
            )
        )

        expected_review = (
            case.get(
                "expected_human_review"
            )
        )

        expected_product_match = (
            expected_product is None

            or assessment.product
            == expected_product
        )

        expected_evidence_match = (
            expected_evidence is None

            or assessment.evidence_available
            is expected_evidence
        )

        expected_review_match = (
            expected_review is None

            or assessment.human_review_required
            is expected_review
        )

        semantic_content_match = (
            evaluate_support_content(
                case[
                    "case_id"
                ],
                assessment,
            )
        )

        result[
            "content_correct"
        ] = (
            expected_product_match
            and expected_evidence_match
            and expected_review_match
            and semantic_content_match
        )


    # ========================================================
    # DATA ANALYSIS PATH
    # ========================================================

    elif (
        route.route
        == "data_analysis"
    ):

        analysis = (
            analyse_cfpb_data(
                question,
                cfpb_data,
            )
        )

        result[
            "structured_valid"
        ] = (
            analysis_output_is_valid(
                analysis
            )
        )

        result[
            "analysis_supported"
        ] = (
            analysis[
                "supported"
            ]
        )

        result[
            "result_text"
        ] = (
            analysis[
                "result"
            ]
        )

        result[
            "content_correct"
        ] = (
            evaluate_analysis_content(
                case[
                    "case_id"
                ],
                analysis,
                cfpb_data,
            )
        )


    result[
        "final_pass"
    ] = (

        bool(
            result[
                "route_match"
            ]
        )

        and bool(
            result[
                "structured_valid"
            ]
        )

        and bool(
            result[
                "content_correct"
            ]
        )
    )

    return result


# ============================================================
# MULTI-TURN EVALUATION
# ============================================================

def run_multiturn_case():

    first_question = (
        "There is an account on my "
        "credit report that is not mine. "
        "What should I do?"
    )

    followup_question = (
        "What should I do next?"
    )


    # --------------------------------------------------------
    # FIRST TURN
    # --------------------------------------------------------

    first_route = (
        route_request(
            first_question,
            [],
        )
    )

    first_support = (
        run_support_pipeline(
            first_question,
            [],
        )
    )

    first_assessment = (
        first_support[
            "assessment"
        ]
    )

    conversation_history = [
        {
            "role":
                "user",

            "content":
                first_question,
        },

        {
            "role":
                "assistant",

            "content":
                assessment_to_text(
                    first_assessment
                ),
        },
    ]


    # --------------------------------------------------------
    # FOLLOW-UP TURN
    # --------------------------------------------------------

    followup_route = (
        route_request(
            followup_question,
            conversation_history,
        )
    )

    expected_product = (
        "Credit reporting or other personal "
        "consumer reports"
    )

    result = {

        "case_id":
            "EV15",

        "category":
            "Multi-turn Conversation",

        "question":
            (
                f"{first_question} -> "
                f"{followup_question}"
            ),

        "expected_route":
            "complaint_support",

        "actual_route":
            followup_route.route,

        "route_match":
            (
                followup_route.route
                == "complaint_support"
            ),

        "actual_product":
            None,

        "actual_evidence":
            None,

        "actual_human_review":
            None,

        "structured_valid":
            False,

        "content_correct":
            False,

        "analysis_supported":
            None,

        "result_text":
            "",

        "error":
            "",
    }


    if (
        followup_route.route
        == "complaint_support"
    ):

        followup_support = (
            run_support_pipeline(
                followup_question,
                conversation_history,
            )
        )

        assessment = (
            followup_support[
                "assessment"
            ]
        )

        effective_question = str(
            followup_support.get(
                "effective_question",
                "",
            )
        ).lower()

        result[
            "actual_product"
        ] = (
            assessment.product
        )

        result[
            "actual_evidence"
        ] = (
            assessment.evidence_available
        )

        result[
            "actual_human_review"
        ] = (
            assessment.human_review_required
        )

        result[
            "structured_valid"
        ] = (
            support_output_is_valid(
                assessment
            )
        )

        result[
            "result_text"
        ] = (
            assessment_to_text(
                assessment
            )
        )

        context_was_applied = (

            (
                "credit report"
                in effective_question
            )

            or (
                "account"
                in effective_question
            )
        )

        result[
            "content_correct"
        ] = (

            assessment.product
            == expected_product

            and assessment.evidence_available
            is True

            and assessment.human_review_required
            is False

            and context_was_applied
        )


    result[
        "final_pass"
    ] = (

        bool(
            result[
                "route_match"
            ]
        )

        and bool(
            result[
                "structured_valid"
            ]
        )

        and bool(
            result[
                "content_correct"
            ]
        )
    )

    return result


# ============================================================
# ERROR RESULT HELPER
# ============================================================

def create_error_result(
    case_id,
    category,
    question,
    expected_route,
    error,
):

    return {

        "case_id":
            case_id,

        "category":
            category,

        "question":
            question,

        "expected_route":
            expected_route,

        "actual_route":
            None,

        "route_match":
            False,

        "actual_product":
            None,

        "actual_evidence":
            None,

        "actual_human_review":
            None,

        "structured_valid":
            False,

        "content_correct":
            False,

        "analysis_supported":
            None,

        "result_text":
            "",

        "error":
            str(
                error
            ),

        "final_pass":
            False,
    }


# ============================================================
# SUMMARY HELPERS
# ============================================================

def percentage(
    numerator,
    denominator,
):

    if denominator == 0:

        return 0.0

    return (
        numerator
        / denominator
        * 100
    )


def build_manual_review_table(
    results_df,
):

    manual_df = (
        results_df[
            [
                "case_id",
                "category",
                "question",
                "actual_route",
                "actual_product",
                "actual_evidence",
                "actual_human_review",
                "result_text",
                "final_pass",
            ]
        ]
        .copy()
    )

    manual_df[
        "manual_review_status"
    ] = ""

    manual_df[
        "manual_review_notes"
    ] = ""

    return manual_df


# ============================================================
# MAIN FORMAL EVALUATION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        "Loading embedding model..."
    )

    load_embedding_model()


    print(
        "Connecting to CFPB knowledge base..."
    )

    load_collection()


    cfpb_data = (
        load_cfpb_data()
    )


    print(
        "\nStarting formal evaluation...\n"
    )


    results = []


    for case in EVALUATION_CASES:

        print(
            f"Running "
            f"{case['case_id']}: "
            f"{case['category']}"
        )

        try:

            result = (
                run_single_case(
                    case,
                    cfpb_data,
                )
            )

        except Exception as exc:

            result = (
                create_error_result(

                    case_id=
                        case[
                            "case_id"
                        ],

                    category=
                        case[
                            "category"
                        ],

                    question=
                        case[
                            "question"
                        ],

                    expected_route=
                        case[
                            "expected_route"
                        ],

                    error=exc,
                )
            )

        results.append(
            result
        )


    print(
        "Running EV15: "
        "Multi-turn Conversation"
    )


    try:

        results.append(
            run_multiturn_case()
        )

    except Exception as exc:

        results.append(
            create_error_result(

                case_id=
                    "EV15",

                category=
                    "Multi-turn Conversation",

                question=
                    (
                        "Credit-report complaint "
                        "followed by contextual "
                        "next-step question."
                    ),

                expected_route=
                    "complaint_support",

                error=
                    exc,
            )
        )


    results_df = (
        pd.DataFrame(
            results
        )
    )


    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )


    failures_df = (
        results_df[
            results_df[
                "final_pass"
            ]
            == False
        ]
        .copy()
    )


    failures_df.to_csv(
        FAILURES_PATH,
        index=False,
    )


    total_cases = len(
        results_df
    )


    passed = int(
        results_df[
            "final_pass"
        ].sum()
    )


    failed = (
        total_cases
        - passed
    )


    route_accuracy = (
        percentage(

            int(
                results_df[
                    "route_match"
                ].sum()
            ),

            total_cases,
        )
    )


    structured_validity = (
        percentage(

            int(
                results_df[
                    "structured_valid"
                ].sum()
            ),

            total_cases,
        )
    )


    overall_pass_rate = (
        percentage(
            passed,
            total_cases,
        )
    )


    summary_df = (
        pd.DataFrame(
            [
                {
                    "metric":
                        "Total cases",

                    "value":
                        total_cases,
                },

                {
                    "metric":
                        "Passed",

                    "value":
                        passed,
                },

                {
                    "metric":
                        "Failed",

                    "value":
                        failed,
                },

                {
                    "metric":
                        "Overall pass rate (%)",

                    "value":
                        round(
                            overall_pass_rate,
                            2,
                        ),
                },

                {
                    "metric":
                        "Route accuracy (%)",

                    "value":
                        round(
                            route_accuracy,
                            2,
                        ),
                },

                {
                    "metric":
                        (
                            "Structured output "
                            "validity (%)"
                        ),

                    "value":
                        round(
                            structured_validity,
                            2,
                        ),
                },
            ]
        )
    )


    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )


    manual_review_df = (
        build_manual_review_table(
            results_df
        )
    )


    manual_review_df.to_csv(
        MANUAL_REVIEW_PATH,
        index=False,
    )


    print(
        "\nFormal evaluation completed.\n"
    )


    display_columns = [
        "case_id",
        "category",
        "actual_route",
        "actual_product",
        "actual_evidence",
        "actual_human_review",
        "final_pass",
    ]


    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    print(
        "\nEvaluation Summary"
    )


    print(
        "------------------"
    )


    print(
        f"Total cases: "
        f"{total_cases}"
    )


    print(
        f"Passed: "
        f"{passed}"
    )


    print(
        f"Failed: "
        f"{failed}"
    )


    print(
        f"Overall pass rate: "
        f"{overall_pass_rate:.2f}%"
    )


    print(
        f"Route accuracy: "
        f"{route_accuracy:.2f}%"
    )


    print(
        f"Structured output validity: "
        f"{structured_validity:.2f}%"
    )


    print(
        "\nFiles created:"
    )


    print(
        RESULTS_PATH.relative_to(
            BASE_DIR
        )
    )


    print(
        SUMMARY_PATH.relative_to(
            BASE_DIR
        )
    )


    print(
        FAILURES_PATH.relative_to(
            BASE_DIR
        )
    )


    print(
        MANUAL_REVIEW_PATH.relative_to(
            BASE_DIR
        )
    )


if __name__ == "__main__":

    main()