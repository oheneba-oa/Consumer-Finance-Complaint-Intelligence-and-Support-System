import unittest

import pandas as pd

from backend import (
    ComplaintAssessment,
    analyse_cfpb_data,
    apply_deterministic_safeguards,
    build_contextual_question,
    build_grounded_fallback_action,
    build_no_evidence_assessment,
    contains_phrase,
    deterministic_route_override,
    find_product_in_question,
)


class TestBackendUtilities(unittest.TestCase):

    def test_contains_phrase_matches_whole_word(self):

        self.assertTrue(
            contains_phrase(
                "show the complaint count",
                "count",
            )
        )

        self.assertFalse(
            contains_phrase(
                "my bank account is frozen",
                "count",
            )
        )


    def test_credit_report_product_detection(self):

        product = (
            find_product_in_question(
                "There is an error on my credit report."
            )
        )

        self.assertEqual(
            product,
            (
                "Credit reporting or other "
                "personal consumer reports"
            ),
        )


    def test_debt_collection_product_detection(self):

        product = (
            find_product_in_question(
                "A debt collector is contacting me."
            )
        )

        self.assertEqual(
            product,
            "Debt collection",
        )


    def test_clear_analytics_route(self):

        decision = (
            deterministic_route_override(
                "How many complaints were published in 2025?"
            )
        )

        self.assertIsNotNone(
            decision
        )

        self.assertEqual(
            decision.route,
            "data_analysis",
        )


    def test_clear_support_route(self):

        decision = (
            deterministic_route_override(
                "There is an error on my credit report."
            )
        )

        self.assertIsNotNone(
            decision
        )

        self.assertEqual(
            decision.route,
            "complaint_support",
        )


    def test_contextual_followup(self):

        history = [
            {
                "role": "user",
                "content": (
                    "There is an account on my "
                    "credit report that is not mine."
                ),
            },
            {
                "role": "assistant",
                "content": "Previous answer.",
            },
        ]

        result = (
            build_contextual_question(
                "What should I do next?",
                history,
            )
        )

        self.assertIn(
            "credit report",
            result.lower(),
        )

        self.assertIn(
            "what should i do next",
            result.lower(),
        )


    def test_no_evidence_requires_human_review(self):

        assessment = (
            build_no_evidence_assessment(
                "My landlord kept my security deposit."
            )
        )

        self.assertEqual(
            assessment.product,
            "Uncertain",
        )

        self.assertFalse(
            assessment.evidence_available
        )

        self.assertTrue(
            assessment.human_review_required
        )


    def test_debt_fallback_contains_validation(self):

        action = (
            build_grounded_fallback_action(
                (
                    "A debt collector is contacting me "
                    "about a debt I do not recognize."
                )
            )
        )

        self.assertIn(
            "validation",
            action.lower(),
        )

        self.assertIn(
            "debt",
            action.lower(),
        )


    def test_unrecognized_debt_safeguard(self):

        original = ComplaintAssessment(
            product="Debt collection",
            summary=(
                "You are being contacted "
                "about a debt."
            ),
            recommended_action=(
                "Contact the CFPB and file a complaint."
            ),
            evidence_available=True,
            human_review_required=False,
        )

        updated = (
            apply_deterministic_safeguards(
                original,
                (
                    "A debt collector keeps contacting "
                    "me about a debt I do not recognize."
                ),
                "CFPB debt collection guidance.",
            )
        )

        self.assertIn(
            "validation",
            updated.recommended_action.lower(),
        )


class TestDeterministicAnalytics(unittest.TestCase):

    def setUp(self):

        self.data = {

            "summary":
                pd.DataFrame(
                    [
                        {
                            "total_complaints":
                                1000,

                            "timely_responses":
                                900,

                            "known_timely_responses":
                                950,

                            "timely_response_rate":
                                94.7368,
                        }
                    ]
                ),

            "products":
                pd.DataFrame(
                    [
                        {
                            "product":
                                (
                                    "Credit reporting or other "
                                    "personal consumer reports"
                                ),

                            "complaint_count":
                                800,

                            "percentage_of_complaints":
                                80.0,
                        },

                        {
                            "product":
                                "Debt collection",

                            "complaint_count":
                                200,

                            "percentage_of_complaints":
                                20.0,
                        },
                    ]
                ),

            "issues":
                pd.DataFrame(
                    [
                        {
                            "issue":
                                "Incorrect information",

                            "complaint_count":
                                500,
                        }
                    ]
                ),

            "companies":
                pd.DataFrame(
                    [
                        {
                            "company":
                                "Example Company",

                            "complaint_count":
                                300,
                        }
                    ]
                ),

            "responses":
                pd.DataFrame(
                    [
                        {
                            "response":
                                "Closed with explanation",

                            "complaint_count":
                                700,
                        }
                    ]
                ),

            "channels":
                pd.DataFrame(
                    [
                        {
                            "submitted_via":
                                "Web",

                            "complaint_count":
                                900,
                        }
                    ]
                ),

            "states":
                pd.DataFrame(
                    [
                        {
                            "state":
                                "CA",

                            "complaint_count":
                                250,
                        }
                    ]
                ),

            "months":
                pd.DataFrame(
                    [
                        {
                            "month":
                                "2025-01",

                            "complaint_count":
                                400,
                        },

                        {
                            "month":
                                "2025-02",

                            "complaint_count":
                                600,
                        },
                    ]
                ),

            "timely_by_product":
                pd.DataFrame(
                    [
                        {
                            "product":
                                "Debt collection",

                            "timely_response_count":
                                180,

                            "known_response_count":
                                190,

                            "timely_response_rate":
                                94.74,
                        }
                    ]
                ),
        }


    def test_total_complaints(self):

        result = (
            analyse_cfpb_data(
                (
                    "How many complaints are in "
                    "the 2025 CFPB data?"
                ),
                self.data,
            )
        )

        self.assertTrue(
            result["supported"]
        )

        self.assertIn(
            "1,000",
            result["result"],
        )


    def test_highest_month(self):

        result = (
            analyse_cfpb_data(
                (
                    "Which month had the highest "
                    "number of complaints in 2025?"
                ),
                self.data,
            )
        )

        self.assertTrue(
            result["supported"]
        )

        self.assertIn(
            "2025-02",
            result["result"],
        )

        self.assertIn(
            "600",
            result["result"],
        )


    def test_unsupported_calculation(self):

        result = (
            analyse_cfpb_data(
                (
                    "What was the average age of "
                    "consumers in 2025?"
                ),
                self.data,
            )
        )

        self.assertFalse(
            result["supported"]
        )


if __name__ == "__main__":

    unittest.main()