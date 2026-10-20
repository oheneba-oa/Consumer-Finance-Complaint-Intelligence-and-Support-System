# Formal Evaluation History

## Overview

The CFPB Consumer Complaint Intelligence & Support Assistant was evaluated iteratively rather than only after development was complete.

The purpose of the evaluation process was to identify failures in routing, retrieval, structured output, analytics and multi-turn behaviour, then make targeted improvements without unnecessarily increasing system complexity.

The final formal evaluation contains 15 test cases representing normal, ambiguous, unsupported, adversarial, analytical and multi-turn interactions.

---

## Evaluation Categories

The formal test suite evaluates:

- consumer complaint support;
- credit reporting complaints;
- debt collection complaints;
- banking complaints;
- out-of-domain requests;
- ambiguous requests;
- invalid or meaningless input;
- adversarial requests;
- deterministic analytics;
- unsupported calculations; and
- multi-turn conversation behaviour.

Each test also checks the expected route and, where applicable, structured output validity.

---

## Initial Evaluation

During the early formal evaluation runs, the system did not pass all test cases consistently.

One run produced:

```text
13 / 15 tests passed
```

The main failures involved:

- identification of the highest complaint month; and
- interpretation of a contextual multi-turn follow-up.

The monthly analytics logic was refined so that questions about the highest complaint month were routed directly to deterministic structured-data analysis.

Conversation context handling was also improved so that short follow-up questions could be interpreted using the previous consumer complaint topic.

---

## Analytics Routing Improvement

A later evaluation produced:

```text
14 / 15 tests passed
```

The remaining failure was caused by inconsistent routing of an analytics question.

To address this, analytics routing was strengthened using deterministic rules for clearly statistical requests.

This reduced dependence on the language model for deciding whether questions involving totals, percentages, counts, maximum values, minimum values and similar statistical operations should use the analytics engine.

---

## Phrase-Matching Regression

During further improvements, a phrase-matching rule introduced an unintended regression.

The word:

```text
count
```

was incorrectly detected inside unrelated words such as:

```text
account
```

This caused some consumer-support questions to be incorrectly routed to the analytics component.

The problem was corrected by replacing loose substring matching with whole-word and phrase-aware matching.

This ensured that analytical keywords were interpreted as independent terms rather than character sequences appearing inside unrelated words.

---

## Singular and Plural Complaint Handling

Another routing inconsistency occurred because some statistical rules detected:

```text
complaints
```

but did not consistently detect:

```text
complaint
```

The routing logic was updated to account for both singular and plural forms.

This improved the system's handling of naturally phrased analytical questions.

---

## Evaluation Logic Refinement

The evaluation suite itself was also reviewed to ensure that failures represented genuine application problems rather than overly rigid test conditions.

For example, an ambiguous question may legitimately retrieve general CFPB complaint-process information while still being classified as uncertain and requiring human review.

The corresponding evaluation expectation was adjusted to assess the intended safety behaviour rather than require an unnecessarily specific retrieval outcome.

Adversarial-response checking was also improved so that unsafe guarantees were detected more accurately without incorrectly failing acceptable responses.

---

## Debt Collection Reliability Issue

Repeated evaluation exposed a stochastic failure involving an unrecognized debt complaint.

The language model occasionally produced a generic recommendation to submit a complaint rather than clearly directing the consumer towards reviewing or requesting debt-validation information.

Because this behaviour varied between model runs, prompt instructions alone were not considered sufficiently reliable.

A narrow deterministic safeguard was therefore added for debt-collection cases where the consumer explicitly states that the debt is not recognized or is not theirs.

The safeguard provides grounded guidance to:

- review or request validation information about the debt; and
- retain records of communications.

This improved consistency while still allowing the language model to handle general consumer-support interactions.

---

## Prompt Engineering Findings

Three primary prompt variants were evaluated during development.

| Prompt Variant | Product Accuracy | Structured Validity | Human Review Consistency |
|---|---:|---:|---:|
| A - Basic | 60.0% | 100% | 93.3% |
| B - Strict Grounding | 73.3% | 100% | 100% |
| C - Evidence Aware | 80.0% | 100% | 100% |

Variant C was selected because it provided the strongest overall balance of:

- grounding;
- classification accuracy;
- consistency;
- evidence awareness;
- human-review behaviour; and
- response quality.

Additional refinements were tested after Variant C.

However, increasing the number of instructions did not consistently improve the performance of the local Llama 3.2 model.

Some predictable behaviours were therefore transferred from prompt instructions into deterministic Python safeguards.

This reduced the amount of behaviour that depended entirely on generative-model compliance.

---

## Generated Link Reliability Improvement

During final application testing, the language model produced an external web link as part of a recommended action even though the URL had not been supplied by verified application metadata.

Although the response remained relevant to the consumer's issue, allowing the language model to independently generate URLs introduced a source-provenance risk.

The backend was therefore improved in two ways:

1. The support prompt was updated to prevent the language model from generating or modifying URLs unless they are explicitly supplied by the application.

2. A deterministic safeguard was added to detect generated URLs and replace affected recommendations with grounded CFPB guidance.

The system was re-evaluated after this change.

Final validation results remained:

```text
Unit tests: 12 / 12 passed
Formal evaluation cases: 15 / 15 passed
Route accuracy: 100%
Structured output validity: 100%
```

A final live application test confirmed that the support response remained useful without exposing an unverified generated link.

This improvement strengthened the system's approach to source provenance and grounded response generation.

---

## Final Formal Evaluation

After the routing improvements, phrase-matching corrections, multi-turn improvements and deterministic safeguards, the final formal evaluation produced:

```text
Total cases: 15
Passed: 15
Failed: 0
Overall pass rate: 100.00%
Route accuracy: 100.00%
Structured output validity: 100.00%
```

The 15 cases covered:

- normal RAG complaint-process requests;
- credit reporting;
- debt collection;
- checking and savings accounts;
- out-of-domain questions;
- ambiguous input;
- invalid input;
- adversarial requests;
- complaint totals;
- response percentages;
- product statistics;
- issue statistics;
- monthly statistics;
- unsupported calculations; and
- multi-turn conversation.

Evaluation outputs are stored in:

```text
evaluation/formal_evaluation/formal_evaluation_results.csv
evaluation/formal_evaluation/formal_evaluation_summary.csv
evaluation/formal_evaluation/formal_evaluation_failures.csv
evaluation/formal_evaluation/formal_manual_review.csv
```

---

## Unit Test Validation

In addition to the formal end-to-end evaluation, a separate backend unit-test suite was created.

The unit tests cover:

- clear analytics routing;
- clear support routing;
- whole-word phrase matching;
- contextual follow-ups;
- credit-report product detection;
- debt-collection product detection;
- debt-validation fallback behaviour;
- insufficient-evidence handling;
- unrecognized-debt safeguards;
- highest complaint month;
- total complaint calculation; and
- unsupported calculations.

The final result was:

```text
Ran 12 tests

OK
```

This provides an additional layer of validation for deterministic backend functionality.

---

## Key Lessons from Evaluation

The evaluation process produced several important findings.

First, deterministic numerical tasks should be performed using conventional software rather than relying on a generative model.

Second, prompt engineering alone does not guarantee consistent model behaviour.

Third, narrow deterministic safeguards can improve reliability when they address clearly defined and repeatable failure modes.

Fourth, the evaluation logic itself must be reviewed carefully so that valid application behaviour is not incorrectly classified as a failure.

Fifth, repeated evaluation is important because generative-model behaviour may vary between runs.

Finally, source-sensitive information such as web links should only be exposed when provenance can be verified by the application.

---

## Final Design Decision

The final application therefore uses a hybrid architecture:

```text
Numerical questions
        ↓
Deterministic Python / Pandas

Knowledge questions
        ↓
Retrieval-Augmented Generation

Structured response validity
        ↓
Pydantic

Predictable safety and routing behaviour
        ↓
Deterministic application logic

Natural-language interpretation and explanation
        ↓
Llama 3.2
```

This approach provides a better balance of usefulness, reliability, explainability and maintainability than relying on the language model for every task.