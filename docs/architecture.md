# System Architecture

## Overview

The CFPB Consumer Complaint Intelligence & Support Assistant uses a hybrid AI architecture.

The system intentionally separates tasks that require deterministic computation from tasks that benefit from natural-language understanding and retrieval.

The main architectural principle is:

```text
Numerical analysis
    → Python / Pandas

Knowledge retrieval
    → Sentence Transformers + ChromaDB

Natural-language explanation
    → Llama 3.2

Structured output validation
    → Pydantic

Reliable routing and safeguards
    → Deterministic Python logic