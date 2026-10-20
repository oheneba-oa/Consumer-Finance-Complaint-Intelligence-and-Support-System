import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(
    BASE_DIR / ".env"
)


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    __name__
)


# Environment variables are loaded before importing the
# application so dependent libraries can read them correctly.
from streamlit_app import run_app


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():

    logger.info(
        "Starting CFPB Consumer Complaint "
        "Intelligence Assistant."
    )

    try:

        run_app()

    except Exception:

        logger.exception(
            "Application startup failed."
        )

        st.error(
            "The application could not be started. "
            "Check that the required CFPB data files "
            "are available and that the configured "
            "services are running."
        )

        st.stop()


if __name__ == "__main__":
    main()