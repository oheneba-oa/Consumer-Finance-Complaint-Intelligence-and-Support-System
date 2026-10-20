import logging
from pathlib import Path

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


# Import after environment variables are loaded.
from streamlit_app import run_app


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():

    logger.info(
        "Starting CFPB Consumer Complaint "
        "Intelligence Assistant."
    )

    run_app()


if __name__ == "__main__":

    main()