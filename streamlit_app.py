import html
import re
import textwrap

import streamlit as st

from backend import (
    MAX_CHUNK_DISTANCE,
    MAX_TOP_DISTANCE,
    MODEL_NAME,
    analyse_cfpb_data,
    assessment_to_text,
    load_cfpb_data,
    load_collection,
    route_request,
    run_support_pipeline,
    stream_analysis_explanation,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CFPB Intelligence Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SAFE HTML RENDERER
# ============================================================

def render_html(markup):

    cleaned = textwrap.dedent(markup)

    cleaned = "\n".join(
        line.strip()
        for line in cleaned.splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned,
        unsafe_allow_html=True,
    )


# ============================================================
# CUSTOM CSS
# ============================================================

def inject_styles():

    render_html(
        """
        <style>

        :root {
            --background: #070a14;
            --sidebar: #0c1425;

            --blue: #5b8cff;
            --cyan: #35d4ff;
            --purple: #916dff;
            --pink: #d66dff;

            --green: #38dda8;
            --yellow: #ffd166;

            --white: #f8faff;
            --text: #dce6f8;
            --muted: #8493b0;

            --border: rgba(145, 169, 216, 0.14);
        }


        html,
        body,
        [class*="css"] {
            font-family:
                Inter,
                ui-sans-serif,
                system-ui,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }


        .stApp {
            background:
                radial-gradient(
                    circle at 8% 5%,
                    rgba(78, 124, 255, 0.17),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 92% 7%,
                    rgba(145, 88, 255, 0.15),
                    transparent 27%
                ),
                radial-gradient(
                    circle at 50% 100%,
                    rgba(53, 212, 255, 0.06),
                    transparent 35%
                ),
                #070a14;

            color: var(--white);
        }


        .block-container {
            max-width: 1240px;
            padding-top: 1.35rem;
            padding-bottom: 8rem;
        }


        #MainMenu {
            visibility: hidden;
        }


        footer {
            visibility: hidden;
        }


        [data-testid="stAppDeployButton"] {
            display: none;
        }


        /*
        Keep Streamlit's sidebar collapse/reopen controls
        visible so the sidebar can always be restored.
        */

        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"] {
            visibility: visible !important;
            display: flex !important;
            opacity: 1 !important;
        }


        /* ==================================================
           SIDEBAR
        ================================================== */

        [data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #0d1628,
                    #09111f
                );

            border-right:
                1px solid
                rgba(139, 163, 209, 0.12);
        }


        [data-testid="stSidebar"] .block-container {
            padding-top: 1.45rem;
        }


        .brand-icon {
            width: 48px;
            height: 48px;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 15px;

            background:
                linear-gradient(
                    135deg,
                    #5b8cff,
                    #966cff
                );

            box-shadow:
                0 14px 34px
                rgba(91, 140, 255, 0.25);

            color: white;

            font-size: 1.35rem;
            font-weight: 900;

            margin-bottom: 0.9rem;
        }


        .brand-name {
            color: white;

            font-size: 1.16rem;
            font-weight: 820;

            letter-spacing: -0.025em;

            margin-bottom: 0.22rem;
        }


        .brand-description {
            color: #8796b4;

            font-size: 0.79rem;
            line-height: 1.55;

            margin-bottom: 1.05rem;
        }


        .system-ready {
            display: flex;
            align-items: center;

            gap: 0.6rem;

            padding:
                0.85rem
                0.9rem;

            margin:
                1rem
                0
                1rem
                0;

            border-radius: 15px;

            background:
                linear-gradient(
                    135deg,
                    rgba(56, 221, 168, 0.12),
                    rgba(53, 212, 255, 0.06)
                );

            border:
                1px solid
                rgba(56, 221, 168, 0.20);
        }


        .system-dot {
            width: 9px;
            height: 9px;

            flex-shrink: 0;

            border-radius: 50%;

            background: #38dda8;

            box-shadow:
                0 0 0 5px
                rgba(56, 221, 168, 0.10);
        }


        .system-ready-text {
            color: #bff8e5;

            font-size: 0.79rem;
            font-weight: 700;
        }


        .sidebar-section-title {
            color: #f3f7ff;

            font-size: 0.82rem;
            font-weight: 760;

            margin-bottom: 0.55rem;
        }


        .sidebar-information-row {
            display: flex;
            justify-content: space-between;
            align-items: center;

            gap: 0.8rem;

            padding: 0.38rem 0;
        }


        .sidebar-information-label {
            color: #7889a8;

            font-size: 0.76rem;
        }


        .sidebar-information-value {
            color: #eef4ff;

            font-size: 0.76rem;
            font-weight: 700;

            text-align: right;
        }


        .sidebar-trust-card {
            margin-top: 1rem;

            padding:
                0.9rem
                0.95rem;

            border-radius: 15px;

            background:
                linear-gradient(
                    145deg,
                    rgba(25, 38, 68, 0.85),
                    rgba(16, 26, 47, 0.85)
                );

            border:
                1px solid
                rgba(138, 164, 215, 0.13);
        }


        .sidebar-trust-title {
            color: #dfe8f8;

            font-size: 0.79rem;
            font-weight: 750;

            margin-bottom: 0.35rem;
        }


        .sidebar-trust-text {
            color: #7f90ad;

            font-size: 0.75rem;
            line-height: 1.5;
        }


        /* ==================================================
           HERO
        ================================================== */

        .hero {
            position: relative;

            overflow: hidden;

            padding:
                2rem
                2.45rem
                1.95rem
                2.45rem;

            margin-bottom: 1.15rem;

            border-radius: 26px;

            border:
                1px solid
                rgba(123, 153, 255, 0.18);

            background:
                linear-gradient(
                    125deg,
                    rgba(31, 55, 111, 0.74),
                    rgba(28, 31, 75, 0.72),
                    rgba(58, 31, 91, 0.63)
                );

            box-shadow:
                0 26px 65px
                rgba(0, 0, 0, 0.28);
        }


        .hero::before {
            content: "";

            position: absolute;

            width: 320px;
            height: 320px;

            top: -205px;
            right: -45px;

            border-radius: 50%;

            background:
                radial-gradient(
                    circle,
                    rgba(53, 212, 255, 0.32),
                    transparent 68%
                );
        }


        .hero::after {
            content: "";

            position: absolute;

            width: 290px;
            height: 290px;

            bottom: -230px;
            left: 28%;

            border-radius: 50%;

            background:
                radial-gradient(
                    circle,
                    rgba(214, 109, 255, 0.24),
                    transparent 70%
                );
        }


        .hero-inner {
            position: relative;
            z-index: 2;
        }


        .hero-badge {
            display: inline-flex;
            align-items: center;

            gap: 0.4rem;

            padding:
                0.4rem
                0.75rem;

            margin-bottom: 0.9rem;

            border-radius: 999px;

            background:
                rgba(91, 140, 255, 0.14);

            border:
                1px solid
                rgba(108, 155, 255, 0.30);

            color: #a9c8ff;

            font-size: 0.70rem;
            font-weight: 800;

            letter-spacing: 0.08em;
            text-transform: uppercase;
        }


        .hero-heading {
            max-width: 920px;

            color: #ffffff;

            font-size:
                clamp(
                    2.1rem,
                    3.8vw,
                    3.35rem
                );

            line-height: 1.04;

            letter-spacing: -0.05em;

            font-weight: 860;

            margin-bottom: 0.85rem;
        }


        .hero-highlight {
            background:
                linear-gradient(
                    90deg,
                    #a7caff,
                    #70dfff,
                    #d19cff
                );

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;

            background-clip: text;
        }


        .hero-description {
            max-width: 840px;

            color: #c2cde2;

            font-size: 0.98rem;
            line-height: 1.65;

            margin-bottom: 1.15rem;
        }


        .hero-tags {
            display: flex;
            flex-wrap: wrap;

            gap: 0.55rem;
        }


        .hero-tag {
            display: inline-flex;
            align-items: center;

            gap: 0.42rem;

            padding:
                0.43rem
                0.72rem;

            border-radius: 999px;

            background:
                rgba(7, 12, 27, 0.38);

            border:
                1px solid
                rgba(181, 199, 234, 0.13);

            color: #dce7fa;

            font-size: 0.74rem;
            font-weight: 620;
        }


        .hero-tag-check {
            color: #38dda8;
            font-weight: 900;
        }


        /* ==================================================
           KPI CARDS
        ================================================== */

        [data-testid="stMetric"] {
            position: relative;

            overflow: hidden;

            min-height: 110px;

            padding:
                1rem
                1.1rem;

            border-radius: 18px;

            border:
                1px solid
                rgba(137, 163, 214, 0.14);

            background:
                linear-gradient(
                    145deg,
                    rgba(20, 31, 56, 0.95),
                    rgba(12, 21, 39, 0.95)
                );

            box-shadow:
                0 12px 34px
                rgba(0, 0, 0, 0.14);
        }


        [data-testid="stMetric"]::before {
            content: "";

            position: absolute;

            top: 0;
            left: 0;
            right: 0;

            height: 3px;

            background:
                linear-gradient(
                    90deg,
                    #5b8cff,
                    #35d4ff,
                    #986dff
                );
        }


        [data-testid="stMetricLabel"] {
            color: #8f9eb9;
            font-weight: 620;
        }


        [data-testid="stMetricValue"] {
            color: #f7faff;
            font-weight: 820;
        }


        /* ==================================================
           SECTION TEXT
        ================================================== */

        .content-title {
            margin-top: 1.55rem;
            margin-bottom: 0.2rem;

            color: #f5f8ff;

            font-size: 1.08rem;
            font-weight: 790;
        }


        .content-description {
            margin-bottom: 0.9rem;

            color: #7e8da9;

            font-size: 0.82rem;
        }


        /* ==================================================
           BUTTONS
        ================================================== */

        .stButton > button {
            min-height: 50px;

            border-radius: 14px;

            border:
                1px solid
                rgba(135, 161, 216, 0.17);

            background:
                linear-gradient(
                    145deg,
                    rgba(26, 38, 67, 0.97),
                    rgba(17, 27, 50, 0.97)
                );

            color: #e0e9f9;

            font-weight: 650;

            transition:
                transform 0.18s ease,
                border-color 0.18s ease,
                box-shadow 0.18s ease,
                background 0.18s ease;
        }


        .stButton > button:hover {
            transform: translateY(-2px);

            border-color:
                rgba(91, 140, 255, 0.60);

            background:
                linear-gradient(
                    145deg,
                    rgba(38, 55, 96, 0.98),
                    rgba(25, 38, 69, 0.98)
                );

            box-shadow:
                0 12px 28px
                rgba(45, 90, 195, 0.17);

            color: #ffffff;
        }


        .stButton > button:focus,
        .stButton > button:focus-visible {
            outline: none !important;

            border-color:
                rgba(91, 140, 255, 0.80) !important;

            box-shadow:
                0 0 0 3px
                rgba(91, 140, 255, 0.14) !important;
        }


        [data-testid="stSidebar"]
        .stButton > button[kind="primary"] {
            border: none;

            color: white;

            background:
                linear-gradient(
                    90deg,
                    #5b8cff,
                    #916dff
                );

            box-shadow:
                0 12px 30px
                rgba(91, 140, 255, 0.21);
        }


        /* ==================================================
           CONVERSATION
        ================================================== */

        .conversation-header-card {
            margin-top: 1.35rem;

            padding:
                1rem
                1.05rem;

            border:
                1px solid
                rgba(137, 163, 214, 0.12);

            border-radius: 20px;

            background:
                linear-gradient(
                    145deg,
                    rgba(13, 21, 39, 0.62),
                    rgba(10, 17, 31, 0.50)
                );

            box-shadow:
                0 16px 42px
                rgba(0, 0, 0, 0.10);
        }


        .conversation-header-row {
            display: flex;

            align-items: center;
            justify-content: space-between;

            gap: 1rem;

            margin-bottom: 0.15rem;
        }


        .conversation-title {
            color: #f5f8ff;

            font-size: 1.08rem;
            font-weight: 790;
        }


        .conversation-badge {
            padding:
                0.32rem
                0.62rem;

            border-radius: 999px;

            color: #8fb5ff;

            background:
                rgba(91, 140, 255, 0.10);

            border:
                1px solid
                rgba(91, 140, 255, 0.16);

            font-size: 0.68rem;
            font-weight: 700;
        }


        .conversation-description {
            color: #7e8da9;

            font-size: 0.82rem;
        }


        [data-testid="stChatMessage"] {
            padding:
                0.55rem
                0.75rem;

            margin-bottom: 0.9rem;

            border:
                1px solid
                rgba(137, 163, 214, 0.11);

            border-radius: 19px;

            background:
                rgba(17, 27, 49, 0.78);

            box-shadow:
                0 10px 30px
                rgba(0, 0, 0, 0.11);
        }


        [data-testid="stChatInput"] {
            overflow: hidden;

            border-radius: 18px;

            border:
                1px solid
                rgba(91, 140, 255, 0.30);

            background:
                rgba(24, 32, 53, 0.98);

            box-shadow:
                0 18px 50px
                rgba(0, 0, 0, 0.32);
        }


        /* ==================================================
           ASSESSMENT
        ================================================== */

        .assessment-grid {
            display: grid;

            grid-template-columns:
                2fr
                1fr
                1fr;

            gap: 0.75rem;

            margin-bottom: 1rem;
        }


        .assessment-box {
            padding:
                0.9rem
                0.95rem;

            min-height: 88px;

            border-radius: 16px;

            border:
                1px solid
                rgba(137, 163, 214, 0.14);

            background:
                linear-gradient(
                    145deg,
                    rgba(25, 37, 66, 0.94),
                    rgba(16, 25, 46, 0.94)
                );
        }


        .assessment-label {
            margin-bottom: 0.42rem;

            color: #71819e;

            font-size: 0.66rem;
            font-weight: 800;

            letter-spacing: 0.08em;
            text-transform: uppercase;
        }


        .assessment-value {
            color: #edf3ff;

            font-size: 0.88rem;
            font-weight: 700;

            line-height: 1.4;
        }


        .positive-value {
            color: #48dfa9;
        }


        .warning-value {
            color: #ffd166;
        }


        .answer-heading {
            margin-top: 1.05rem;
            margin-bottom: 0.32rem;

            color: #f8faff;

            font-size: 0.92rem;
            font-weight: 770;
        }


        /* ==================================================
           ANALYTICS RESULT
        ================================================== */

        .analytics-box {
            position: relative;

            overflow: hidden;

            padding:
                1rem
                1.1rem;

            margin-bottom: 1rem;

            border-radius: 17px;

            border:
                1px solid
                rgba(91, 140, 255, 0.26);

            background:
                linear-gradient(
                    135deg,
                    rgba(37, 83, 190, 0.20),
                    rgba(89, 64, 184, 0.14)
                );
        }


        .analytics-box::before {
            content: "";

            position: absolute;

            width: 5px;

            top: 0;
            bottom: 0;
            left: 0;

            background:
                linear-gradient(
                    180deg,
                    #35d4ff,
                    #916dff
                );
        }


        .analytics-label {
            margin-bottom: 0.38rem;

            color: #70ceff;

            font-size: 0.68rem;
            font-weight: 800;

            letter-spacing: 0.09em;
            text-transform: uppercase;
        }


        .analytics-value {
            color: #eef5ff;

            font-size: 0.96rem;
            line-height: 1.6;
        }


        /* ==================================================
           SCOPE NOTE
        ================================================== */

        .scope-card {
            display: flex;

            align-items: flex-start;

            gap: 0.7rem;

            margin-top: 1.3rem;

            padding:
                0.88rem
                1rem;

            border-radius: 16px;

            border:
                1px solid
                rgba(137, 163, 214, 0.12);

            background:
                linear-gradient(
                    135deg,
                    rgba(18, 28, 50, 0.77),
                    rgba(12, 20, 38, 0.77)
                );

            color: #8796b1;

            font-size: 0.79rem;
            line-height: 1.55;
        }


        .scope-icon {
            color: #70ceff;
            font-size: 1rem;
        }


        .scope-title {
            color: #bdcbe3;
            font-weight: 750;
        }


        /* ==================================================
           EXPANDERS
        ================================================== */

        [data-testid="stExpander"] {
            border-radius: 14px;

            border:
                1px solid
                rgba(137, 163, 214, 0.11);

            background:
                rgba(13, 21, 38, 0.60);
        }


        [data-testid="stExpander"] summary:focus,
        [data-testid="stExpander"] summary:focus-visible {
            outline: none !important;
            box-shadow: none !important;
        }


        /* ==================================================
           MOBILE
        ================================================== */

        @media (max-width: 800px) {

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }


            .hero {
                padding: 1.4rem;
                border-radius: 20px;
            }


            .hero-heading {
                font-size: 2rem;
            }


            .assessment-grid {
                grid-template-columns: 1fr;
            }

        }

        </style>
        """
    )


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_text(value):

    return html.escape(
        str(value)
    )


def clean_history_for_backend():

    return [
        {
            "role":
                message["role"],

            "content":
                message["content"],
        }

        for message
        in st.session_state.messages
    ]


# ============================================================
# SUPPORT RESULT
# ============================================================

def render_support_result(
    assessment,
):

    product = safe_text(
        assessment[
            "product"
        ]
    )

    evidence_available = bool(
        assessment[
            "evidence_available"
        ]
    )

    human_review = bool(
        assessment[
            "human_review_required"
        ]
    )

    evidence_text = (
        "Available"
        if evidence_available
        else "Insufficient"
    )

    evidence_class = (
        "positive-value"
        if evidence_available
        else "warning-value"
    )

    review_text = (
        "Required"
        if human_review
        else "Not required"
    )

    review_class = (
        "warning-value"
        if human_review
        else "positive-value"
    )

    render_html(
        f"""
        <div class="assessment-grid">

        <div class="assessment-box">
        <div class="assessment-label">
        CFPB product
        </div>
        <div class="assessment-value">
        {product}
        </div>
        </div>

        <div class="assessment-box">
        <div class="assessment-label">
        Evidence
        </div>
        <div class="assessment-value {evidence_class}">
        {evidence_text}
        </div>
        </div>

        <div class="assessment-box">
        <div class="assessment-label">
        Human review
        </div>
        <div class="assessment-value {review_class}">
        {review_text}
        </div>
        </div>

        </div>
        """
    )

    render_html(
        """
        <div class="answer-heading">
        Summary
        </div>
        """
    )

    st.write(
        assessment[
            "summary"
        ]
    )

    render_html(
        """
        <div class="answer-heading">
        Recommended next step
        </div>
        """
    )

    st.write(
        assessment[
            "recommended_action"
        ]
    )


# ============================================================
# ANALYTICS RESULT
# ============================================================

def render_analytics_result(
    result,
):

    render_html(
        f"""
        <div class="analytics-box">

        <div class="analytics-label">
        Verified calculation
        </div>

        <div class="analytics-value">
        {safe_text(result)}
        </div>

        </div>
        """
    )


# ============================================================
# OLD RESPONSE PARSER
# ============================================================

def parse_old_support_response(
    text,
):

    pattern = re.compile(
        r"Product:\s*(.*?)\n\n"
        r"Summary:\s*(.*?)\n\n"
        r"Recommended Action:\s*(.*?)\n\n"
        r"(?:Supporting Evidence Available|Evidence Available):\s*(.*?)\n\n"
        r"Human Review Required:\s*(.*)",
        re.DOTALL,
    )

    match = pattern.match(
        text.strip()
    )

    if not match:
        return None

    return {
        "product":
            match.group(1).strip(),

        "summary":
            match.group(2).strip(),

        "recommended_action":
            match.group(3).strip(),

        "evidence_available":
            (
                match.group(4)
                .strip()
                .lower()
                == "true"
            ),

        "human_review_required":
            (
                match.group(5)
                .strip()
                .lower()
                == "true"
            ),
    }


# ============================================================
# CHAT HISTORY
# ============================================================

def render_history():

    for message in (
        st.session_state.messages
    ):

        role = (
            message[
                "role"
            ]
        )

        avatar = (
            "👤"
            if role == "user"
            else "🤖"
        )

        with st.chat_message(
            role,
            avatar=avatar,
        ):

            if role == "user":

                st.write(
                    message[
                        "content"
                    ]
                )

                continue


            if (
                message.get(
                    "kind"
                )
                == "support"
            ):

                render_support_result(
                    message[
                        "assessment"
                    ]
                )

                continue


            if (
                message.get(
                    "kind"
                )
                == "analysis"
            ):

                render_analytics_result(
                    message[
                        "calculated_result"
                    ]
                )

                render_html(
                    """
                    <div class="answer-heading">
                    What this means
                    </div>
                    """
                )

                st.write(
                    message[
                        "explanation"
                    ]
                )

                continue


            parsed = (
                parse_old_support_response(
                    message[
                        "content"
                    ]
                )
            )

            if parsed:

                render_support_result(
                    parsed
                )

            else:

                st.write(
                    message[
                        "content"
                    ]
                )


# ============================================================
# APPLICATION
# ============================================================

def run_app():

    inject_styles()


    # ========================================================
    # SESSION STATE
    # ========================================================

    if (
        "messages"
        not in st.session_state
    ):

        st.session_state.messages = []


    if (
        "pending_prompt"
        not in st.session_state
    ):

        st.session_state.pending_prompt = None


    # ========================================================
    # DATA
    # ========================================================

    cfpb_data = (
        load_cfpb_data()
    )

    summary = (
        cfpb_data[
            "summary"
        ]
        .iloc[0]
    )

    try:

        knowledge_chunks = (
            load_collection()
            .count()
        )

    except Exception:

        knowledge_chunks = "—"


    sample_records = None

    if (
        "sample_records"
        in summary.index
    ):

        sample_records = int(
            summary[
                "sample_records"
            ]
        )


    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        render_html(
            """
            <div class="brand-icon">
            ✦
            </div>

            <div class="brand-name">
            CFPB Intelligence
            </div>

            <div class="brand-description">
            Grounded consumer complaint guidance
            and complaint-data intelligence.
            </div>
            """
        )


        if st.button(
            "＋  New conversation",
            use_container_width=True,
            type="primary",
        ):

            st.session_state.messages = []

            st.session_state.pending_prompt = None

            st.rerun()


        render_html(
            """
            <div class="system-ready">
            <span class="system-dot"></span>
            <span class="system-ready-text">
            AI system ready
            </span>
            </div>
            """
        )


        # ----------------------------------------------------
        # SYSTEM
        # ----------------------------------------------------

        render_html(
            """
            <div class="sidebar-section-title">
            System
            </div>
            """
        )


        render_html(
            f"""
            <div class="sidebar-information-row">
            <span class="sidebar-information-label">
            Model
            </span>
            <span class="sidebar-information-value">
            {safe_text(MODEL_NAME)}
            </span>
            </div>

            <div class="sidebar-information-row">
            <span class="sidebar-information-label">
            Retrieval
            </span>
            <span class="sidebar-information-value">
            ChromaDB + MiniLM
            </span>
            </div>

            <div class="sidebar-information-row">
            <span class="sidebar-information-label">
            Analytics
            </span>
            <span class="sidebar-information-value">
            Python / Pandas
            </span>
            </div>

            <div class="sidebar-information-row">
            <span class="sidebar-information-label">
            Validation
            </span>
            <span class="sidebar-information-value">
            Pydantic
            </span>
            </div>
            """
        )


        st.divider()


        # ----------------------------------------------------
        # NON-DUPLICATED DATASET DETAILS
        # ----------------------------------------------------

        render_html(
            """
            <div class="sidebar-section-title">
            Dataset details
            </div>
            """
        )


        render_html(
            f"""
            <div class="sidebar-information-row">
            <span class="sidebar-information-label">
            Companies
            </span>
            <span class="sidebar-information-value">
            {int(summary['unique_companies']):,}
            </span>
            </div>
            """
        )


        if (
            sample_records
            is not None
        ):

            render_html(
                f"""
                <div class="sidebar-information-row">
                <span class="sidebar-information-label">
                Record sample
                </span>
                <span class="sidebar-information-value">
                {sample_records:,}
                </span>
                </div>
                """
            )


        # ----------------------------------------------------
        # TRUST CARD
        # ----------------------------------------------------

        render_html(
            """
            <div class="sidebar-trust-card">

            <div class="sidebar-trust-title">
            Grounded by design
            </div>

            <div class="sidebar-trust-text">
            Uses official CFPB material for guidance
            and deterministic calculations for
            complaint statistics.
            </div>

            </div>
            """
        )


        st.divider()


        # ----------------------------------------------------
        # TECHNICAL EXPLANATION
        # ----------------------------------------------------

        with st.expander(
            "How it works"
        ):

            st.markdown(
                "**Routing**"
            )

            st.caption(
                "Requests are automatically directed "
                "to complaint support or deterministic "
                "data analysis."
            )

            st.markdown(
                "**Knowledge retrieval**"
            )

            st.caption(
                "Semantic retrieval uses MiniLM "
                "embeddings and ChromaDB."
            )

            st.markdown(
                "**Analytics**"
            )

            st.caption(
                "Numerical results are calculated "
                "with Python and Pandas."
            )

            st.markdown(
                "**Structured output**"
            )

            st.caption(
                "Pydantic validates complaint "
                "classification and response fields."
            )

            st.markdown(
                "**Knowledge base**"
            )

            st.caption(
                f"{knowledge_chunks} indexed "
                "CFPB document chunks."
            )


    # ========================================================
    # HERO
    # ========================================================

    render_html(
        """
        <div class="hero">
        <div class="hero-inner">

        <div class="hero-badge">
        ✦ Consumer Finance Intelligence
        </div>

        <div class="hero-heading">
        Understand complaints.
        <span class="hero-highlight">
        Explore the data.
        </span>
        </div>

        <div class="hero-description">
        Ask questions about consumer-finance complaints,
        understand CFPB complaint guidance and explore
        more than five million published complaint
        records from 2025.
        </div>

        <div class="hero-tags">

        <span class="hero-tag">
        <span class="hero-tag-check">✓</span>
        Official CFPB evidence
        </span>

        <span class="hero-tag">
        <span class="hero-tag-check">✓</span>
        Verified calculations
        </span>

        <span class="hero-tag">
        <span class="hero-tag-check">✓</span>
        Multi-turn conversation
        </span>

        <span class="hero-tag">
        <span class="hero-tag-check">✓</span>
        Human-review safeguards
        </span>

        </div>

        </div>
        </div>
        """
    )


    # ========================================================
    # MAIN DASHBOARD METRICS
    # ========================================================

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    with metric1:

        st.metric(
            "Published complaints",
            f"{int(summary['total_complaints']):,}",
        )


    with metric2:

        st.metric(
            "Financial products",
            f"{int(summary['unique_products'])}",
        )


    with metric3:

        st.metric(
            "Timely response rate",
            (
                f"{float(summary['timely_response_rate']):.2f}%"
            ),
        )


    with metric4:

        st.metric(
            "Knowledge chunks",
            str(
                knowledge_chunks
            ),
        )


    # ========================================================
    # EXAMPLE PROMPTS
    # ========================================================

    if not st.session_state.messages:

        render_html(
            """
            <div class="content-title">
            What would you like to explore?
            </div>

            <div class="content-description">
            Choose an example or ask your own
            question below.
            </div>
            """
        )


        left, right = (
            st.columns(2)
        )


        with left:

            if st.button(
                "💳  Credit report contains an account that isn't mine",
                use_container_width=True,
            ):

                st.session_state.pending_prompt = (
                    "There is an account on my "
                    "credit report that is not mine. "
                    "What should I do?"
                )

                st.rerun()


            if st.button(
                "📊  Find the busiest complaint month in 2025",
                use_container_width=True,
            ):

                st.session_state.pending_prompt = (
                    "Which month had the highest "
                    "number of published complaints "
                    "in 2025?"
                )

                st.rerun()


        with right:

            if st.button(
                "📞  A collector is contacting me about an unknown debt",
                use_container_width=True,
            ):

                st.session_state.pending_prompt = (
                    "A debt collector keeps contacting "
                    "me about a debt I do not recognize. "
                    "What should I do?"
                )

                st.rerun()


            if st.button(
                "⏱️  Check the CFPB company response timeframe",
                use_container_width=True,
            ):

                st.session_state.pending_prompt = (
                    "How long does a company have "
                    "to provide a timely response "
                    "to a CFPB complaint?"
                )

                st.rerun()


    # ========================================================
    # CONVERSATION
    # ========================================================

    render_html(
        """
        <div class="conversation-header-card">

        <div class="conversation-header-row">

        <div class="conversation-title">
        Ask CFPB Intelligence
        </div>

        <div class="conversation-badge">
        AI ASSISTANT
        </div>

        </div>

        <div class="conversation-description">
        Describe a consumer issue or ask a question
        about the 2025 CFPB complaint dataset.
        </div>

        </div>
        """
    )


    render_history()


    typed_input = st.chat_input(
        "Ask about complaints, CFPB guidance or complaint statistics..."
    )


    if (
        st.session_state.pending_prompt
        is not None
    ):

        user_input = (
            st.session_state.pending_prompt
        )

        st.session_state.pending_prompt = None

    else:

        user_input = typed_input


    # ========================================================
    # PROCESS REQUEST
    # ========================================================

    if user_input:

        previous_messages = (
            clean_history_for_backend()
        )


        st.session_state.messages.append(
            {
                "role":
                    "user",

                "content":
                    user_input,
            }
        )


        with st.chat_message(
            "user",
            avatar="👤",
        ):

            st.write(
                user_input
            )


        try:

            progress = st.status(
                "Understanding your question...",
                expanded=False,
            )


            route = route_request(
                user_input,
                previous_messages,
            )


            # =================================================
            # DATA ANALYSIS
            # =================================================

            if (
                route.route
                == "data_analysis"
            ):

                progress.update(
                    label=(
                        "Calculating from the "
                        "2025 CFPB dataset..."
                    ),
                    state="running",
                )


                analysis = (
                    analyse_cfpb_data(
                        user_input,
                        cfpb_data,
                    )
                )


                if (
                    analysis[
                        "supported"
                    ]
                ):

                    calculated_result = (
                        analysis[
                            "result"
                        ]
                    )


                    progress.update(
                        label=(
                            "Calculation complete. "
                            "Preparing explanation..."
                        ),
                        state="running",
                    )


                    with st.chat_message(
                        "assistant",
                        avatar="🤖",
                    ):

                        render_analytics_result(
                            calculated_result
                        )


                        if (
                            analysis[
                                "table"
                            ]
                            is not None
                        ):

                            with st.expander(
                                "View supporting data"
                            ):

                                st.dataframe(
                                    analysis[
                                        "table"
                                    ],

                                    use_container_width=True,

                                    hide_index=True,
                                )


                        render_html(
                            """
                            <div class="answer-heading">
                            What this means
                            </div>
                            """
                        )


                        explanation = (
                            st.write_stream(
                                stream_analysis_explanation(
                                    user_input,
                                    calculated_result,
                                )
                            )
                        )


                        if not isinstance(
                            explanation,
                            str,
                        ):

                            explanation = str(
                                explanation
                            )


                        st.caption(
                            "The numerical result was "
                            "calculated directly using "
                            "Python/Pandas. The language "
                            "model only explains the "
                            "computed result."
                        )


                        with st.expander(
                            "⚙️ Technical details"
                        ):

                            st.write(
                                "**Capability:** "
                                "Deterministic CFPB "
                                "data analysis"
                            )

                            st.write(
                                f"**Routing reason:** "
                                f"{route.reason}"
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
                                stored_response,

                            "kind":
                                "analysis",

                            "calculated_result":
                                calculated_result,

                            "explanation":
                                explanation,
                        }
                    )


                    progress.update(
                        label="Analysis ready",
                        state="complete",
                    )


                else:

                    response_text = (
                        analysis[
                            "result"
                        ]
                    )


                    st.session_state.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                response_text,
                        }
                    )


                    progress.update(
                        label=(
                            "Calculation not supported"
                        ),
                        state="complete",
                    )


                    with st.chat_message(
                        "assistant",
                        avatar="🤖",
                    ):

                        st.warning(
                            response_text
                        )


            # =================================================
            # RAG / SUPPORT
            # =================================================

            else:

                progress.update(
                    label=(
                        "Searching official "
                        "CFPB guidance..."
                    ),
                    state="running",
                )


                support_result = (
                    run_support_pipeline(
                        user_input,
                        previous_messages,
                    )
                )


                assessment = (
                    support_result[
                        "assessment"
                    ]
                )


                assessment_data = (
                    assessment.model_dump()
                )


                assistant_text = (
                    assessment_to_text(
                        assessment
                    )
                )


                st.session_state.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            assistant_text,

                        "kind":
                            "support",

                        "assessment":
                            assessment_data,

                        "sources":
                            support_result[
                                "sources"
                            ],

                        "best_distance":
                            support_result[
                                "best_distance"
                            ],

                        "retrieval_query":
                            support_result[
                                "retrieval_query"
                            ],

                        "effective_question":
                            support_result[
                                "effective_question"
                            ],
                    }
                )


                progress.update(
                    label="Grounded answer ready",
                    state="complete",
                )


                with st.chat_message(
                    "assistant",
                    avatar="🤖",
                ):

                    render_support_result(
                        assessment_data
                    )


                    sources = (
                        support_result[
                            "sources"
                        ]
                    )


                    if sources:

                        with st.expander(
                            f"📚 Sources & evidence ({len(sources)})"
                        ):

                            for (
                                index,
                                source,
                            ) in enumerate(
                                sources,
                                start=1,
                            ):

                                st.markdown(
                                    f"**{index}.** "
                                    f"{source}"
                                )


                    else:

                        st.warning(
                            "The current CFPB knowledge "
                            "base did not return "
                            "sufficiently relevant evidence. "
                            "The system therefore avoided "
                            "generating an unsupported "
                            "substantive answer."
                        )


                    with st.expander(
                        "⚙️ Technical details"
                    ):

                        st.write(
                            "**Capability:** "
                            "CFPB Knowledge Retrieval / RAG"
                        )


                        st.write(
                            f"**Routing reason:** "
                            f"{route.reason}"
                        )


                        effective_question = (
                            support_result[
                                "effective_question"
                            ]
                        )


                        if (
                            effective_question
                            != user_input
                        ):

                            st.write(
                                "**Conversation context "
                                "applied:** Yes"
                            )

                            st.code(
                                effective_question
                            )


                        st.write(
                            "**Retrieval query**"
                        )

                        st.code(
                            support_result[
                                "retrieval_query"
                            ]
                        )


                        best_distance = (
                            support_result[
                                "best_distance"
                            ]
                        )


                        if (
                            best_distance
                            is not None
                        ):

                            st.write(
                                f"**Best retrieval distance:** "
                                f"{best_distance:.4f}"
                            )

                            st.write(
                                f"**Acceptance threshold:** "
                                f"{MAX_TOP_DISTANCE:.2f}"
                            )

                            st.write(
                                f"**Chunk threshold:** "
                                f"{MAX_CHUNK_DISTANCE:.2f}"
                            )


                        st.write(
                            "**Validated structured output**"
                        )

                        st.json(
                            assessment_data
                        )


        except Exception as error:

            if (
                "progress"
                in locals()
            ):

                progress.update(
                    label=(
                        "The request could not "
                        "be completed"
                    ),
                    state="error",
                )


            st.error(
                "Something went wrong while "
                "processing the request. "
                "Confirm that Ollama is running "
                "and try again."
            )


            with st.expander(
                "Technical error"
            ):

                st.write(
                    error
                )


    # ========================================================
    # SCOPE NOTE
    # ========================================================

    render_html(
        """
        <div class="scope-card">

        <div class="scope-icon">
        ⓘ
        </div>

        <div>

        <span class="scope-title">
        Data scope.
        </span>

        Analytics represent records published in the
        CFPB Consumer Complaint Database for 2025.
        Guidance is informational and grounded in
        official CFPB materials loaded into this
        application.

        </div>

        </div>
        """
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_app()