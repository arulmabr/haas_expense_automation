import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import base64
from datetime import datetime
from typing import Dict, List, Optional
import gspread
from google.oauth2.service_account import Credentials
import openai
import os
from dataclasses import dataclass
import logging
import time
import PyPDF2
import fitz
import re
import io
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.exceptions import ClientError

# Handle different Streamlit versions
try:
    from streamlit.runtime.scriptrunner.script_runner import RerunException
    from streamlit.runtime.scriptrunner.exceptions import StopException
except ImportError:
    try:
        from streamlit.script_runner import RerunException

        StopException = None
    except ImportError:
        # For newer Streamlit versions, RerunException may not be needed
        RerunException = None
        StopException = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Haas Expense Report Automation",
    page_icon="🐻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add Haas UC Berkeley branding with custom CSS
st.markdown(
    """
<style>
    /* Berkeley Blue and California Gold colors */
    :root {
        --berkeley-blue: #003262;
        --california-gold: #FDB515;
        --founders-rock: #3B7EA1;
        --pacific-blue: #46535E;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--berkeley-blue) 0%, var(--founders-rock) 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .haas-subtitle {
        color: var(--california-gold);
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--berkeley-blue);
        color: white;
        border: 2px solid var(--california-gold);
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: var(--founders-rock);
        border-color: var(--california-gold);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--berkeley-blue);
        color: white;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--california-gold);
        color: var(--berkeley-blue);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--berkeley-blue) 0%, var(--pacific-blue) 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background-color: rgba(253, 181, 21, 0.1);
        border-left: 5px solid var(--california-gold);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--berkeley-blue);
        font-weight: 700;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Pydantic model for structured output from OpenAI
class ExpenseExtraction(BaseModel):
    """Schema for structured expense extraction from OpenAI"""

    amount: float
    currency: str
    description: str
    date: str
    category: str
    confidence: float
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    # UC Berkeley specific fields
    expense_type: Optional[str] = None  # TRANSPORTATION, MISCELLANEOUS, DAILY
    meal_type: Optional[str] = None  # BREAKFAST, LUNCH, DINNER, INCIDENTAL
    destination: Optional[str] = None  # Extracted location/city


@dataclass
class ExpenseData:
    """Data structure for expense information"""

    amount: float
    currency: str
    description: str
    date: str
    category: str
    filename: str
    confidence: float = 0.0
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    # UC Berkeley specific fields
    expense_type: Optional[str] = None  # TRANSPORTATION, MISCELLANEOUS, DAILY
    meal_type: Optional[str] = None  # BREAKFAST, LUNCH, DINNER, INCIDENTAL
    destination: Optional[str] = None  # Extracted location/city


# Constants - UC Berkeley Travel Reimbursement Categories
EXPENSE_CATEGORIES = {
    # Transportation (Location Expenses)
    "AIRFARE": "Airfare",
    "AIRFARE_CHANGE_FEE": "Airfare Change Fee",
    "RENTAL_CAR": "Rental Car",
    "PERSONAL_VEHICLE": "Personal Vehicle",
    "GROUND_TRANSPORT": "Other Ground Transportation",
    # Miscellaneous (Location Expenses)
    "CONFERENCE_FEE": "Conference/Event Registration",
    "SUPPLIES": "Business Meeting Supplies",
    "OTHER_MISC": "Other Business Expenses",
    # Daily Expenses
    "MEAL": "Meal",
    "LODGING": "Lodging",
}

# Category groupings for UC Berkeley form
TRANSPORTATION_CATEGORIES = [
    "AIRFARE",
    "AIRFARE_CHANGE_FEE",
    "RENTAL_CAR",
    "PERSONAL_VEHICLE",
    "GROUND_TRANSPORT",
]
MISCELLANEOUS_CATEGORIES = ["CONFERENCE_FEE", "SUPPLIES", "OTHER_MISC"]
DAILY_CATEGORIES = ["MEAL", "LODGING"]

# Meal types for daily expenses
MEAL_TYPES = ["BREAKFAST", "LUNCH", "DINNER", "INCIDENTAL"]

# Trip duration options
TRIP_DURATIONS = [
    "total travel is less than 30 days",
    "total travel is 30 days or more",
]

# Trip legs options
TRIP_LEGS = ["One Destination", "Multiple Destinations"]

CURRENCY_OPTIONS = [
    "USD",
    "EUR",
    "GBP",
    "CAD",
    "AUD",
    "JPY",
    "CHF",
    "CNY",
    "INR",
    "MXN",
    "BRL",
]


# PII patterns for redaction
PII_PATTERNS = {
    "email": re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'),
    "us_phone": re.compile(r'(?<!\d)(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}(?!\d)'),
    "intl_phone": re.compile(r'(?<!\d)\+\d{1,3}[\s\-.]?\d{1,4}[\s\-.]?\d{2,4}[\s\-.]?\d{2,4}(?:[\s\-.]?\d{2,4})?(?!\d)'),
}

# Personal email domains to redact — business/service/institutional emails are preserved
PERSONAL_EMAIL_DOMAINS = re.compile(
    r'@(?:gmail|yahoo|hotmail|outlook|aol|icloud|protonmail|zoho|ymail|live|msn|me|mac)\.',
    re.IGNORECASE,
)

# Lines that look like amounts/prices — skip these during label-based address redaction
AMOUNT_LINE_PATTERN = re.compile(r'^\s*[\$€£¥]?\s*\d+[.,]\d{2}\s*$')

ADDRESS_LABELS = re.compile(
    r'(?:'
    r'Pick\s*-?\s*up|Drop\s*-?\s*off|Pickup|Dropoff'  # Ride-sharing
    r'|(?:Billing|Shipping|Mailing|Property|Hotel|Departure|Arrival|From|To)\s+Address'
    r'|(?:Check.?in|Check.?out)\s+(?:at|location|address)'
    r'|Location|Destination|Address|Venue'
    r')',
    re.IGNORECASE,
)

ADDRESS_PATTERNS = [
    # US street address: 123 N Main St, Suite 100, City, CA 90210
    re.compile(
        r'\d{1,6}\s+(?:[NSEW]\.?\s+)?(?:[A-Z][a-zA-Z]*\.?\s+){1,4}'
        r'(?:St(?:reet)?|Ave(?:nue)?|Blvd|Boulevard|Dr(?:ive)?|Ln|Lane|Rd|Road|Way|Ct|Court'
        r'|Pl(?:ace)?|Pkwy|Parkway|Cir(?:cle)?|Ter(?:race)?|Hwy|Highway)\b'
        r'(?:\.?,?\s*(?:Ste|Suite|Apt|Unit|#)\s*\d+[A-Za-z]?)?'
        r'(?:\.?,?\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)?'
        r'(?:\.?,?\s+[A-Z]{2})?'
        r'(?:\.?,?\s+\d{5}(?:\-\d{4})?)?',
        re.IGNORECASE,
    ),
    # City, State ZIP (standalone line)
    re.compile(
        r'[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s+\d{5}(?:\-\d{4})?',
    ),
    # PO Box
    re.compile(
        r'P\.?O\.?\s*Box\s+\d+',
        re.IGNORECASE,
    ),
    # International postal codes (UK, Canada, Australia, etc.)
    re.compile(
        r'[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}',  # UK: SW1A 1AA
    ),
    re.compile(
        r'[A-Z]\d[A-Z]\s*\d[A-Z]\d',  # Canada: K1A 0B1
    ),
]


class ExpenseReportApp:
    def __init__(self):
        logger.info("Initializing ExpenseReportApp")
        self.setup_session_state()
        logger.info("ExpenseReportApp initialized successfully")

    def setup_session_state(self):
        """Initialize session state variables"""
        logger.info("Setting up session state")
        if "expenses" not in st.session_state:
            st.session_state.expenses = []
        if "metadata" not in st.session_state:
            st.session_state.metadata = {}
        if "processing_complete" not in st.session_state:
            st.session_state.processing_complete = False
        if "additional_context" not in st.session_state:
            st.session_state.additional_context = ""
        if "include_external_emails" not in st.session_state:
            st.session_state.include_external_emails = False
        if "use_ai_business_purpose" not in st.session_state:
            st.session_state.use_ai_business_purpose = False
        if "uploaded_files_data" not in st.session_state:
            st.session_state.uploaded_files_data = {}
        if "auto_correction_done" not in st.session_state:
            st.session_state.auto_correction_done = False
        if "submission_success" not in st.session_state:
            st.session_state.submission_success = False
        if "s3_upload_message" not in st.session_state:
            st.session_state.s3_upload_message = None
        if "s3_presigned_url" not in st.session_state:
            st.session_state.s3_presigned_url = None
        if "merged_pdf_bytes" not in st.session_state:
            st.session_state.merged_pdf_bytes = None
        logger.info("Session state setup complete")

    def is_valid_email(self, email: str, allow_external: bool = False) -> bool:
        """
        Validate email domain based on Berkeley policy.

        Args:
            email: Email address to validate
            allow_external: If True, allows non-Berkeley emails (for external guests)

        Returns:
            True if email is valid according to policy, False otherwise
        """
        if not email:
            return False

        # Always allow Berkeley emails
        if email.lower().endswith("@berkeley.edu"):
            return True

        # Allow non-Berkeley emails only if external guests flag is set
        return allow_external

    def detect_duplicate_expenses(self, expenses: List[ExpenseData]) -> List[tuple]:
        """
        Detect potential duplicate expenses based on amount and date.

        Returns:
            List of tuples containing pairs of potentially duplicate expenses
        """
        duplicates = []
        seen = []

        for exp in expenses:
            for other in seen:
                # Check if same amount and same date (potential duplicate)
                if (
                    abs(exp.amount - other.amount) < 0.01
                    and exp.date == other.date
                    and exp.currency == other.currency
                ):
                    duplicates.append((other, exp))
            seen.append(exp)

        return duplicates

    def get_openai_client(self):
        """Initialize OpenAI client"""
        logger.info("Getting OpenAI client")
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found")
            st.error(
                "OpenAI API key not found. Please set OPENAI_API_KEY in secrets or environment variables."
            )
            return None
        try:
            logger.info("Creating OpenAI client")
            client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client created successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
            return None

    def get_google_sheets_client(self):
        """Initialize Google Sheets client"""
        try:
            # Try to get credentials from Streamlit secrets
            if "google_credentials" in st.secrets:
                creds_dict = dict(st.secrets["google_credentials"])
                # Define the required scopes for Google Sheets
                scopes = [
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive",
                ]
                creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            else:
                # Fallback to environment variable
                creds_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if not creds_file:
                    st.error(
                        "Google credentials not found. Please configure google_credentials in secrets."
                    )
                    return None
                scopes = [
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive",
                ]
                creds = Credentials.from_service_account_file(creds_file, scopes=scopes)

            return gspread.authorize(creds)
        except Exception as e:
            st.error(f"Failed to initialize Google Sheets client: {str(e)}")
            return None

    def extract_text_from_pdf(self, pdf_file) -> Optional[str]:
        """Extract text from PDF file using PyPDF2"""
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip() if text.strip() else None
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {str(e)}")
            return None

    def auto_correct_category(
        self, description: str, category: str, meal_type: Optional[str]
    ) -> tuple:
        """
        Minimal validation of category - relies on GPT for proper categorization.
        Only maps category to expense_type for grouping purposes.
        """
        # Normalize category to match expected values
        category_mapping = {
            "MEAL": ("Meal", "DAILY"),
            "Meal": ("Meal", "DAILY"),
            "LODGING": ("Lodging", "DAILY"),
            "Lodging": ("Lodging", "DAILY"),
            "AIRFARE": ("Airfare", "TRANSPORTATION"),
            "Airfare": ("Airfare", "TRANSPORTATION"),
            "AIRFARE_CHANGE_FEE": ("Airfare Change Fee", "TRANSPORTATION"),
            "Airfare Change Fee": ("Airfare Change Fee", "TRANSPORTATION"),
            "RENTAL_CAR": ("Rental Car", "TRANSPORTATION"),
            "Rental Car": ("Rental Car", "TRANSPORTATION"),
            "PERSONAL_VEHICLE": ("Personal Vehicle", "TRANSPORTATION"),
            "Personal Vehicle": ("Personal Vehicle", "TRANSPORTATION"),
            "GROUND_TRANSPORT": ("Other Ground Transportation", "TRANSPORTATION"),
            "Other Ground Transportation": ("Other Ground Transportation", "TRANSPORTATION"),
            "CONFERENCE_FEE": ("Conference/Event Registration", "MISCELLANEOUS"),
            "Conference/Event Registration": ("Conference/Event Registration", "MISCELLANEOUS"),
            "SUPPLIES": ("Business Meeting Supplies", "MISCELLANEOUS"),
            "Business Meeting Supplies": ("Business Meeting Supplies", "MISCELLANEOUS"),
            "OTHER_MISC": ("Other Business Expenses", "MISCELLANEOUS"),
            "Other Business Expenses": ("Other Business Expenses", "MISCELLANEOUS"),
        }

        if category in category_mapping:
            normalized_category, expense_type = category_mapping[category]
            return normalized_category, expense_type, meal_type

        # Default to miscellaneous if category not recognized
        logger.warning(f"Unrecognized category '{category}' for: {description}")
        return "Other Business Expenses", "MISCELLANEOUS", meal_type

    def get_expense_analysis_prompt(self, context: str = "") -> str:
        """Get the standard expense analysis prompt for GPT - aligned with UC Berkeley Travel Reimbursement"""
        base_prompt = """Analyze this expense document and extract the following information in JSON format.
This is for UC Berkeley's Travel Reimbursement system.

REQUIRED FIELDS:
1. amount: the total amount paid (as a number, no currency symbols)
2. currency: the currency code (USD, EUR, CAD, etc.) - default to USD if unclear
3. description: a brief description of what this expense is for
4. date: the transaction date in YYYY-MM-DD format. For hotels, use the check-in date
5. category: categorize into ONE of these EXACT UC Berkeley categories:

   *** CATEGORIZATION PRIORITY (check in this order) ***

   FIRST - Check if it's FOOD/DINING (category="MEAL"):
   - Any restaurant, cafe, bar, coffee shop, fast food, food court
   - Any food or beverage purchase
   - Credit card transactions at eating establishments
   - Includes: breakfast, lunch, dinner, snacks, drinks
   - Even if purchased during travel, food is ALWAYS "MEAL", never transportation

   SECOND - Check if it's LODGING (category="LODGING"):
   - Hotels, motels, Airbnb, hostels, any overnight accommodation

   THIRD - Check if it's a MEMBERSHIP/REGISTRATION FEE (category="CONFERENCE_FEE"):
   - Professional association memberships
   - Conference registration fees
   - Event registration fees
   - Society dues or memberships

   FOURTH - Check if it's TRANSPORTATION:
   - "AIRFARE" - ONLY for airline flights and tickets
   - "AIRFARE_CHANGE_FEE" - flight change/cancellation fees
   - "RENTAL_CAR" - car rentals only
   - "PERSONAL_VEHICLE" - mileage, gas for personal car
   - "GROUND_TRANSPORT" - taxi, Uber, Lyft, train, bus, parking, tolls

   FIFTH - Everything else:
   - "SUPPLIES" - business meeting supplies
   - "OTHER_MISC" - other business expenses

6. confidence: a confidence score from 0.0 to 1.0

7. expense_type: classify as one of:
   - "DAILY" - for meals and lodging
   - "MISCELLANEOUS" - for conference fees, supplies, other
   - "TRANSPORTATION" - for airfare, car rental, ground transport

8. meal_type (only for MEAL category): one of "BREAKFAST", "LUNCH", "DINNER", or "INCIDENTAL"

9. destination: Extract the city/location for this expense

PERSONAL INFORMATION (extract if visible):
10. first_name: Extract from passenger name, guest name, cardholder
11. last_name: Extract from same locations
12. email: Extract any email address visible

CRITICAL CATEGORIZATION RULES:
- Food/restaurant purchases are ALWAYS "MEAL", even if they appear on a travel day
- Professional memberships (like ASA, IEEE, ACM) are "CONFERENCE_FEE", not transportation
- Only actual flights are "AIRFARE" - not food bought at airports
- When in doubt about food vs other: choose "MEAL"
- When in doubt about membership vs other: choose "CONFERENCE_FEE"

Return ONLY valid JSON with these fields, nothing else."""

        if context and context.strip():
            context_addition = f"\n\nADDITIONAL CONTEXT PROVIDED BY USER:\n{context.strip()}\n\nUse this context to better understand the purpose and nature of the expenses, but still extract the specific details from the document itself."
            return base_prompt + context_addition

        return base_prompt

    def analyze_expense_with_gpt_image(
        self, image_file, filename: str, context: str = "", max_retries: int = 3
    ) -> Optional[ExpenseData]:
        """Analyze expense image directly using GPT-5 vision API"""
        client = self.get_openai_client()
        if not client:
            return None

        for attempt in range(max_retries):
            try:
                # Read and encode image
                image_file.seek(0)
                image_bytes = image_file.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                # Determine image format
                image_format = "jpeg"
                if filename.lower().endswith(".png"):
                    image_format = "png"
                elif filename.lower().endswith(".gif"):
                    image_format = "gif"
                elif filename.lower().endswith(".webp"):
                    image_format = "webp"

                # Use structured outputs for reliable JSON extraction
                response = client.beta.chat.completions.parse(
                    model="gpt-5.2",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert expense document analyzer. Extract information accurately from receipts, invoices, and expense documents.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.get_expense_analysis_prompt(context),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{image_format};base64,{image_base64}"
                                    },
                                },
                            ],
                        },
                    ],
                    response_format=ExpenseExtraction,
                )

                # Get the parsed structured output
                extracted_data = response.choices[0].message.parsed

                if not extracted_data:
                    if attempt < max_retries - 1:
                        st.warning(
                            f"Empty response from GPT-5 for {filename}, retrying... (Attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(2)
                        continue
                    else:
                        st.error(
                            f"Empty response from GPT-5 after {max_retries} attempts"
                        )
                        return None

                # Log extracted personal information
                if (
                    extracted_data.first_name
                    or extracted_data.last_name
                    or extracted_data.email
                ):
                    logger.info(
                        f"✅ Extracted personal info from {filename}: "
                        f"first_name={extracted_data.first_name}, "
                        f"last_name={extracted_data.last_name}, "
                        f"email={extracted_data.email}"
                    )
                else:
                    logger.warning(f"⚠️ No personal info extracted from {filename}")

                # Auto-correct category based on description
                description = extracted_data.description or "Unknown expense"
                category = extracted_data.category or "OTHER_MISC"
                corrected_category, corrected_expense_type, corrected_meal_type = (
                    self.auto_correct_category(
                        description, category, extracted_data.meal_type
                    )
                )

                return ExpenseData(
                    filename=filename,
                    description=description,
                    amount=float(extracted_data.amount or 0),
                    currency=extracted_data.currency or "USD",
                    date=extracted_data.date or datetime.now().strftime("%Y-%m-%d"),
                    category=corrected_category,
                    confidence=float(extracted_data.confidence or 0),
                    first_name=extracted_data.first_name,
                    last_name=extracted_data.last_name,
                    email=extracted_data.email,
                    expense_type=corrected_expense_type,
                    meal_type=corrected_meal_type,
                    destination=extracted_data.destination,
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(
                        f"Error processing {filename}, retrying... (Attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    time.sleep(2)
                    continue
                else:
                    st.error(
                        f"Error with GPT-5 image processing after {max_retries} attempts: {str(e)}"
                    )
                    return None

        return None  # All retries exhausted

    def analyze_expense_with_gpt_direct_pdf(
        self, pdf_file, filename: str, context: str = "", max_retries: int = 3
    ) -> Optional[ExpenseData]:
        """Analyze PDF directly using GPT-5 vision without text extraction"""
        client = self.get_openai_client()
        if not client:
            return None

        for attempt in range(max_retries):
            try:
                # Upload the PDF file to OpenAI
                # Read the file content and create a proper file object
                pdf_file.seek(0)  # Reset file pointer
                file_content = pdf_file.read()

                # Validate file content
                if not file_content or len(file_content) == 0:
                    raise ValueError(f"File {filename} is empty or could not be read")

                logger.info(f"Read {len(file_content)} bytes from {filename}")

                # Create a BytesIO object with proper name attribute
                file_obj = io.BytesIO(file_content)
                file_obj.name = filename

                uploaded_file = client.files.create(file=file_obj, purpose="user_data")
                logger.info(
                    f"Successfully uploaded {filename} to OpenAI (file_id: {uploaded_file.id})"
                )

                # Use structured outputs for reliable JSON extraction
                response = client.beta.chat.completions.parse(
                    model="gpt-5.2",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert expense document analyzer. Extract information accurately from receipts, invoices, and expense documents.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "file",
                                    "file": {"file_id": uploaded_file.id},
                                },
                                {
                                    "type": "text",
                                    "text": self.get_expense_analysis_prompt(context),
                                },
                            ],
                        },
                    ],
                    response_format=ExpenseExtraction,
                )

                extracted_data = response.choices[0].message.parsed

                # Clean up the uploaded file
                try:
                    client.files.delete(uploaded_file.id)
                except Exception:
                    pass  # File cleanup failed, but continue

                if not extracted_data:
                    if attempt < max_retries - 1:
                        st.warning(
                            f"Empty response from GPT-5 for {filename}, retrying... (Attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(2)
                        continue
                    else:
                        st.error(
                            f"Empty response from GPT-5 after {max_retries} attempts"
                        )
                        return None

                # Log extracted personal information
                if (
                    extracted_data.first_name
                    or extracted_data.last_name
                    or extracted_data.email
                ):
                    logger.info(
                        f"✅ Extracted personal info from {filename}: "
                        f"first_name={extracted_data.first_name}, "
                        f"last_name={extracted_data.last_name}, "
                        f"email={extracted_data.email}"
                    )
                else:
                    logger.warning(f"⚠️ No personal info extracted from {filename}")

                # Auto-correct category based on description
                description = extracted_data.description or "Unknown expense"
                category = extracted_data.category or "OTHER_MISC"
                corrected_category, corrected_expense_type, corrected_meal_type = (
                    self.auto_correct_category(
                        description, category, extracted_data.meal_type
                    )
                )

                return ExpenseData(
                    amount=float(extracted_data.amount or 0),
                    currency=extracted_data.currency or "USD",
                    description=description,
                    date=extracted_data.date or datetime.now().strftime("%Y-%m-%d"),
                    category=corrected_category,
                    filename=filename,
                    confidence=float(extracted_data.confidence or 0.5),
                    first_name=extracted_data.first_name,
                    last_name=extracted_data.last_name,
                    email=extracted_data.email,
                    expense_type=corrected_expense_type,
                    meal_type=corrected_meal_type,
                    destination=extracted_data.destination,
                )

            except Exception as e:
                logger.error(
                    f"Error processing {filename} (Attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    st.warning(
                        f"Error processing {filename}, retrying... (Attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    time.sleep(2)
                    continue
                else:
                    st.error(
                        f"Error with direct PDF processing after {max_retries} attempts: {str(e)}"
                    )
                    logger.error(f"Failed to process {filename} after all retries")
                    return None

        return None  # All retries exhausted

    def analyze_expense_with_gpt_fallback(
        self, text: str, filename: str, context: str = ""
    ) -> Optional[ExpenseData]:
        """Fallback method: Analyze expense text using GPT-5 (for non-PDF files)"""
        client = self.get_openai_client()
        if not client:
            return None

        prompt = f"""{self.get_expense_analysis_prompt(context)}

Document text:
{text}"""

        try:
            response = client.beta.chat.completions.parse(
                model="gpt-5.2",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert expense document analyzer. Extract information accurately from receipts, invoices, and expense documents.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=ExpenseExtraction,
            )

            extracted_data = response.choices[0].message.parsed

            if not extracted_data:
                st.error("Empty response from GPT-5")
                return None

            # Log extracted personal information
            if (
                extracted_data.first_name
                or extracted_data.last_name
                or extracted_data.email
            ):
                logger.info(
                    f"✅ Extracted personal info from {filename}: "
                    f"first_name={extracted_data.first_name}, "
                    f"last_name={extracted_data.last_name}, "
                    f"email={extracted_data.email}"
                )
            else:
                logger.warning(f"⚠️ No personal info extracted from {filename}")

            # Auto-correct category based on description
            description = extracted_data.description or "Unknown expense"
            category = extracted_data.category or "OTHER_MISC"
            corrected_category, corrected_expense_type, corrected_meal_type = (
                self.auto_correct_category(
                    description, category, extracted_data.meal_type
                )
            )

            return ExpenseData(
                amount=float(extracted_data.amount or 0),
                currency=extracted_data.currency or "USD",
                description=description,
                date=extracted_data.date or datetime.now().strftime("%Y-%m-%d"),
                category=corrected_category,
                filename=filename,
                confidence=float(extracted_data.confidence or 0.5),
                first_name=extracted_data.first_name,
                last_name=extracted_data.last_name,
                email=extracted_data.email,
                expense_type=corrected_expense_type,
                meal_type=corrected_meal_type,
                destination=extracted_data.destination,
            )

        except Exception as e:
            st.error(f"Error calling OpenAI API: {str(e)}")
            logger.error(f"Error in fallback processing: {str(e)}")
            return None

    def get_exchange_rates(self, currencies: List[str], _date: str) -> Dict[str, float]:
        """Get exchange rates with USD as base currency"""
        rates = {"USD": 1.0}

        # For Haas School of Business, USD is the default currency
        # If other currencies are needed, they can be added here
        # For now, we'll just return 1.0 for all currencies since USD is default
        for currency in currencies:
            if currency == "USD":
                rates[currency] = 1.0
            else:
                # For non-USD currencies, you could integrate with a different API
                # For now, defaulting to 1.0 (manual conversion expected)
                rates[currency] = 1.0
                st.info(
                    f"Exchange rate for {currency} set to 1.0. Manual conversion may be needed."
                )

        return rates

    def auto_prefill_event_info(self, expenses: List[ExpenseData], context: str = ""):
        """Auto-prefill event information from extracted expense data"""
        if not expenses:
            return

        # Generate event name from common patterns
        event_name = self.generate_event_name_from_expenses(expenses, context)

        # Generate business purpose from expense descriptions
        business_purpose = self.generate_business_purpose_from_expenses(
            expenses, context
        )

        # Get date range from expenses
        dates = [
            datetime.strptime(exp.date, "%Y-%m-%d") for exp in expenses if exp.date
        ]
        if dates:
            start_date = min(dates).date()
            end_date = max(dates).date()
        else:
            start_date = datetime.now().date()
            end_date = datetime.now().date()

        # Calculate trip duration
        trip_days = (end_date - start_date).days
        trip_duration = TRIP_DURATIONS[1] if trip_days >= 30 else TRIP_DURATIONS[0]

        # Extract unique destinations from expenses
        destinations_set = set()
        for exp in expenses:
            if exp.destination:
                destinations_set.add(exp.destination.strip())

        destinations_list = sorted(list(destinations_set))
        destinations_str = ", ".join(destinations_list) if destinations_list else ""

        # Determine number of trip legs
        num_trip_legs = TRIP_LEGS[1] if len(destinations_list) > 1 else TRIP_LEGS[0]

        # Auto-prefill metadata if not already set
        if not st.session_state.metadata:
            # Extract name and email from GPT-extracted data
            first_name = ""
            last_name = ""
            email = ""

            # Get the allow_external flag from session state
            allow_external = st.session_state.get("include_external_emails", False)

            # Find the first expense with valid personal information
            for exp in expenses:
                if exp.first_name and not first_name:
                    first_name = exp.first_name
                if exp.last_name and not last_name:
                    last_name = exp.last_name
                if exp.email and not email:
                    # Only use email if it passes domain validation
                    if self.is_valid_email(exp.email, allow_external):
                        email = exp.email
                    else:
                        logger.info(
                            f"Filtered out non-Berkeley email: {exp.email} (allow_external={allow_external})"
                        )
                # Break early if we have all three
                if first_name and last_name and email:
                    break

            st.session_state.metadata = {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "event_name": event_name,
                "description": business_purpose,
                "start_date": start_date,
                "end_date": end_date,
                "trip_duration": trip_duration,
                "destinations": destinations_str,
                "num_trip_legs": num_trip_legs,
                "currencies": ["USD"],  # Default to USD
                "exchange_rates": {"USD": 1.0},
                "median_date": start_date.strftime(
                    "%Y-%m-%d"
                ),  # Use start date as median
            }
        else:
            # Update existing metadata with extracted info
            # Extract personal information from expenses if not already set
            allow_external = st.session_state.get("include_external_emails", False)

            if not st.session_state.metadata.get("first_name"):
                for exp in expenses:
                    if exp.first_name:
                        st.session_state.metadata["first_name"] = exp.first_name
                        break

            if not st.session_state.metadata.get("last_name"):
                for exp in expenses:
                    if exp.last_name:
                        st.session_state.metadata["last_name"] = exp.last_name
                        break

            if not st.session_state.metadata.get("email"):
                for exp in expenses:
                    if exp.email and self.is_valid_email(exp.email, allow_external):
                        st.session_state.metadata["email"] = exp.email
                        break
                    elif exp.email:
                        logger.info(
                            f"Filtered out non-Berkeley email: {exp.email} (allow_external={allow_external})"
                        )

            if not st.session_state.metadata.get("event_name"):
                st.session_state.metadata["event_name"] = event_name
            if not st.session_state.metadata.get("description"):
                st.session_state.metadata["description"] = business_purpose
            if not st.session_state.metadata.get("start_date"):
                st.session_state.metadata["start_date"] = start_date
            if not st.session_state.metadata.get("end_date"):
                st.session_state.metadata["end_date"] = end_date
            if not st.session_state.metadata.get("trip_duration"):
                st.session_state.metadata["trip_duration"] = trip_duration
            if not st.session_state.metadata.get("destinations"):
                st.session_state.metadata["destinations"] = destinations_str
            if not st.session_state.metadata.get("num_trip_legs"):
                st.session_state.metadata["num_trip_legs"] = num_trip_legs

    def generate_event_name_from_expenses(
        self, expenses: List[ExpenseData], context: str = ""
    ) -> str:
        """Generate event name from expense patterns"""
        if not expenses:
            return "Business Event"

        # If context is provided, prioritize extracting event name from it
        if context and context.strip():
            context_lower = context.lower()
            # Check for common event name indicators in context
            conference_keywords = [
                "conference",
                "meeting",
                "summit",
                "workshop",
                "seminar",
                "training",
            ]
            for keyword in conference_keywords:
                if keyword in context_lower:
                    # Try to extract the event name from the sentence containing the keyword
                    sentences = context.split(".")
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            # Use the first 50 characters of the sentence as the event name
                            return sentence.strip()[:50]

        # Look for common business event patterns in expense descriptions
        descriptions = [exp.description.lower() for exp in expenses if exp.description]

        # Check for conference/meeting patterns
        conference_keywords = [
            "conference",
            "meeting",
            "summit",
            "workshop",
            "seminar",
            "training",
        ]
        for desc in descriptions:
            for keyword in conference_keywords:
                if keyword in desc:
                    return desc.title()

        # Check for travel patterns
        travel_keywords = ["hotel", "flight", "airfare", "travel", "trip"]
        if any(
            any(keyword in desc for keyword in travel_keywords) for desc in descriptions
        ):
            return "Business Travel"

        # Check for meal patterns
        meal_keywords = ["restaurant", "dining", "meal", "lunch", "dinner", "breakfast"]
        if any(
            any(keyword in desc for keyword in meal_keywords) for desc in descriptions
        ):
            return "Business Meals"

        # Default based on most common category
        categories = [exp.category for exp in expenses]
        if categories:
            most_common = max(set(categories), key=categories.count)
            if "AIRFARE" in most_common:
                return "Business Travel"
            elif "ACCOMMODATION" in most_common:
                return "Hotel Stay"
            elif "MEALS" in most_common:
                return "Business Meals"

        return "Business Event"

    def generate_business_purpose_with_ai(
        self, expenses: List[ExpenseData], context: str = ""
    ) -> str:
        """
        Use AI to generate a compliant business purpose following UC Berkeley best practices.
        """
        if not expenses:
            return "Business expenses"

        client = self.get_openai_client()
        if not client:
            # Fallback to simple generation if OpenAI not available
            return self.generate_business_purpose_simple(expenses, context)

        # Extract locations and dates from expenses
        locations = set()
        dates = []
        categories = []

        for exp in expenses:
            categories.append(exp.category)
            if exp.date:
                dates.append(exp.date)
            # Try to extract location from description
            if exp.description:
                # Simple heuristic: look for city/location patterns
                if "to " in exp.description.lower():
                    parts = exp.description.split("to ")
                    if len(parts) > 1:
                        locations.add(parts[-1].strip())
                if " in " in exp.description.lower():
                    parts = exp.description.split(" in ")
                    if len(parts) > 1:
                        locations.add(parts[-1].strip())

        # Build context for AI
        expense_summary = f"""
Trip Details:
- Number of expenses: {len(expenses)}
- Categories: {', '.join(set(categories))}
- Date range: {min(dates) if dates else 'N/A'} to {max(dates) if dates else 'N/A'}
- Possible locations: {', '.join(locations) if locations else 'Not specified'}
- Sample expenses: {', '.join([exp.description for exp in expenses[:3] if exp.description])}

User-provided context: {context if context else 'None provided'}
"""

        prompt = f"""You are writing a business purpose statement for a UC Berkeley Haas School of Business travel expense report.

BEST PRACTICES FOR BUSINESS PURPOSE:
1. Focus on what was ACCOMPLISHED and how UC Berkeley BENEFITED
2. Use action verbs like "attending" (not "registering for"), "presenting", "collaborating", "conducting research"
3. Explain why each leg of the trip was taken and what was accomplished
4. Be specific but concise (maximum 200 characters)
5. Focus on the academic/professional benefit to the university

EXAMPLES OF GOOD BUSINESS PURPOSES:
- "Attending Annual Economics Conference in Boston to present research and network with colleagues from peer institutions"
- "Collaborating with MIT researchers on behavioral finance project and presenting findings at Harvard seminar"
- "Conducting field research in New York and meeting with industry partners to advance curriculum development"

{expense_summary}

Generate a business purpose statement (maximum 200 characters) that:
- Follows the best practices above
- Explains how UC Berkeley benefited from this travel
- Uses the user's context if provided, but reframes it in proper business purpose language
- Is professional and suitable for university financial compliance

Return ONLY the business purpose statement, nothing else."""

        try:
            response = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at writing compliant business purpose statements for university expense reports.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=100,
            )

            business_purpose = response.choices[0].message.content.strip()

            # Ensure it's within 200 characters
            if len(business_purpose) > 200:
                business_purpose = business_purpose[:197] + "..."

            return business_purpose

        except Exception as e:
            logger.error(f"Error generating business purpose with AI: {str(e)}")
            # Fallback to simple generation
            return self.generate_business_purpose_simple(expenses, context)

    def generate_business_purpose_simple(
        self, expenses: List[ExpenseData], context: str = ""
    ) -> str:
        """Simple fallback business purpose generation (non-AI)"""
        if not expenses:
            return "Business expenses"

        # If context is provided, use it as the primary business purpose
        if context and context.strip():
            # Use the context directly, truncated to a reasonable length
            return (
                context.strip()[:200] if len(context.strip()) > 200 else context.strip()
            )

        # Get unique descriptions (first 3)
        descriptions = list(
            set(exp.description for exp in expenses if exp.description)
        )[:3]

        if len(descriptions) == 1:
            return f"Business expense: {descriptions[0]}"
        elif len(descriptions) == 2:
            return f"Business expenses: {descriptions[0]} and {descriptions[1]}"
        else:
            return f"Business expenses: {', '.join(descriptions)} and {len(expenses) - 3} other items"

    def generate_business_purpose_from_expenses(
        self, expenses: List[ExpenseData], context: str = ""
    ) -> str:
        """Generate business purpose from expense descriptions - uses AI if enabled"""
        # Check if user wants AI-powered generation
        use_ai = st.session_state.get("use_ai_business_purpose", False)

        if use_ai:
            # Use AI-powered generation for better compliance (slower)
            return self.generate_business_purpose_with_ai(expenses, context)
        else:
            # Use simple/fast generation (default)
            return self.generate_business_purpose_simple(expenses, context)

    def extract_name_from_expenses(self, expenses):
        """Try to extract name from expense documents"""
        # Look for common name patterns in descriptions
        for expense in expenses:
            if expense.description:
                # Look for patterns like "Invoice for John Doe" or "Payment to Jane Smith"
                import re

                name_patterns = [
                    r"(?:for|to|from)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
                    r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:invoice|receipt|payment)",
                    r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:expense|travel)",
                ]

                for pattern in name_patterns:
                    match = re.search(pattern, expense.description, re.IGNORECASE)
                    if match:
                        full_name = match.group(1).strip()
                        name_parts = full_name.split()
                        if len(name_parts) >= 2:
                            return {
                                "first_name": name_parts[0],
                                "last_name": " ".join(name_parts[1:]),
                            }

        return {"first_name": "", "last_name": ""}

    def extract_email_from_expenses(self, expenses):
        """Try to extract email from expense documents"""
        # Look for email patterns in descriptions
        for expense in expenses:
            if expense.description:
                import re

                email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                match = re.search(email_pattern, expense.description)
                if match:
                    return match.group(0)

        return ""

    def render_inline_event_form(self):
        """Render UC Berkeley Travel Reimbursement aligned form"""
        with st.form("inline_metadata_form"):
            # === SECTION 1: TRAVELER INFO ===
            st.markdown("#### 1. Traveler Info")
            col1, col2 = st.columns(2)

            with col1:
                first_name = st.text_input(
                    "First Name*", value=st.session_state.metadata.get("first_name", "")
                )
                last_name = st.text_input(
                    "Last Name*", value=st.session_state.metadata.get("last_name", "")
                )

            with col2:
                email = st.text_input(
                    "Email*", value=st.session_state.metadata.get("email", "")
                )

            st.markdown("---")

            # === SECTION 2: TRIP INFO ===
            st.markdown("#### 2. Trip Info")

            description = st.text_area(
                "Business Purpose* (max 200 characters)",
                value=st.session_state.metadata.get("description", ""),
                max_chars=200,
                help="Explain how UC Berkeley benefited from this travel. Focus on what was ACCOMPLISHED.",
            )

            col1, col2 = st.columns(2)
            with col1:
                trip_duration = st.selectbox(
                    "Trip Duration*",
                    options=TRIP_DURATIONS,
                    index=(
                        TRIP_DURATIONS.index(
                            st.session_state.metadata.get(
                                "trip_duration", TRIP_DURATIONS[0]
                            )
                        )
                        if st.session_state.metadata.get("trip_duration")
                        in TRIP_DURATIONS
                        else 0
                    ),
                )
                num_trip_legs = st.selectbox(
                    "Number of Trip Legs*",
                    options=TRIP_LEGS,
                    index=(
                        TRIP_LEGS.index(
                            st.session_state.metadata.get("num_trip_legs", TRIP_LEGS[0])
                        )
                        if st.session_state.metadata.get("num_trip_legs") in TRIP_LEGS
                        else 0
                    ),
                )

            with col2:
                destinations = st.text_input(
                    "Trip Destination(s)*",
                    value=st.session_state.metadata.get("destinations", ""),
                    help="City and state/country (e.g., 'Boston, MA' or 'London, UK')",
                )
                event_name = st.text_input(
                    "Event/Conference Name",
                    value=st.session_state.metadata.get("event_name", ""),
                    help="Name of conference or event attended",
                )

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Travel Start Date*",
                    value=st.session_state.metadata.get(
                        "start_date", datetime.now().date()
                    ),
                )
            with col2:
                end_date = st.date_input(
                    "Travel End Date*",
                    value=st.session_state.metadata.get(
                        "end_date", datetime.now().date()
                    ),
                )

            st.markdown("---")

            # === CURRENCIES ===
            non_usd_options = [c for c in CURRENCY_OPTIONS if c != "USD"]
            current_currencies = st.session_state.metadata.get("currencies", [])
            default_currencies = [c for c in current_currencies if c != "USD"]

            currencies = st.multiselect(
                "Additional Currencies (if expenses are not in USD)",
                options=non_usd_options,
                default=default_currencies,
            )

            submitted = st.form_submit_button(
                "💾 Save Travel Information", type="primary"
            )

            if submitted:
                # Validate required fields - collect all missing fields
                missing_fields = []
                if not first_name:
                    missing_fields.append("First Name")
                if not last_name:
                    missing_fields.append("Last Name")
                if not email:
                    missing_fields.append("Email")
                if not destinations:
                    missing_fields.append("Trip Destination(s)")
                if not description:
                    missing_fields.append("Business Purpose")

                # Validate email domain
                allow_external = st.session_state.get("include_external_emails", False)
                email_valid = True
                if email and not self.is_valid_email(email, allow_external):
                    email_valid = False

                if missing_fields:
                    # Show all missing fields at once
                    st.error("⚠️ **Missing Required Fields:**")
                    for field in missing_fields:
                        st.markdown(f"- {field}")
                    st.info("Please fill in all fields marked with *")
                elif not email_valid:
                    st.error(
                        f"⚠️ Email '{email}' is not a Berkeley email address (@berkeley.edu). "
                        "If this is for an external guest/speaker, please check the "
                        "'Include external guest/speaker expenses' checkbox above."
                    )
                else:
                    median_date = start_date
                    all_currencies = ["USD"] + currencies

                    with st.spinner("Fetching exchange rates..."):
                        exchange_rates = self.get_exchange_rates(
                            all_currencies, median_date.strftime("%Y-%m-%d")
                        )

                    st.session_state.metadata = {
                        # Traveler Info
                        "first_name": first_name,
                        "last_name": last_name,
                        "email": email,
                        # Trip Info
                        "event_name": event_name,
                        "description": description,
                        "trip_duration": trip_duration,
                        "num_trip_legs": num_trip_legs,
                        "destinations": destinations,
                        "start_date": start_date,
                        "end_date": end_date,
                        # Currency
                        "currencies": all_currencies,
                        "exchange_rates": exchange_rates,
                        "median_date": median_date.strftime("%Y-%m-%d"),
                    }

                    st.toast("✅ Travel information saved! Go to the Review tab to check your expenses.")

    def submit_to_google_sheets(
        self, expenses: List[ExpenseData], metadata: Dict
    ) -> bool:
        """Submit data to Google Sheets - UC Berkeley Travel Reimbursement format"""
        client = self.get_google_sheets_client()
        if not client:
            return False

        try:
            sheet_id = st.secrets.get("GOOGLE_SHEET_ID") or os.getenv("GOOGLE_SHEET_ID")
            if not sheet_id:
                st.error("Google Sheet ID not configured")
                return False

            spreadsheet = client.open_by_key(sheet_id)

            # Group expenses by category type and calculate totals
            transport_total = 0.0
            misc_total = 0.0
            meals_total = 0.0
            lodging_total = 0.0

            transportation_expenses = []
            miscellaneous_expenses = []
            meal_expenses = []
            lodging_expenses = []

            for expense in expenses:
                exchange_rate = metadata.get("exchange_rates", {}).get(
                    expense.currency, 1.0
                )
                amount_usd = expense.amount * exchange_rate

                # Find the category key
                cat_key = None
                for key, val in EXPENSE_CATEGORIES.items():
                    if val == expense.category or key == expense.category:
                        cat_key = key
                        break

                if cat_key in TRANSPORTATION_CATEGORIES:
                    transport_total += amount_usd
                    transportation_expenses.append((expense, amount_usd))
                elif cat_key in MISCELLANEOUS_CATEGORIES:
                    misc_total += amount_usd
                    miscellaneous_expenses.append((expense, amount_usd))
                elif cat_key == "MEAL":
                    meals_total += amount_usd
                    meal_expenses.append((expense, amount_usd))
                elif cat_key == "LODGING":
                    lodging_total += amount_usd
                    lodging_expenses.append((expense, amount_usd))
                else:
                    misc_total += amount_usd
                    miscellaneous_expenses.append((expense, amount_usd))

            grand_total = transport_total + misc_total + meals_total + lodging_total

            # Sheet1: Summary in UC Berkeley order
            sheet1 = spreadsheet.sheet1

            # Convert dates to strings
            start_date = metadata.get("start_date", "")
            end_date = metadata.get("end_date", "")
            if hasattr(start_date, "strftime"):
                start_date = start_date.strftime("%Y-%m-%d")
            if hasattr(end_date, "strftime"):
                end_date = end_date.strftime("%Y-%m-%d")

            # UC Berkeley ordered summary row
            summary_row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 1. Timestamp
                metadata.get("first_name", ""),  # 2. First Name
                metadata.get("last_name", ""),  # 3. Last Name
                metadata.get("email", ""),  # 4. Email
                metadata.get("description", ""),  # 5. Business Purpose
                metadata.get("trip_duration", ""),  # 6. Trip Duration
                metadata.get("destinations", ""),  # 7. Destinations
                start_date,  # 8. Start Date
                end_date,  # 9. End Date
                round(transport_total, 2),  # 10. Transportation Total
                round(misc_total, 2),  # 11. Miscellaneous Total
                round(meals_total, 2),  # 12. Meals Total
                round(lodging_total, 2),  # 13. Lodging Total
                round(grand_total, 2),  # 14. Grand Total
            ]

            sheet1.append_row(summary_row)

            # Details sheet: Individual expenses organized by section
            try:
                details_sheet = spreadsheet.worksheet("Details")
            except Exception:
                details_sheet = spreadsheet.add_worksheet(
                    title="Details", rows="10000", cols="20"
                )
                # Add headers matching UC Berkeley format
                headers = [
                    "Timestamp",
                    "Name",
                    "Section",
                    "Category",
                    "Description",
                    "Date",
                    "Amount",
                    "Currency",
                    "Amount USD",
                    "Meal Type",
                    "Destination",
                    "Filename",
                    "Confidence",
                ]
                details_sheet.append_row(headers)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            name = f"{metadata.get('first_name', '')} {metadata.get('last_name', '')}".strip()

            # Helper function to add expense rows
            def add_expense_row(expense, amount_usd, section):
                row_data = [
                    timestamp,
                    name,
                    section,
                    expense.category,
                    expense.description,
                    expense.date,
                    expense.amount,
                    expense.currency,
                    round(amount_usd, 2),
                    expense.meal_type or "",
                    expense.destination or "",
                    expense.filename,
                    expense.confidence,
                ]
                details_sheet.append_row(row_data)

            # Add Transportation expenses
            for expense, amount_usd in transportation_expenses:
                add_expense_row(expense, amount_usd, "TRANSPORTATION")

            # Add Miscellaneous expenses
            for expense, amount_usd in miscellaneous_expenses:
                add_expense_row(expense, amount_usd, "MISCELLANEOUS")

            # Add Meals (sorted by date)
            for expense, amount_usd in sorted(meal_expenses, key=lambda x: x[0].date):
                add_expense_row(expense, amount_usd, "DAILY - MEALS")

            # Add Lodging (sorted by date)
            for expense, amount_usd in sorted(
                lodging_expenses, key=lambda x: x[0].date
            ):
                add_expense_row(expense, amount_usd, "DAILY - LODGING")

            return True

        except Exception as e:
            st.error(f"Error submitting to Google Sheets: {str(e)}")
            logger.error(f"Google Sheets error: {str(e)}")
            return False

    def get_s3_client(self):
        """Initialize AWS S3 client"""
        try:
            aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID") or os.getenv(
                "AWS_ACCESS_KEY_ID"
            )
            aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY") or os.getenv(
                "AWS_SECRET_ACCESS_KEY"
            )
            aws_region = st.secrets.get("AWS_REGION") or os.getenv(
                "AWS_REGION", "us-west-2"
            )

            if not aws_access_key or not aws_secret_key:
                logger.error("AWS credentials not found")
                return None

            return boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
            )
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            return None

    def merge_files_to_pdf(self, files_data: Dict[str, bytes]) -> Optional[bytes]:
        """
        Merge all PDFs and images into a single PDF.

        Args:
            files_data: dict of {filename: bytes}

        Returns:
            Combined PDF as bytes, or None if failed
        """
        from PIL import Image
        from PyPDF2 import PdfReader, PdfWriter

        try:
            pdf_writer = PdfWriter()

            # Sort files by name for consistent ordering
            for filename in sorted(files_data.keys()):
                file_bytes = files_data[filename]

                if filename.lower().endswith(".pdf"):
                    # Add PDF pages directly
                    pdf_reader = PdfReader(io.BytesIO(file_bytes))
                    for page in pdf_reader.pages:
                        pdf_writer.add_page(page)
                    logger.info(
                        f"Added PDF to merge: {filename} ({len(pdf_reader.pages)} pages)"
                    )

                elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    # Convert image to PDF page
                    image = Image.open(io.BytesIO(file_bytes))

                    # Convert to RGB if necessary (for PNG with transparency)
                    if image.mode in ("RGBA", "LA", "P"):
                        background = Image.new("RGB", image.size, (255, 255, 255))
                        if image.mode == "P":
                            image = image.convert("RGBA")
                        background.paste(
                            image,
                            mask=image.split()[-1] if image.mode == "RGBA" else None,
                        )
                        image = background

                    # Save image as PDF to bytes
                    img_pdf_bytes = io.BytesIO()
                    image.save(img_pdf_bytes, "PDF", resolution=100.0)
                    img_pdf_bytes.seek(0)

                    # Add image PDF to writer
                    img_pdf_reader = PdfReader(img_pdf_bytes)
                    for page in img_pdf_reader.pages:
                        pdf_writer.add_page(page)
                    logger.info(f"Added image to merge: {filename}")

            # Write combined PDF to bytes
            output = io.BytesIO()
            pdf_writer.write(output)
            output.seek(0)
            logger.info(
                f"Merged {len(files_data)} files into single PDF ({len(pdf_writer.pages)} pages)"
            )
            return output.read()

        except Exception as e:
            logger.error(f"Failed to merge files to PDF: {str(e)}")
            return None

    def redact_pii_from_pdf(self, pdf_bytes: bytes) -> bytes:
        """
        Redact PII (emails, phone numbers, addresses) from a PDF's text layer.
        Returns redacted PDF bytes. Image-only pages pass through unchanged.
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            redaction_count = 0

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if not page_text.strip():
                    continue  # Skip image-only pages with no text layer

                # Redact personal emails and phone numbers
                for pattern_name, pattern in PII_PATTERNS.items():
                    for match in pattern.finditer(page_text):
                        matched_text = match.group()
                        # Only redact emails from personal domains
                        if pattern_name == "email" and not PERSONAL_EMAIL_DOMAINS.search(matched_text):
                            continue
                        rects = page.search_for(matched_text)
                        for rect in rects:
                            page.add_redact_annot(rect, fill=(0, 0, 0))
                            redaction_count += 1

                # Redact addresses (ride locations, street addresses, etc.)
                self._redact_addresses(page)

                # Apply all redactions on this page
                page.apply_redactions()

            logger.info(f"PII redaction complete: {redaction_count} items redacted across {len(doc)} pages")
            redacted_bytes = doc.tobytes()
            doc.close()
            return redacted_bytes

        except Exception as e:
            logger.error(f"PII redaction failed, returning original PDF: {str(e)}")
            return pdf_bytes  # Fail safe: return unredacted PDF

    def _redact_addresses(self, page):
        """
        Redact addresses from a PDF page.
        - Label-based: detects address-related labels and redacts text on the same + next line
        - Pattern-based: matches US street addresses, City/State/ZIP, PO Boxes, international postal codes
        """
        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])

        # Collect all text lines with their positions
        all_lines = []
        for block in blocks:
            if block.get("type") != 0:  # text blocks only
                continue
            for line in block.get("lines", []):
                line_text = ""
                line_bbox = None
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                    span_bbox = fitz.Rect(span["bbox"])
                    if line_bbox is None:
                        line_bbox = span_bbox
                    else:
                        line_bbox |= span_bbox  # union
                if line_text.strip() and line_bbox:
                    all_lines.append({"text": line_text, "bbox": line_bbox})

        # Label-based: find address labels and redact text on the same + next line
        for i, line_info in enumerate(all_lines):
            if ADDRESS_LABELS.search(line_info["text"]):
                # Redact address text on the same line (after the label)
                label_match = ADDRESS_LABELS.search(line_info["text"])
                after_label = line_info["text"][label_match.end():]
                if after_label.strip():
                    rects = page.search_for(after_label.strip())
                    for rect in rects:
                        page.add_redact_annot(rect, fill=(0, 0, 0))

                # Redact the next line (likely the address continuation)
                # Skip if it looks like an amount/price or another label
                if i + 1 < len(all_lines):
                    next_text = all_lines[i + 1]["text"].strip()
                    if (next_text
                            and not ADDRESS_LABELS.search(next_text)
                            and not AMOUNT_LINE_PATTERN.match(next_text)):
                        rects = page.search_for(next_text)
                        for rect in rects:
                            page.add_redact_annot(rect, fill=(0, 0, 0))

        # Pattern-based: match addresses anywhere on the page
        page_text = page.get_text()
        for pattern in ADDRESS_PATTERNS:
            for match in pattern.finditer(page_text):
                matched_text = match.group()
                rects = page.search_for(matched_text)
                for rect in rects:
                    page.add_redact_annot(rect, fill=(0, 0, 0))

    def upload_files_to_s3(self, files_data: Dict[str, bytes], merged_pdf: bytes = None) -> tuple:
        """
        Upload merged PDF to AWS S3. Uses pre-generated merged_pdf if provided, otherwise generates it.

        Args:
            files_data: dict of {filename: bytes}
            merged_pdf: Optional pre-generated merged PDF bytes

        Returns:
            (success: bool, uploaded_count: int, error_message: str or None, s3_key: str or None, merged_pdf: bytes or None)
        """
        if not files_data and not merged_pdf:
            return True, 0, None, None, None

        # Get bucket name from secrets
        bucket_name = st.secrets.get("S3_BUCKET_NAME") or os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            logger.info("S3 bucket not configured, skipping file upload")
            return True, 0, None, None, None  # Silent skip if not configured

        s3_client = self.get_s3_client()
        if not s3_client:
            return False, 0, "Failed to initialize S3 client", None, None

        # Use pre-generated merged PDF or generate new one
        if not merged_pdf:
            merged_pdf = self.merge_files_to_pdf(files_data)
            if merged_pdf:
                merged_pdf = self.redact_pii_from_pdf(merged_pdf)
        if not merged_pdf:
            return False, 0, "Failed to merge files into PDF", None, None

        try:
            # Create S3 key with date and timestamp
            today = datetime.now().strftime("%Y-%m-%d")
            timestamp = datetime.now().strftime("%H%M%S")
            s3_key = f"{today}/{timestamp}_expense_receipts.pdf"

            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=merged_pdf,
                ContentType="application/pdf",
            )

            logger.info(f"Uploaded merged PDF to S3: {s3_key}")
            return True, len(files_data), None, s3_key, merged_pdf

        except ClientError as e:
            error_msg = f"Failed to upload merged PDF: {str(e)}"
            logger.error(error_msg)
            return False, 0, error_msg, None, None

    def generate_presigned_url(self, s3_key: str, expiration: int = 604800) -> str:
        """
        Generate a presigned URL for an S3 object.

        Args:
            s3_key: The S3 object key
            expiration: URL expiration time in seconds (default: 7 days = 604800 seconds)

        Returns:
            Presigned URL string or None if failed
        """
        bucket_name = st.secrets.get("S3_BUCKET_NAME") or os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            return None

        s3_client = self.get_s3_client()
        if not s3_client:
            return None

        try:
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": s3_key},
                ExpiresIn=expiration,
            )
            logger.info(f"Generated presigned URL for {s3_key}, expires in {expiration} seconds")
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {str(e)}")
            return None

    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        logger.info("Rendering sidebar")

        # Haas branded sidebar header
        st.sidebar.markdown(
            """
        <div style="text-align: center; padding: 1rem 0; border-bottom: 3px solid #FDB515; margin-bottom: 1rem;">
            <h2 style="color: #FDB515; margin: 0; font-size: 1.5rem;">🐻 Haas FSU</h2>
            <p style="color: #FDB515; font-size: 0.9rem; margin: 0.3rem 0 0 0;">Expense Automation</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.sidebar.title("🔧 Configuration")

        # API Status
        st.sidebar.subheader("API Status")

        # Check OpenAI - only check if API key exists, don't initialize
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            st.sidebar.success("✅ OpenAI API Key Found")
            logger.info("OpenAI API key found")
        else:
            st.sidebar.error("❌ OpenAI API Key Missing")
            logger.warning("OpenAI API key missing")

        # Check Google Sheets - only check if credentials exist, don't initialize
        if "google_credentials" in st.secrets or os.path.exists(
            "google-credentials.json"
        ):
            st.sidebar.success("✅ Google Sheets Credentials Found")
            logger.info("Google Sheets credentials found")
        else:
            st.sidebar.warning("⚠️ Google Sheets Credentials Missing")
            logger.warning("Google Sheets credentials missing")

        st.sidebar.markdown("---")
        logger.info("Sidebar rendered successfully")

        # Clear data button
        if st.sidebar.button("🗑️ Clear All Data"):
            st.session_state.expenses = []
            st.session_state.metadata = {}
            st.session_state.processing_complete = False
            st.session_state.show_event_info = False
            st.session_state.uploaded_files_data = {}
            st.session_state.auto_correction_done = False
            st.session_state.submission_success = False
            st.session_state.s3_upload_message = None
            st.session_state.s3_presigned_url = None
            st.session_state.merged_pdf_bytes = None
            st.rerun()

    def render_metadata_form(self):
        """Render the metadata collection form"""
        st.header("📝 Event Information")

        # Show helpful message if expenses are already processed
        if st.session_state.expenses:
            st.success(
                f"✅ {len(st.session_state.expenses)} expense(s) already processed! Event details have been auto-prefilled from your documents."
            )

        with st.form("metadata_form"):
            col1, col2 = st.columns(2)

            with col1:
                first_name = st.text_input(
                    "First Name*", value=st.session_state.metadata.get("first_name", "")
                )
                last_name = st.text_input(
                    "Last Name*", value=st.session_state.metadata.get("last_name", "")
                )
                email = st.text_input(
                    "Email*", value=st.session_state.metadata.get("email", "")
                )

            with col2:
                event_name = st.text_input(
                    "Event Name*", value=st.session_state.metadata.get("event_name", "")
                )
                start_date = st.date_input(
                    "Start Date*",
                    value=st.session_state.metadata.get(
                        "start_date", datetime.now().date()
                    ),
                )
                end_date = st.date_input(
                    "End Date*",
                    value=st.session_state.metadata.get(
                        "end_date", datetime.now().date()
                    ),
                )

            description = st.text_area(
                "Business Purpose / Event Description (max 200 characters)",
                value=st.session_state.metadata.get("description", ""),
                max_chars=200,
                help="Explain how UC Berkeley benefited from this travel. Focus on what was ACCOMPLISHED (e.g., 'Attending conference to present research' not 'Registering for conference'). Include why each leg of the trip was taken.",
            )

            # Filter out USD from both options and defaults
            non_usd_options = [c for c in CURRENCY_OPTIONS if c != "USD"]
            current_currencies = st.session_state.metadata.get("currencies", [])
            # Remove USD from defaults if it exists
            default_currencies = [c for c in current_currencies if c != "USD"]

            currencies = st.multiselect(
                "Currencies (other than USD)",
                options=non_usd_options,
                default=default_currencies,
            )

            submitted = st.form_submit_button("Save Event Information")

            if submitted:
                if first_name and last_name and email and event_name:
                    # Validate email domain
                    allow_external = st.session_state.get(
                        "include_external_emails", False
                    )
                    if not self.is_valid_email(email, allow_external):
                        st.error(
                            f"⚠️ Email '{email}' is not a Berkeley email address (@berkeley.edu). "
                            "If this is for an external guest/speaker, please check the "
                            "'Include external guest/speaker expenses' checkbox above."
                        )
                    else:
                        median_date = start_date  # Use start date as median
                        all_currencies = ["USD"] + currencies

                        with st.spinner("Fetching exchange rates..."):
                            exchange_rates = self.get_exchange_rates(
                                all_currencies, median_date.strftime("%Y-%m-%d")
                            )

                        st.session_state.metadata = {
                            "first_name": first_name,
                            "last_name": last_name,
                            "email": email,
                            "event_name": event_name,
                            "start_date": start_date,
                            "end_date": end_date,
                            "description": description,
                            "currencies": all_currencies,
                            "exchange_rates": exchange_rates,
                            "median_date": median_date.strftime("%Y-%m-%d"),
                        }

                        st.success("✅ Event information saved!")
                        # Streamlit will automatically rerun after form submission
                else:
                    st.error("Please fill in all required fields (marked with *)")

    def render_file_upload(self):
        """Render the file upload and processing section"""
        st.header("📎 Upload Expense Documents")

        # Show a note about event information
        if not st.session_state.metadata:
            st.info(
                "💡 You can upload documents first, then fill in event details. The AI will extract expense information from your documents."
            )

        # Additional context text area
        st.markdown("#### 📝 Additional Context (Optional)")
        st.text_area(
            "Provide any additional details about this trip or event",
            value=st.session_state.additional_context,
            placeholder="Example: Attended the Annual Economics Conference in Boston. Met with research collaborators from MIT and Harvard to discuss ongoing project on behavioral finance...",
            help="This information helps the AI better understand the purpose of your expenses and generate more accurate business purpose descriptions.",
            height=100,
            key="additional_context",
        )

        # Email privacy settings
        st.checkbox(
            "Include external guest/speaker expenses (allows non-Berkeley emails)",
            value=st.session_state.include_external_emails,
            help="By default, only @berkeley.edu emails are extracted to protect privacy. Check this if submitting expenses for external speakers or guests from other universities.",
            key="include_external_emails",
        )

        # AI business purpose toggle
        st.checkbox(
            "🤖 Use AI to generate compliant business purpose (slower, more accurate)",
            value=st.session_state.use_ai_business_purpose,
            help="Enable this to use AI (GPT-5) to generate a UC Berkeley-compliant business purpose. Disabled by default for faster processing.",
            key="use_ai_business_purpose",
        )

        uploaded_files = st.file_uploader(
            "Choose PDF or image files",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload receipts, invoices, or other expense documents",
        )

        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary"):
                self.process_uploaded_files(uploaded_files)

        # Show inline event form if expenses are processed
        if st.session_state.get("expenses") and st.session_state.get(
            "processing_complete"
        ):
            st.markdown("---")
            st.markdown("### 📝 **Complete Your Event Information**")
            st.info(
                "💡 **Personal details** (name, email) and **Event details** should be auto-extracted from your documents. Please verify and update if needed!"
            )

            # Render the inline event form
            self.render_inline_event_form()

    def process_single_file(self, file, context: str = ""):
        """Process a single file and return expense data"""
        try:
            # Use GPT-5 direct processing for both PDFs and images
            if file.type == "application/pdf":
                # Check file size before processing
                file.seek(0)
                file_content = file.read()
                file.seek(0)  # Reset for processing

                if len(file_content) == 0:
                    return (
                        None,
                        f"❌ {file.name} is empty. Please check the file and try again.",
                        "error",
                    )

                logger.info(f"Processing PDF {file.name} ({len(file_content)} bytes)")
                expense_data = self.analyze_expense_with_gpt_direct_pdf(
                    file, file.name, context
                )
                if expense_data:
                    success_msg = (
                        f"✅ Processed {file.name} with GPT-5 direct PDF analysis"
                    )
                    # Add personal info to message
                    if (
                        expense_data.first_name
                        or expense_data.last_name
                        or expense_data.email
                    ):
                        personal_parts = []
                        if expense_data.first_name or expense_data.last_name:
                            name = f"{expense_data.first_name or ''} {expense_data.last_name or ''}".strip()
                            personal_parts.append(f"Name: {name}")
                        if expense_data.email:
                            personal_parts.append(f"Email: {expense_data.email}")
                        success_msg += f" | 📋 Extracted: {', '.join(personal_parts)}"
                    return expense_data, success_msg, "success"
                else:
                    # Fallback: Try text extraction method
                    text = self.extract_text_from_pdf(file)
                    if text:
                        expense_data = self.analyze_expense_with_gpt_fallback(
                            text, file.name, context
                        )
                        if expense_data:
                            success_msg = f"✅ Processed {file.name} with text extraction fallback"
                            if (
                                expense_data.first_name
                                or expense_data.last_name
                                or expense_data.email
                            ):
                                personal_parts = []
                                if expense_data.first_name or expense_data.last_name:
                                    name = f"{expense_data.first_name or ''} {expense_data.last_name or ''}".strip()
                                    personal_parts.append(f"Name: {name}")
                                if expense_data.email:
                                    personal_parts.append(
                                        f"Email: {expense_data.email}"
                                    )
                                success_msg += (
                                    f" | 📋 Extracted: {', '.join(personal_parts)}"
                                )
                            return expense_data, success_msg, "success"
                        else:
                            return (
                                None,
                                f"❌ Failed to analyze {file.name} even with text extraction",
                                "error",
                            )
                    else:
                        return (
                            None,
                            f"❌ Could not extract text from {file.name}",
                            "error",
                        )
            else:
                # For images, use GPT-5 vision directly
                expense_data = self.analyze_expense_with_gpt_image(
                    file, file.name, context
                )
                if expense_data:
                    success_msg = f"✅ Processed {file.name} with GPT-5 vision"
                    if (
                        expense_data.first_name
                        or expense_data.last_name
                        or expense_data.email
                    ):
                        personal_parts = []
                        if expense_data.first_name or expense_data.last_name:
                            name = f"{expense_data.first_name or ''} {expense_data.last_name or ''}".strip()
                            personal_parts.append(f"Name: {name}")
                        if expense_data.email:
                            personal_parts.append(f"Email: {expense_data.email}")
                        success_msg += f" | 📋 Extracted: {', '.join(personal_parts)}"
                    return expense_data, success_msg, "success"
                else:
                    return None, f"❌ Failed to analyze {file.name}", "error"

        except Exception as e:
            return None, f"❌ Error processing {file.name}: {str(e)}", "error"

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files in parallel and extract expense data"""
        # Reset auto-correction flag for new uploads
        st.session_state.auto_correction_done = False

        progress_bar = st.progress(0)
        status_text = st.empty()

        processed_expenses = []
        total_files = len(uploaded_files)
        completed_count = 0

        # Create a container for status messages
        message_container = st.container()

        # Get context once
        context = st.session_state.additional_context

        # Store file bytes for later S3 upload
        for file in uploaded_files:
            file.seek(0)
            st.session_state.uploaded_files_data[file.name] = file.read()
            file.seek(0)  # Reset for processing

        # Process files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(5, total_files)) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self.process_single_file, file, context): file
                for file in uploaded_files
            }

            # Process results as they complete
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                completed_count += 1

                # Update progress
                progress = completed_count / total_files
                progress_bar.progress(progress)
                status_text.text(
                    f"Processing complete: {completed_count}/{total_files} files"
                )

                # Get result from future
                expense_data, message, status = future.result()

                # Display message based on status
                with message_container:
                    if status == "success" and expense_data:
                        processed_expenses.append(expense_data)
                        st.success(message)
                    elif status == "error":
                        st.error(message)
                    else:
                        st.warning(message)

        # Update session state
        st.session_state.expenses.extend(processed_expenses)
        st.session_state.processing_complete = True

        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")

        if processed_expenses:
            st.success(f"Successfully processed {len(processed_expenses)} documents!")

            # Auto-prefill event information from extracted data
            self.auto_prefill_event_info(
                processed_expenses, st.session_state.additional_context
            )

            # Show what personal information was extracted
            extracted_info = []
            if st.session_state.metadata.get("first_name"):
                extracted_info.append(
                    f"👤 Name: {st.session_state.metadata['first_name']} "
                    f"{st.session_state.metadata.get('last_name', '')}"
                )
            if st.session_state.metadata.get("email"):
                extracted_info.append(f"📧 Email: {st.session_state.metadata['email']}")

            if extracted_info:
                st.info(
                    "🎯 **Auto-extracted personal information:**\n"
                    + "\n".join(f"- {info}" for info in extracted_info)
                )

            # Show note about business purpose generation
            if st.session_state.metadata.get("description"):
                if st.session_state.get("use_ai_business_purpose", False):
                    st.success(
                        f"🤖 **AI-generated business purpose:** {st.session_state.metadata['description']}\n\n"
                        "✏️ **Please review and edit** the business purpose below to ensure it accurately reflects "
                        "how UC Berkeley benefited from this travel."
                    )
                else:
                    st.info(
                        f"📝 **Auto-generated business purpose:** {st.session_state.metadata['description']}\n\n"
                        "✏️ **Please review and edit** the business purpose below. "
                        "You can enable AI-powered generation (slower but more compliant) using the checkbox above."
                    )

            # Set flag to show Event Info content prominently
            st.session_state.show_event_info = True

            # Streamlit will automatically rerun and show the inline form

    def render_expense_review(self):
        """Render the expense review interface organized by UC Berkeley sections"""
        if not st.session_state.expenses:
            return

        # Auto-correct categories for all expenses (only run once per upload)
        if not st.session_state.get("auto_correction_done", False):
            corrections_made = []
            for exp in st.session_state.expenses:
                old_category = exp.category
                corrected_category, corrected_expense_type, corrected_meal_type = (
                    self.auto_correct_category(
                        exp.description, exp.category, exp.meal_type
                    )
                )
                if corrected_category != old_category:
                    exp.category = corrected_category
                    exp.expense_type = corrected_expense_type
                    if corrected_meal_type is not None:
                        exp.meal_type = corrected_meal_type
                    corrections_made.append(
                        f"'{exp.description[:30]}...' → {corrected_category}"
                    )

            if corrections_made:
                st.toast(f"🔧 Auto-corrected {len(corrections_made)} expense(s)")

            st.session_state.auto_correction_done = True

        st.header("📊 Review Extracted Expenses")

        # Group expenses by category type
        transportation_expenses = []
        miscellaneous_expenses = []
        meal_expenses = []
        lodging_expenses = []

        for i, exp in enumerate(st.session_state.expenses):
            # Find the category key from the value
            cat_key = None
            for key, val in EXPENSE_CATEGORIES.items():
                if val == exp.category or key == exp.category:
                    cat_key = key
                    break

            if cat_key in TRANSPORTATION_CATEGORIES:
                transportation_expenses.append((i, exp))
            elif cat_key in MISCELLANEOUS_CATEGORIES:
                miscellaneous_expenses.append((i, exp))
            elif cat_key == "MEAL":
                meal_expenses.append((i, exp))
            elif cat_key == "LODGING":
                lodging_expenses.append((i, exp))
            else:
                # Default to miscellaneous
                miscellaneous_expenses.append((i, exp))

        # Calculate totals by section
        def calc_section_total(expenses_list):
            return sum(
                exp.amount
                * st.session_state.metadata.get("exchange_rates", {}).get(
                    exp.currency, 1.0
                )
                for _, exp in expenses_list
            )

        transport_total = calc_section_total(transportation_expenses)
        misc_total = calc_section_total(miscellaneous_expenses)
        meals_total = calc_section_total(meal_expenses)
        lodging_total = calc_section_total(lodging_expenses)
        grand_total = transport_total + misc_total + meals_total + lodging_total

        # Duplicate detection
        duplicates = self.detect_duplicate_expenses(st.session_state.expenses)
        if duplicates:
            with st.expander("⚠️ Potential Duplicate Expenses Detected", expanded=True):
                st.warning(
                    "The following expenses may be duplicates. Please review and delete if necessary:"
                )
                for dup_group in duplicates:
                    exp1, exp2 = dup_group
                    st.markdown(
                        f"- **{exp1.description}** (${exp1.amount:.2f}) on {exp1.date} vs **{exp2.description}** (${exp2.amount:.2f}) on {exp2.date}"
                    )

        # Summary by section
        st.markdown("### Summary by Section")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Transportation", f"${transport_total:.2f}")
        with col2:
            st.metric("Miscellaneous", f"${misc_total:.2f}")
        with col3:
            st.metric("Meals", f"${meals_total:.2f}")
        with col4:
            st.metric("Lodging", f"${lodging_total:.2f}")

        st.metric("**Grand Total**", f"${grand_total:.2f}")
        st.markdown("---")

        # Helper function to render expense editor
        def render_expense_editor(i, expense, section_prefix):
            # Determine confidence level for visual indicator
            confidence_emoji = (
                "✅"
                if expense.confidence >= 0.8
                else "⚠️" if expense.confidence >= 0.5 else "❓"
            )
            expander_label = f"{confidence_emoji} {expense.filename} - ${expense.amount:.2f} {expense.currency}"

            with st.expander(expander_label):
                # Low confidence warning
                if expense.confidence < 0.7:
                    st.warning(
                        f"⚠️ **Low confidence ({expense.confidence:.0%})** - "
                        "Please verify this expense was extracted correctly."
                    )

                # Receipt preview (if available)
                if expense.filename in st.session_state.uploaded_files_data:
                    file_data = st.session_state.uploaded_files_data[expense.filename]
                    if expense.filename.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".gif")
                    ):
                        st.image(file_data, caption="Original Receipt", width=200)
                    elif expense.filename.lower().endswith(".pdf"):
                        st.caption(
                            f"📎 PDF: {expense.filename} ({len(file_data):,} bytes)"
                        )

                col1, col2 = st.columns(2)

                with col1:
                    new_description = st.text_input(
                        "Description",
                        value=expense.description,
                        key=f"{section_prefix}_desc_{i}",
                    )
                    new_amount = st.number_input(
                        "Amount",
                        value=float(expense.amount),
                        min_value=0.0,
                        step=0.01,
                        key=f"{section_prefix}_amount_{i}",
                    )
                    new_currency = st.selectbox(
                        "Currency",
                        options=st.session_state.metadata.get("currencies", ["USD"]),
                        index=(
                            st.session_state.metadata.get("currencies", ["USD"]).index(
                                expense.currency
                            )
                            if expense.currency
                            in st.session_state.metadata.get("currencies", ["USD"])
                            else 0
                        ),
                        key=f"{section_prefix}_currency_{i}",
                    )

                with col2:
                    new_date = st.date_input(
                        "Date",
                        value=datetime.strptime(expense.date, "%Y-%m-%d").date(),
                        key=f"{section_prefix}_date_{i}",
                    )
                    new_category = st.selectbox(
                        "Category",
                        options=list(EXPENSE_CATEGORIES.values()),
                        index=(
                            list(EXPENSE_CATEGORIES.values()).index(expense.category)
                            if expense.category in EXPENSE_CATEGORIES.values()
                            else 0
                        ),
                        key=f"{section_prefix}_category_{i}",
                    )

                    exchange_rate = st.session_state.metadata.get(
                        "exchange_rates", {}
                    ).get(new_currency, 1.0)
                    amount_usd = new_amount * exchange_rate
                    st.info(
                        f"Confidence: {expense.confidence:.1%} | USD: ${amount_usd:.2f}"
                    )

                # Delete button
                if st.button(
                    f"🗑️ Delete this expense",
                    key=f"{section_prefix}_delete_{i}",
                    type="secondary",
                ):
                    st.session_state.expenses.pop(i)
                    st.success("Expense deleted!")
                    st.rerun()

                # Update expense data
                st.session_state.expenses[i] = ExpenseData(
                    amount=new_amount,
                    currency=new_currency,
                    description=new_description,
                    date=new_date.strftime("%Y-%m-%d"),
                    category=new_category,
                    filename=expense.filename,
                    confidence=expense.confidence,
                    first_name=expense.first_name,
                    last_name=expense.last_name,
                    email=expense.email,
                    expense_type=expense.expense_type,
                    meal_type=expense.meal_type,
                    destination=expense.destination,
                )

        # === LOCATION EXPENSES: TRANSPORTATION ===
        if transportation_expenses:
            st.markdown("### 🚗 Transportation Expenses")
            st.caption("Airfare, Rental Car, Ground Transportation")
            for i, expense in transportation_expenses:
                render_expense_editor(i, expense, "trans")
            st.info(f"**Transportation Subtotal: ${transport_total:.2f}**")
            st.markdown("---")

        # === LOCATION EXPENSES: MISCELLANEOUS ===
        if miscellaneous_expenses:
            st.markdown("### 📋 Miscellaneous Expenses")
            st.caption("Conference Fees, Supplies, Other Business Expenses")
            for i, expense in miscellaneous_expenses:
                render_expense_editor(i, expense, "misc")
            st.info(f"**Miscellaneous Subtotal: ${misc_total:.2f}**")
            st.markdown("---")

        # === DAILY EXPENSES: MEALS ===
        if meal_expenses:
            st.markdown("### 🍽️ Daily Expenses - Meals")
            st.caption("Meals organized by date")

            # Check daily meal limits ($92/day federal rate)
            FEDERAL_DAILY_MEAL_LIMIT = 92.00
            meals_by_date = {}
            for _, expense in meal_expenses:
                date = expense.date
                exchange_rate = st.session_state.metadata.get("exchange_rates", {}).get(
                    expense.currency, 1.0
                )
                amount_usd = expense.amount * exchange_rate
                meals_by_date[date] = meals_by_date.get(date, 0) + amount_usd

            # Show warning for dates over limit
            dates_over_limit = [
                (date, total)
                for date, total in meals_by_date.items()
                if total > FEDERAL_DAILY_MEAL_LIMIT
            ]
            if dates_over_limit:
                with st.expander("⚠️ Daily Meal Limits Exceeded", expanded=True):
                    st.warning(
                        f"The federal daily meal limit is **${FEDERAL_DAILY_MEAL_LIMIT:.2f}**. "
                        "The following dates exceed this limit and may require special approval:"
                    )
                    for date, total in sorted(dates_over_limit):
                        over_by = total - FEDERAL_DAILY_MEAL_LIMIT
                        st.markdown(
                            f"- **{date}**: ${total:.2f} (${over_by:.2f} over limit)"
                        )

            # Sort by date
            meal_expenses_sorted = sorted(meal_expenses, key=lambda x: x[1].date)
            for i, expense in meal_expenses_sorted:
                render_expense_editor(i, expense, "meal")
            st.info(f"**Meals Subtotal: ${meals_total:.2f}**")
            st.markdown("---")

        # === DAILY EXPENSES: LODGING ===
        if lodging_expenses:
            st.markdown("### 🏨 Daily Expenses - Lodging")
            st.caption("Lodging organized by date")
            lodging_expenses_sorted = sorted(lodging_expenses, key=lambda x: x[1].date)
            for i, expense in lodging_expenses_sorted:
                render_expense_editor(i, expense, "lodging")
            st.info(f"**Lodging Subtotal: ${lodging_total:.2f}**")

        # === ADD MANUAL EXPENSE ===
        st.markdown("---")
        st.markdown("### ➕ Add Manual Expense")
        st.caption("Add expenses not on receipts (mileage, tips, etc.)")

        with st.expander("Click to add a manual expense", expanded=False):
            with st.form("manual_expense_form", clear_on_submit=True):
                col1, col2 = st.columns(2)

                with col1:
                    manual_description = st.text_input(
                        "Description*",
                        placeholder="e.g., Mileage to airport, Tips for hotel staff",
                    )
                    manual_amount = st.number_input(
                        "Amount*", min_value=0.0, step=0.01, format="%.2f"
                    )
                    manual_currency = st.selectbox(
                        "Currency", options=CURRENCY_OPTIONS, index=0
                    )

                with col2:
                    manual_date = st.date_input("Date*", value=datetime.now().date())
                    # Category selection with all options
                    category_options = list(EXPENSE_CATEGORIES.values())
                    manual_category = st.selectbox(
                        "Category*", options=category_options
                    )
                    # Show meal type if meal is selected
                    manual_meal_type = None
                    if manual_category == "Meal":
                        manual_meal_type = st.selectbox("Meal Type", options=MEAL_TYPES)

                manual_destination = st.text_input(
                    "Destination (optional)", placeholder="e.g., Boston, MA"
                )

                add_manual = st.form_submit_button("➕ Add Expense", type="primary")

                if add_manual:
                    if manual_description and manual_amount > 0:
                        # Find the category key from value
                        cat_key = None
                        for key, val in EXPENSE_CATEGORIES.items():
                            if val == manual_category:
                                cat_key = key
                                break

                        new_expense = ExpenseData(
                            filename="Manual Entry",
                            description=manual_description,
                            amount=manual_amount,
                            currency=manual_currency,
                            date=manual_date.strftime("%Y-%m-%d"),
                            category=cat_key or manual_category,
                            confidence=1.0,
                            meal_type=manual_meal_type,
                            destination=(
                                manual_destination if manual_destination else None
                            ),
                        )
                        st.session_state.expenses.append(new_expense)
                        st.success(
                            f"✅ Added: {manual_description} - ${manual_amount:.2f}"
                        )
                        st.rerun()
                    else:
                        st.error("Please fill in description and amount")

    def render_submission(self):
        """Render the final submission interface with UC Berkeley copy-paste preview"""
        if not st.session_state.expenses or not st.session_state.processing_complete:
            if not st.session_state.expenses:
                st.info("📎 Please upload and process some expense documents first.")
            return

        st.header("🚀 Submit to Google Sheets")

        # Show warning if event information is missing
        if not st.session_state.metadata:
            st.warning(
                "⚠️ Please fill in the event information in the 'Event Info' tab before submitting."
            )
            return

        metadata = st.session_state.metadata
        expenses = st.session_state.expenses

        # Group expenses by category and calculate totals
        transport_expenses = []
        misc_expenses = []
        meal_expenses = []
        lodging_expenses = []

        transport_total = 0.0
        misc_total = 0.0
        meals_total = 0.0
        lodging_total = 0.0

        for expense in expenses:
            exchange_rate = metadata.get("exchange_rates", {}).get(
                expense.currency, 1.0
            )
            amount_usd = expense.amount * exchange_rate

            cat_key = None
            for key, val in EXPENSE_CATEGORIES.items():
                if val == expense.category or key == expense.category:
                    cat_key = key
                    break

            if cat_key in TRANSPORTATION_CATEGORIES:
                transport_total += amount_usd
                transport_expenses.append((expense, amount_usd))
            elif cat_key in MISCELLANEOUS_CATEGORIES:
                misc_total += amount_usd
                misc_expenses.append((expense, amount_usd))
            elif cat_key == "MEAL":
                meals_total += amount_usd
                meal_expenses.append((expense, amount_usd))
            elif cat_key == "LODGING":
                lodging_total += amount_usd
                lodging_expenses.append((expense, amount_usd))
            else:
                misc_total += amount_usd
                misc_expenses.append((expense, amount_usd))

        grand_total = transport_total + misc_total + meals_total + lodging_total

        # Convert dates to strings
        start_date = metadata.get("start_date", "")
        end_date = metadata.get("end_date", "")
        if hasattr(start_date, "strftime"):
            start_date = start_date.strftime("%m/%d/%Y")
        if hasattr(end_date, "strftime"):
            end_date = end_date.strftime("%m/%d/%Y")

        # === COPY-PASTE FRIENDLY PREVIEW ===
        st.subheader("📋 UC Berkeley Travel Reimbursement Preview")
        st.caption(
            "Copy this text directly into the UC Berkeley Travel Reimbursement form"
        )

        # Build the copy-paste text
        preview_lines = []

        # TRAVELER INFO
        preview_lines.append("=== TRAVELER INFO ===")
        preview_lines.append(
            f"Name: {metadata.get('first_name', '')} {metadata.get('last_name', '')}"
        )
        preview_lines.append(f"Email: {metadata.get('email', '')}")
        preview_lines.append("")

        # TRIP INFO
        preview_lines.append("=== TRIP INFO ===")
        preview_lines.append(f"Business Purpose: {metadata.get('description', '')}")
        preview_lines.append(f"Trip Duration: {metadata.get('trip_duration', '')}")
        preview_lines.append(f"Destinations: {metadata.get('destinations', '')}")
        preview_lines.append(f"Travel Dates: {start_date} - {end_date}")
        if metadata.get("event_name"):
            preview_lines.append(f"Event/Conference: {metadata.get('event_name', '')}")
        preview_lines.append("")

        # LOCATION EXPENSES
        preview_lines.append("=== LOCATION EXPENSES ===")

        # Transportation
        preview_lines.append("Transportation:")
        if transport_expenses:
            for expense, amount_usd in transport_expenses:
                preview_lines.append(f"  {expense.description}: ${amount_usd:.2f}")
            preview_lines.append(f"  Subtotal: ${transport_total:.2f}")
        else:
            preview_lines.append("  (none)")
        preview_lines.append("")

        # Miscellaneous
        preview_lines.append("Miscellaneous:")
        if misc_expenses:
            for expense, amount_usd in misc_expenses:
                preview_lines.append(f"  {expense.description}: ${amount_usd:.2f}")
            preview_lines.append(f"  Subtotal: ${misc_total:.2f}")
        else:
            preview_lines.append("  (none)")
        preview_lines.append("")

        # DAILY EXPENSES
        preview_lines.append("=== DAILY EXPENSES ===")

        # Meals (grouped by date)
        preview_lines.append("Meals:")
        if meal_expenses:
            # Group meals by date
            meals_by_date = {}
            for expense, amount_usd in meal_expenses:
                date = expense.date
                if date not in meals_by_date:
                    meals_by_date[date] = {"total": 0.0, "types": []}
                meals_by_date[date]["total"] += amount_usd
                if expense.meal_type:
                    meals_by_date[date]["types"].append(expense.meal_type.capitalize())

            for date in sorted(meals_by_date.keys()):
                meal_info = meals_by_date[date]
                types_str = (
                    f" ({', '.join(meal_info['types'])})" if meal_info["types"] else ""
                )
                preview_lines.append(f"  {date}: ${meal_info['total']:.2f}{types_str}")
            preview_lines.append(f"  Subtotal: ${meals_total:.2f}")
        else:
            preview_lines.append("  (none)")
        preview_lines.append("")

        # Lodging (grouped by date)
        preview_lines.append("Lodging:")
        if lodging_expenses:
            # Group lodging by date
            lodging_by_date = {}
            for expense, amount_usd in lodging_expenses:
                date = expense.date
                lodging_by_date[date] = lodging_by_date.get(date, 0) + amount_usd

            for date in sorted(lodging_by_date.keys()):
                preview_lines.append(f"  {date}: ${lodging_by_date[date]:.2f}")
            preview_lines.append(f"  Subtotal: ${lodging_total:.2f}")
        else:
            preview_lines.append("  (none)")
        preview_lines.append("")

        # TOTALS
        preview_lines.append("=== TOTALS ===")
        preview_lines.append(f"Transportation: ${transport_total:.2f}")
        preview_lines.append(f"Miscellaneous: ${misc_total:.2f}")
        preview_lines.append(f"Meals: ${meals_total:.2f}")
        preview_lines.append(f"Lodging: ${lodging_total:.2f}")
        preview_lines.append(f"Trip Total: ${grand_total:.2f}")

        preview_text = "\n".join(preview_lines)

        # Display in a text area for easy copy-paste
        st.text_area(
            "Copy-Paste Preview",
            value=preview_text,
            height=500,
            help="Select all (Ctrl+A / Cmd+A) and copy this text to paste into the UC Berkeley form",
        )

        st.markdown("---")

        # Exchange rates used
        if len(metadata.get("currencies", [])) > 1:
            st.subheader("💱 Exchange Rates Used")
            rates_data = []
            for currency, rate in metadata.get("exchange_rates", {}).items():
                if currency != "USD":
                    rates_data.append(
                        {
                            "Currency": currency,
                            "Rate to USD": f"{rate:.4f}",
                            "Date": metadata.get("median_date", "N/A"),
                        }
                    )

            if rates_data:
                rates_df = pd.DataFrame(rates_data)
                st.dataframe(rates_df, use_container_width=True)

        # Submission options
        st.markdown("---")
        st.subheader("📤 Submit to Google Sheets")

        st.markdown(
            """
        Export your expense data to Google Sheets:
        - **Sheet1**: Summary row with totals in UC Berkeley format
        - **Details**: Individual expenses organized by section (Transportation, Miscellaneous, Meals, Lodging)
        """
        )

        # Generate merged PDF if we have uploaded files and haven't generated it yet
        if st.session_state.uploaded_files_data and not st.session_state.get("merged_pdf_bytes"):
            merged_pdf = self.merge_files_to_pdf(st.session_state.uploaded_files_data)
            if merged_pdf:
                merged_pdf = self.redact_pii_from_pdf(merged_pdf)
                st.session_state.merged_pdf_bytes = merged_pdf

        # Show download button for combined PDF (before submission)
        if st.session_state.get("merged_pdf_bytes"):
            st.markdown("### 📄 Combined Receipts PDF")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=st.session_state.merged_pdf_bytes,
                    file_name="expense_receipts.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with col2:
                if st.session_state.get("s3_presigned_url"):
                    st.link_button(
                        label="🔗 Open S3 Link",
                        url=st.session_state.s3_presigned_url,
                        use_container_width=True,
                    )
            if st.session_state.get("s3_presigned_url"):
                st.caption("⏰ S3 link expires in 7 days")

        st.markdown("---")

        # Show success message if already submitted
        if st.session_state.get("submission_success", False):
            st.success("✅ Data successfully submitted to both Sheet1 and Details!")
            if st.session_state.get("s3_upload_message"):
                st.success(st.session_state.s3_upload_message)
            st.info("Click '🔄 Start New Report' below to submit another expense report.")
        else:
            # Use a form to prevent tab switching on submit
            with st.form("submit_form", clear_on_submit=False):
                submitted = st.form_submit_button(
                    "📤 Submit to Google Sheets", type="primary", use_container_width=True
                )

                if submitted:
                    with st.spinner("Submitting to Sheet1 and Details sheet..."):
                        success = self.submit_to_google_sheets(
                            st.session_state.expenses, st.session_state.metadata
                        )

                        if success:
                            st.session_state.submission_success = True
                            st.success("✅ Data successfully submitted to both Sheet1 and Details!")

                            # Upload files to AWS S3
                            if st.session_state.uploaded_files_data or st.session_state.get("merged_pdf_bytes"):
                                with st.spinner("Uploading files to S3..."):
                                    s3_success, uploaded_count, s3_error, s3_key, merged_pdf = (
                                        self.upload_files_to_s3(
                                            st.session_state.uploaded_files_data,
                                            st.session_state.get("merged_pdf_bytes")
                                        )
                                    )

                                    if s3_success and (uploaded_count > 0 or s3_key):
                                        file_count = uploaded_count or len(st.session_state.uploaded_files_data)
                                        st.session_state.s3_upload_message = f"📁 {file_count} file(s) merged into single PDF and uploaded to S3!"

                                        # Generate presigned URL (7 days expiration)
                                        if s3_key:
                                            presigned_url = self.generate_presigned_url(s3_key)
                                            st.session_state.s3_presigned_url = presigned_url

                                        st.success(st.session_state.s3_upload_message)
                                    elif not s3_success:
                                        st.warning(f"📁 S3 upload failed: {s3_error}")

                            st.balloons()
                            st.rerun()
                        else:
                            st.error(
                                "❌ Failed to submit data. Please check your configuration."
                            )

        # Start over option
        st.markdown("---")
        if st.button("🔄 Start New Report", use_container_width=True):
            st.session_state.expenses = []
            st.session_state.metadata = {}
            st.session_state.processing_complete = False
            st.session_state.show_event_info = False
            st.session_state.additional_context = ""
            st.session_state.include_external_emails = False
            st.session_state.use_ai_business_purpose = False
            st.session_state.uploaded_files_data = {}
            st.session_state.auto_correction_done = False
            st.session_state.submission_success = False
            st.session_state.s3_upload_message = None
            st.session_state.s3_presigned_url = None
            st.session_state.merged_pdf_bytes = None
            st.rerun()

    def render_progress_indicator(self):
        """Render a visual progress indicator showing the user's current step"""
        # Determine current step
        has_expenses = bool(st.session_state.get("expenses"))
        has_metadata = bool(st.session_state.get("metadata"))
        processing_complete = st.session_state.get("processing_complete", False)

        # Calculate step completion
        step1_complete = has_expenses and processing_complete
        step2_complete = step1_complete and has_metadata
        step3_ready = step2_complete

        # Determine current step number
        if not step1_complete:
            current_step = 1
        elif not step2_complete:
            current_step = 2
        else:
            current_step = 3

        # Render progress bar
        col1, col2, col3 = st.columns(3)

        with col1:
            if step1_complete:
                st.success("✅ **Step 1: Upload**\nDocuments processed")
            elif current_step == 1:
                st.info("📎 **Step 1: Upload**\nUpload & process documents")
            else:
                st.write("⬜ **Step 1: Upload**")

        with col2:
            if step2_complete:
                st.success("✅ **Step 2: Review**\nExpenses verified")
            elif current_step == 2:
                st.info("📊 **Step 2: Review**\nReview & edit expenses")
            else:
                st.write("⬜ **Step 2: Review**")

        with col3:
            if step3_ready:
                st.info("🚀 **Step 3: Submit**\nReady to submit!")
            else:
                st.write("⬜ **Step 3: Submit**")

        # Progress percentage
        progress = (1 if step1_complete else 0) + (1 if step2_complete else 0)
        st.progress(progress / 3, text=f"Progress: {progress}/3 steps complete")
        st.markdown("---")

    def run(self):
        """Main application runner"""
        logger.info("Starting main run method")
        try:
            logger.info("Setting title and markdown")

            # Haas branded header
            st.markdown(
                """
            <div class="main-header">
                <h1 class="main-title">🐻 Haas Expense Report Automation</h1>
                <p class="haas-subtitle">UC Berkeley Haas School of Business | AI-Powered with GPT-5</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "**Upload your expense documents and let AI extract and categorize the information automatically.** "
                "Perfect for faculty, staff, and researchers managing travel and business expenses."
            )

            # Render sidebar
            logger.info("About to render sidebar")
            self.render_sidebar()
            logger.info("Sidebar rendered")
        except Exception as e:
            logger.error(f"Error in run method header: {str(e)}")
            st.error(f"Error in run method: {str(e)}")
            import traceback

            st.code(traceback.format_exc())
            logger.error(traceback.format_exc())

        # Progress indicator
        self.render_progress_indicator()

        # Main content area - simplified tabs without Event Info
        logger.info("Creating tabs")


        tab1, tab2, tab3 = st.tabs(["📎 Upload & Event Info", "📊 Review", "🚀 Submit"])

        logger.info("Rendering tab 1")
        with tab1:
            self.render_file_upload()

        logger.info("Rendering tab 2")
        with tab2:
            self.render_expense_review()

        logger.info("Rendering tab 3")
        with tab3:
            self.render_submission()

        logger.info("Run method completed successfully")


# Run the application
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Application starting")
    logger.info("=" * 50)
    try:
        app = ExpenseReportApp()
        logger.info("App instance created, calling run()")
        app.run()
        logger.info("App run() completed")
    except Exception as e:
        # Check if it's a RerunException or StopException (normal Streamlit behavior)
        if RerunException is not None and isinstance(e, RerunException):
            # RerunException is normal Streamlit behavior, not an error
            # It's raised when st.rerun() is called to refresh the app
            raise
        if StopException is not None and isinstance(e, StopException):
            # StopException is normal Streamlit behavior, not an error
            # It's raised when the script execution is stopped for a rerun
            raise
        logger.error(f"FATAL ERROR in main: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        st.error(f"Fatal error: {str(e)}")
        st.code(traceback.format_exc())
