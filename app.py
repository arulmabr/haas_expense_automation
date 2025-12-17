import streamlit as st
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


# Constants
EXPENSE_CATEGORIES = {
    "AIRFARE": "AIRFARE",
    "ACCOMMODATION_US": "ACCOMMODATION (In US)",
    "ACCOMMODATION_INT": "ACCOMMODATION (Outside US)",
    "TRANSIT_US": "RAILWAY/BUS/TAXI (In US)",
    "TRANSIT_INT": "RAILWAY/BUS/TAXI (Outside US)",
    "CAR_RENTAL_US": "CAR RENTAL (In US)",
    "CAR_RENTAL_INT": "CAR RENTAL (Outside US)",
    "MEALS_US": "MEALS (In US)",
    "MEALS_INT": "MEALS (Outside US)",
    "OTHER": "OTHER",
}

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

    def get_expense_analysis_prompt(self, context: str = "") -> str:
        """Get the standard expense analysis prompt for GPT"""
        base_prompt = """Analyze this expense document and extract the following information in JSON format:

REQUIRED FIELDS:
1. amount: the total amount paid (as a number, no currency symbols)
2. currency: the currency code (USD, EUR, CAD, etc.) - default to USD if unclear
3. description: a brief description of what this expense is for (e.g., "Flight from SFO to NYC", "Hotel stay in Boston")
4. date: the transaction date in YYYY-MM-DD format. For hotels, use the first day of the stay
5. category: categorize this into one of the following EXACT categories:
   - "AIRFARE"
   - "ACCOMMODATION (In US)"
   - "ACCOMMODATION (Outside US)"
   - "RAILWAY/BUS/TAXI (In US)"
   - "RAILWAY/BUS/TAXI (Outside US)"
   - "CAR RENTAL (In US)"
   - "CAR RENTAL (Outside US)"
   - "MEALS (In US)"
   - "MEALS (Outside US)"
   - "OTHER"
6. confidence: a confidence score from 0.0 to 1.0 indicating how confident you are in the extraction

CRITICAL - PERSONAL INFORMATION (extract if ANY name or email is present):
7. first_name: Extract the first name from ANY of these locations:
   - Passenger name, traveler name, guest name
   - Cardholder name, account holder
   - "Bill to", "Billed to", "Customer"
   - "Name:", "Guest:", "Passenger:"
   - Email sender or recipient name
   - Any name on the receipt or invoice
   - Name in email header "From:" or "To:"
8. last_name: Extract the last name from the same locations
9. email: Extract email address from ANY location including:
   - Email headers (From:, To:, Reply-to:)
   - Contact information sections
   - Customer information
   - Account details
   - Any email address visible on the document

IMPORTANT INSTRUCTIONS:
- THOROUGHLY scan the ENTIRE document for names and email addresses
- For email documents: Extract sender/recipient name and email from headers
- For receipts: Look for customer name, cardholder, or guest information
- Determine if this is in the US or not based on address, city, state, country, or currency indicators
- If personal information is truly not found after thorough search, set those fields to null
- Return ONLY valid JSON with these fields, nothing else. No explanations or markdown."""

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
                    model="gpt-5",
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

                return ExpenseData(
                    filename=filename,
                    description=extracted_data.description or "Unknown expense",
                    amount=float(extracted_data.amount or 0),
                    currency=extracted_data.currency or "USD",
                    date=extracted_data.date or datetime.now().strftime("%Y-%m-%d"),
                    category=extracted_data.category or "OTHER",
                    confidence=float(extracted_data.confidence or 0),
                    first_name=extracted_data.first_name,
                    last_name=extracted_data.last_name,
                    email=extracted_data.email,
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
                    model="gpt-5",
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

                return ExpenseData(
                    amount=float(extracted_data.amount or 0),
                    currency=extracted_data.currency or "USD",
                    description=extracted_data.description or "Unknown expense",
                    date=extracted_data.date or datetime.now().strftime("%Y-%m-%d"),
                    category=extracted_data.category or "OTHER",
                    filename=filename,
                    confidence=float(extracted_data.confidence or 0.5),
                    first_name=extracted_data.first_name,
                    last_name=extracted_data.last_name,
                    email=extracted_data.email,
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
                model="gpt-5",
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

            return ExpenseData(
                amount=float(extracted_data.amount or 0),
                currency=extracted_data.currency or "USD",
                description=extracted_data.description or "Unknown expense",
                date=extracted_data.date or datetime.now().strftime("%Y-%m-%d"),
                category=extracted_data.category or "OTHER",
                filename=filename,
                confidence=float(extracted_data.confidence or 0.5),
                first_name=extracted_data.first_name,
                last_name=extracted_data.last_name,
                email=extracted_data.email,
            )

        except Exception as e:
            st.error(f"Error calling OpenAI API: {str(e)}")
            logger.error(f"Error in fallback processing: {str(e)}")
            return None

    def get_exchange_rates(self, currencies: List[str], date: str) -> Dict[str, float]:
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
        business_purpose = self.generate_business_purpose_from_expenses(expenses, context)

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

    def generate_event_name_from_expenses(self, expenses: List[ExpenseData], context: str = "") -> str:
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
                    sentences = context.split('.')
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
                model="gpt-5",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at writing compliant business purpose statements for university expense reports.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=100,
                temperature=0.7,
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
            return context.strip()[:200] if len(context.strip()) > 200 else context.strip()

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
        """Render a simplified event form inline in the upload tab"""
        with st.form("inline_metadata_form"):
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

            submitted = st.form_submit_button(
                "💾 Save Event Information", type="primary"
            )

            if submitted:
                if first_name and last_name and email and event_name:
                    # Validate email domain
                    allow_external = st.session_state.get("include_external_emails", False)
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

                        # Set flag to indicate user should go to review tab
                        st.session_state.show_review_next = True

                        st.success(
                            "✅ Event information saved! Please go to the **Review** tab to verify your expenses."
                        )
                        st.balloons()
                        # Don't rerun - let user see the message and balloons
                else:
                    st.error("Please fill in all required fields (marked with *)")

    def submit_to_google_sheets(
        self, expenses: List[ExpenseData], metadata: Dict
    ) -> bool:
        """Submit data to Google Sheets - individual expenses and summary"""
        client = self.get_google_sheets_client()
        if not client:
            return False

        try:
            # You'll need to replace this with your actual Google Sheet ID
            sheet_id = st.secrets.get("GOOGLE_SHEET_ID") or os.getenv("GOOGLE_SHEET_ID")
            if not sheet_id:
                st.error("Google Sheet ID not configured")
                return False

            spreadsheet = client.open_by_key(sheet_id)

            # Calculate totals first
            total_amount_usd = 0
            category_totals = {}

            for expense in expenses:
                exchange_rate = metadata.get("exchange_rates", {}).get(
                    expense.currency, 1.0
                )
                amount_usd = expense.amount * exchange_rate
                total_amount_usd += amount_usd

                # Track category totals
                category = expense.category
                category_totals[category] = (
                    category_totals.get(category, 0) + amount_usd
                )

            # Sheet1: Event summary (one row per event)
            sheet1 = spreadsheet.sheet1

            # Prepare category breakdown string
            category_breakdown = ", ".join(
                [f"{cat}: ${amt:.2f}" for cat, amt in category_totals.items()]
            )

            # Submit summary row to Sheet1
            # Convert dates to strings if they're date objects
            start_date = metadata.get("start_date", "")
            end_date = metadata.get("end_date", "")

            if hasattr(start_date, "strftime"):
                start_date = start_date.strftime("%Y-%m-%d")
            if hasattr(end_date, "strftime"):
                end_date = end_date.strftime("%Y-%m-%d")

            summary_row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
                metadata.get("event_name", ""),
                metadata.get("first_name", ""),
                metadata.get("last_name", ""),
                metadata.get("email", ""),
                start_date,
                end_date,
                metadata.get("description", ""),
                round(total_amount_usd, 2),
                len(expenses),
                category_breakdown,
                "USD",
            ]

            sheet1.append_row(summary_row)

            # Details sheet: Individual expense details
            try:
                details_sheet = spreadsheet.worksheet("Details")
            except Exception:
                # Create Details sheet if it doesn't exist
                details_sheet = spreadsheet.add_worksheet(
                    title="Details", rows="10000", cols="20"
                )
                # Add headers
                headers = [
                    "Timestamp",
                    "Event Name",
                    "First Name",
                    "Last Name",
                    "Email",
                    "Invoice Name",
                    "Amount",
                    "Currency",
                    "Amount in USD",
                    "Category",
                    "Event Date",
                    "Invoice Files",
                    "Accuracy",
                ]
                details_sheet.append_row(headers)

            # Submit individual expenses to Details sheet
            for expense in expenses:
                exchange_rate = metadata.get("exchange_rates", {}).get(
                    expense.currency, 1.0
                )
                amount_usd = expense.amount * exchange_rate

                row_data = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
                    metadata.get("event_name", ""),
                    metadata.get("first_name", ""),
                    metadata.get("last_name", ""),
                    metadata.get("email", ""),
                    expense.description,
                    expense.amount,
                    expense.currency,
                    amount_usd,
                    expense.category,
                    expense.date,
                    expense.filename,
                    expense.confidence,
                ]

                details_sheet.append_row(row_data)

            return True

        except Exception as e:
            st.error(f"Error submitting to Google Sheets: {str(e)}")
            logger.error(f"Google Sheets error: {str(e)}")
            return False

    def get_s3_client(self):
        """Initialize AWS S3 client"""
        try:
            aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = st.secrets.get("AWS_REGION") or os.getenv("AWS_REGION", "us-west-2")

            if not aws_access_key or not aws_secret_key:
                logger.error("AWS credentials not found")
                return None

            return boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
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

                if filename.lower().endswith('.pdf'):
                    # Add PDF pages directly
                    pdf_reader = PdfReader(io.BytesIO(file_bytes))
                    for page in pdf_reader.pages:
                        pdf_writer.add_page(page)
                    logger.info(f"Added PDF to merge: {filename} ({len(pdf_reader.pages)} pages)")

                elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Convert image to PDF page
                    image = Image.open(io.BytesIO(file_bytes))

                    # Convert to RGB if necessary (for PNG with transparency)
                    if image.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P':
                            image = image.convert('RGBA')
                        background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                        image = background

                    # Save image as PDF to bytes
                    img_pdf_bytes = io.BytesIO()
                    image.save(img_pdf_bytes, 'PDF', resolution=100.0)
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
            logger.info(f"Merged {len(files_data)} files into single PDF ({len(pdf_writer.pages)} pages)")
            return output.read()

        except Exception as e:
            logger.error(f"Failed to merge files to PDF: {str(e)}")
            return None

    def upload_files_to_s3(self, files_data: Dict[str, bytes]) -> tuple:
        """
        Merge all files into a single PDF and upload to AWS S3.

        Args:
            files_data: dict of {filename: bytes}

        Returns:
            (success: bool, uploaded_count: int, error_message: str or None)
        """
        if not files_data:
            return True, 0, None

        # Get bucket name from secrets
        bucket_name = st.secrets.get("S3_BUCKET_NAME") or os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            logger.info("S3 bucket not configured, skipping file upload")
            return True, 0, None  # Silent skip if not configured

        s3_client = self.get_s3_client()
        if not s3_client:
            return False, 0, "Failed to initialize S3 client"

        # Merge all files into single PDF
        merged_pdf = self.merge_files_to_pdf(files_data)
        if not merged_pdf:
            return False, 0, "Failed to merge files into PDF"

        try:
            # Create S3 key with date and timestamp
            today = datetime.now().strftime("%Y-%m-%d")
            timestamp = datetime.now().strftime("%H%M%S")
            s3_key = f"{today}/{timestamp}_expense_receipts.pdf"

            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=merged_pdf,
                ContentType='application/pdf'
            )

            logger.info(f"Uploaded merged PDF to S3: {s3_key}")
            return True, len(files_data), None

        except ClientError as e:
            error_msg = f"Failed to upload merged PDF: {str(e)}"
            logger.error(error_msg)
            return False, 0, error_msg

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
            st.session_state.show_review_next = False
            st.session_state.uploaded_files_data = {}
            # Streamlit will automatically rerun after state changes

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
                    allow_external = st.session_state.get("include_external_emails", False)
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
        st.session_state.additional_context = st.text_area(
            "Provide any additional details about this trip or event",
            value=st.session_state.additional_context,
            placeholder="Example: Attended the Annual Economics Conference in Boston. Met with research collaborators from MIT and Harvard to discuss ongoing project on behavioral finance...",
            help="This information helps the AI better understand the purpose of your expenses and generate more accurate business purpose descriptions.",
            height=100,
        )

        # Email privacy settings
        st.session_state.include_external_emails = st.checkbox(
            "Include external guest/speaker expenses (allows non-Berkeley emails)",
            value=st.session_state.include_external_emails,
            help="By default, only @berkeley.edu emails are extracted to protect privacy. Check this if submitting expenses for external speakers or guests from other universities.",
        )

        # AI business purpose toggle
        st.session_state.use_ai_business_purpose = st.checkbox(
            "🤖 Use AI to generate compliant business purpose (slower, more accurate)",
            value=st.session_state.use_ai_business_purpose,
            help="Enable this to use AI (GPT-5) to generate a UC Berkeley-compliant business purpose. Disabled by default for faster processing.",
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
                    return None, f"❌ {file.name} is empty. Please check the file and try again.", "error"

                logger.info(
                    f"Processing PDF {file.name} ({len(file_content)} bytes)"
                )
                expense_data = self.analyze_expense_with_gpt_direct_pdf(
                    file, file.name, context
                )
                if expense_data:
                    success_msg = f"✅ Processed {file.name} with GPT-5 direct PDF analysis"
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
                                if (
                                    expense_data.first_name
                                    or expense_data.last_name
                                ):
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
                            return None, f"❌ Failed to analyze {file.name} even with text extraction", "error"
                    else:
                        return None, f"❌ Could not extract text from {file.name}", "error"
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
                        success_msg += (
                            f" | 📋 Extracted: {', '.join(personal_parts)}"
                        )
                    return expense_data, success_msg, "success"
                else:
                    return None, f"❌ Failed to analyze {file.name}", "error"

        except Exception as e:
            return None, f"❌ Error processing {file.name}: {str(e)}", "error"

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files in parallel and extract expense data"""
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
        """Render the expense review and editing interface"""
        if not st.session_state.expenses:
            return

        st.header("📊 Review Extracted Expenses")

        # Show helpful message if user just saved event info
        if st.session_state.get("show_review_next"):
            st.info(
                "👀 **Review your extracted expenses below.** You can edit any details if needed, then proceed to the Submit tab."
            )
            # Clear the flag after showing
            st.session_state.show_review_next = False

        # Summary statistics
        total_expenses = len(st.session_state.expenses)
        total_amount_usd = sum(
            exp.amount
            * st.session_state.metadata.get("exchange_rates", {}).get(exp.currency, 1.0)
            for exp in st.session_state.expenses
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Expenses", total_expenses)
        with col2:
            st.metric("Total Amount (USD)", f"${total_amount_usd:.2f}")
        with col3:
            avg_confidence = (
                sum(exp.confidence for exp in st.session_state.expenses)
                / total_expenses
            )
            st.metric("Avg. Confidence", f"{avg_confidence:.1%}")

        st.markdown("---")

        # Expense editing interface
        for i, expense in enumerate(st.session_state.expenses):
            with st.expander(f"📄 {expense.filename} - {expense.description[:50]}..."):
                col1, col2 = st.columns(2)

                with col1:
                    new_description = st.text_input(
                        "Description", value=expense.description, key=f"desc_{i}"
                    )
                    new_amount = st.number_input(
                        "Amount",
                        value=float(expense.amount),
                        min_value=0.0,
                        step=0.01,
                        key=f"amount_{i}",
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
                        key=f"currency_{i}",
                    )

                with col2:
                    new_date = st.date_input(
                        "Date",
                        value=datetime.strptime(expense.date, "%Y-%m-%d").date(),
                        key=f"date_{i}",
                    )
                    new_category = st.selectbox(
                        "Category",
                        options=list(EXPENSE_CATEGORIES.values()),
                        index=(
                            list(EXPENSE_CATEGORIES.values()).index(expense.category)
                            if expense.category in EXPENSE_CATEGORIES.values()
                            else 0
                        ),
                        key=f"category_{i}",
                    )

                    # Show confidence and USD amount
                    exchange_rate = st.session_state.metadata.get(
                        "exchange_rates", {}
                    ).get(new_currency, 1.0)
                    amount_usd = new_amount * exchange_rate
                    st.info(f"Confidence: {expense.confidence:.1%}")
                    st.info(f"USD Amount: ${amount_usd:.2f}")

                # Update expense data
                st.session_state.expenses[i] = ExpenseData(
                    amount=new_amount,
                    currency=new_currency,
                    description=new_description,
                    date=new_date.strftime("%Y-%m-%d"),
                    category=new_category,
                    filename=expense.filename,
                    confidence=expense.confidence,
                )

    def render_submission(self):
        """Render the final submission interface"""
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

        # Final summary
        st.subheader("📋 Final Summary")

        # Create summary DataFrame
        summary_data = []
        for expense in st.session_state.expenses:
            exchange_rate = st.session_state.metadata.get("exchange_rates", {}).get(
                expense.currency, 1.0
            )
            amount_usd = expense.amount * exchange_rate

            summary_data.append(
                {
                    "File": expense.filename,
                    "Description": expense.description,
                    "Amount": f"{expense.amount:.2f} {expense.currency}",
                    "USD Amount": f"${amount_usd:.2f}",
                    "Category": expense.category,
                    "Date": expense.date,
                    "Confidence": f"{expense.confidence:.1%}",
                }
            )

        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)

        # Exchange rates used
        if len(st.session_state.metadata.get("currencies", [])) > 1:
            st.subheader("💱 Exchange Rates Used")
            rates_data = []
            for currency, rate in st.session_state.metadata.get(
                "exchange_rates", {}
            ).items():
                if currency != "USD":
                    rates_data.append(
                        {
                            "Currency": currency,
                            "Rate to USD": f"{rate:.4f}",
                            "Date": st.session_state.metadata.get("median_date", "N/A"),
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
        - **Sheet1**: Event totals (one row per event with aggregated data)
        - **Details**: Individual invoice details (one row per expense)
        """
        )

        if st.button(
            "📤 Submit to Google Sheets", type="primary", use_container_width=True
        ):
            with st.spinner("Submitting to Sheet1 and Details sheet..."):
                success = self.submit_to_google_sheets(
                    st.session_state.expenses, st.session_state.metadata
                )

                if success:
                    st.success(
                        "✅ Data successfully submitted to both Sheet1 and Details!"
                    )

                    # Upload files to AWS S3
                    if st.session_state.uploaded_files_data:
                        with st.spinner("Uploading files to S3..."):
                            s3_success, uploaded_count, s3_error = self.upload_files_to_s3(
                                st.session_state.uploaded_files_data
                            )

                            if s3_success and uploaded_count > 0:
                                st.success(f"📁 {uploaded_count} file(s) merged into single PDF and uploaded to S3!")
                            elif not s3_success:
                                st.warning(f"📁 S3 upload failed: {s3_error}")

                    st.balloons()
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
            st.session_state.show_review_next = False
            st.session_state.additional_context = ""
            st.session_state.include_external_emails = False
            st.session_state.use_ai_business_purpose = False
            st.session_state.uploaded_files_data = {}
            # Streamlit will automatically rerun after state changes

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
