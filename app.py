import streamlit as st
import pandas as pd
import json
import base64
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import gspread
from google.oauth2.service_account import Credentials
import openai
from PIL import Image
import requests
import os
from dataclasses import dataclass
import urllib.parse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="AI Expense Report Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
        logger.info("Session state setup complete")

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

    def analyze_expense_with_gpt_image(
        self, image_file, filename: str
    ) -> Optional[ExpenseData]:
        """Analyze expense image directly using GPT-5 vision API"""
        try:
            # Get OpenAI client
            client = self.get_openai_client()
            if not client:
                return None

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

            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.get_expense_analysis_prompt(),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                max_completion_tokens=1024,
            )

            content = response.choices[0].message.content
            
            if not content:
                st.error("Empty response from GPT-5")
                return None
            
            content = content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
                
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Try to find JSON in the response if it's not pure JSON
            if not content.startswith("{"):
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    content = content[start_idx:end_idx]
                else:
                    st.error(f"No JSON object found in image response: {content[:200]}")
                    return None
            
            data = json.loads(content)

            return ExpenseData(
                filename=filename,
                description=data.get("description") or "Unknown expense",
                amount=float(data.get("amount") or 0),
                currency=data.get("currency") or "USD",
                date=data.get("date") or datetime.now().strftime("%Y-%m-%d"),
                category=data.get("category") or "Other",
                confidence=float(data.get("confidence") or 0),
            )

        except Exception as e:
            st.error(f"Error with GPT-5 image processing: {str(e)}")
            return None

    def analyze_expense_with_gpt_direct_pdf(
        self, pdf_file, filename: str
    ) -> Optional[ExpenseData]:
        """Analyze PDF directly using GPT-5 vision without text extraction"""
        client = self.get_openai_client()
        if not client:
            return None

        try:
            # Upload the PDF file to OpenAI
            pdf_file.seek(0)  # Reset file pointer
            uploaded_file = client.files.create(file=pdf_file, purpose="user_data")

            # Create chat completion with direct PDF input using GPT-5
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert expense document analyzer. Analyze the PDF and return ONLY valid JSON with the specified fields.",
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
                                "text": """Analyze this expense document and extract the following information in JSON format:
1. amount: the total amount paid (as a number, no currency symbols)
2. currency: the currency code (USD, EUR, CAD, etc.) - default to USD if unclear
3. description: a brief description of what this expense is for
4. date: the transaction date in YYYY-MM-DD format. For hotels, use the first day of the stay
5. category: categorize this into one of the following exact categories:
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

Determine if this is in the US or not based on address, currency, or other indicators.
Return ONLY valid JSON with these fields, nothing else.""",
                            },
                        ],
                    },
                ],
                max_completion_tokens=1024,
            )

            result_text = response.choices[0].message.content
            
            # Clean up the uploaded file
            try:
                client.files.delete(uploaded_file.id)
            except:
                pass  # File cleanup failed, but continue

            # Parse the JSON response
            try:
                if not result_text:
                    st.error("Empty response from GPT-5")
                    return None
                    
                result_text = result_text.strip()
                
                # Handle markdown code blocks
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                elif result_text.startswith("```"):
                    result_text = result_text[3:]
                    
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                result_text = result_text.strip()
                
                # Try to find JSON in the response if it's not pure JSON
                if not result_text.startswith("{"):
                    # Look for JSON object in the text
                    start_idx = result_text.find("{")
                    end_idx = result_text.rfind("}") + 1
                    if start_idx != -1 and end_idx > start_idx:
                        result_text = result_text[start_idx:end_idx]
                    else:
                        st.error(f"No JSON object found in response: {result_text[:200]}")
                        return None

                data = json.loads(result_text)

                return ExpenseData(
                    amount=float(data.get("amount", 0) or 0),
                    currency=data.get("currency", "USD") or "USD",
                    description=data.get("description", "Unknown expense")
                    or "Unknown expense",
                    date=data.get("date", datetime.now().strftime("%Y-%m-%d"))
                    or datetime.now().strftime("%Y-%m-%d"),
                    category=data.get("category", "OTHER") or "OTHER",
                    filename=filename,
                    confidence=float(data.get("confidence", 0.5) or 0.5),
                )

            except json.JSONDecodeError as e:
                st.error(f"Failed to parse GPT response as JSON: {str(e)}")
                st.error(f"Response was: {result_text}")
                return None

        except Exception as e:
            st.error(f"Error with direct PDF processing: {str(e)}")
            return None

    def analyze_expense_with_gpt_fallback(
        self, text: str, filename: str
    ) -> Optional[ExpenseData]:
        """Fallback method: Analyze expense text using GPT-5 (for non-PDF files)"""
        client = self.get_openai_client()
        if not client:
            return None

        prompt = f"""
        You are an expert at parsing and analyzing receipts and expense documents.

        Analyze this expense document text and extract the following information in JSON format:
        1. amount: the total amount paid (as a number, no currency symbols)
        2. currency: the currency code (USD, EUR, CAD, etc.) - default to USD if unclear
        3. description: a brief description of what this expense is for
        4. date: the transaction date in YYYY-MM-DD format. For hotels, use the first day of the stay
        5. category: categorize this into one of the following exact categories:
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

        Determine if this is in the US or not based on address, currency, or other indicators.

        Document text:
        {text}

        Return ONLY valid JSON with these fields, nothing else.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-5",  # Use latest GPT-5 model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise expense document analyzer. Always return valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=1024,
            )

            result_text = response.choices[0].message.content

            # Try to extract JSON from the response
            try:
                if not result_text:
                    st.error("Empty response from GPT-5")
                    return None
                    
                result_text = result_text.strip()
                
                # Remove any markdown formatting
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                elif result_text.startswith("```"):
                    result_text = result_text[3:]
                    
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                result_text = result_text.strip()
                
                # Try to find JSON in the response if it's not pure JSON
                if not result_text.startswith("{"):
                    start_idx = result_text.find("{")
                    end_idx = result_text.rfind("}") + 1
                    if start_idx != -1 and end_idx > start_idx:
                        result_text = result_text[start_idx:end_idx]
                    else:
                        st.error(f"No JSON object found in response: {result_text[:200]}")
                        return None

                data = json.loads(result_text)

                return ExpenseData(
                    amount=float(data.get("amount", 0) or 0),
                    currency=data.get("currency", "USD") or "USD",
                    description=data.get("description", "Unknown expense")
                    or "Unknown expense",
                    date=data.get("date", datetime.now().strftime("%Y-%m-%d"))
                    or datetime.now().strftime("%Y-%m-%d"),
                    category=data.get("category", "OTHER") or "OTHER",
                    filename=filename,
                    confidence=float(data.get("confidence", 0.5) or 0.5),
                )

            except json.JSONDecodeError as e:
                st.error(f"Failed to parse GPT response as JSON: {str(e)}")
                st.error(f"Response was: {result_text}")
                return None

        except Exception as e:
            st.error(f"Error calling OpenAI API: {str(e)}")
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

    def auto_prefill_event_info(self, expenses: List[ExpenseData]):
        """Auto-prefill event information from extracted expense data"""
        if not expenses:
            return

        # Extract common patterns from expense descriptions
        descriptions = [exp.description for exp in expenses if exp.description]

        # Generate event name from common patterns
        event_name = self.generate_event_name_from_expenses(expenses)

        # Generate business purpose from expense descriptions
        business_purpose = self.generate_business_purpose_from_expenses(expenses)

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
            # Try to extract name and email from documents
            extracted_name = self.extract_name_from_expenses(expenses)
            extracted_email = self.extract_email_from_expenses(expenses)

            st.session_state.metadata = {
                "first_name": extracted_name.get("first_name", ""),
                "last_name": extracted_name.get("last_name", ""),
                "email": extracted_email,
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
            if not st.session_state.metadata.get("event_name"):
                st.session_state.metadata["event_name"] = event_name
            if not st.session_state.metadata.get("description"):
                st.session_state.metadata["description"] = business_purpose
            if not st.session_state.metadata.get("start_date"):
                st.session_state.metadata["start_date"] = start_date
            if not st.session_state.metadata.get("end_date"):
                st.session_state.metadata["end_date"] = end_date

    def generate_event_name_from_expenses(self, expenses: List[ExpenseData]) -> str:
        """Generate event name from expense patterns"""
        if not expenses:
            return "Business Event"

        # Look for common business event patterns
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

    def generate_business_purpose_from_expenses(
        self, expenses: List[ExpenseData]
    ) -> str:
        """Generate business purpose from expense descriptions"""
        if not expenses:
            return "Business expenses"

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
                "Event Description",
                value=st.session_state.metadata.get("description", ""),
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

    def generate_google_form_prefill_url(
        self, expenses: List[ExpenseData], metadata: Dict
    ) -> str:
        """Generate a prefilled Google Form URL for FSU Travel and Entertainment Request"""
        base_url = "https://docs.google.com/forms/d/e/1FAIpQLScmcTFMDnAxU8FKQyns4BVJDFOa35-B4sCD1VzUFpWvdXfedw/viewform"

        # Calculate total estimated amount
        total_amount = sum(exp.amount for exp in expenses)

        # Generate business purpose from expense descriptions
        business_purpose = f"Business expenses including: {', '.join(set(exp.description[:50] for exp in expenses[:3]))}"
        if len(expenses) > 3:
            business_purpose += f" and {len(expenses) - 3} other expenses"

        # Determine reimbursement type based on expense categories
        has_travel = any(
            "AIRFARE" in exp.category or "ACCOMMODATION" in exp.category
            for exp in expenses
        )
        reimbursement_type = "Travel" if has_travel else "Entertainment"

        # Get the latest expense date
        latest_date = max(datetime.strptime(exp.date, "%Y-%m-%d") for exp in expenses)

        # URL parameters for form prefilling (these are example field IDs - you'll need to inspect the actual form)
        # Note: You'll need to inspect the Google Form HTML to get the actual field entry IDs
        params = {
            # These are placeholder field IDs - replace with actual form field IDs
            "entry.123456789": reimbursement_type,  # Reimbursement Type
            "entry.987654321": latest_date.strftime(
                "%m/%d/%Y"
            ),  # Last date of business travel
            "entry.456789123": metadata.get("event_name", ""),  # Academic Group/Event
            "entry.789123456": f"{metadata.get('first_name', '')} {metadata.get('last_name', '')}",  # Faculty name
            "entry.321654987": business_purpose,  # Business Purpose
            "entry.654987321": f"${total_amount:.2f}",  # Estimated amount
        }

        # URL encode the parameters
        query_string = urllib.parse.urlencode(params)
        prefill_url = f"{base_url}?{query_string}"

        return prefill_url

    def submit_to_google_sheets(
        self, expenses: List[ExpenseData], metadata: Dict
    ) -> bool:
        """Submit data to Google Sheets"""
        client = self.get_google_sheets_client()
        if not client:
            return False

        try:
            # You'll need to replace this with your actual Google Sheet ID
            sheet_id = st.secrets.get("GOOGLE_SHEET_ID") or os.getenv("GOOGLE_SHEET_ID")
            if not sheet_id:
                st.error("Google Sheet ID not configured")
                return False

            sheet = client.open_by_key(sheet_id).sheet1

            # Prepare data for submission
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

                sheet.append_row(row_data)

            return True

        except Exception as e:
            st.error(f"Error submitting to Google Sheets: {str(e)}")
            return False

    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        logger.info("Rendering sidebar")
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
            st.rerun()

    def render_metadata_form(self):
        """Render the metadata collection form"""
        st.header("📝 Event Information")

        # Show helpful message if expenses are already processed
        if st.session_state.expenses:
            st.success(
                f"✅ {len(st.session_state.expenses)} expense(s) already processed! Event details have been auto-prefilled from your documents."
            )

            # Show auto-generated form link
            if st.session_state.metadata:
                prefill_url = self.generate_google_form_prefill_url(
                    st.session_state.expenses, st.session_state.metadata
                )
                st.markdown("---")
                st.success("🎉 **Your FSU Form is Ready!**")
                st.markdown(
                    f"**[🔗 Click here to open your prefilled FSU Travel & Entertainment Form]({prefill_url})**"
                )
                st.info(
                    "📋 The form has been automatically prefilled with your expense data. Review and submit!"
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
                "Event Description",
                value=st.session_state.metadata.get("description", ""),
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
                    st.rerun()
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
                "💡 **Personal details** (name, email) need to be filled manually. **Event details** have been auto-prefilled from your documents!"
            )

            # Show FSU form link if metadata exists
            if st.session_state.metadata:
                prefill_url = self.generate_google_form_prefill_url(
                    st.session_state.expenses, st.session_state.metadata
                )
                st.markdown("---")
                st.success("🎉 **Your FSU Form is Ready!**")
                st.markdown(
                    f"**[🔗 Click here to open your prefilled FSU Travel & Entertainment Form]({prefill_url})**"
                )
                st.info(
                    "📋 The form has been automatically prefilled with your expense data. Review and submit!"
                )

            # Render the inline event form
            self.render_inline_event_form()

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files and extract expense data"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        processed_expenses = []

        for i, file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {file.name}...")

            try:
                # Use GPT-5 direct processing for both PDFs and images
                if file.type == "application/pdf":
                    st.info(f"🚀 Using GPT-5 direct PDF processing for {file.name}")
                    expense_data = self.analyze_expense_with_gpt_direct_pdf(
                        file, file.name
                    )
                    if expense_data:
                        processed_expenses.append(expense_data)
                        st.success(
                            f"✅ Processed {file.name} with GPT-5 direct PDF analysis"
                        )
                    else:
                        st.error(f"❌ Failed to analyze {file.name}")
                else:
                    # For images, use GPT-5 vision directly
                    st.info(f"🚀 Using GPT-5 vision for {file.name}")
                    expense_data = self.analyze_expense_with_gpt_image(file, file.name)
                    if expense_data:
                        processed_expenses.append(expense_data)
                        st.success(f"✅ Processed {file.name} with GPT-5 vision")
                    else:
                        st.error(f"❌ Failed to analyze {file.name}")

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")

        # Update session state
        st.session_state.expenses.extend(processed_expenses)
        st.session_state.processing_complete = True

        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")

        if processed_expenses:
            st.success(f"Successfully processed {len(processed_expenses)} documents!")

            # Auto-prefill event information from extracted data
            self.auto_prefill_event_info(processed_expenses)

            # Set flag to show Event Info content prominently
            st.session_state.show_event_info = True

            # The inline form will be shown in the file upload section
            st.rerun()

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
        st.subheader("📤 Submission Options")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🏫 FSU Travel & Entertainment Form**")
            if st.button(
                "🔗 Open Prefilled FSU Form", type="secondary", use_container_width=True
            ):
                prefill_url = self.generate_google_form_prefill_url(
                    st.session_state.expenses, st.session_state.metadata
                )
                st.markdown(
                    f"[Click here to open the prefilled FSU form]({prefill_url})"
                )
                st.info(
                    "📋 The form has been prefilled with your expense data. You may need to adjust field mappings and add additional required information."
                )

        with col2:
            st.markdown("**📊 Google Sheets Backup**")
            if st.button(
                "📤 Submit to Google Sheets", type="primary", use_container_width=True
            ):
                with st.spinner("Submitting data..."):
                    success = self.submit_to_google_sheets(
                        st.session_state.expenses, st.session_state.metadata
                    )

                    if success:
                        st.success("✅ Data successfully submitted to Google Sheets!")
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
            st.rerun()

    def run(self):
        """Main application runner"""
        logger.info("Starting main run method")
        try:
            logger.info("Setting title and markdown")
            st.title("🤖 AI-Powered Expense Report Generator (GPT-5)")
            st.markdown(
                "Upload your expense documents and let GPT-5 extract and categorize the information automatically with enhanced accuracy and speed!"
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
        logger.error(f"FATAL ERROR in main: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        st.error(f"Fatal error: {str(e)}")
        st.code(traceback.format_exc())
