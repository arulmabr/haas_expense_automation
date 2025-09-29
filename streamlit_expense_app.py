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
from pdf2image import convert_from_bytes
import PyPDF2
import pytesseract
from PIL import Image
import requests
import os
from dataclasses import dataclass

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
        self.setup_session_state()

    def setup_session_state(self):
        """Initialize session state variables"""
        if "expenses" not in st.session_state:
            st.session_state.expenses = []
        if "metadata" not in st.session_state:
            st.session_state.metadata = {}
        if "processing_complete" not in st.session_state:
            st.session_state.processing_complete = False

    def get_openai_client(self):
        """Initialize OpenAI client"""
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error(
                "OpenAI API key not found. Please set OPENAI_API_KEY in secrets or environment variables."
            )
            return None
        try:
            return openai.OpenAI(api_key=api_key)
        except Exception as e:
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

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF using PyPDF2 and OCR fallback"""
        try:
            # First try PyPDF2 for text-based PDFs
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n"

            # If we got substantial text, return it
            if len(text.strip()) > 50:
                return text

            # Fallback to OCR for scanned PDFs
            st.info("PDF appears to be scanned. Using OCR to extract text...")
            pdf_file.seek(0)  # Reset file pointer
            images = convert_from_bytes(pdf_file.read())
            ocr_text = ""

            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image)
                ocr_text += f"Page {i+1}:\n{page_text}\n\n"

            return ocr_text

        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def extract_text_from_image(self, image_file) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(image_file)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            st.error(f"Error extracting text from image: {str(e)}")
            return ""

    def analyze_expense_with_gpt(
        self, text: str, filename: str
    ) -> Optional[ExpenseData]:
        """Analyze expense text using GPT-4o"""
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
                model="gpt-4o",  # Use latest GPT-4o frontier model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise expense document analyzer. Always return valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )

            result_text = response.choices[0].message.content.strip()

            # Try to extract JSON from the response
            try:
                # Remove any markdown formatting
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]

                data = json.loads(result_text)

                return ExpenseData(
                    amount=float(data.get("amount", 0)),
                    currency=data.get("currency", "USD"),
                    description=data.get("description", "Unknown expense"),
                    date=data.get("date", datetime.now().strftime("%Y-%m-%d")),
                    category=data.get("category", "OTHER"),
                    filename=filename,
                    confidence=float(data.get("confidence", 0.5)),
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
        st.sidebar.title("🔧 Configuration")

        # API Status
        st.sidebar.subheader("API Status")

        # Check OpenAI
        if self.get_openai_client():
            st.sidebar.success("✅ OpenAI Connected")
        else:
            st.sidebar.error("❌ OpenAI Not Connected")

        # Check Google Sheets
        if self.get_google_sheets_client():
            st.sidebar.success("✅ Google Sheets Connected")
        else:
            st.sidebar.error("❌ Google Sheets Not Connected")

        st.sidebar.markdown("---")

        # Clear data button
        if st.sidebar.button("🗑️ Clear All Data"):
            st.session_state.expenses = []
            st.session_state.metadata = {}
            st.session_state.processing_complete = False
            st.rerun()

    def render_metadata_form(self):
        """Render the metadata collection form"""
        st.header("📝 Event Information")

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
                    median_date = start_date + (end_date - start_date) / 2
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
        if not st.session_state.metadata:
            st.warning("Please fill in the event information first.")
            return

        st.header("📎 Upload Expense Documents")

        uploaded_files = st.file_uploader(
            "Choose PDF or image files",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload receipts, invoices, or other expense documents",
        )

        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary"):
                self.process_uploaded_files(uploaded_files)

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
                # Extract text based on file type
                if file.type == "application/pdf":
                    text = self.extract_text_from_pdf(file)
                else:
                    text = self.extract_text_from_image(file)

                if text.strip():
                    # Analyze with GPT
                    expense_data = self.analyze_expense_with_gpt(text, file.name)
                    if expense_data:
                        processed_expenses.append(expense_data)
                        st.success(f"✅ Processed {file.name}")
                    else:
                        st.error(f"❌ Failed to analyze {file.name}")
                else:
                    st.warning(f"⚠️ No text found in {file.name}")

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")

        # Update session state
        st.session_state.expenses.extend(processed_expenses)
        st.session_state.processing_complete = True

        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")

        if processed_expenses:
            st.success(f"Successfully processed {len(processed_expenses)} documents!")
            st.rerun()

    def render_expense_review(self):
        """Render the expense review and editing interface"""
        if not st.session_state.expenses:
            return

        st.header("📊 Review Extracted Expenses")

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
            return

        st.header("🚀 Submit to Google Sheets")

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

        # Submission button
        st.markdown("---")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📤 Submit to Google Sheets", type="primary"):
                with st.spinner("Submitting data..."):
                    success = self.submit_to_google_sheets(
                        st.session_state.expenses, st.session_state.metadata
                    )

                    if success:
                        st.success("✅ Data successfully submitted to Google Sheets!")
                        st.balloons()

                        # Option to start over
                        if st.button("🔄 Start New Report"):
                            st.session_state.expenses = []
                            st.session_state.metadata = {}
                            st.session_state.processing_complete = False
                            st.rerun()
                    else:
                        st.error(
                            "❌ Failed to submit data. Please check your configuration."
                        )

    def run(self):
        """Main application runner"""
        st.title("🤖 AI-Powered Expense Report Generator")
        st.markdown(
            "Upload your expense documents and let AI extract and categorize the information automatically!"
        )

        # Render sidebar
        self.render_sidebar()

        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📝 Event Info", "📎 Upload", "📊 Review", "🚀 Submit"]
        )

        with tab1:
            self.render_metadata_form()

        with tab2:
            self.render_file_upload()

        with tab3:
            self.render_expense_review()

        with tab4:
            self.render_submission()


# Run the application
if __name__ == "__main__":
    app = ExpenseReportApp()
    app.run()
