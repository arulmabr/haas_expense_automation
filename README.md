# AI Expense Report Generator

GPT-5.5 powered expense report automation for Haas School of Business.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Configure secrets in `.streamlit/secrets.toml`
   - `APP_PASSWORD`: shared password required to open the app
   - `OPENAI_API_KEY`: required for document processing
   - `OPENAI_MODEL`: optional, defaults to `gpt-5.5`
   - `GOOGLE_SHEET_ID` and `google_credentials`: required for Google Sheets submission
   - `S3_BUCKET_NAME` and AWS credentials: optional, for receipt PDF upload links
3. Run: `streamlit run app.py`

## Streamlit Cloud Deployment

The app is deployed at Streamlit Cloud with:
- Python 3.11
- GPT-5.5 by default for document processing
- Google Sheets integration
