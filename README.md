# AI-Powered Expense Report Generator

A Streamlit-based application that uses GPT-4/5 to automatically extract expense information from PDF receipts and images, then submits the data to Google Sheets for form processing.

## Features

- 🤖 **AI-Powered Extraction**: Uses OpenAI GPT-4/5 to intelligently parse expense documents
- 📄 **Multi-Format Support**: Handles PDFs (text and scanned) and images (PNG, JPG, JPEG)
- 🔄 **OCR Fallback**: Automatically uses OCR for scanned documents
- 💱 **Exchange Rate Integration**: Fetches real-time rates from Bank of Canada
- 📊 **Interactive Review**: Edit and verify extracted data before submission
- 📤 **Google Sheets Integration**: Automatically submits data to configured Google Sheet
- 🎨 **Modern UI**: Clean, responsive Streamlit interface

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- OpenAI API key (GPT-4 access recommended)
- Google Cloud Project with Sheets API enabled
- Tesseract OCR installed (for image processing)

### 2. Installation

```bash
# Clone or download the project
cd expense-report-generator

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (platform-specific)
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Configuration

#### OpenAI API Setup

1. Get your OpenAI API key from https://platform.openai.com/api-keys
2. Ensure you have access to GPT-4 (recommended for better accuracy)

#### Google Sheets Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Sheets API
4. Create a Service Account:
   - Go to IAM & Admin > Service Accounts
   - Click "Create Service Account"
   - Download the JSON credentials file
5. Create a Google Sheet and note the Sheet ID from the URL
6. Share the Google Sheet with your service account email (with Editor permissions)

#### Configure Secrets

Edit `.streamlit/secrets.toml` with your actual values:

```toml
# OpenAI Configuration
OPENAI_API_KEY = "sk-your-actual-openai-key"

# Google Sheets Configuration
GOOGLE_SHEET_ID = "your-actual-sheet-id"

# Google Service Account Credentials
[google_credentials]
type = "service_account"
project_id = "your-actual-project-id"
private_key_id = "your-actual-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR-ACTUAL-PRIVATE-KEY\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-actual-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
```

### 4. Running the Application

```bash
streamlit run streamlit_expense_app.py
```

The app will be available at `http://localhost:8501`

## Usage Guide

### Step 1: Event Information

- Fill in your personal details and event information
- Select any currencies other than CAD that appear in your receipts
- The app will automatically fetch exchange rates from Bank of Canada

### Step 2: Upload Documents

- Upload PDF receipts, invoices, or images
- The app supports multiple file formats and batch processing
- AI will automatically extract key information from each document

### Step 3: Review & Edit

- Review the extracted expense data
- Edit any fields that need correction
- Check confidence scores for extraction quality
- View CAD conversions using current exchange rates

### Step 4: Submit

- Review the final summary
- Submit all data to your configured Google Sheet
- Data includes timestamps and confidence scores for audit purposes

## Supported Expense Categories

- AIRFARE
- ACCOMMODATION (In Canada / Outside Canada)
- RAILWAY/BUS/TAXI (In Canada / Outside Canada)
- CAR RENTAL (In Canada / Outside Canada)
- MEALS (In Canada / Outside Canada)
- OTHER

## Deployment Options

### Streamlit Community Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Configure secrets in the Streamlit dashboard
5. Deploy!

### Heroku

```bash
# Add Procfile
echo "web: streamlit run streamlit_expense_app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your-key
heroku config:set GOOGLE_SHEET_ID=your-sheet-id
# Add other config vars as needed
git push heroku main
```

### Railway

1. Connect your GitHub repo to Railway
2. Add environment variables in the Railway dashboard
3. Deploy automatically on git push

## Security Notes

- Never commit API keys or credentials to version control
- Use environment variables or Streamlit secrets for sensitive data
- Regularly rotate API keys and service account credentials
- Consider implementing additional access controls for production use

## Troubleshooting

### Common Issues

1. **"OpenAI Not Connected"**

   - Check your API key is correct and has GPT-4 access
   - Verify you have sufficient API credits

2. **"Google Sheets Not Connected"**

   - Ensure the service account JSON is properly formatted
   - Check that the Google Sheet is shared with the service account email
   - Verify the Sheets API is enabled in your Google Cloud project

3. **OCR Not Working**

   - Install Tesseract OCR for your platform
   - For Docker deployments, add Tesseract to your Dockerfile

4. **Exchange Rate Errors**
   - Bank of Canada API may be temporarily unavailable
   - The app will fall back to manual entry prompts

### Performance Tips

- For large batches of documents, process in smaller chunks
- Higher confidence scores generally indicate better extraction quality
- Review low-confidence extractions manually

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## License

This project is open source and available under the MIT License.
