# 🚀 Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Repository**: Your code should be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Google Cloud Project**: With Sheets API enabled
4. **OpenAI API Key**: From OpenAI platform (supports GPT-5)

## 🔒 Security Checklist

### Before Deployment:

- [ ] Verify `.gitignore` excludes all sensitive files
- [ ] Remove any hardcoded credentials from code
- [ ] Test app works with environment variables locally

### Files to NEVER commit:

- `.streamlit/secrets.toml`
- `*.json` (Google credentials)
- `.env` files
- Any file containing API keys

## 📋 Step-by-Step Deployment

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Select your repository and branch (main)
5. Set main file path: `streamlit_expense_app.py`
6. Click "Deploy"

### Step 3: Configure Secrets in Streamlit Cloud

1. In your deployed app, click the "⚙️ Settings" menu
2. Go to "Secrets" tab
3. Copy and paste the content from `secrets_template.toml`
4. Replace all placeholder values with your actual credentials:

#### OpenAI API Key:

```toml
OPENAI_API_KEY = "sk-proj-your-actual-key-here"
```

#### Google Sheet ID:

```toml
GOOGLE_SHEET_ID = "1bR8xFXIwwD6Ll9usmBlYsG9kUY5NJ_kuqbxgPXsagv0"
```

#### Google Credentials:

```toml
[google_credentials]
type = "service_account"
project_id = "haas-expense-automation"
private_key_id = "31f56dafa983ec7330962beefea0ffb310b5a36d"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_ACTUAL_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n"
client_email = "haas-expense-automation-servic@haas-expense-automation.iam.gserviceaccount.com"
client_id = "112859806853894524774"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/haas-expense-automation-servic%40haas-expense-automation.iam.gserviceaccount.com"
universe_domain = "googleapis.com"
```

### Step 4: Save and Restart

1. Click "Save" in the secrets section
2. The app will automatically restart with the new secrets

## 🔧 Google Cloud Setup

### Enable Required APIs:

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Select your project: `haas-expense-automation`
3. Navigate to "APIs & Services" > "Library"
4. Enable these APIs:
   - Google Sheets API
   - Google Drive API

### Service Account Permissions:

1. Go to "IAM & Admin" > "Service Accounts"
2. Find your service account: `haas-expense-automation-servic@...`
3. Ensure it has these roles:
   - Editor (or custom role with Sheets/Drive access)

### Google Sheet Permissions:

1. Open your target Google Sheet
2. Click "Share" button
3. Add your service account email with "Editor" permissions:
   `haas-expense-automation-servic@haas-expense-automation.iam.gserviceaccount.com`

## 🧪 Testing Deployment

### Test Checklist:

- [ ] App loads without errors
- [ ] OpenAI connection shows green ✅
- [ ] Google Sheets connection shows green ✅
- [ ] Can upload and process documents
- [ ] Can submit data to Google Sheets

## 🚨 Security Best Practices

1. **Never commit secrets to git**
2. **Use environment-specific secrets**
3. **Regularly rotate API keys**
4. **Monitor usage and access logs**
5. **Use least-privilege principle for service accounts**

## 🔍 Troubleshooting

### Common Issues:

**"OpenAI Not Connected"**

- Check OPENAI_API_KEY in secrets
- Verify key is valid and has credits

**"Google Sheets Not Connected"**

- Verify all google_credentials fields are correct
- Check service account has proper permissions
- Ensure Google Sheets/Drive APIs are enabled

**"Invalid OAuth scope"**

- Verify service account has Sheets/Drive access
- Check the scopes in the code are correct

### Debug Mode:

Add this to your secrets for debugging:

```toml
DEBUG = true
```

## 📞 Support

If you encounter issues:

1. Check Streamlit Cloud logs
2. Verify all secrets are properly formatted
3. Test credentials locally first
4. Check Google Cloud audit logs

---

**Remember**: Keep your credentials secure and never share them publicly!
