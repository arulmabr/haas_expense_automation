# How to View Streamlit Cloud Logs

## 📋 Viewing Logs on Streamlit Cloud

### Method 1: From the Streamlit Cloud Dashboard
1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Find your app: **haas-expenses-ai**
4. Click on the app name
5. Click the **"⋮" menu** (three dots) on the right
6. Select **"Manage app"**
7. Click on the **"Logs"** tab
8. You'll see real-time logs with our debug output!

### Method 2: Direct URL
Go to: `https://share.streamlit.io/[your-username]/haas_expense_automation`

### Method 3: From the App
If the app is stuck or erroring:
1. The error should appear on screen
2. Check the browser console (F12) for JavaScript errors
3. Check the Streamlit Cloud logs for Python errors

## 🔍 What to Look For in Logs

Our logging shows exactly where the app is:

```
INFO - ================================================== 
INFO - Application starting
INFO - ==================================================
INFO - Initializing ExpenseReportApp
INFO - Setting up session state
INFO - Session state setup complete
INFO - ExpenseReportApp initialized successfully
INFO - Starting main run method
INFO - Setting title and markdown
INFO - About to render sidebar
INFO - Rendering sidebar
INFO - OpenAI API key found
INFO - Google Sheets credentials found
INFO - Sidebar rendered successfully
INFO - Creating tabs
INFO - Rendering tab 1
INFO - Rendering tab 2
INFO - Rendering tab 3
INFO - Run method completed successfully
```

### If the app is stuck, the logs will stop at a specific line, showing exactly where the issue is!

## 🐛 Common Issues

### Stuck after "Preparing system..."
- Check logs for import errors
- Missing dependencies in requirements.txt

### Stuck after "Spinning up manager process..."
- Check if secrets are configured
- Look for "OpenAI API key missing" or "Google Sheets credentials missing"

### Blank page
- Check browser console (F12)
- Look for JavaScript errors

## ✅ Expected Output (Healthy App)

When working correctly, you should see ALL these log messages in order:
1. "Application starting"
2. "Initializing ExpenseReportApp"
3. "Session state setup complete"
4. "Starting main run method"
5. "Sidebar rendered successfully"
6. "Creating tabs"
7. "Run method completed successfully"

If logs stop before "Run method completed successfully", that's where the issue is!
