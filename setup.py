from setuptools import setup, find_packages

setup(
    name="ai-expense-report-generator",
    version="1.0.0",
    description="AI-powered expense report generator using Streamlit and GPT",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "openai>=1.0.0",
        "gspread>=5.10.0",
        "google-auth>=2.20.0",
        "google-auth-oauthlib>=1.0.0",
        "google-auth-httplib2>=0.1.0",
        "PyPDF2>=3.0.0",
        "pdf2image>=1.16.0",
        "pytesseract>=0.3.10",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
