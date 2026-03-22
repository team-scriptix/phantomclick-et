# PhantomClick Python Backend

Backend service for PhantomClick built with FastAPI and Tesseract OCR for extracting text from SMS screenshots.

---

## Setup Guide

### 1. Clone the project

git clone <repo-url>  
cd server/python-server

---

### 2. Create virtual environment (recommended)

#### Windows
python -m venv venv  
venv\Scripts\activate

#### macOS / Linux
python3 -m venv venv  
source venv/bin/activate

---

### 3. Install dependencies

pip install -r requirements.txt

---

## Install Tesseract OCR (Required)

This project uses pytesseract to extract text from SMS screenshots.

pytesseract will not work unless Tesseract is installed on your system.

### Windows

Download installer:  
https://github.com/UB-Mannheim/tesseract/wiki

Downloaded this:

tesseract-ocr-w64-setup-5.5.0.20241111.exe (64 bit)

Install Tesseract and add the following to your PATH:

C:\Program Files\Tesseract-OCR

Verify installation:

tesseract --version

---

## Environment Variables

Create a `.env` file in the root directory.

GEMINI_API_KEY=

Add any other required environment variables used by the project.

---

## Run the server

From the backend folder:

uvicorn main:app --host 0.0.0.0 --port 8001 --reload

---

## API

After starting the server:

Application  
http://localhost:8001


