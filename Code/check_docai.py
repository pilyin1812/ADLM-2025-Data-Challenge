# save as check_docai.py and run:  python check_docai.py
from google.oauth2 import service_account
from google.cloud import documentai_v1 as documentai
import json, pathlib

PROJECT_ID = ""
LOCATION = "us"
PROCESSOR_ID = ""   # try your LayoutParser ID; if pages=0, try DocOCR ID too
SERVICE_ACCOUNT_JSON = ""

creds = service_account.Credentials.from_service_account_info(
    json.loads(SERVICE_ACCOUNT_JSON),
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
client = documentai.DocumentProcessorServiceClient(credentials=creds)
name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)
print("Processor:", name)

proc = client.get_processor(name=name)
print("State:", proc.state.name, "| Type:", proc.type_)

pdf_path = "C:\Users\piatm\OneDrive\Documents\VCapture\PeterIlyinESE3040HW2.pdf"
raw = documentai.RawDocument(content=pathlib.Path(pdf_path).read_bytes(), mime_type="application/pdf")
resp = client.process_document(request=documentai.ProcessRequest(name=name, raw_document=raw))
doc = resp.document
print("len(doc.pages):", len(doc.pages), "| text length:", len(doc.text or ""))
