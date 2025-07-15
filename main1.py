import os
import re
import uuid
import time
import base64
import fitz
import torch
import hashlib
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ---- Global Setup ----
UPLOAD_FOLDER = "temp_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading model...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-7B-0225-preview",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
).eval()

# ✅ Use compatible processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

model.eval()
print("Model loaded on", device)

image_cache = {}

# ---- FastAPI App ----
app = FastAPI()

# ---- Utilities ----
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_file(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def render_pdf_to_base64png(filepath, page_number=1, target_dim=1024):
    doc = fitz.open(filepath)
    if not 1 <= page_number <= len(doc):
        raise ValueError("Invalid PDF page number")
    page = doc[page_number - 1]
    zoom = target_dim / max(page.rect.width, page.rect.height)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_anchor_text(filepath, page_number=1, max_len=4000):
    doc = fitz.open(filepath)
    if not 1 <= page_number <= len(doc):
        return ""
    return doc[page_number - 1].get_text()[:max_len]

def build_finetuning_prompt(anchor_text):
    return f"Extract and summarize the document. Use this hint if useful: {anchor_text}"

def process_document(filepath):
    start_time = time.time()
    file_hash = hash_file(filepath)

    if file_hash in image_cache:
        image_base64 = image_cache[file_hash]['image_base64']
        image_pil = image_cache[file_hash]['image_pil']
    else:
        if filepath.endswith('.pdf'):
            image_base64 = render_pdf_to_base64png(filepath)
            anchor_text = get_anchor_text(filepath)
        else:
            with open(filepath, 'rb') as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            anchor_text = ""

        image_pil = Image.open(BytesIO(base64.b64decode(image_base64)))
        if max(image_pil.size) > 1024:
            image_pil.thumbnail((1024, 1024))
        image_cache[file_hash] = {
            'image_base64': image_base64,
            'image_pil': image_pil
        }

    prompt = build_finetuning_prompt(anchor_text if filepath.endswith('.pdf') else "")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], images=[image_pil], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=512  # ✅ Only supported generation arg
    )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_len:]
    response = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    print(f"Processed in {time.time() - start_time:.2f} seconds")
    return response, time.time() - start_time

def parse_output(raw):
    structured_data = {
        "entities": {"names": [], "dates": [], "addresses": []},
        "tables": [],
        "raw_text": raw
    }

    name_patterns = [
        r'(Customer Name|Patient Name|Name):\s*(.+)',
        r'([A-Z][a-z]+ [A-Z][a-z]+)',
    ]
    for pattern in name_patterns:
        matches = re.findall(pattern, raw, re.IGNORECASE)
        for match in matches:
            name = match[1].strip() if isinstance(match, tuple) else match.strip()
            if name and name not in structured_data["entities"]["names"]:
                structured_data["entities"]["names"].append(name)

    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{2}/\d{2}/\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{2,4})',
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, raw)
        for match in matches:
            if match not in structured_data["entities"]["dates"]:
                structured_data["entities"]["dates"].append(match)

    address_patterns = [
        r'(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl|Way|Circle|Cir|Terrace|Ter))',
        r'Address:\s*(.+)',
    ]
    for pattern in address_patterns:
        matches = re.findall(pattern, raw, re.IGNORECASE)
        for match in matches:
            if match and match not in structured_data["entities"]["addresses"]:
                structured_data["entities"]["addresses"].append(match.strip())

    lines = raw.split('\n')
    table_data, in_table, headers = [], False, []
    for line in lines:
        line = line.strip()
        if '|' in line and any(char.isdigit() for char in line):
            if not in_table:
                headers = [h.strip() for h in line.split('|')]
                in_table = True
            else:
                row = [cell.strip() for cell in line.split('|')]
                if len(row) == len(headers):
                    table_data.append(row)
        elif in_table and not line:
            break
    if table_data and headers:
        structured_data["tables"].append({"headers": headers, "rows": table_data})

    return structured_data


# ---- Routes ----
@app.get("/")
def read_root():
    return {"message": "API is running! Send files to /extract"}

@app.post("/extract")
async def extract_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)

        with open(filepath, "wb") as f:
            f.write(await file.read())

        print(f"Processing: {filepath}")
        raw_text, duration = process_document(filepath)
        parsed = parse_output(raw_text)

        return JSONResponse({
            "message": "File processed successfully!",
            "filename": file.filename,
            "time_taken_sec": round(duration, 2),
            "extracted_data": parsed
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleaned up: {filepath}")

# (base) PS C:\Users\saivi> conda activate PIAZZA
# (PIAZZA) PS C:\Users\saivi> cd downloads/PIAZZA
# (PIAZZA) PS C:\Users\saivi\downloads\PIAZZA> uvicorn main:app --reload --port 5000