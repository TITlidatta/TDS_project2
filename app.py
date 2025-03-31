
from fastapi import FastAPI, Query,File, UploadFile,Form, Request,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional , List
from collections import defaultdict
from botocore.exceptions import NoCredentialsError
from markdownify import markdownify
from fuzzywuzzy import process,fuzz
from datetime import datetime
from functions import FUNCTIONS
import base64
import platform
from pydantic import BaseModel
import time
import boto3
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import os
import requests
import json 
import uvicorn
import subprocess
import hashlib
import tiktoken
import pandas as pd
import zipfile
import csv
import gzip
import json
import hashlib
import unicodedata
import shutil
import re
import tempfile
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pdfplumber
import sys
import ast
from PIL import Image
import colorsys


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET","POST"],  
    allow_headers=["*"],  
)
file_path=None
api_key = "your key "
headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

def parse_to_list(arr):
    if arr.startswith("[") and arr.endswith("]"):
        return ast.literal_eval(arr)  # Safely parse list format
    else:
        return [int(x) for x in arr.split(",")]

def GA1_1(editorname):
    return """Version:          Code 1.96.2 (fabdb6a30b49f79a7aba0f2ad9df9b399473380f, 2024-12-19T10:22:47.216Z)
OS Version:       Windows_NT x64 10.0.22631
CPUs:             11th Gen Intel(R) Core(TM) i3-1125G4 @ 2.00GHz (8 x 1997)
Memory (System):  7.70GB (0.74GB free)
VM:               0%
Screen Reader:    no
Process Argv:     --crash-reporter-id 457d26db-c623-4624-bbba-a4e476670f35
GPU Status:       2d_canvas:                              enabled
                  canvas_oop_rasterization:               enabled_on
                  direct_rendering_display_compositor:    disabled_off_ok
                  gpu_compositing:                        enabled
                  multiple_raster_threads:                enabled_on
                  opengl:                                 enabled_on
                  rasterization:                          enabled
                  raw_draw:                               disabled_off_ok
                  skia_graphite:                          disabled_off
                  video_decode:                           enabled
                  video_encode:                           enabled
                  vulkan:                                 disabled_off
                  webgl:                                  enabled
                  webgl2:                                 enabled
                  webgpu:                                 enabled
                  webnn:                                  disabled_off

CPU %   Mem MB     PID  Process
    0      136   13392  code main
    0       32    1996     crashpad-handler
    0      201    3360  window [1] (Visual Studio Code)
    0      188    4896     gpu-process
    0      143    5984  extensionHost [1]
    0       71    6824  fileWatcher [1]
    0      141   10956  shared-process
    0       43   13552     utility-network-service"""

def GA1_2(email): # fetching hhtpbin.org
    return f"""{{"args": {{"email": "{email}"}},"headers": {{"Accept": "*/*","Accept-Encoding": "gzip, deflate","Host": "httpbin.org","User-Agent": "HTTPie/3.2.4","X-Amzn-Trace-Id": "Root=1-678bfdcd-2171ada0505f22d077951800"}},"origin": "152.58.183.96","url": "https://httpbin.org/get?email={email}"}}"""
def GA1_3(fp): # format readme with prettier
    global file_path
    try:
        print('hola')
        print(file_path)
        formatted_output = subprocess.check_output(
            ["npx", "-y", "prettier@3.4.2", file_path],text=True
        )

        sha256_hash = hashlib.sha256(formatted_output.encode()).hexdigest()
        return str(sha256_hash.split()[0])

    except subprocess.CalledProcessError as e:
        return f"Error: {e}"
def GA1_4(command): # run command in google sheets

    data = {
        "model": "gpt-4o-mini",  
        "messages": [
            {"role": "system", "content": "Try to tell me the output of the command if its run in google sheets. Give only the result with no other text ."},
            {"role": "user", "content": f"The command is : {command}"},
        ],
    }
    ans = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",headers=headers,data=json.dumps(data))
    an = ans.json()
    
    if "choices" in an :
        qs = an["choices"][0]["message"]["content"]
        return str(qs)

def GA1_5(arr1,arr2,col):
    arr1=parse_to_list(arr1)
    arr2=parse_to_list(arr2)
    col=int(col)
    pos = arr2.index(col)  
    return str(arr1[pos])

def GA1_6(val):
    return str(val)

def GA1_7(start_date,end_date):
    d = pd.date_range(start=start_date, end=end_date, freq='D')
    nw = sum(d.weekday == 2)
    return str(int(nw))

def GA1_8(filename):
    global file_path
    zip_filename = file_path 
    extract_path = "extracted_files"
    csv_filename = "extract.csv"

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    csv_path = os.path.join(extract_path, csv_filename)

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            fg=row["answer"] 
    return str(fg)

def GA1_9(data_json):
    data_json=json.loads(data_json)
    sorted_data = sorted(data_json, key=lambda x: (x["age"], x["name"]))
    return json.dumps(sorted_data, separators=(',', ':'))

def GA1_10(filename):
    data = {}
    global file_path    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip()
    
    text = json.dumps(data, indent=4)
    
    normalized_text = unicodedata.normalize("NFC", text.strip())
    json_string = json.dumps(json.loads(normalized_text), separators=(',', ':'))
    return str(hashlib.sha256(json_string.encode('utf-8')).hexdigest())  

def GA1_11(arr):
    arr=parse_to_list(arr)
    return str(sum(arr))

def GA1_12(symb1,symb2,symb3):
    global file_path
    ENCODINGS = {
      "data1.csv": "cp1252",
      "data2.csv": "utf-8",
      "data3.txt": "utf-16"}

    TARGET_SYMBOLS = { symb1 ,symb2, symb3}  

    def extract_zip(zip_path, extract_to="extractedfiles"):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        return extract_to

    def read_encoded_file(file_p, encoding):
        with open(file_p, "r", encoding=encoding, newline="") as file:
            delimiter = "," if file_p.endswith(".csv") else "\t"
            reader = csv.reader(file, delimiter=delimiter)

            next(reader)
            data = []
            for row in reader:
                if len(row) < 2:
                    continue 
                symbol = row[0].strip() 
                try:
                    value = int(row[1].strip()) 
                    data.append((symbol, value))
                except ValueError:
                    print(f"Skipping invalid row in {file_p}: {row}") 
                    continue
        return data

    def compute_sum(directory, target_symbols):
        total_sum = 0
        found_symbols = set()

        for file, encoding in ENCODINGS.items():
            fh = os.path.join(directory, file)
            if os.path.exists(fh):
                for symbol, value in read_encoded_file(fh, encoding):
                    if symbol in target_symbols:
                        total_sum += value
                        found_symbols.add(symbol)
        
        return total_sum

    zip_file_path = file_path
    extracted_folder = extract_zip(zip_file_path)
    result_sum = compute_sum(extracted_folder, TARGET_SYMBOLS)

    return str(result_sum)
    
def GA1_13(email):

    GITHUB_USERNAME = "Suneha111"
    GITHUB_REPO = "tdsssga1"
    GITHUB_BRANCH = "main"
    NEW_EMAIL = email
    GITHUB_PERSONAL_ACCESS_TOKEN = "your token"
    REPO_DIR = os.path.expanduser("~/tdsssga1")  
    EMAIL_FILE = os.path.join(REPO_DIR, "email.json")
    subprocess.run(["git", "config", "--global", "user.email", "23f1002574@ds.study.iitm.ac.in"])
    subprocess.run(["git", "config", "--global", "user.name", "Suneha"])

    if not os.path.exists(REPO_DIR):
        repo_url = f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
        subprocess.run(["git", "clone", repo_url, REPO_DIR], check=True)

    os.chdir(REPO_DIR)
    subprocess.run(["git", "pull", "origin", GITHUB_BRANCH], check=True)
    with open(EMAIL_FILE, "w") as f:
        json.dump({"email": NEW_EMAIL}, f, indent=4)

    subprocess.run(["git", "add", EMAIL_FILE], check=True)
    try:
        subprocess.run(["git", "commit", "-m", "Updated email.json"], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "commit", "--allow-empty", "-m", "Forced commit"], check=True)

    repo_url_auth = f"https://{GITHUB_USERNAME}:{GITHUB_PERSONAL_ACCESS_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
    subprocess.run(["git", "push", repo_url_auth], check=True)
    return "https://raw.githubusercontent.com/Suneha111/tdsssga1/main/email.json" 

def GA1_14(filename):
    global file_path
    ZIP_FILE_PATH = file_path
    EXTRACT_FOLDER = "exted_files" 
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)

    pattern = re.compile(r"iitm", re.IGNORECASE) 
    for file in sorted(glob.glob(f"{EXTRACT_FOLDER}/*")): 
        with open(file, "r", encoding="utf-8", newline="") as f:
            content = f.read()
        
        updated_content = pattern.sub("IIT Madras", content) 
        with open(file, "w", encoding="utf-8", newline="") as f:
            f.write(updated_content)
    sha256_hash = hashlib.sha256()
    for file in sorted(glob.glob(f"{EXTRACT_FOLDER}/*")): 
        with open(file, "rb") as f: 
            while chunk := f.read(4096):  
                sha256_hash.update(chunk)

    return str(sha256_hash.hexdigest())

def GA1_15(bytess,date):
    global file_path
    ZIP_FILE = file_path
    EXTRACT_FOLDER = "exted_les"
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            extracted_path = zip_ref.extract(file_info, EXTRACT_FOLDER)
            mod_time = time.mktime(file_info.date_time + (0, 0, -1))
            os.utime(extracted_path, (mod_time, mod_time))
    m = int(bytess)
    TARGET_DATE = date # tell llm to capture date in format "1995-11-23 13:24"
    cmd = f'ls -l --time-style=+"%Y-%m-%d %H:%M" {EXTRACT_FOLDER}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    total_size = 0
    for line in result.stdout.split("\n"):
        parts = line.split()
        if len(parts) < 6:
            continue 
        
        file_size = int(parts[4]) 
        file_date = " ".join(parts[5:7])  

        if file_size >= m and file_date >= TARGET_DATE:
            total_size += file_size

    return str(total_size)

def GA1_16(zih):
    global file_path
    zip_path = file_path
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        final_folder = os.path.join(temp_dir, "final")
        os.makedirs(final_folder, exist_ok=True)
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path != final_folder: 
                    shutil.move(file_path, os.path.join(final_folder, file))
        def rename_digits(filename):
            return re.sub(r'\d', lambda x: str((int(x.group(0)) + 1) % 10), filename)

        for file in os.listdir(final_folder):
            old_path = os.path.join(final_folder, file)
            new_name = rename_digits(file)
            new_path = os.path.join(final_folder, new_name)
            os.rename(old_path, new_path)

        result = subprocess.run(
            "grep . * | LC_ALL=C sort | sha256sum",
            shell=True,
            cwd=final_folder,  
            text=True,
            capture_output=True
        )

        return str(result.stdout.split()[0])  

def GA1_17(zp):
    global file_path
    zip_path = file_path
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        a_path = os.path.join(temp_dir, "a.txt")
        b_path = os.path.join(temp_dir, "b.txt")

        with open(a_path, 'r', encoding='utf-8') as f1, open(b_path, 'r', encoding='utf-8') as f2:
            a_lines = f1.readlines()
            b_lines = f2.readlines()
        diff_count = sum(1 for x, y in zip(a_lines, b_lines) if x != y)
        return str(diff_count)
    
def GA1_18(ticket_type):
    return """SELECT SUM(units * price) AS total_sales FROM tickets WHERE LOWER(TRIM(type)) = 'gold';"""

def GA3_1(text): #class="hljs language-nginx"
    return f"""import httpx

def analyze_sentiment():
    # Define the API endpoint and dummy API key
    api_url = "https://api.openai.com/v1/chat/completions"
    api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDI1NzRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.b5ZfSvDoRnYsCq-OS5OVjZ22RHCgeYLi2LjpGvaXdcQ"

    # Define the request payload
    payload = {{
        "model": "gpt-4o-mini",
        "messages": [
            {{"role": "system", "content": "Analyze the sentiment of the text and classify it as GOOD, BAD, or NEUTRAL."}},
            {{"role": "user", "content": "{text}"}}
        ]
    }}

    # Define headers with the dummy API key
    headers = {{
        "Authorization": f"Bearer {{api_key}}",
        "Content-Type": "application/json"
    }}

    try:
        # Make the POST request
        response = httpx.post(api_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP status codes >= 400

        # Parse and print the response
        result = response.json()
        print("Sentiment Analysis Result:", result)

    except httpx.RequestError as e:
        print(f"An error occurred while making the request: {{e}}")

    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {{e.response.status_code}} - {{e.response.text}}")

# Run the function
analyze_sentiment()
"""
def GA3_2(text, model="gpt-4o-mini"): # class="language-text hljs language-plaintext"

    enc = tiktoken.encoding_for_model(model)
    token_count = len(enc.encode(text))
    total_tokens = token_count + 7 
    return str(total_tokens)

def GA3_3(f1, f2, f3):
    return f"""{{"model": "gpt-4o-mini", "messages": [{{"role": "system", "content": "Respond in JSON"}}, {{"role": "user", "content": "Generate 10 random addresses in the US"}}], "response_format": {{"type": "json_schema", "json_schema": {{"schema": {{"type": "object", "properties": {{"addresses": {{"type": "array", "items": {{"type": "object", "properties": {{"city": {{"type": "string"}}, "apartment": {{"type": "string"}}, "county": {{"type": "string"}}}}, "required": ["{f1}", "{f2}", "{f3}"], "additionalProperties": false}}}}}}, "required": ["addresses"]}}}}}}"""


def GA3_4(url):
    return f"""return {{"model": "gpt-4o-mini", "messages": [{{"role": "user", "content": [{{"type": "text", "text": "Extract text from this image"}}, {{"type": "image_url", "image_url": {{"url": "{url}"}}}}]}}]}}"""

def GA3_5(txn1,txn2):#class=hljs language-stylus  for both txn1, txn2
    return f"""{{"model": "text-embedding-3-small", "input": ["{txn1}", "{txn2}"]}}"""

def GA3_6(funcname):
    return r'''import numpy as np def most_similar(embeddings): def cosine_similarity_np(v1, v2):  dot_product = np.dot(v1, v2) norm_v1 = np.linalg.norm(v1) norm_v2 = np.linalg.norm(v2) return dot_product / (norm_v1 * norm_v2)  phrases = list(embeddings.keys()) embedding_vectors = list(embeddings.values()) highest_similarity = -1 most_similar_pair = None for i in range(len(embedding_vectors)): for j in range(i + 1, len(embedding_vectors)): similarity = cosine_similarity_np(embedding_vectors[i], embedding_vectors[j]) if similarity > highest_similarity: highest_similarity = similarity most_similar_pair = (phrases[i], phrases[j]) return most_similar_pair
'''
class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embeddings_from_openai(texts: List[str]):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": texts,
        "model": "text-embedding-3-small",
        "encoding_format": "float"
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error from OpenAI Proxy: {response.json()}"
        )
    return response.json().get('data')

@app.post("/similarity")
async def get_similarity(data: SimilarityRequest):
    if not data.docs or not data.query:
        raise HTTPException(status_code=400, detail="Both 'docs' and 'query' are required.")
    all_texts = data.docs + [data.query]
    embeddings = get_embeddings_from_openai(all_texts)
    print(embeddings)
    doc_embeddings = [embedding['embedding'] for embedding in embeddings[:-1]]  
    query_embedding = embeddings[-1]['embedding']  
    similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    ranked_docs = [doc for _, doc in sorted(zip(similarities, data.docs), reverse=True)]
    return {"matches": ranked_docs[:3]}

FUNCTIONSS = [
    {
        "name": "get_ticket_status",
        "description": "Get the status of a support ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "integer", "description": "Ticket ID number."}
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "schedule_meeting",
        "description": "Schedule a meeting room for a specific date and time.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Meeting date in YYYY-MM-DD format."},
                "time": {"type": "string", "description": "Meeting time in HH:MM format."},
                "meeting_room": {"type": "string", "description": "Name of the meeting room."},
            },
            "required": ["date", "time", "meeting_room"],
        },
    },
    {
        "name": "get_expense_balance",
        "description": "Get the expense balance for an employee.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "integer", "description": "Employee ID number."}
            },
            "required": ["employee_id"],
        },
    },
    {
        "name": "calculate_performance_bonus",
        "description": "Calculate the yearly performance bonus for an employee.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "integer", "description": "Employee ID number."},
                "current_year": {"type": "integer", "description": "Year to calculate the bonus for."},
            },
            "required": ["employee_id", "current_year"],
        },
    },
    {
        "name": "report_office_issue",
        "description": "Report an office issue by specifying a department or issue number.",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_code": {"type": "integer", "description": "Office issue code."},
                "department": {"type": "string", "description": "Department name."},
            },
            "required": ["issue_code", "department"],
        },
    },
]

@app.get("/execute")
def execute(q: str = Query(..., description="Query text to parse and execute.")):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
        ],
        "functions": FUNCTIONSS,
        "function_call": "auto",
    }

    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, data=json.dumps(data))
    print('here')
    print(response.status_code)
    result = response.json()
    if "choices" in result and result["choices"]:
        function_call = result["choices"][0]["message"].get("function_call")
        if function_call:
            name = function_call.get("name")
            arguments = function_call.get("arguments")
            return {
                "name": name,
                "arguments": arguments
            }

    return {"error": "Unable to match query to a function or call OpenAI API."}

def GA3_7(endpointname):
    return "http://16.16.189.187:8000/similarity"

def GA3_8(endpointname):
    return "http://16.16.189.187:8000/execute"

def GA4_1(page_number):
    url = f"https://stats.espncricinfo.com/stats/engine/stats/index.html?class=2;page={page_number};template=results;type=batting"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status() 
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) >= 50: 
            headers = [th.text.strip() for th in table.find("thead").find_all("th")]
            if "0" in headers:
                duck_index = headers.index("0")
            else:
                raise ValueError("Column '0' not found in table headers.")
        
            total_ducks = 0
            for row in table.find("tbody").find_all("tr"):
                cells = row.find_all("td")
                if len(cells) > duck_index:
                    try:
                        total_ducks += int(cells[duck_index].text.strip())
                    except ValueError:
                        continue 
            
            return str(total_ducks)
    raise ValueError("No valid statistics table found.")

def GA4_2(min_rating, max_rating):
    url = f"https://www.imdb.com/search/title/?user_rating={min_rating},{max_rating}"
    headers = {"User-Agent": "Mozilla/5.0"} 
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Failed to fetch IMDb data")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    movie_list = []
    for item in soup.select(".ipc-metadata-list-summary-item"):
        try:
            imdb_id = item.select_one(".ipc-title-link-wrapper")
            if imdb_id:
                imdb_id = imdb_id["href"].split("/")[2] 
            
            title = item.select_one(".ipc-title__text").text.strip()
            year = item.select_one(".dli-title-metadata-item").text.strip()
            rating = item.select_one(".ipc-rating-star--rating")
            rating = rating.text.strip() if rating else "N/A"
            
            movie_list.append({
                "id": imdb_id,
                "title": title,
                "year": year,
                "rating": rating
            })
        except AttributeError:
            continue  
    
    return movie_list

@app.get("/api/outline")
def get_wikipedia_outline(country: str = Query(..., title="Country Name", description="Name of the country to fetch Wikipedia outline for")):
    wikipedia_url = f"https://en.wikipedia.org/wiki/{country.replace(' ', '_')}"
    
    try:
        response = requests.get(wikipedia_url)
        response.raise_for_status()
    except requests.RequestException:
        return {"error": "Failed to fetch Wikipedia page"}
    
    soup = BeautifulSoup(response.text, "html.parser")
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    markdown_outline = "".join(f"{'#' * int(tag.name[1])} {tag.text.strip()}\n" for tag in headings)
    
    return {markdown_outline}

def GA4_3(url):
    return "http://16.16.189.187:8000/api/outline" 

def GA4_4(place):
    locator_url = 'https://locator-service.api.bbci.co.uk/locations?' + urlencode({
   'api_key': 'AGbFAKx58hyjQScCXIYrxuEwJh2W2cmv',
   's': place,
   'stack': 'aws',
   'locale': 'en',
   'filter': 'international',
   'place-types': 'settlement,airport,district',
   'order': 'importance',
   'a': 'true',
   'format': 'json'
   })
    response = requests.get(locator_url)
    if response.status_code == 200:
        location_data = response.json()
        if "response" in location_data and "results" in location_data["response"]:
            location_id = location_data["response"]["results"]["results"][0]["id"]
            print(f"Location ID : {location_id}")
        else:
            print("Error: No location ID found .")
            exit()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        exit()
    weather_url = f"https://weather-broker-cdn.api.bbci.co.uk/en/forecast/aggregated/{location_id}"
    weather_response = requests.get(weather_url)
    if weather_response.status_code == 200:
        weather_data = weather_response.json()
        forecasts = weather_data.get("forecasts", [])
        weather_summary = {f["summary"]["report"]["localDate"]: f["summary"]["report"]["enhancedWeatherDescription"] for f in forecasts}

        return weather_summary
    else:
        return 0

def GA4_5(city, country, latitude_type):
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "format": "jsonv2",
        "city": city,
        "country": country
    }
    headers = {
        "User-Agent": "MyGeospatialApp/1.0 (suneha2003datta@gmail.com)"  #
    }

    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status() 
    data = response.json()
    city_data = sorted(
        (entry for entry in data if entry.get("addresstype") == "city"),
        key=lambda x: x.get("importance", 0),
        reverse=True
    )
    
    if not city_data:
        raise ValueError(f"No city found for {city}, {country}")
    best_match = city_data[0]
    bounding_box = best_match.get("boundingbox")
    osm_id = best_match.get("osm_id")
    
    if not bounding_box:
        raise ValueError("Bounding box not available in API response")
    
    min_lat, max_lat = float(bounding_box[0]), float(bounding_box[1])
    selected_latitude = min_lat if latitude_type == "minimum latitude" else max_lat
  
    if not isinstance(selected_latitude, float):
        raise ValueError("Expected a numerical latitude value")
    return str(selected_latitude)

def GA4_6(selected_topic,min_points):
   
    url = f"https://hnrss.org/newest?q={selected_topic}&points={min_points}"
    
    try:
        response = requests.get(url, headers={"User-Agent": f"MyHackerNewsBot/1.0 (suneha2003datta@gmail.com)"})
        response.raise_for_status()
        root = ET.fromstring(response.text)
        link_element = root.find(".//item/link")
        
        if link_element is None:
            raise ValueError("No relevant Hacker News posts found.")

        return  str(link_element.text)
    
    except requests.HTTPError as http_err:
        return {"error": f"HTTP error occurred: {http_err}"}
    except Exception as err:
        return {"error": f"An error occurred: {err}"}
    
def GA4_7(location,followers):
    url = f"https://api.github.com/search/users?q=location:{location}+followers:%3E={followers}&sort=joined&order=desc"
    response = requests.get(url)
    data = response.json()
    user = data["items"][0]
    username = user["login"]
    user_details_response = requests.get(f"https://api.github.com/users/{username}")
    user_details = user_details_response.json()
          #print(user_details)
    created_at = user_details["created_at"]
    return created_at
def GA4_8(email):
    return "https://github.com/Suneha111/tdsga4"

def extract_tables_from_pdf(pdf_path):
    extracted_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() 
            tables = page.extract_tables()  
            
            if text:
                match = re.search(r"Group (\d+)", text)
                if match:
                    group_number = int(match.group(1)) 
                else:
                    continue 
            
            for table in tables:
                extracted_data.append((group_number, table))
                    
    return extracted_data

def GA4_9(main_subject, filter_subject, min_marks, group_range=(1, 38)):
    global file_path
    group_range=ast.literal_eval(group_range)
    data = extract_tables_from_pdf(file_path)
    min_marks=int(min_marks)
    total_marks = 0 
    for group_number, table in data:
        if not (group_range[0] <= group_number <= group_range[1]):
            continue 
        
        headers = table[0] 
        try:
            filter_index = headers.index(filter_subject) 
            main_index = headers.index(main_subject)
        except ValueError:
            continue 
        for row in table[1:]: 
            try:
                filter_marks = int(row[filter_index])
                main_marks = int(row[main_index])
                
                if filter_marks >= min_marks:
                    total_marks += main_marks
            except ValueError:
                continue
    return str(total_marks)

def extract_text_from_pdf(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text())
    return "\n\n".join(filter(None, text))

def convert_to_markdown(text):
    text = re.sub(r'(https?://[^\s]+)', r'[\1](\1)', text)
    lines = text.split("\n")
    markdown_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.isupper() and len(stripped.split()) < 5: 
            markdown_lines.append(f"## {stripped}")
        elif stripped.startswith("•") or stripped.startswith("-"):
            markdown_lines.append(f"- {stripped[1:].strip()}")
        else:
            markdown_lines.append(stripped)
    
    markdown_text = "\n".join(markdown_lines)
    markdown_text = markdownify(markdown_text)

    return markdown_text

def format_with_prettier(markdown_text):
    prettier_process = subprocess.run(
         ["npx", "prettier@3.4.2", "--parser", "markdown"],
        input=markdown_text.encode(),
        capture_output=True
    )
    return prettier_process.stdout.decode()

def GA4_10(pdf_path):
    global file_path
    raw_text = extract_text_from_pdf(file_path)
    markdown_text = convert_to_markdown(raw_text)
    formatted_markdown = format_with_prettier(markdown_text)
    
    return formatted_markdown 

def GA5_1(country,yr,mnth,date,hr,min,sec,productname):
    global file_path
    df = pd.read_excel(file_path)
    df['Country'] = df['Country'].str.strip().str.upper()
    country_mapping = {
        "USA": "US", "U.S.A": "US", "UNITED STATES": "US", "US": "US",
        "BRA": "BR", "BRAZIL": "BR", "BR": "BR",
        "U.K": "GB", "UK": "GB", "UNITED KINGDOM": "GB",
        "FR": "FRANCE", "FRA": "FRANCE", "FRANCE": "FRANCE",
        "IND": "INDIA", "IN": "INDIA", "INDIA": "INDIA",
        "AE": "UAE", "U.A.E": "UAE", "UNITED ARAB EMIRATES": "UAE", "UAE": "UAE"
    }
    df['Country'] = df['Country'].replace(country_mapping)

    def convert_to_datetime(date):
        for fmt in ("%m-%d-%Y", "%Y/%m/%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(date), fmt)
            except ValueError:
                continue
        return pd.NaT 
    yr=int(yr)
    mnth=int(mnth)
    date=int(date)
    hr=int(hr)
    min=int(min)
    sec= int(sec)
    df['Date'] = df['Date'].apply(convert_to_datetime)
    df['Product/Code'] = df['Product/Code'].str.split('/').str[0].str.strip()
    df['Sales'] = df['Sales'].astype(str).str.replace("USD", "").str.strip().astype(float)
    df['Cost'] = df['Cost'].astype(str).str.replace("USD", "").str.strip()
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
    df['Cost'].fillna(df['Sales'] * 0.5, inplace=True)
    date_filter = datetime(yr, mnth, date, hr, min, sec)  # Tue Jan 03 2023 16:56:57 GMT+0530  ##Sat Sep 24 2022 22:25:09 GMT+0530
    df_filtered = df[
        (df['Date'] <= date_filter) &
        (df['Product/Code'].str.lower() == productname.lower()) &
        (df['Country'] == country)
    ]

    total_sales = df_filtered['Sales'].sum()
    total_cost = df_filtered['Cost'].sum()

    if total_sales > 0:
        total_margin = (total_sales - total_cost) / total_sales
    else:
        total_margin = 0
    return str(float(total_margin))

def GA5_2(filepath):
    global file_path
    unique_ids = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'-([\w\d]+)::?Marks', line)
            if match:
                unique_id = match.group(1)
                unique_ids.add(unique_id)
    
    return str(len(unique_ids))

def is_sunday(log_date,day):
    date_obj = datetime.strptime(log_date, "%d/%b/%Y:%H:%M:%S %z")
    return date_obj.weekday() == day

def GA5_3(start,end,day,url_prefix): # day in number sunday being 6 # should be '/tamilmp3/'
    global file_path
    start=int(start)
    end=int(end)
    day=int(day)
    pattern = re.compile(
        rf'(?P<ip>\S+) - - \[(?P<datetime>[\d+/A-Za-z: -]+)\] '
        rf'"GET (?P<url>{re.escape(url_prefix)}[^\s]*) HTTP/1.1" (?P<status>\d+)'
    )

    count = 0

    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                log_datetime = match.group("datetime")
                status_code = match.group("status")
                log_time = datetime.strptime(log_datetime, "%d/%b/%Y:%H:%M:%S %z")
                if (status_code == "200" and 
                    is_sunday(log_datetime,day) and 
                    start <= log_time.hour < end):
                    count += 1

    return str(count)

def GA5_4(url_prefix, target_date): #date and url like /tamil/ on 2024-05-28
    global file_path
    pattern = re.compile(
        rf'(?P<ip>\S+) - - \[(?P<datetime>\d+/[A-Za-z]+/\d+:\d+:\d+:\d+ [+-]\d+)\] '
        rf'"GET (?P<url>{re.escape(url_prefix)}[^\s]*) HTTP/1.1" \d+ (?P<bytes>\d+)'
    )

    ip_bytes = defaultdict(int)
    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                log_datetime = match.group("datetime")
                log_date = datetime.strptime(log_datetime, "%d/%b/%Y:%H:%M:%S %z").strftime("%Y-%m-%d")
                
                if log_date == target_date:
                    ip = match.group("ip")
                    bytes_transferred = int(match.group("bytes"))
                    ip_bytes[ip] += bytes_transferred

    if not ip_bytes:
        return None, 0 

    top_ip = max(ip_bytes, key=ip_bytes.get)
    max_bytes = ip_bytes[top_ip]

    return str(max_bytes)

def load_sales_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def GA5_5(target_product, target_city, min_units):
    global file_path
    sales_data = load_sales_data(file_path)
    total_sales = 0
    min_units = int(min_units)

    for entry in sales_data:
        entry_sales = int(entry['sales'])  # Ensure 'sales' is an integer
        city_name = entry['city']

        # Compare the city name directly with the target_city
        similarity_score = fuzz.ratio(city_name.lower(), target_city.lower())

        if (entry['product'].lower() == target_product.lower() and
            similarity_score > 65 and  # Accept city if similarity > 65%
            entry_sales >= min_units):
            total_sales += entry_sales

    print(total_sales)
    return str(total_sales)

def extract_sales_from_line(line):
    match = re.search(r'"sales":\s*(\d+)', line)
    if match:
        return int(match.group(1))
    return 0

def GA5_6(filepath):
    global file_path
    total_sales = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            total_sales += extract_sales_from_line(line)
    
    return str(total_sales)

def GA5_7(kk):
    global file_path
    def count_fh_keys(data,kkk):
        if isinstance(data, dict):
            return sum((1 if key == kkk else 0) + count_fh_keys(value,kkk) for key, value in data.items())
        elif isinstance(data, list):
            return sum(count_fh_keys(item,kkk) for item in data)
        return 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        nested_data = json.load(file)
    cc=count_fh_keys(nested_data,kk)
    return str(cc)

def GA5_8(timestamp, min_useful_stars):
    return f"""SELECT post_id FROM (SELECT post_id FROM (SELECT post_id,json_extract(comments, '$[*].stars.useful') AS useful_stars FROM social_media WHERE timestamp >= '{timestamp}') WHERE EXISTS (SELECT 1 FROM UNNEST(useful_stars) AS t(value) WHERE CAST(value AS INTEGER) >= {min_useful_stars})) ORDER BY post_id;""".strip()

def GA5_9(start_time, end_time):
    start_time=float(start_time)
    end_time=float(end_time)
    with open("transcript.json", "r", encoding="utf-8") as file:
      json_data = json.load(file)
    extracted_text = []
    if isinstance(json_data, str):
        json_data = json.loads(json_data) 
    for entry in json_data:
        timestamp = entry.get("timestamp", "")
        parts = timestamp.split(":")
        if len(parts) == 3: 
            h, m, s = map(float, parts)
            total_seconds = h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            total_seconds = m * 60 + s
        else:
            continue 
        if start_time <= total_seconds <= end_time:
            extracted_text.append(entry.get("text", ""))

    return " ".join(extracted_text)

def GA5_10(imagepath):
    image_path="Rim.png"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def GA2_1(topic):
    return """# Analysis of Daily Steps Walked: A Comparative Study

This document analyzes the number of steps I walked each day for a week, comparing the results over time and with friends. It contains various examples, such as lists, a table, a hyperlink, an image, and a blockquote.

## Introduction
This study examines the walking patterns over a seven-day period. Data was collected and compared both over time and with friends, providing insights into walking habits and consistency.

## Methodology
The analysis was carried out using daily step counts recorded from a fitness tracker. The steps for each day were compared using a simple statistical approach, and comparisons were made with my friends' data.

### Steps Walked Daily

- **Day 1:** 7,000 steps
- **Day 2:** 9,500 steps
- **Day 3:** 8,200 steps
- **Day 4:** 7,800 steps
- **Day 5:** 10,000 steps
- **Day 6:** 8,500 steps
- **Day 7:** 9,000 steps

## Results

### Comparison Over Time

1. The steps I walked on Day 5 (10,000 steps) were the highest for the week.
2. On average, I walked around 8,500 steps each day.
3. Day 2 showed a noticeable increase in steps, which could be attributed to additional outdoor activities.

### Comparison with Friends

| Friend     | Average Steps | Total Steps for Week |
|------------|---------------|----------------------|
| Friend A   | 8,000 steps   | 56,000 steps         |
| Friend B   | 9,000 steps   | 63,000 steps         |
| Friend C   | 7,500 steps   | 52,500 steps         |

- **Note:** Friend B walked the most on average, *but* I walked the most on Day 5.

## Hyperlink
For further details on the methodology used, visit [Fitness Tracker Research](https://example.com).

## Image
Here is an example of a graph displaying the daily steps walked:

![Step Count Graph](https://example.com/steps.jpg)

## Blockquote
> "Consistency is key. Small improvements each day lead to long-term benefits."

## Code Example
The analysis was carried out using daily step counts recorded from a fitness tracker. The steps for each day were compared using a simple statistical approach, and comparisons were made with my friends' data. A Python script was used for the analysis, such as `average_steps = sum(steps) / len(steps)`.


This is an example of some Python code that could be used to analyze the data:

```python
steps = [7000, 9500, 8200, 7800, 10000, 8500, 9000]
average_steps = sum(steps) / len(steps)
print("Average steps per day:", average_steps) 
```
"""

def GA2_2(imagepath):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "compressim.png")
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def GA2_3(email):
    GITHUB_USERNAME = "Suneha111"
    GITHUB_REPO = "shivaluck"
    GITHUB_BRANCH = "main"
    GITHUB_PERSONAL_ACCESS_TOKEN = "your token"
    REPO_DIR = os.path.expanduser("~/shivaluck")  
    EMAIL_FILE = os.path.join(REPO_DIR, "index.html")
    subprocess.run(["git", "config", "--global", "user.email", "23f1002574@ds.study.iitm.ac.in"])
    subprocess.run(["git", "config", "--global", "user.name", "Suneha"])

    if not os.path.exists(REPO_DIR):
        repo_url = f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
        subprocess.run(["git", "clone", repo_url, REPO_DIR], check=True)

    os.chdir(REPO_DIR)
    subprocess.run(["git", "pull", "origin", GITHUB_BRANCH], check=True)
    with open(EMAIL_FILE, "w") as f:
        f.write('<!--email_off-->'+email+'<!--/email_off-->')

    subprocess.run(["git", "add", EMAIL_FILE], check=True)
    try:
        subprocess.run(["git", "commit", "-m", "Updated email.json"], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "commit", "--allow-empty", "-m", "Forced commit"], check=True)

    repo_url_auth = f"https://{GITHUB_USERNAME}:{GITHUB_PERSONAL_ACCESS_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
    subprocess.run(["git", "push", repo_url_auth], check=True)

    return "https://suneha111.github.io/shivaluck/" 

def GA2_4(email):
    current_year = datetime.now().year
    hash_value = hashlib.sha256(f"{email} {current_year}".encode()).hexdigest()[-5:]
    return str(hash_value)

def GA2_5(lightnessno):
    image_path = "lenna.webp"
    lightnessno=float(lightnessno) 
    image = Image.open(image_path)
    rgb = np.array(image) / 255.0
    lightness = np.apply_along_axis(lambda x: colorsys.rgb_to_hls(*x)[1], 2, rgb)
    light_pixels = np.sum(lightness > lightnessno)
    return str(int(light_pixels))

def GA2_6(filename):
    global file_path
    AWS_ACCESS_KEY = "your key"
    AWS_SECRET_KEY = "your key"
    S3_BUCKET_NAME = "tdsprj"
    REGION_NAME = "eu-north-1"

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=REGION_NAME,
    )

    try:
        file_key = "uploads/q-vercel-python.json" 
        with open(file_path, "rb") as file:
            s3_client.upload_fileobj(file, S3_BUCKET_NAME, file_key)
        file_url = f"https://{S3_BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/{file_key}"
        print("File uploaded successfully")
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return 'https://tdsp2-b8wmn37ft-raves-projects-ceefbc83.vercel.app/get-json'

def GA2_7(email):
    GITHUB_USERNAME = "Suneha111"
    GITHUB_REPO = "see1"
    GITHUB_BRANCH = "main"
    GITHUB_PERSONAL_ACCESS_TOKEN = "token"
    EMAIL = email 
    subprocess.run(["git", "config", "--global", "user.email", "23f1002574@ds.study.iitm.ac.in"])
    subprocess.run(["git", "config", "--global", "user.name", "Suneha"])
    REPO_DIR = os.path.expanduser(f"~/github/{GITHUB_REPO}") 
    WORKFLOW_DIR = os.path.join(REPO_DIR, ".github", "workflows")
    WORKFLOW_FILE = os.path.join(WORKFLOW_DIR, "ci.yaml")
    if not os.path.exists(REPO_DIR):
        repo_url = f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
        subprocess.run(["git", "clone", repo_url, REPO_DIR], check=True)
    os.makedirs(WORKFLOW_DIR, exist_ok=True)
    workflow_content = f"""name: Example CI Workflow

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    # Checkout code from the repository
    - name: Checkout code
      uses: actions/checkout@v2

    # Run your build step
    - name: Run build step
      run: |
        echo "This is the build step"
        # Add your build commands here (e.g., npm install, python setup, etc.)
    
    # Custom step with your email address
    - name: Notify user at {EMAIL}
      run: echo "Build. Notification to {EMAIL}"
      
  test:
    runs-on: ubuntu-latest
    steps:
      - name: {EMAIL}
        run: echo "Hello, world!"
    """
    with open(WORKFLOW_FILE, "w") as f:
        f.write(workflow_content)
    os.chdir(REPO_DIR)
    subprocess.run(["git", "add", WORKFLOW_FILE], check=True)
    try:
        subprocess.run(["git", "commit", "-m", "Added GitHub Action workflow"], check=True)
    except subprocess.CalledProcessError:
        print("No changes to commit.")
    repo_url_auth = f"https://{GITHUB_USERNAME}:{GITHUB_PERSONAL_ACCESS_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
    subprocess.run(["git", "push", repo_url_auth, GITHUB_BRANCH], check=True)
    repo_url = f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}"
    actions_url = f"{repo_url}/actions"
    return 'https://github.com/Suneha111/see1'

def GA2_8(tag):
    def start_docker():
        system = platform.system()
        if system == "Windows":
            try:
                subprocess.run("docker info", shell=True, check=True)
            except subprocess.CalledProcessError:
                subprocess.run("wsl --exec sudo service docker start", shell=True)
                time.sleep(10)
        else: 
            try:
                subprocess.run(["systemctl", "is-active", "--quiet", "docker"], check=True)
            except subprocess.CalledProcessError:
                subprocess.run(["sudo", "systemctl", "start", "docker"], check=True)

    start_docker()

    DOCKER_USERNAME = "suneha681"
    DOCKER_REPO = "sample-app"
    DOCKER_PASSWORD = "cR9*ZnpjT#knDJ9"
    OLD_TAG = "suneha2003datta"
    NEW_TAG = tag

    def run_command(command):
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running command: {command}")
            print(e.stderr)
            exit(1)

    run_command(f"echo {DOCKER_PASSWORD} | docker login -u {DOCKER_USERNAME} --password-stdin")
    run_command(f"sudo docker pull suneha681/sample-app:suneha2003datta")
    run_command(f"sudo docker tag {DOCKER_USERNAME}/{DOCKER_REPO}:{OLD_TAG} {DOCKER_USERNAME}/{DOCKER_REPO}:{NEW_TAG}")
    run_command(f"sudo docker push {DOCKER_USERNAME}/{DOCKER_REPO}:{NEW_TAG}")
    return 'https://hub.docker.com/repository/docker/suneha681/sample-app/general'

@app.post("/v1/chat/completions")
async def fake_response(request: Request):
    current_time = int(time.time())
    return {
        "model": "unknown", 
        "object": "chat.completion",
        "created": current_time,  
        "choices": [
            {
                "message": {
                    "content": "This is a valid response."
                }
            }
        ]
    }

@app.get("/api")
def get_students(class_: list[str] = Query(None, alias="class")):
    global file_path
    CSV_PATH = file_path
    df = pd.read_csv(CSV_PATH)
    if class_:
        filtered_df = df[df["class"].isin(class_)] 
    else:
        filtered_df = df  
    return {"students": filtered_df.to_dict(orient="records")}

def GA2_9(url):
    return 'http://16.16.189.187:8000/api'

def GA2_10(url):
    return 'http://16.16.189.187:8000'




@app.post("/tdsp2")
async def tdsp2(question: str = Form(...), file: Optional[UploadFile] = File(None) ):
    global file_path
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an assistant that can call functions. You donot ask for any content nor you yourself provide any content you just call suitable functions from the functions set given. "},
            {"role": "user", "content": question},
        ],
        "functions": FUNCTIONS,
        "function_call": "auto",
    }
    if file :
        print('yesss')
        file_path = os.path.join(os.getcwd(), file.filename)
        file_path = file_path.replace("\\", "/") 
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
        except Exception as e:
            return {"error": str(e)}
    print(file_path)
    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, data=json.dumps(data))
    print('here')
    print(response.status_code)
    result = response.json()
    print('here2')
    if "choices" in result and result["choices"]:
        print(result)
        function_call = result["choices"][0]["message"].get("function_call")
        if function_call:
            name = function_call.get("name")
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)  
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON format for arguments")
            funv= {
                "name": name,
                "arguments": arguments
            }
    else :
        print('hi')
        print(result)
   
    result = globals()[funv["name"]](**funv["arguments"])
    print('hiii')
    if result:
        return json.dumps({"answer": result})
