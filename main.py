import azure.functions as func
import json
import os
import logging
from docx import Document
import io
import pytesseract
from pdf2image import convert_from_bytes
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import pandas as pd
from io import BytesIO
import uuid
import requests
import openai
from openai import AzureOpenAI
import re
import uuid
import json
from datetime import datetime, timedelta
import os
from azure.data.tables import TableServiceClient
from azure.core.exceptions import ResourceExistsError
from azure.functions import HttpResponse

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

connection_string = os.getenv("AzureWebJobsStorage")  # Default in Azure Functions
table_name = "UserSessions"  # Your table name

try:
    table_service = TableServiceClient.from_connection_string(connection_string)
    table_client = table_service.create_table(table_name)
except ResourceExistsError:
    table_client = table_service.get_table_client(table_name)

app = func.FunctionApp()
def get_supported_api_versions():
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_KEY")
    
    url = f"{endpoint}/?api-version=2020-06-30"
    headers = {"api-key": key}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            versions = [v['version'] for v in response.json()['apiVersion']]
            logging.info(f"✅ Supported API versions: {versions}")
            return versions
        else:
            logging.error(f"❌ Failed to get API versions: {response.status_code} {response.text}")
    except Exception as e:
        logging.error(f"❌ Error checking API versions: {e}")
    
    return None
# ---------------------------------- PARSERS ----------------------------------

def parse_image(blob_bytes):
    endpoint = os.environ["DOC_INTELLIGENCE_ENDPOINT"]
    key = os.environ["DOC_INTELLIGENCE_KEY"]
    client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
    
    poller = client.begin_analyze_document(model_id ="prebuilt-read", document=blob_bytes)
    result = poller.result()

    lines = []
    for page in result.pages:
        for line in page.lines:
            lines.append(line.content)
    return "\n".join(lines)

def parse_docx(blob_bytes):
    doc = Document(io.BytesIO(blob_bytes))
    return '\n'.join([para.text for para in doc.paragraphs])

def analyze_excel(excel_bytes):
    endpoint = os.environ["DOC_INTELLIGENCE_ENDPOINT"]
    key = os.environ["DOC_INTELLIGENCE_KEY"]
    client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
    poller = client.begin_analyze_document("prebuilt-document", document=excel_bytes)
    result = poller.result()
    return [element.content for element in result.content_elements]

def parse_with_doc_intelligence(blob_bytes):
    endpoint = os.environ["DOC_INTELLIGENCE_ENDPOINT"]
    key = os.environ["DOC_INTELLIGENCE_KEY"]
    client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
    
    poller = client.begin_analyze_document(model_id ="prebuilt-read", document=blob_bytes)
    result = poller.result()

    lines = []
    for page in result.pages:
        for line in page.lines:
            lines.append(line.content)
    return "\n".join(lines)
# ---------------------------- TEXT CHUNKING FUNCTION ----------------------------------
chunk_size = 1500
def chunk_text_with_metadata(text, source_id=None, chunk_size = None, overlap=200):
    if chunk_size is None:
        chunk_size = globals()['chunk_size']
        
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks = []
    start = 0
    chunk_index = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end]
        chunks.append({
            'text': chunk_text,
            'source_id': source_id,
            'chunk_index': chunk_index,
            'start_char': start,
            'end_char': end
        })
        if end == text_len:
            break
        start = end - overlap
        chunk_index += 1

    return chunks

# ---------------------------- EMBEDDING FUNCTION ----------------------------

# Configure OpenAI Azure settings here
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def get_embedding(text_chunk):
    response = client.embeddings.create(
        input=[text_chunk],
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )
    return response.data[0].embedding

# ---------------------------- AZURE COGNITIVE SEARCH SETUP ----------------------------
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile
)
from azure.search.documents.models import VectorizedQuery

# Setup
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_api_key = os.getenv("AZURE_SEARCH_KEY")   
credential = AzureKeyCredential(search_api_key)
#index_name = "document-chunkss"
index_client = SearchIndexClient(endpoint=search_service_endpoint, credential=credential)

# Define fields
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="text", type=SearchFieldDataType.String),
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536,
        vector_search_profile_name="vector-profile",
    ),
    SimpleField(name="source_id", type=SearchFieldDataType.String),
    SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
    SimpleField(name="start_char", type=SearchFieldDataType.Int32),
    SimpleField(name="end_char", type=SearchFieldDataType.Int32),
]

# vector search configuration (as per 11.5.2 stable)
vector_search_config = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="vector-config",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters={
                "metric": "cosine",
                "m": 8,
                "efConstruction": 100,
                "efSearch": 100
            }
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="vector-profile",
            algorithm_configuration_name="vector-config"
        )
    ]
)

# Define and create index

def create_index_if_not_exists(index_name):
    try:
        index = SearchIndex(
            name=index_name,
            fields=[f for f in fields],  # Create new field instances
            vector_search=VectorSearch(
                algorithms=[a for a in vector_search_config.algorithms],
                profiles=[p for p in vector_search_config.profiles]
            )
        )        
        existing = [idx.name for idx in index_client.list_indexes()]
        logging.info(f"🧾 Existing indexes before creation: {existing}")
        
        if index_name not in existing:
            logging.info(f"🔧 Creating index '{index_name}'...")
            index_client.create_index(index)
            
            # Immediately re-check
            post_create = [idx.name for idx in index_client.list_indexes()]
            logging.info(f"📦 Indexes after creation attempt: {post_create}")

            if index_name in post_create:
                logging.info(f"[✓] Index '{index_name}' created successfully.")
            else:
                logging.error(f"⚠️ Tried creating '{index_name}', but it's still not listed.")
        else:
            logging.info(f"[X] Index '{index_name}' already exists.")
    except Exception as e:
        logging.error(f"❌ Error creating index '{index_name}': {e}")

#-----------------------------PUSHING DATA TO INDEXES---------------------------------------------


def upload_chunks_to_index(embedded_chunks,index_name):
    search_client = SearchClient(
    endpoint=search_service_endpoint, 
    index_name=index_name, 
    credential=credential,
    api_version="2023-11-01"  # Updated to stable version
    )
    docs = []
    for chunk in embedded_chunks:
        doc = {
            "id": str(uuid.uuid4()),
            "text": chunk["text"],
            "embedding": chunk["embedding"],
            "source_id": chunk["metadata"]["source_id"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "start_char": chunk["metadata"]["start_char"],
            "end_char": chunk["metadata"]["end_char"],
        }
        docs.append(doc)

    result = search_client.upload_documents(documents=docs)
    logging.info(f"✅ Uploaded {len(result)} documents. Statuses: {[r.succeeded for r in result]}")

#------------------------- VECTOR SEARCH FUNCTION ----------------------------------
def vector_search(query_vector,index_name):
    try:
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=index_name,
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
            api_version="2023-11-01"  # Updated to stable version
        )

        vector_query = VectorizedQuery(
            vector=query_vector, 
            k_nearest_neighbors=4,  # Retrieving top 4 results 
            fields="embedding"
        )


        results = search_client.search(
            search_text="",
            vector_queries=[vector_query],  # FIXED: Correct parameter name
            top=4, # Limit to top 4 results
            filter=f"vertical_id eq '{index_name}'"
            
        )

        return [doc for doc in results]
    except Exception as e:
        logging.error(f"❌ Error during vector search: {e}")
        return []

#------------------------GENERATING BACK THE RESPONSE------------------
def generate_answer(query, chunks):
    context = "\n\n".join([c["content"] for c in chunks])

    system_prompt = "You are an AI query assistant. Use the following contexts to answer the question. If the context does not provide enough information, say 'I dont know'."
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"
    try:
        
        headers = {
            "Content-Type": "application/json",
            "api-key": "54fr8UBCi6mhFAvAD1iBz500wy6EkDJX8MHJWdRFlu0SlVxmoeP3JQQJ99BFACHYHv6XJ3w3AAABACOGJGCy"
        }

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.8,
            "max_tokens": 4000
        }

        endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
        deployment = "gpt-35-turbo"


        url = "https://queryassistantaihelper.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"
        
        response = requests.post(url, headers=headers, data=json.dumps(payload),timeout=60)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.exception("❌ Exception in generate_answer")
        return f"Error generating answer: {str(e)}" 
    
def validate_session(req: func.HttpRequest):
    session_id = req.cookies.get("session_id")
    if not session_id:
        return None

    try:
        session = table_client.get_entity(partition_key="session", row_key=session_id)
        if datetime.fromisoformat(session["Expires"]) > datetime.utcnow():
            return {
                "username": session["Username"],
                "vertical_id": session["VerticalId"],
                "role": session["Role"]
            }
    except Exception as e:
        logging.error(f"Session validation failed: {str(e)}")
    
    return None

# ------------------------ MAIN FUNCTION --------------------------------------

@app.blob_trigger(arg_name="myblob", path="queryassistantcontainer/{vertical_id}/{name}", connection="queryassistant_STORAGE")
def BlobTrigger1(myblob: func.InputStream):
    supported_versions = get_supported_api_versions()
    if supported_versions:
        logging.info(f"ℹ️ Supported API versions: {supported_versions}")
    else:
        logging.warning("⚠️ Could not determine supported API versions")
        
    filename = myblob.name
    extension = os.path.splitext(filename)[1].lower()
    blob_bytes = myblob.read()
    vertical_id = myblob.name.split('/')[1]

    logging.info(f"✅ BlobTrigger hit! File: {filename}, Size: {myblob.length} bytes")

    text = ""
    try:
        if extension in [".pdf", ".docx", ".pptx", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            logging.info(f"📄 File type '{extension}' supported by Document Intelligence. Using it for parsing.")
            text = parse_with_doc_intelligence(blob_bytes)
            logging.info(f"🧠 Extracted text (preview 1st {chunk_size} chars):\n{text[:chunk_size]}")
        elif extension == ".png":
            text = parse_image(blob_bytes)
            logging.info("🖼️ PNG image detected. Decoding using OCR.")
            text = parse_with_doc_intelligence(blob_bytes,".png")
        elif extension == ".txt":
            logging.info("📃 Plain text file detected. Decoding directly.")
            text = blob_bytes.decode("utf-8")
            logging.info(f"🧾 Extracted TXT (preview 1st 200 chars):\n{text[:200]}")

        elif extension == ".xlsx":
            logging.info("📊 Excel file detected — reading using pandas.")
            try:
                df = pd.read_excel(BytesIO(blob_bytes), engine="openpyxl")
                text = df.to_string(index=False)
                safe_text = text.encode('utf-8', errors='replace').decode('utf-8')
                logging.info(f"📊 Excel content preview (1st {chunk_size} chars):\n{safe_text[:chunk_size]}")
            except Exception as e:
                logging.error(f" Failed to read Excel file: {e}")

        else:
            logging.warning(f"⚠️ Unsupported file type: {extension}")

    except Exception as e:
        logging.error(f" Parsing error for file '{filename}': {e}")
        text = ""

    # ------------------------ Chunking & Embedding ------------------------------
    if text:
        try:
            chunks = chunk_text_with_metadata(text, source_id=filename)
            logging.info(f"🔖 Generated {len(chunks)} text chunks from {filename}")
            logging.info(f"🔍 First chunk preview (1st {chunk_size} chars):\n{chunks[0]['text']}")
            
            embedded_chunks = []
            for i, chunk in enumerate(chunks):
                logging.info(f" Processing chunk {i+1} (chars {chunk['start_char']} to {chunk['end_char']})")

                # Generate embedding for the chunk text
                embedding = get_embedding(chunk['text'])
                logging.info(f" Chunk {i + 1} embedding (first 5 dims): {embedding[:5]}")
                
                embedded_chunks.append({
                    'embedding': embedding,
                    'metadata': {
                        'source_id': chunk['source_id'],
                        'chunk_index': chunk['chunk_index'],
                        'start_char': chunk['start_char'],
                        'end_char': chunk['end_char'],
                    },
                    'text': chunk['text']
                })
                
            # Create index if needed
            create_index_if_not_exists(vertical_id)
            
            # Upload to search index
            upload_chunks_to_index(embedded_chunks,vertical_id)
            logging.info(f"✅ Uploaded {len(embedded_chunks)} embedded chunks to search index.")
            logging.info(f"✅ Prepared {len(embedded_chunks)} embedded chunks for indexing.")

                
        except Exception as e:
            logging.error(f"❌ Chunking or embedding failed for file '{filename}': {e}")
    else:
        logging.warning("⚠️ No text extracted from file. Skipping chunking and embedding.")
        
        
@app.route(route="search", methods=["POST"])
def search(req: func.HttpRequest) -> func.HttpResponse:
    try:
        print(" Incoming search request")
        
        # ===== 1. Session Validation =====
        session = validate_session(req)
        if not session:
            print(" Unauthorized access attempt")
            return func.HttpResponse(
                json.dumps({"error": "Unauthorized"}),
                status_code=401,
                mimetype="application/json"
            )
        
        print(f" Authenticated user: {session['username']} (Vertical: {session['vertical_id']})")

        # ===== 2. Request Processing =====
        data = req.get_json()
        print(" Request JSON:", json.dumps(data, indent=2))
        
        query_text = data.get("query", "").strip()
        if not query_text:
            print(" Empty query received")
            return func.HttpResponse(
                json.dumps({"error": "Query missing"}),
                status_code=400,
                mimetype="application/json"
            )

        print(" Processing query:", query_text[:50] + ("..." if len(query_text) > 50 else ""))

        # ===== 3. Search Execution =====
        query_embedding = get_embedding(query_text)
        print(" Embedding generated (dims:", len(query_embedding), ")")
        
        # Modified to filter by vertical_id
        results = vector_search(
            embedding=query_embedding,
            vertical_id=session["vertical_id"]  # Critical security filter
        )
        print(f" Found {len(results)} relevant documents")

        # ===== 4. Response Generation =====
        response_data = []
        for doc in results:
            response_data.append({
                "content": re.sub(r'\s+', ' ', doc["text"]).strip(),
                "source": doc.get("source_id", "unknown")
            })
        
        answer = generate_answer(query_text, response_data)
        print(" Generated answer:", answer[:100] + ("..." if len(answer) > 100 else ""))
        
        return func.HttpResponse(
            json.dumps({
                "query": query_text,
                "answer": answer,
                "vertical": session["vertical_id"],  # Include vertical in response
                "sources": [doc["source"] for doc in response_data]
            }, ensure_ascii=False),
            mimetype="application/json",
            status_code=200
        )
        
    except json.JSONDecodeError:
        print(" Malformed JSON received")
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON format"}),
            status_code=400,
            mimetype="application/json"
        )
        
    except Exception as e:
        print(" Critical error:", str(e))
        return func.HttpResponse(
            json.dumps({
                "error": "Internal server error",
                "details": str(e)
            }),
            status_code=500,
            mimetype="application/json"
        )



@app.route(route="login", auth_level=func.AuthLevel.ANONYMOUS)
def login(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Login endpoint hit.')

    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({ "success": False, "message": "Invalid JSON" }),
            status_code=400,
            mimetype="application/json"
        )

    username = req_body.get("username")
    password = req_body.get("password")

    if username == "admin1" and password == "admin1pwd":
        role,vertical_id = "admin","vertical1"
    elif username == "admin2" and password == "admin2pwd":
        role ,vertical_id = "admin","vertical2"
    elif username == "admin3" and password == "admin3pwd":
        role ,vertical_id = "admin","vertical3"
    elif username == "admin4" and password == "admin4pwd":
        role ,vertical_id = "admin","vertical4"
    elif username == "user1" and password == "user1pwd":
        role ,vertical_id = "role","vertical1"
    elif username == "user2" and password == "user2pwd":
        role,vertical_id = "user","vertical1"
    elif username == "user3" and password == "user3pwd":
        role,vertical_id = "user","vertical3"
    elif username == "user4" and password == "user4pwd":
        role = "user"
        vertical_id = "vertical4"
    else:
        return func.HttpResponse(
            json.dumps({ "success": False, "message": "Invalid credentials" }),
            status_code=401,
            mimetype="application/json"
        )
    session_id = str(uuid.uuid4())
    expires = (datetime.utcnow() + timedelta(hours=24)).isoformat()

    session_entity = {
        "PartitionKey": "user_session",
        "RowKey": session_id,
        "Username": username,
        "VerticalId": vertical_id,
        "Role": role,
        "Expires": expires,
        "CreatedAt": datetime.utcnow().isoformat()
    }

    try:
        table_client.upsert_entity(session_entity)
    except Exception as e:
        logging.error(f"Failed to create session: {str(e)}")
        return func.HttpResponse(
            json.dumps({"success": False, "message": "Login failed"}),
            status_code=500,
            mimetype="application/json"
        )

    # ===== 3. Response with Secure Cookie =====
    response = func.HttpResponse(
        json.dumps({
            "success": True,
            "role": role,
            "vertical": vertical_id
        }),
        status_code=200,
        mimetype="application/json"
    )

    response.set_cookie(
        "session_id",
        value=session_id,
        expires=datetime.strptime(expires, "%Y-%m-%dT%H:%M:%S.%f"),
        httponly=True,
        secure=True,  # Enable in production (requires HTTPS)
        samesite="Lax",
        path="/"
    )

