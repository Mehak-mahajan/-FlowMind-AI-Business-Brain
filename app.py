import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os
import docx
import pptx
from langdetect import detect
import pandas as pd
from datetime import datetime
import pickle

load_dotenv()

# ---- PAGE SETUP ----
st.set_page_config(
    page_title="FlowMind — AI Business Brain",
    page_icon="🧠",
    layout="wide"
)

# ---- CSS ----
st.markdown("""
<style>
    .main { background-color: #ECE5DD; }
    .header-box {
        background-color: #075E54;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .user-bubble {
        background-color: #DCF8C6;
        padding: 10px 15px;
        border-radius: 15px 15px 0px 15px;
        margin: 5px 0;
        margin-left: 20%;
        color: #000;
        font-size: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .assistant-bubble {
        background-color: #FFFFFF;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0px;
        margin: 5px 0;
        margin-right: 20%;
        color: #000;
        font-size: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .source-tag {
        font-size: 11px;
        color: #075E54;
        margin-top: 3px;
    }
    .lead-box {
        background-color: #FFF3CD;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #FFC107;
        margin: 10px 0;
    }
    .alert-box {
        background-color: #FFE0E0;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #FF0000;
        margin: 10px 0;
    }
    .welcome-box {
        text-align: center;
        padding: 40px;
        background: white;
        border-radius: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("""
<div class="header-box">
    <h2 style="color:white;margin:0;">
        🧠 FlowMind — AI Business Brain
    </h2>
    <p style="color:#25D366;margin:0;">
        ● Online | Powered by AI
    </p>
</div>
""", unsafe_allow_html=True)

# ---- LOAD MODELS ----
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

embedding_model = load_model()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---- SESSION STATE ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "leads" not in st.session_state:
    st.session_state.leads = []
if "tickets" not in st.session_state:
    st.session_state.tickets = []
if "questions_count" not in st.session_state:
    st.session_state.questions_count = 0
if "angry_count" not in st.session_state:
    st.session_state.angry_count = 0
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# ---- SAVE KNOWLEDGE BASE ----
def save_knowledge_base(chunks, metadata, index):
    with open("knowledge_base.pkl", "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "metadata": metadata
        }, f)
    faiss.write_index(index, "faiss_index.bin")
    st.success("✅ Knowledge base saved permanently!")

# ---- LOAD KNOWLEDGE BASE ----
def load_knowledge_base():
    if os.path.exists("knowledge_base.pkl") and \
       os.path.exists("faiss_index.bin"):
        with open("knowledge_base.pkl", "rb") as f:
            data = pickle.load(f)
        index = faiss.read_index("faiss_index.bin")
        return data["chunks"], data["metadata"], index
    return None, None, None

# ---- TEXT EXTRACTION ----
def extract_text(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
    pages = []

    if file_type == "pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    pages.append((page_text, i + 1))

    elif file_type == "txt":
        text = uploaded_file.read().decode("utf-8")
        pages.append((text, 1))

    elif file_type == "docx":
        doc = docx.Document(uploaded_file)
        full_text = ""
        for paragraph in doc.paragraphs:
            full_text += paragraph.text + "\n"
        pages.append((full_text, 1))

    elif file_type == "pptx":
        prs = pptx.Presentation(uploaded_file)
        for i, slide in enumerate(prs.slides):
            slide_text = ""
            for shape in slide.shapes:
                if shape.has_text_frame:
                    slide_text += shape.text + "\n"
            if slide_text:
                pages.append((slide_text, i + 1))

    elif file_type == "csv":
        text = uploaded_file.read().decode("utf-8")
        pages.append((text, 1))

    return pages

# ---- PROCESS FILES ----
def process_files(uploaded_files):
    all_chunks = []
    all_metadata = []

    for uploaded_file in uploaded_files:
        pages = extract_text(uploaded_file)
        chunk_size = 500

        for page_text, page_num in pages:
            for i in range(0, len(page_text), chunk_size):
                chunk = page_text[i:i+chunk_size]
                all_chunks.append(chunk)
                all_metadata.append({
                    "file": uploaded_file.name,
                    "page": page_num
                })

        st.write(f"✅ {uploaded_file.name} processed!")

    if not all_chunks:
        return None, None, None

    vectors = embedding_model.encode(
        all_chunks,
        show_progress_bar=False,
        batch_size=32
    )

    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors))

    return all_chunks, all_metadata, index

# ---- SENTIMENT ANALYSIS ----
def detect_sentiment(text):
    angry_words = [
        "angry", "frustrated", "terrible", "worst",
        "horrible", "useless", "scam", "fraud",
        "cheat", "fake", "disgusting", "pathetic",
        "waste", "refund", "complaint",
        "खराब", "बेकार", "धोखा", "गुस्सा",
        "disappointed", "ridiculous", "cheated"
    ]
    text_lower = text.lower()
    for word in angry_words:
        if word in text_lower:
            return "angry"
    return "neutral"

# ---- DETECT LEAD INTENT ----
def detect_lead_intent(text):
    lead_words = [
        "buy", "purchase", "order", "interested",
        "want to", "how to buy", "price", "cost",
        "bulk order", "wholesale", "contact",
        "खरीदना", "ऑर्डर", "कीमत", "चाहिए"
    ]
    text_lower = text.lower()
    for word in lead_words:
        if word in text_lower:
            return True
    return False

# ---- ASK QUESTION ----
def ask_question(question, chunks, metadata, index):
    try:
        lang = detect(question)
    except:
        lang = "en"

    question_vector = embedding_model.encode([question])
    distances, indices = index.search(
        np.array(question_vector), 3
    )

    context = ""
    sources = []

    for i in indices[0]:
        context += chunks[i] + "\n\n"
        source = metadata[i]
        if source not in sources:
            sources.append(source)

    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are FlowMind, 
                an intelligent AI business assistant 
                for an e-commerce store.
                Answer in same language as question 
                (detected: {lang}).
                Use ONLY the provided context.
                Be helpful, friendly and concise.
                If answer not in context say:
                'Let me connect you with our 
                support team for this query.'"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\n"
                           f"Question: {question}"
            }
        ],
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content, sources

# ---- SUMMARIZE ----
def summarize_document(chunks):
    sample = "\n\n".join(chunks[:15])
    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Summarize this business document in clear bullet points."
            },
            {
                "role": "user",
                "content": f"Summarize:\n{sample}"
            }
        ],
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

# ---- GENERATE FAQS ----
def generate_faqs(chunks):
    sample = "\n\n".join(chunks[:15])
    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Generate top 10 FAQs with answers from this business document."
            },
            {
                "role": "user",
                "content": f"Generate FAQs:\n{sample}"
            }
        ],
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

# ---- LOAD EXISTING KNOWLEDGE BASE ----
chunks, metadata, index = load_knowledge_base()

# ---- SIDEBAR ----
with st.sidebar:

    # ADMIN LOGIN
    st.markdown("### 🔐 Admin Panel")
    password = st.text_input(
        "Admin Password",
        type="password",
        placeholder="Enter password"
    )

    if password == "flowmind123":
        st.session_state.is_admin = True
        st.success("✅ Admin mode active!")
    elif password != "":
        st.error("❌ Wrong password!")
        st.session_state.is_admin = False

    # ADMIN FEATURES
    if st.session_state.is_admin:
        st.divider()
        st.markdown("### 📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload catalogs, policies, FAQs",
            type=["pdf", "txt", "docx", "pptx", "csv"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("🧠 Process & Save Documents"):
                with st.spinner("Processing..."):
                    new_chunks, new_metadata, new_index = \
                        process_files(uploaded_files)
                    if new_chunks:
                        save_knowledge_base(
                            new_chunks,
                            new_metadata,
                            new_index
                        )
                        chunks = new_chunks
                        metadata = new_metadata
                        index = new_index
                        st.rerun()

        st.divider()

        # Admin Analytics
        st.markdown("### 📊 Analytics")
        st.metric(
            "Questions Asked",
            st.session_state.questions_count
        )
        st.metric(
            "Leads Captured",
            len(st.session_state.leads)
        )
        st.metric(
            "Support Tickets",
            len(st.session_state.tickets)
        )
        st.metric(
            "Angry Alerts",
            st.session_state.angry_count
        )

        st.divider()

        # ROI Calculator
        st.markdown("### 💰 ROI Calculator")
        staff_cost = st.number_input(
            "Staff cost/month (₹)",
            value=25000,
            step=1000
        )
        st.success(f"💰 Monthly saving: ₹{staff_cost:,}")
        st.success(f"🚀 Yearly saving: ₹{staff_cost*12:,}")

        st.divider()

        # Admin Tools
        st.markdown("### 🛠️ Admin Tools")

        if chunks:
            if st.button("📝 Summarize Documents"):
                with st.expander("Summary", expanded=True):
                    st.write_stream(
                        summarize_document(chunks)
                    )

            if st.button("❓ Generate FAQs"):
                with st.expander("FAQs", expanded=True):
                    st.write_stream(
                        generate_faqs(chunks)
                    )

        # Download Leads
        if st.session_state.leads:
            leads_df = pd.DataFrame(st.session_state.leads)
            st.download_button(
                "📥 Download Leads CSV",
                leads_df.to_csv(index=False),
                "leads.csv",
                "text/csv"
            )

        # Download Tickets
        if st.session_state.tickets:
            tickets_df = pd.DataFrame(
                st.session_state.tickets
            )
            st.download_button(
                "📥 Download Tickets CSV",
                tickets_df.to_csv(index=False),
                "tickets.csv",
                "text/csv"
            )

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# ---- MAIN CHAT AREA ----
if chunks:
    # Customer chat interface
    st.markdown("""
    <div style="background:#075E54;padding:10px;
    border-radius:8px;margin-bottom:10px;">
        <p style="color:white;margin:0;">
            🛒 ShopEasy India — Customer Support
        </p>
        <p style="color:#25D366;margin:0;font-size:12px;">
            ● Online | Ask us anything!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">'
                f'👤 {message["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">'
                f'🧠 {message["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )
            if "sources" in message:
                for source in message["sources"]:
                    st.markdown(
                        f'<div class="source-tag">'
                        f'📄 {source["file"]} | '
                        f'Page {source["page"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # Chat input
    question = st.chat_input(
        "Ask about products, shipping, returns..."
    )

    if question:
        st.session_state.questions_count += 1

        # Show user message
        st.markdown(
            f'<div class="user-bubble">'
            f'👤 {question}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Sentiment check
        sentiment = detect_sentiment(question)
        if sentiment == "angry":
            st.session_state.angry_count += 1
            st.markdown("""
            <div class="alert-box">
                🚨 <strong>Priority Alert!</strong>
                Angry customer detected!
                Human agent needed!
            </div>
            """, unsafe_allow_html=True)

            ticket = {
                "id": f"TKT{len(st.session_state.tickets)+1:03d}",
                "issue": question,
                "priority": "HIGH",
                "time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "status": "Open"
            }
            st.session_state.tickets.append(ticket)
            st.error(f"🎫 Ticket: {ticket['id']} created!")

        # Lead detection
        if detect_lead_intent(question):
            st.markdown("""
            <div class="lead-box">
                🎯 <strong>Potential Lead Detected!</strong>
                Customer interested in buying!
            </div>
            """, unsafe_allow_html=True)

            with st.form("lead_form"):
                st.markdown("**📋 Save Customer Details:**")
                name = st.text_input("Name")
                phone = st.text_input("Phone")
                email = st.text_input("Email")
                submit = st.form_submit_button("Save Lead")

                if submit and name:
                    lead = {
                        "name": name,
                        "phone": phone,
                        "email": email,
                        "query": question,
                        "time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M"
                        )
                    }
                    st.session_state.leads.append(lead)
                    st.success(
                        f"✅ Lead saved! "
                        f"Total: {len(st.session_state.leads)}"
                    )

        # Get AI answer
        answer_text = ""
        answer_sources = []
        placeholder = st.empty()

        for content, sources in ask_question(
            question, chunks, metadata, index
        ):
            answer_text += content
            answer_sources = sources
            placeholder.markdown(
                f'<div class="assistant-bubble">'
                f'🧠 {answer_text}'
                f'</div>',
                unsafe_allow_html=True
            )

        for source in answer_sources:
            st.markdown(
                f'<div class="source-tag">'
                f'📄 {source["file"]} | '
                f'Page {source["page"]}'
                f'</div>',
                unsafe_allow_html=True
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": answer_sources
        })

else:
    # No knowledge base yet
    st.markdown("""
    <div class="welcome-box">
        <h1>🧠 FlowMind</h1>
        <h3 style="color:#075E54;">
            AI Business Brain for E-commerce
        </h3>
        <br>
        <p style="font-size:16px;">
            Welcome to FlowMind! Our AI assistant 
            is being set up.
        </p>
        <p style="font-size:14px;color:#888;">
            Please contact the store admin to 
            activate the AI assistant.
        </p>
        <br>
        <div style="display:flex;
        justify-content:center;gap:30px;
        flex-wrap:wrap;">
            <div>📦 Product Info</div>
            <div>🚚 Shipping</div>
            <div>↩️ Returns</div>
            <div>💬 Support</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.is_admin:
        st.info(
            "👈 Admin? Login from the sidebar "
            "to upload business documents!"
        )