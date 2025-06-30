import streamlit as st
import whisper
import spacy
import re
from datetime import datetime
from fpdf import FPDF
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import tempfile, os
import librosa

# 1. Whisper
@st.cache_resource(show_spinner=False)
def load_whisper():
    return whisper.load_model("tiny")  # <<< switch to tiny

whisper_model = load_whisper()

# 2. spaCy
@st.cache_resource(show_spinner=False)
def load_spacy():
    return spacy.load("en_core_web_sm")

spacy_nlp = load_spacy()

# 3. HF pipelines & LangChain
@st.cache_resource(show_spinner=False)
def load_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa = pipeline("text2text-generation", model="google/flan-t5-large")
    return HuggingFacePipeline(pipeline=summarizer), HuggingFacePipeline(pipeline=qa)

llm_summarizer, llm_qa = load_pipelines()

# ——— Chains & Prompts ———
summary_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="Summarize the following meeting transcript:\n\n{transcript}"
)
qa_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="{question}\n\n{context}"
)
summary_chain = LLMChain(llm=llm_summarizer, prompt=summary_prompt)
qa_chain = LLMChain(llm=llm_qa, prompt=qa_prompt)

# ——— PDF helper ———
class MeetingMinutesPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "MEETING MINUTES", ln=True, align="C")
        self.ln(5)
        self.rect(5.0, 5.0, 200.0, 287.0)
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
    def add_section(self, title, content):
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, title, ln=True, border="B")
        self.set_font("Arial", size=11)
        self.multi_cell(0, 7, content)
        self.ln(5)
    def add_bullet_points(self, title, items):
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, title, ln=True, border="B")
        self.set_font("Arial", size=11)
        for item in items:
            self.cell(5)
            self.cell(0, 7, f"- {item.strip()}", ln=True)
        self.ln(5)

def extract_metadata(text):
    doc = spacy_nlp(text)
    date_match = re.search(r"\b(\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})\b", text)
    date = date_match.group(0) if date_match else datetime.today().strftime("%d %B %Y")
    attendees = list({ent.text for ent in doc.ents if ent.label_ == "PERSON"})
    return date, attendees

# ——— Streamlit UI ———
st.title("🎤 Meeting Minutes Generator")

uploaded = st.file_uploader("Upload audio (.mp3/.wav)", type=["mp3","wav"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

        st.info("Transcribing…")
    transcript = whisper_model.transcribe(path)["text"]
       # load via librosa to avoid needing a system ffmpeg
    audio_array, sr = librosa.load(path, sr=16000)
    transcript = whisper_model.transcribe(audio_array)["text"]
    st.success("Transcription complete")

    st.subheader("Transcript Preview")
    st.text_area("", transcript[:1000] + "…", height=200)

    st.info("Generating summary & extracting…")
    summary = summary_chain.run(transcript[:2000])
    date, attendees = extract_metadata(transcript)
    action_items = qa_chain.run({"question": "Extract the action items and responsibilities:", "context": transcript})
    next_steps = qa_chain.run({"question": "List next steps and follow‑ups:", "context": transcript})
    closing = qa_chain.run({"question": "Summarize the closing remarks:", "context": transcript})
    st.success("Done")

    st.subheader("📄 Preview")
    st.write(f"**Date:** {date}")
    st.write(f"**Attendees:** {', '.join(attendees)}")
    st.write("**Summary:**", summary)
    st.write("**Action Items:**", action_items)
    st.write("**Next Steps:**", next_steps)
    st.write("**Closing Remarks:**", closing)

    pdf = MeetingMinutesPDF()
    pdf.add_page()
    pdf.add_section("Date", date)
    pdf.add_section("Time", datetime.now().strftime("%I:%M %p"))
    pdf.add_bullet_points("Attendees", attendees)
    pdf.add_section("Meeting Overview", summary)
    pdf.add_section("Action Items", action_items)
    pdf.add_section("Next Steps", next_steps)
    pdf.add_section("Closing Remarks", closing)

    pdf_path = os.path.join(tempfile.gettempdir(), "minutes.pdf")
    pdf.output(pdf_path)
    with open(pdf_path, "rb") as f:
        st.download_button("Download PDF", f, file_name="Meeting_Minutes.pdf", mime="application/pdf")
