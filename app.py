import streamlit as st
import whisper
import librosa
import os
import spacy
import re
from datetime import datetime
from fpdf import FPDF
from transformers import pipeline

# New imports:
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# ——— Model loading caches ———
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa         = pipeline("text2text-generation", model="google/flan-t5-large")
    return (
      HuggingFacePipeline(pipeline=summarizer),
      HuggingFacePipeline(pipeline=qa),
    )

whisper_model = load_whisper()
spacy_nlp     = load_spacy()
llm_summarizer, llm_qa = load_pipelines()

# ——— Prompts & Runnables ———
summary_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="Summarize the following meeting transcript:\n\n{transcript}"
)
qa_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="{question}\n\n{context}"
)

# Build RunnableSequences instead of deprecated LLMChains
summary_chain = summary_prompt | llm_summarizer
qa_chain      = qa_prompt | llm_qa

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
    # 1) Save uploaded file
    suffix = os.path.splitext(uploaded.name)[1]
    tmp_path = f"temp_audio{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # 2) Load with librosa (avoids ffmpeg)
    st.info("Loading audio into memory…")
    audio_array, sr = librosa.load(tmp_path, sr=16_000, mono=True)
    st.write(f"✅ Audio loaded: shape={audio_array.shape}, sr={sr}")

    # 3) Transcribe from array
    st.info("Transcribing…")
    result = whisper_model.transcribe(audio_array)
    transcript = result["text"]
    st.success("✅ Transcription complete")

    st.text_area("Transcript preview", transcript[:1000] + "…", height=200)

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
