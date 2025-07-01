import streamlit as st
import whisper
import os
import spacy
import re
from datetime import datetime
from fpdf import FPDF
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import tempfile

# ——— Load Models ———
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_langchain_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_model   = pipeline("text2text-generation", model="google/flan-t5-large")
    return HuggingFacePipeline(pipeline=summarizer), HuggingFacePipeline(pipeline=qa_model)

whisper_model = load_whisper_model()
spacy_nlp = load_spacy_model()
llm_summarizer, llm_qa = load_langchain_pipelines()

# ——— LangChain Prompts ———
summary_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="Summarize the following meeting transcript:\n\n{transcript}"
)

qa_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="{question}\n\n{context}"
)

# Use `|` as RunnableSequence (no import of RunnableSequence needed)
summary_chain = summary_prompt | llm_summarizer
qa_chain = qa_prompt | llm_qa

# ——— PDF Generator ———
class MeetingPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Meeting Minutes", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def add_section(self, title, content):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 8, content)
        self.ln(5)

    def add_bullet_points(self, title, points):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", "", 11)
        for point in points:
            self.cell(5)
            self.cell(0, 8, f"• {point.strip()}", ln=True)
        self.ln(5)

# ——— Helper: Extract Metadata ———
def extract_metadata(text):
    doc = spacy_nlp(text)
    date_match = re.search(r"\b\d{1,2} \w+ \d{4}\b", text)
    date = date_match.group() if date_match else datetime.now().strftime("%d %B %Y")
    attendees = list({ent.text for ent in doc.ents if ent.label_ == "PERSON"})
    return date, attendees

# ——— Streamlit UI ———
st.title("🎤 Meeting Minutes Generator")

uploaded_file = st.file_uploader("Upload your meeting audio (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.info("Transcribing using Whisper...")
    try:
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]
        st.success("Transcription complete!")
        st.text_area("Transcript", transcript[:2000], height=300)
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        os.remove(audio_path)
        st.stop()
    os.remove(audio_path)

    st.info("Generating Meeting Summary and Insights...")
    summary = summary_chain.invoke({"transcript": transcript})
    date, attendees = extract_metadata(transcript)
    action_items = qa_chain.invoke({"question": "What are the action items and responsibilities?", "context": transcript})
    next_steps = qa_chain.invoke({"question": "What are the next steps?", "context": transcript})
    closing = qa_chain.invoke({"question": "Summarize the closing remarks.", "context": transcript})

    st.success("Analysis complete!")

    # Display results
    st.subheader("📝 Meeting Summary")
    st.write("**Date:**", date)
    st.write("**Attendees:**", ", ".join(attendees))
    st.write("**Summary:**", summary)
    st.write("**Action Items:**", action_items)
    st.write("**Next Steps:**", next_steps)
    st.write("**Closing Remarks:**", closing)

    # Generate PDF
    pdf = MeetingPDF()
    pdf.add_page()
    pdf.add_section("Date", date)
    pdf.add_bullet_points("Attendees", attendees)
    pdf.add_section("Meeting Summary", summary)
    pdf.add_section("Action Items", action_items)
    pdf.add_section("Next Steps", next_steps)
    pdf.add_section("Closing Remarks", closing)

    pdf_path = os.path.join(tempfile.gettempdir(), "meeting_minutes.pdf")
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Meeting Minutes PDF", f, file_name="Meeting_Minutes.pdf", mime="application/pdf")
