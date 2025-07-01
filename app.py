import streamlit as st
import whisper
import os
import spacy
import re
from datetime import datetime
from fpdf import FPDF
from transformers import pipeline
import tempfile

# LangChain 0.x imports
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ─── CACHING MODELS ─────────────────────────────────

@st.cache_resource
def get_whisper():
    return whisper.load_model("tiny")  # tiny for faster loads

@st.cache_resource
def get_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def get_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa         = pipeline("text2text-generation", model="google/flan-t5-large")
    return (
        HuggingFacePipeline(pipeline=summarizer),
        HuggingFacePipeline(pipeline=qa),
    )

whisper_model = get_whisper()
spacy_nlp     = get_spacy()
llm_summarizer, llm_qa = get_pipelines()

# ─── PROMPTS & CHAINS ────────────────────────────────

summary_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="Summarize the following meeting transcript:\n\n{transcript}"
)
qa_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="{question}\n\n{context}"
)

summary_chain = LLMChain(llm=llm_summarizer, prompt=summary_prompt)
qa_chain      = LLMChain(llm=llm_qa,      prompt=qa_prompt)

# ─── PDF BUILDER ─────────────────────────────────────

class MeetingMinutesPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "MEETING MINUTES", ln=True, align="C")
        self.ln(5)
        self.rect(5, 5, 200, 287)

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

    def add_bullet(self, title, items):
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, title, ln=True, border="B")
        self.set_font("Arial", size=11)
        for i, item in enumerate(items):
            # limit to first 10, then "... +n more"
            if i == 10:
                self.cell(5)
                self.cell(0, 7, f"+{len(items)-10} more", ln=True)
                break
            self.cell(5)
            self.cell(0, 7, f"- {item.strip()}", ln=True)
        self.ln(5)

# ─── METADATA EXTRACTION ─────────────────────────────

def extract_meta(text):
    doc = spacy_nlp(text)
    date_m = re.search(r"\b\d{1,2} (?:January|February|...) \d{4}\b", text)
    date = date_m.group(0) if date_m else datetime.today().strftime("%d %B %Y")
    ppl  = list({ent.text for ent in doc.ents if ent.label_=="PERSON"})
    return date, ppl

# ─── STREAMLIT UI ────────────────────────────────────

st.title("🎤 Meeting Minutes Generator")

audio = st.file_uploader("Upload audio (.mp3/.wav/.m4a)", type=["mp3","wav","m4a"])
if audio:
    # save to temp file
    ext = os.path.splitext(audio.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(audio.read())
    tmp.close()
    tmp_path = tmp.name

    # transcribe
    st.info("Transcribing…")
    try:
        res = whisper_model.transcribe(tmp_path)
        transcript = res["text"]
        st.success("Transcription done")
    except Exception as e:
        st.error(f"Transcription error: {e}")
        os.unlink(tmp_path)
        st.stop()

    os.unlink(tmp_path)

    st.text_area("🔊 Transcript", transcript[:2000]+"…", height=200)

    # summarize & extract
    st.info("Generating summary & extract…")
    summary = summary_chain.run(transcript[:2000])
    date, attendees = extract_meta(transcript)
    actions = qa_chain.run({"question":"Extract action items:", "context":transcript})
    steps   = qa_chain.run({"question":"List next steps:",    "context":transcript})
    closing = qa_chain.run({"question":"Summarize closing remarks:", "context":transcript})
    st.success("Done")

    # preview
    st.subheader("📄 Preview")
    st.write(f"**Date:** {date}")
    st.write(f"**Attendees:** {', '.join(attendees)}")
    st.write("**Summary:**", summary)
    st.write("**Action Items:**", actions)
    st.write("**Next Steps:**", steps)
    st.write("**Closing Remarks:**", closing)

    # build PDF
    pdf = MeetingMinutesPDF()
    pdf.add_page()
    pdf.add_section("Date", date)
    pdf.add_section("Time", datetime.now().strftime("%I:%M %p"))
    pdf.add_bullet("Attendees", attendees)
    pdf.add_section("Meeting Overview", summary)
    pdf.add_section("Action Items", actions)
    pdf.add_section("Next Steps", steps)
    pdf.add_section("Closing Remarks", closing)

    out_path = os.path.join(tempfile.gettempdir(), "minutes.pdf")
    pdf.output(out_path)
    with open(out_path, "rb") as f:
        st.download_button("📥 Download PDF", f, "Meeting_Minutes.pdf", "application/pdf")
