import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")  # silence Whisper FP16 warning

import streamlit as st
import whisper
import os
import spacy
import re
from datetime import datetime
from fpdf import FPDF
from transformers import pipeline
import tempfile

# ——— New LangChain/HuggingFace imports ———
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# ─── CACHE MODELS ──────────────────────────

@st.cache_resource
def load_whisper():
    # smaller model for speed
    return whisper.load_model("tiny")

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_pipelines():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    qa         = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    return (
        HuggingFacePipeline(pipeline=summarizer),
        HuggingFacePipeline(pipeline=qa),
    )

whisper_model   = load_whisper()
spacy_nlp       = load_spacy()
llm_summarizer, llm_qa = load_pipelines()

# ─── PROMPTS & SEQUENCES ────────────────────

summary_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="Summarize the following meeting transcript:\n\n{transcript}"
)
qa_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="{question}\n\n{context}"
)

# build RunnableSequences
summary_chain = summary_prompt | llm_summarizer
qa_chain      = qa_prompt      | llm_qa

# ─── PDF BUILDER ────────────────────────────

class MeetingMinutesPDF(FPDF):
    def header(self):
        self.set_font("Arial","B",16)
        self.cell(0,10,"MEETING MINUTES",ln=True,align="C")
        self.ln(5)
        self.rect(5,5,200,287)
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial","I",10)
        self.cell(0,10,f"Page {self.page_no()}",align="C")
    def section(self, title, text):
        self.set_font("Arial","B",12)
        self.cell(0,8,title,ln=True,border="B")
        self.set_font("Arial","",11)
        self.multi_cell(0,7,text)
        self.ln(5)
    def bullets(self, title, items):
        self.set_font("Arial","B",12)
        self.cell(0,8,title,ln=True,border="B")
        self.set_font("Arial","",11)
        for i,item in enumerate(items):
            if i==10:
                self.cell(5); self.cell(0,7,f"+{len(items)-10} more",ln=True); break
            self.cell(5); self.cell(0,7,f"- {item.strip()}",ln=True)
        self.ln(5)

# ─── METADATA EXTRACTION ───────────────────

def extract_meta(text):
    # date
    m = re.search(r"\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b", text)
    date = m.group(0) if m else datetime.today().strftime("%d %B %Y")
    # attendees
    doc = spacy_nlp(text)
    attendees = list({ent.text for ent in doc.ents if ent.label_=="PERSON"})
    return date, attendees

# ─── STREAMLIT UI ───────────────────────────

st.title("🎤 Meeting Minutes Generator")

uploaded = st.file_uploader("Upload audio (.mp3/.wav/.m4a)", type=["mp3","wav","m4a"])
if not uploaded:
    st.info("Please upload an audio file to begin.")
    st.stop()

# save to temp file
ext = os.path.splitext(uploaded.name)[1]
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
tmp.write(uploaded.read())
tmp.close()

# transcribe
st.info("⏳ Transcribing with Whisper…")
try:
    res = whisper_model.transcribe(tmp.name)
    transcript = res["text"]
    st.success("✅ Transcription complete")
except Exception as e:
    st.error(f"Transcription error: {e}")
    os.unlink(tmp.name)
    st.stop()
os.unlink(tmp.name)

st.text_area("🔊 Transcript Preview", transcript[:2000]+"…", height=200)

# summarize & extract
st.info("🧠 Generating summary & extraction…")
summary     = summary_chain.invoke({"transcript": transcript[:2000]})
date, ppl    = extract_meta(transcript)
actions     = qa_chain.invoke({"question":"Extract the action items and responsibilities:", "context":transcript})
next_steps  = qa_chain.invoke({"question":"List next steps and follow‑ups:", "context":transcript})
closing     = qa_chain.invoke({"question":"Summarize the closing remarks:",       "context":transcript})
st.success("🎉 Done")

# preview
st.subheader("📄 Preview")
st.write(f"**Date:** {date}")
st.write(f"**Attendees:** {', '.join(ppl)}")
st.write("**Summary:**", summary)
st.write("**Action Items:**", actions)
st.write("**Next Steps:**", next_steps)
st.write("**Closing Remarks:**", closing)

# build PDF
pdf = MeetingMinutesPDF()
pdf.add_page()
pdf.section("Date",     date)
pdf.section("Time",     datetime.now().strftime("%I:%M %p"))
pdf.bullets("Attendees", ppl)
pdf.section("Meeting Overview", summary)
pdf.section("Action Items",     actions)
pdf.section("Next Steps",       next_steps)
pdf.section("Closing Remarks",  closing)

out = os.path.join(tempfile.gettempdir(), "minutes.pdf")
pdf.output(out)
with open(out,"rb") as f:
    st.download_button("📥 Download PDF", f, "Meeting_Minutes.pdf", "application/pdf")
