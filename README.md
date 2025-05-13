# Minute Meet

This project automates the process of transcribing, summarizing, and generating meeting minutes from an audio file using OpenAI Whisper, Facebook BART, Google FLAN-T5, and SpaCy. It outputs a well-formatted PDF with all essential meeting details.

## Features

-  Audio transcription using Whisper
-  Summarization using BART
-  Information extraction (action items, next steps, closing remarks) using FLAN-T5
-  Attendee identification using SpaCy
-  PDF report generation using FPDF
-  Runs in Google Colab

## Installation (Colab)

```bash
!pip install -U openai-whisper
!apt-get install ffmpeg
!python -m spacy download en_core_web_sm
!apt-get install fonts-noto
!pip install fpdf
!pip install transformers torch
```

## Usage

1. Upload your audio file (e.g., `.mp3`) to Colab.
2. Set the path in `audio_file_path`.
3. Run the notebook cells in order.
4. Download the generated PDF with the meeting summary.

## Output PDF Includes

* Date and Time
* Attendees
* Meeting Overview
* Action Items & Responsibilities
* Next Steps & Follow-ups
* Closing Remarks

## Models Used

* `openai/whisper-base` for transcription
* `facebook/bart-large-cnn` for summarization
* `google/flan-t5-large` for extraction
* `en_core_web_sm` (SpaCy) for named entity recognition

## Notes

* Works best with clear audio.
* Limited to \~700 words for summarization.
* Only top 10 attendees are listed explicitly in the PDF.

