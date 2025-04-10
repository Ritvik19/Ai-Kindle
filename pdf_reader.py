import streamlit as st
import fitz
from PIL import Image
import io
import requests
import os
import re
from tqdm.auto import trange
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_KWARGS = dict(temperature=0, max_tokens=None, api_key=GEMINI_API_KEY)
LEARN_LM = ChatGoogleGenerativeAI(model="learnlm-1.5-pro-experimental", **GEMINI_KWARGS)
GEMINI_2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash", **GEMINI_KWARGS)

RAG_PROMPT = """
**Task:** Answer questions based on a given context.

**Additional Details:**
- The context will be provided as a string.
- Questions will be in natural language and require reasoning before providing an answer.
- Answers should be formatted using markdown with appropriate headings, bullet points, tables, etc., as required by the question.

# Steps

1. **Understand the Context:** Read and comprehend the given context to extract relevant information.
2. **Analyze the Question:** Break down the question into smaller parts if necessary. Identify what specific information is being asked for.
3. **Reason and Infer:** Use logical reasoning and inference based on the context to derive the answer.
4. **Format the Answer:** Structure the answer using markdown formatting, including headings, bullet points, tables, etc., as appropriate.

# Output Format

The output should be in markdown format for the question asked. The answer should be structured appropriately using bullet points, tables, or other markdown elements as necessary.

# Notes
- Always ensure that the answer is based on the given context. If the question cannot be answered from the provided context, clearly state "Cannot be determined from the given context."
- Use tables sparingly and only when necessary to present complex data in a structured format.

{context}

---

Question: {query}
""".strip()

REFORMAT_PROMPT = """
Reformat the given text as markdown.

The goal is to convert the provided text into valid markdown format. Preserve the original content and structure as much as possible while applying appropriate markdown syntax for headings, lists, emphasis, links, and other common elements.

# Output Format

The output should be a single string containing the reformatted text in markdown format.

```markdown
The reformatted text goes here.
```

# Text to Reformat
{text}
"""
# --- Streamlit Configuration ---

st.set_page_config(layout="wide", page_title="AI PDF Reader")


# --- Helper Functions ---

def pdf_to_images_and_text(file_bytes):
    """Extracts images and text from each page of a PDF."""
    images = []
    texts = []
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in trange(len(pdf_document)):
            page = pdf_document.load_page(page_num)

            # Extract text
            page_text = page.get_text("text")
            reformatted_text = reformat_text(page_text)
            texts.append(reformatted_text)

            # Render page to an image (pixmap)
            pix = page.get_pixmap(dpi=150) # Increase DPI for better quality if needed
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            images.append(img_byte_arr.getvalue())

        pdf_document.close()
        return images, texts
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return [], []

def reformat_text(text):
    messages = [{"role": "user", "content": REFORMAT_PROMPT.format(text=text)}]
    try:
        response = GEMINI_2.invoke(messages).content
        response = re.findall(r"```markdown\n(.*?)\n```", response, re.DOTALL)[0]
        return response
    except Exception as e:
        st.error(f"Error during text reformatting: {e}")
        return text

def ask_ai(context, query):
    """
    Placeholder function to interact with your AI model.
    Replace this with your specific AI model API call.
    """
    st.info("Asking the AI...", icon="🤖")
    messages = [{"role": "user", "content": RAG_PROMPT.format(context=context, query=query)}]
    try:
        response = LEARN_LM.invoke(messages).content
        return response
    except Exception as e:
        st.error(f"An error occurred during AI interaction: {e}")
        return f"Error: An unexpected error occurred. Details: {e}"


# --- Streamlit App ---
st.title("📄 AI-Kindle")

# --- Initialize Session State ---
# Stores data across reruns
if 'pdf_images' not in st.session_state:
    st.session_state.pdf_images = []
if 'pdf_texts' not in st.session_state:
    st.session_state.pdf_texts = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'notes' not in st.session_state:
    st.session_state.notes = [] # List to store notes (highlighted text or AI responses)
if 'selected_text' not in st.session_state:
    st.session_state.selected_text = ""
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = ""
if 'pdf_file_name' not in st.session_state:
    st.session_state.pdf_file_name = ""


# --- PDF Upload Section ---
st.header("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Check if it's a new file
    if uploaded_file.name != st.session_state.get('pdf_file_name', ''):
        st.session_state.pdf_file_name = uploaded_file.name
        st.info("Processing PDF...")
        file_bytes = uploaded_file.getvalue()
        st.session_state.pdf_images, st.session_state.pdf_texts = pdf_to_images_and_text(file_bytes)
        st.session_state.current_page = 0
        st.session_state.notes = [] # Reset notes for new PDF
        st.session_state.selected_text = ""
        st.session_state.ai_response = ""
        st.success("PDF Processed!")
        st.rerun() # Rerun to update the main view immediately
    else:
        st.write(f"Loaded: `{st.session_state.pdf_file_name}`")

st.divider()

# --- Main Area for PDF Display and Interaction ---
if st.session_state.pdf_images:
    total_pages = len(st.session_state.pdf_images)

    # --- Page Navigation ---
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("⬅️ Previous Page", disabled=(st.session_state.current_page == 0)):
            st.session_state.current_page -= 1
            st.session_state.selected_text = "" # Clear selection on page change
            st.session_state.ai_response = ""
            st.rerun()
    with col_nav2:
        st.write(f"Page: **{st.session_state.current_page + 1} / {total_pages}**")
    with col_nav3:
        if st.button("Next Page ➡️", disabled=(st.session_state.current_page == total_pages - 1)):
            st.session_state.current_page += 1
            st.session_state.selected_text = "" # Clear selection on page change
            st.session_state.ai_response = ""
            st.rerun()

    st.divider()

    # --- PDF Page Display and Text Interaction in Three Columns ---
    current_page_index = st.session_state.current_page
    col_image, col_text, col_interact = st.columns([1, 1, 1]) # Adjust widths as needed

    with col_image:
        st.subheader("📄 PDF View")
        st.image(st.session_state.pdf_images[current_page_index], caption=f"Page {current_page_index + 1}", use_column_width=True)

    with col_text:
        st.subheader("📖 Page Text")
        page_text = st.session_state.pdf_texts[current_page_index]
        st.text_area("Extracted Text (Copy from here)", value=page_text, height=400, key=f"text_disp_{current_page_index}", disabled=True)


    with col_interact:
        st.subheader("💬 Interaction")

        # Area for user to paste selected text
        st.session_state.selected_text = st.text_area(
            "Paste selected text here:",
            height=150,
            key="text_selection_area", # Give it a key to help preserve state if needed
            value=st.session_state.selected_text # Persist value across simple interactions
        )

        # --- Actions for Selected Text ---
        if st.button("📌 Save Selected Text as Note", disabled=not st.session_state.selected_text):
            note_content = f"Highlight from Page {current_page_index + 1}\n\n---\n\n{st.session_state.selected_text}\n\n==="
            st.session_state.notes.append(note_content)
            st.success("Text saved as a note!")
            st.session_state.selected_text = "" # Clear selection after saving
            st.rerun() # Update immediately

        st.divider()

        # --- AI Query Section ---
        st.subheader("🤖 Ask AI about Selected Text")
        ai_query = st.text_input("Enter your question about the selected text:")

        if st.button("❓ Ask AI", disabled=not st.session_state.selected_text or not ai_query):
            # Call the AI function
            st.session_state.ai_response = ask_ai(st.session_state.selected_text, ai_query)
            # No need to rerun here, response will display below

        # Display AI response if available
        if st.session_state.ai_response:
            st.text_area("AI Response:", value=st.session_state.ai_response, height=150, key="ai_response_display", disabled=True)
            if not st.session_state.ai_response.startswith("Error:"): # Only allow saving valid responses
                if st.button("💾 Save AI Response as Note"):
                    note_content = f"AI Query (Page {current_page_index + 1})\nQuery: {ai_query}\n\n---\n\n{st.session_state.ai_response}\n\n==="
                    st.session_state.notes.append(note_content)
                    st.success("AI response saved as a note!")
                    # Optionally clear parts of the state after saving
                    # st.session_state.ai_response = ""
                    # st.session_state.selected_text = ""
                    st.rerun() # Update immediately

    st.divider()

    # --- Display Notes in Four Columns ---
    if st.session_state.notes:
        st.subheader("📝 My Notes")
        note_cols = st.columns(4)
        for i, note in enumerate(st.session_state.notes):
            with note_cols[i % 4]:
                st.text_area(f"Note {i+1}", value=note, height=150, key=f"note_{i}")
                if st.button(f"❌ Delete Note {i+1}", key=f"delete_note_{i}"):
                    st.session_state.notes.pop(i)
                    st.success(f"Note {i+1} deleted!")
                    st.rerun()  # Rerun to update the UI immediately

        # --- Export Notes ---
        notes_text = "\n\n---\n\n".join(st.session_state.notes)
        st.download_button(
            label="📥 Export Notes to TXT",
            data=notes_text,
            file_name=f"{st.session_state.pdf_file_name.replace('.', '_')}_notes.txt" if st.session_state.pdf_file_name else "notes.txt",
            mime="text/plain",
        )
    else:
        st.info("No notes saved yet.")

else:
    st.info("Please upload a PDF file to get started.")