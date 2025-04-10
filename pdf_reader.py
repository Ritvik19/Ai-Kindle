import io
import os
import re
import time

import fitz
import requests
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from tqdm.auto import trange

from prompts import RAG_PROMPT, REFORMAT_PROMPT, USER_GUIDE

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_KWARGS = dict(temperature=0, max_tokens=None, api_key=GEMINI_API_KEY)
LEARN_LM = ChatGoogleGenerativeAI(model="learnlm-1.5-pro-experimental", **GEMINI_KWARGS)
GEMINI_2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash", **GEMINI_KWARGS)

# --- Streamlit Configuration ---
st.set_page_config(layout="wide", page_title="AI Kindle")

# --- Helper Functions ---
@st.cache_data
def pdf_to_images_and_text(file_bytes, reformat_enabled):
    """Extracts images and text from each page of a PDF."""
    images = []
    texts = []
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        num_pages = len(pdf_document)
        for page_num in trange(num_pages):
            page = pdf_document.load_page(page_num)

            # Extract text
            page_text = page.get_text("text")
            processed_text = reformat_text(page_text) if reformat_enabled else page_text
            texts.append(processed_text)

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
    max_retries = 3
    retry_delay = 10

    for attempt in range(max_retries):
        try:
            response = GEMINI_2.invoke(messages).content
            extracted_response = re.findall(r"```markdown\n(.*?)\n```", response, re.DOTALL)[0]
            return extracted_response
        except Exception as e:
            if "429" in str(e):  # Check if the error is a 429 Too Many Requests
                print(f"429 error encountered. Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Error during text reformatting (attempt {attempt + 1}): {e}")
            
            if attempt == max_retries - 1:  # If it's the last attempt, return the original text
                print("Max retries reached. Returning original text.")
                return text

def ask_ai(context, query):
    """
    Placeholder function to interact with your AI model.
    Replace this with your specific AI model API call.
    """
    
    if context.strip().startswith("@"):  # If context starts with @, process page numbers
        try:
            pages = context[1:].split(",")  # Remove @ and split by commas
            selected_texts = []
            selected_pages = []  # To track selected page numbers
            for page in pages:
                if "-" in page:  # Handle ranges (e.g., 3-5)
                    start, end = map(int, page.split("-"))
                    selected_texts.extend(st.session_state.pdf_texts[start - 1:end])
                    selected_pages.extend(range(start, end + 1))
                else:  # Handle single page (e.g., 1)
                    selected_texts.append(st.session_state.pdf_texts[int(page) - 1])
                    selected_pages.append(int(page))
            context = "\n\n".join(selected_texts)
            page_info = f"Selected pages: {', '.join(map(str, selected_pages))}"
        except (ValueError, IndexError) as e:
            st.error(f"Invalid page specification: {e}")
            return f"Error: Invalid page specification. Details: {e}"
    elif context.strip() == "":  # If context is empty, use all text from all pages
        context = "\n\n".join(st.session_state.pdf_texts)
        page_info = "Using all pages"
    else:  # If context is provided directly
        page_info = "Using selected text"

    model = LEARN_LM if len(context.split()) < 20_000 else GEMINI_2

    st.info(f"Asking the AI... ({page_info})", icon="🤖")
    messages = [{"role": "user", "content": RAG_PROMPT.format(context=context, query=query)}]
    try:
        response = model.invoke(messages).content
        return response
    except Exception as e:
        st.error(f"An error occurred during AI interaction: {e}")
        return f"Error: An unexpected error occurred. Details: {e}"


# --- Streamlit App ---
st.title("📔 AI-Kindle")

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
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
reformat_enabled_checkbox = st.sidebar.checkbox("Reformat Text (slower)", value=True)

with st.sidebar.expander("User Guide"):
    st.write(USER_GUIDE)

if uploaded_file is not None:
    # Check if it's a new file
    if uploaded_file.name != st.session_state.get('pdf_file_name', ''):
        st.session_state.pdf_file_name = uploaded_file.name
        st.info("Processing PDF...")
        file_bytes = uploaded_file.getvalue()
        st.session_state.pdf_images, st.session_state.pdf_texts = pdf_to_images_and_text(file_bytes, reformat_enabled_checkbox)
        st.session_state.current_page = 0
        st.session_state.notes = [] # Reset notes for new PDF
        st.session_state.selected_text = ""
        st.session_state.ai_response = ""
        st.success("PDF Processed!")
        st.rerun() # Rerun to update the main view immediately
    else:
        st.sidebar.write(f"Loaded: `{st.session_state.pdf_file_name}`")

# --- Main Area for PDF Display and Interaction ---
if st.session_state.pdf_images:
    total_pages = len(st.session_state.pdf_images)

    # --- PDF Page Display and Text Interaction in Three Columns ---
    current_page_index = st.session_state.current_page
    col_image, col_text, col_interact = st.columns([1, 1, 1]) # Adjust widths as needed

    with col_image:
        st.subheader("📄 PDF View")
        st.image(st.session_state.pdf_images[current_page_index], use_column_width=True)
        # --- Page Navigation ---
        col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
        with col_nav1:
            if st.button("1 ⬅️", disabled=current_page_index == 0):
                st.session_state.current_page = 0 
                st.rerun()
        with col_nav2:
            page_num_input = st.number_input("Go to page:", min_value=1, max_value=total_pages, value=st.session_state.current_page + 1, step=1, label_visibility="collapsed")
            if page_num_input != st.session_state.current_page + 1:
                st.session_state.current_page = page_num_input - 1
                st.rerun()
        with col_nav3:
            if st.button(f"➡️ {total_pages}", disabled=current_page_index == total_pages - 1):
                st.session_state.current_page = total_pages - 1
                st.rerun()

    with col_text:
        st.subheader("📖 Page Text")
        page_text = st.session_state.pdf_texts[current_page_index]
        st.text_area("Extracted Text (Copy from here)", value=page_text, height=500, key=f"text_disp_{current_page_index}", disabled=True)


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
            st.session_state.show_response_dialog = False # Show dialog for response
            # No need to rerun here, response will display below

        # Display AI response if available
        if st.session_state.ai_response:
                st.text_area("AI Response Preview:", value=st.session_state.ai_response, height=100, key="ai_response_display_preview", disabled=True)

                col_resp1, col_resp2 = st.columns(2)
                with col_resp1:
                    # --- Button to open the dialog ---
                    if st.button("👁️ View Full Response", use_container_width=True):
                        st.session_state.show_response_dialog = True
                        st.rerun() # Rerun to make the dialog appear

                with col_resp2:
                    # --- Button to save the response ---
                    # Only allow saving valid responses
                    is_error_response = st.session_state.ai_response.startswith("Error:")
                    if st.button("💾 Save AI Response", use_container_width=True, disabled=is_error_response):
                        # Determine context for the note title
                        context_desc = "Selected Text"
                        if st.session_state.selected_text.strip().startswith("@"):
                            context_desc = f"Pages {st.session_state.selected_text.strip()[1:]}"
                        elif not st.session_state.selected_text.strip():
                            context_desc = "All Pages"

                        note_content = f"AI Query ({context_desc})\nQuery: {ai_query}\n\n---\n\n{st.session_state.ai_response}\n\n==="
                        st.session_state.notes.append(note_content)
                        st.success("AI response saved as a note!")
                        st.session_state.show_response_dialog = False
                        # Optionally clear parts of the state after saving
                        # st.session_state.ai_response = ""
                        # st.session_state.selected_text = ""
                        st.rerun()

    if st.session_state.get("show_response_dialog", False) and st.session_state.ai_response:    
        @st.dialog("Full AI Response")
        def show_full_response():
            st.text_area(label="ai_reponse", value=st.session_state.ai_response, height=400)
            if st.button("Close Dialog"):
                st.session_state.show_response_dialog = False
                st.rerun() # Rerun to close the dialog

        # Call the dialog function to display it
        show_full_response()

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