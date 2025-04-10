import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import requests # For generic API calls, replace with your AI SDK if needed
import os

# --- Configuration ---
# !! IMPORTANT: Replace with your actual AI model details !!
AI_API_ENDPOINT = "YOUR_AI_API_ENDPOINT_HERE" # e.g., https://api.example.com/v1/chat/completions
AI_API_KEY = os.environ.get("YOUR_AI_API_KEY") # Best practice: use environment variables
st.set_page_config(layout="wide", page_title="AI PDF Reader")
if not AI_API_KEY:
    st.warning("AI API Key not found. Please set the YOUR_AI_API_KEY environment variable.", icon="⚠️")
    # You might want to add st.stop() here if the key is essential for core functionality
    # Or allow users to enter it via st.text_input(type="password")


# --- Helper Functions ---

def pdf_to_images_and_text(file_bytes):
    """Extracts images and text from each page of a PDF."""
    images = []
    texts = []
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)

            # Extract text
            texts.append(page.get_text("text"))

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

def ask_ai(context_text, query):
    """
    Placeholder function to interact with your AI model.
    Replace this with your specific AI model API call.
    """
    if not AI_API_KEY or not AI_API_ENDPOINT:
         return "Error: AI API Key or Endpoint not configured."

    st.info("Asking the AI...", icon="🤖") # Provide feedback

    # --- Replace this section with your specific AI API call ---
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }
    # Example payload structure (adjust based on your AI model's requirements)
    payload = {
        "model": "your-model-name-here", # Specify the model if required by the API
        "messages": [
            {"role": "system", "content": "You are a helpful assistant analyzing PDF text."},
            {"role": "user", "content": f"Based on the following text, answer the question:\n\nText:\n{context_text}\n\nQuestion:\n{query}"}
        ],
        # Add other parameters like temperature, max_tokens etc. if needed
    }

    try:
        response = requests.post(AI_API_ENDPOINT, headers=headers, json=payload, timeout=60)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # --- Parse the response (adjust based on your AI model's output format) ---
        # Example for OpenAI-like structure:
        # result = response.json()
        # ai_response = result.get('choices', [{}])[0].get('message', {}).get('content', 'No response content found.')

        # Example for a simpler JSON response:
        ai_response = response.json().get("response", "Error: Could not parse AI response.") # Adjust the key 'response' as needed

        return ai_response
    except requests.exceptions.RequestException as e:
        st.error(f"AI API call failed: {e}")
        return f"Error: Failed to connect to AI service. Details: {e}"
    except Exception as e:
        st.error(f"An error occurred during AI interaction: {e}")
        return f"Error: An unexpected error occurred. Details: {e}"
    # --- End of AI API call section ---


# --- Streamlit App ---
st.title("📄 AI-Powered PDF Reader & Notetaker")

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
            note_content = f"**Highlight from Page {current_page_index + 1}:**\n{st.session_state.selected_text}"
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
                    note_content = f"**AI Query (Page {current_page_index + 1}):**\n*Query:* {ai_query}\n*Context Text:*\n{st.session_state.selected_text}\n\n*AI Response:*\n{st.session_state.ai_response}"
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
            note_cols[i % 4].text_area(f"Note {i+1}", value=note, height=150, key=f"note_{i}", disabled=True)

        # --- Export Notes ---
        notes_text = "\n\n---\n\n".join(st.session_state.notes)
        st.download_button(
            label="📥 Export Notes to TXT",
            data=notes_text,
            file_name=f"{st.session_state.pdf_file_name}_notes.txt" if st.session_state.pdf_file_name else "notes.txt",
            mime="text/plain",
        )
    else:
        st.info("No notes saved yet.")

else:
    st.info("Please upload a PDF file to get started.")