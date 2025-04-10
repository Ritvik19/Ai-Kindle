USER_GUIDE = """
**Welcome to AI-Kindle!**

This app helps you read PDFs, extract text, take notes, and interact with an AI assistant based on the document content.

**1. Upload a PDF:**
    - Click "Browse files" or drag and drop your PDF into the box above.
    - Wait for the app to process the PDF (extraction and reformatting). A progress bar will show the status.

**2. Navigate Pages:**
    - Once processed, the first page image and its text will appear.
    - Use the "⬅️ Previous Page" and "Next Page ➡️" buttons below the main content area to move through the document.
    - The current page number is displayed between the buttons.

**3. View Text:**
    - The middle column ("📖 Page Text") shows the text extracted and reformatted from the current page.
    - You can manually **copy text** from this box.

**4. Interact & Take Notes:**
    - **Select Text:** Copy the text you're interested in from the "📖 Page Text" box.
    - **Paste Text:** Paste the copied text into the "Paste selected text or enter page context..." area in the right column ("💬 Interaction").
    - **Save Highlight:** Click "📌 Save Selected Text as Note" to save the pasted text as a highlight note, linked to the current page. The pasted text area will clear.

**5. Ask the AI:**
    - **Provide Context:**
        - **Option A (Selected Text):** Paste the relevant text into the "Paste selected text or enter page context..." area.
        - **Option B (Specific Pages):** Type `@` followed by page numbers or ranges (e.g., `@1`, `@3-5`, `@1,3,5-7`). This tells the AI to use the text from those specific pages.
        - **Option C (All Pages):** Leave the context area **empty**. The AI will use the text from the *entire* document (may be slow for large PDFs).
    - **Enter Question:** Type your question about the provided context (pasted text or specified pages) into the "Enter your question..." input box.
    - **Get Answer:** Click "❓ Ask AI". The AI's response will appear below.
    - **Save AI Response:** If the response is helpful, click "💾 Save AI Response as Note" to save the question and answer.

**6. Manage Notes:**
    - All saved highlights and AI responses appear in the "📝 My Notes" section at the bottom.
    - Notes are displayed in columns.
    - Click "❌ Delete Note" to remove a specific note.
    - Click "📥 Export Notes to TXT" to download all your notes as a text file.

**Tips:**
- Text reformatting during PDF processing uses an AI and might take a moment per page.
- AI responses can also take a few seconds, especially with large contexts.
- If you upload a new PDF, your current notes will be cleared. Export them first if needed!
"""

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