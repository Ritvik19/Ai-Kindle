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