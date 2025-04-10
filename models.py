from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_KWARGS = dict(temperature=0, max_tokens=None, api_key=GEMINI_API_KEY)
LEARN_LM = ChatGoogleGenerativeAI(model="learnlm-1.5-pro-experimental", **GEMINI_KWARGS)
GEMINI_2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash", **GEMINI_KWARGS)