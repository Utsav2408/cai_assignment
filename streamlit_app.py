import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Chat Interface with RAG Options", page_icon="💬", layout="wide")
st.title("💬 RAG-Powered Chat Interface with Immediate Response")

st.markdown("""
Welcome! Use the toggle below to switch between **Basic RAG** and **Advanced RAG** modes. 
Each mode responds differently based on the retrieval logic used.
""")

# --- Toggle Between RAG Modes ---
rag_mode = st.radio(
    "Select RAG Mode:",
    ("Basic RAG", "Advanced RAG"),
    horizontal=True
)

# --- Session State for Message History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Basic RAG Placeholder Function ---
def basic_rag(query):
    # Placeholder for real basic RAG logic, simple keyword match or naive search
    return f"[Basic RAG] Here is a simple answer to: '{query}'"

# --- Advanced RAG Placeholder Function ---
def advanced_rag(query):
    # Placeholder for advanced RAG (e.g., using embeddings, vector DB, advanced ranking)
    return f"[Advanced RAG] This is a detailed response to: '{query}', leveraging deeper context and semantic search."


# --- Input Field ---
user_input = st.chat_input("Type your message here...")


# --- Immediately Display User Message and Generate Response ---
if user_input:
    # Append user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Render user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and append bot response based on selected RAG mode
    if rag_mode == "Basic RAG":
        response = basic_rag(user_input)
    else:
        response = advanced_rag(user_input)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Render assistant response immediately
    with st.chat_message("assistant"):
        st.markdown(response)


# --- Render Previous Chat History (if any) ---
# NOTE: Render this AFTER handling current input to keep order proper
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
