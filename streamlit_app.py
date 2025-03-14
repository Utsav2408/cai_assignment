import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Chat Interface with RAG Options", page_icon="💬", layout="wide")
st.title("💬 RAG-Powered Chat Interface with Immediate Response")

st.markdown("""
Welcome! Use the toggle below to switch between **Basic RAG** and **Advanced RAG** modes.  
Each mode responds differently based on the retrieval logic used.
""")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    # Initial greeting message
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]

if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "Basic RAG"  # Default mode

# --- Sidebar for Controls (RAG Mode + Clear Chat) ---
with st.sidebar:
    st.header("⚙️ Settings")
    # RAG Mode toggle
    rag_mode = st.radio(
        "Select RAG Mode:",
        ("Basic RAG", "Advanced RAG"),
        horizontal=False,
        index=0 if st.session_state.rag_mode == "Basic RAG" else 1
    )

    # Clear Chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [{"role": "assistant", "content": f"Chat cleared. You are now using **{rag_mode}** mode. How can I assist you?"}]
        st.session_state.rag_mode = rag_mode  # Make sure mode stays in sync
        st.rerun()  # Refresh app to show cleared state

# --- Keep chat history on RAG Mode change ---
# Simply update mode without clearing chat
if st.session_state.rag_mode != rag_mode:
    st.session_state.rag_mode = rag_mode  # Update mode only, keep messages intact
    st.rerun()  # Optional: refresh to update UI if necessary

# --- RAG Placeholder Functions ---
def basic_rag(query):
    return f"[Basic RAG] Here is a simple answer to: '{query}'"

def advanced_rag(query):
    return f"[Advanced RAG] This is a detailed response to: '{query}', leveraging deeper context and semantic search."

# --- Input Field ---
user_input = st.chat_input("Type your message here...")

# --- Handle New User Input and Generate Response (Adding to Chat History) ---
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Select appropriate RAG mode to generate a response
    response = basic_rag(user_input) if st.session_state.rag_mode == "Basic RAG" else advanced_rag(user_input)

    # Handle empty/None responses
    response = response if response else "I'm sorry, I couldn't find an answer. Could you rephrase your question?"

    # Add assistant's response
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Render All Messages in Order ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
