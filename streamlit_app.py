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
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]  # Initial greeting
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
        st.session_state.rag_mode = rag_mode
        st.experimental_rerun()  # Refresh to show cleared chat

# --- Reset Chat on RAG Mode Change ---
if st.session_state.rag_mode != rag_mode:
    st.session_state.rag_mode = rag_mode
    st.session_state.messages = [{"role": "assistant", "content": f"You are now using **{rag_mode}**. How can I help you?"}]
    st.experimental_rerun()  # Refresh to apply mode change

# --- RAG Placeholder Functions ---
def basic_rag(query):
    return f"[Basic RAG] Here is a simple answer to: '{query}'"

def advanced_rag(query):
    return f"[Advanced RAG] This is a detailed response to: '{query}', leveraging deeper context and semantic search."

# --- Input Field ---
user_input = st.chat_input("Type your message here...")

# --- Handle User Input and Generate Response ---
if user_input:
    user_message = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Select and generate response based on RAG mode
    response = basic_rag(user_input) if rag_mode == "Basic RAG" else advanced_rag(user_input)
    response = response if response else "I'm sorry, I couldn't find an answer. Could you rephrase your question?"

    assistant_message = {"role": "assistant", "content": response}
    st.session_state.messages.append(assistant_message)

    with st.chat_message("assistant"):
        st.markdown(response)

# --- Render Previous Chat History (excluding current user interaction if present) ---
history_to_render = st.session_state.messages[:-2] if user_input else st.session_state.messages
for message in history_to_render:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
