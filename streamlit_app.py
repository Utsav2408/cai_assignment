import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Chat Interface with RAG Options", page_icon="💬", layout="wide")
st.title("💬 RAG-Powered Chat Interface with Immediate Response")

st.markdown("""
Welcome! Use the toggle below to switch between **Basic RAG** and **Advanced RAG** modes.  
Each mode keeps a separate chat history to help you stay organized.
""")

# --- Session State Initialization for Both Modes ---
if "basic_messages" not in st.session_state:
    st.session_state.basic_messages = [{"role": "assistant", "content": "Hi! You are now in **Basic RAG** mode. How can I help you?"}]
if "advanced_messages" not in st.session_state:
    st.session_state.advanced_messages = [{"role": "assistant", "content": "Hi! You are now in **Advanced RAG** mode. How can I help you?"}]
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

    # Clear Chat button for current mode
    if st.button("🗑️ Clear Current Chat"):
        if rag_mode == "Basic RAG":
            st.session_state.basic_messages = [{"role": "assistant", "content": "Chat cleared. You are now in **Basic RAG** mode. How can I assist you?"}]
        else:
            st.session_state.advanced_messages = [{"role": "assistant", "content": "Chat cleared. You are now in **Advanced RAG** mode. How can I assist you?"}]
        st.session_state.rag_mode = rag_mode  # Ensure sync
        st.rerun()

# --- Update mode on switch without clearing chat ---
if st.session_state.rag_mode != rag_mode:
    st.session_state.rag_mode = rag_mode  # Just update the active mode

# --- RAG Placeholder Functions ---
def basic_rag(query):
    return f"[Basic RAG] Here is a simple answer to: '{query}'"

def advanced_rag(query):
    return f"[Advanced RAG] This is a detailed response to: '{query}', leveraging deeper context and semantic search."

# --- Select Correct Message Store Based on Mode ---
current_chat = st.session_state.basic_messages if st.session_state.rag_mode == "Basic RAG" else st.session_state.advanced_messages

# --- Input Field ---
user_input = st.chat_input(f"Type your message here for {st.session_state.rag_mode}...")

# --- Handle New User Input and Generate Response (Adding to Current Mode's Chat History) ---
if user_input:
    # Add user message
    current_chat.append({"role": "user", "content": user_input})

    # Generate response based on selected mode
    if st.session_state.rag_mode == "Basic RAG":
        response = basic_rag(user_input)
    else:
        response = advanced_rag(user_input)

    # Handle empty/None responses
    response = response if response else "I'm sorry, I couldn't find an answer. Could you rephrase your question?"

    # Add assistant's response
    current_chat.append({"role": "assistant", "content": response})

# --- Render All Messages for the Current Mode ---
for message in current_chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
