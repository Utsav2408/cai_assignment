import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Group 97's Chat Interface with RAG Options", page_icon="💬", layout="wide")
st.title("💬 Group 97's RAG-Powered Chat Interface")

st.markdown("""
Welcome! Use the toggle below to switch between **Basic RAG** and **Advanced RAG** modes.  
Each mode responds differently based on the retrieval logic used.
""")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]  # Initial greeting
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "Basic RAG"  # Default mode

# --- Toggle Between RAG Modes ---
rag_mode = st.radio(
    "Select RAG Mode:",
    ("Basic RAG", "Advanced RAG"),
    horizontal=True
)

# --- Reset Chat on RAG Mode Change ---
if st.session_state.rag_mode != rag_mode:
    st.session_state.rag_mode = rag_mode
    st.session_state.messages = [{"role": "assistant", "content": f"You are now using **{rag_mode}**. How can I help you?"}]

# --- RAG Placeholder Functions ---
def basic_rag(query):
    # Placeholder for real Basic RAG logic
    return f"[Basic RAG] Here is a simple answer to: '{query}'"

def advanced_rag(query):
    # Placeholder for real Advanced RAG logic
    return f"[Advanced RAG] This is a detailed response to: '{query}', leveraging deeper context and semantic search."

# --- Input Field ---
user_input = st.chat_input("Type your message here...")

# --- Handle User Input and Generate Response ---
if user_input:
    # Append user's message
    user_message = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_message)

    # Render user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Select and generate response based on RAG mode
    if rag_mode == "Basic RAG":
        response = basic_rag(user_input)
    else:
        response = advanced_rag(user_input)

    # Ensure fallback if response is empty
    response = response if response else "I'm sorry, I couldn't find an answer. Could you rephrase your question?"

    # Append assistant's response
    assistant_message = {"role": "assistant", "content": response}
    st.session_state.messages.append(assistant_message)

    # Render assistant response immediately
    with st.chat_message("assistant"):
        st.markdown(response)

# --- Render Previous Chat History (except the latest exchange already shown above) ---
# If new input was processed, exclude last 2 (user and assistant), else show all
history_to_render = st.session_state.messages[:-2] if user_input else st.session_state.messages

for message in history_to_render:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
