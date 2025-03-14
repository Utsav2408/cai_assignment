import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Chat Interface with RAG Options", page_icon="💬", layout="wide")
st.title("💬 RAG-Powered Chat Interface")

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

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Basic RAG Placeholder Function ---
def basic_rag(query):
    # Placeholder for real basic RAG logic, simple keyword match or naive search
    return f"[Basic RAG] Here is a simple answer to: '{query}'"

# --- Advanced RAG Placeholder Function ---
def advanced_rag(query):
    # Placeholder for advanced RAG (e.g., using embeddings, vector DB, advanced ranking)
    return f"[Advanced RAG] This is a detailed response to: '{query}', leveraging deeper context and semantic search."

# --- Handle User Input and Generate Response ---
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user input to session
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response based on selected RAG mode
    if rag_mode == "Basic RAG":
        response = basic_rag(user_input)
    else:
        response = advanced_rag(user_input)
    
    # Add bot response to session
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(response)
