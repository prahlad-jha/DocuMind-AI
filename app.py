import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Disable LangSmith tracing to avoid serialization issues
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

# Configure page first
st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

# Import templates
from htmlTemplates import css, bot_template, user_template

# Apply CSS
st.markdown(css, unsafe_allow_html=True)


class GeminiLLM:
    """Gemini LLM wrapper for text generation"""
    
    def __init__(self, model_name: str = None, temperature: float = 0.3):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Provide a helpful message but don't fail immediately
            print("⚠️  Warning: GEMINI_API_KEY not set in environment variables")
        else:
            genai.configure(api_key=api_key)
        
        self.temperature = temperature
        
        # If no model specified, find the first available text-generation model
        if model_name is None:
            try:
                # List all available models
                models = genai.list_models()
                for model in models:
                    # Look for models that support generateContent
                    if "generateContent" in model.supported_generation_methods:
                        model_name = model.name
                        break
            except Exception as e:
                print(f"Error listing models: {e}")
                model_name = "gemini-pro"  # Fallback default
        
        self.model_name = model_name or "gemini-pro"
    
    def __call__(self, inputs: str) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                inputs,
                generation_config=genai.types.GenerationConfig(temperature=self.temperature)
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            print(f"Error with model {self.model_name}: {error_str}")
            
            # Try to find and use an alternative model
            try:
                models = genai.list_models()
                for model in models:
                    if "generateContent" in model.supported_generation_methods:
                        alt_model = genai.GenerativeModel(model.name)
                        response = alt_model.generate_content(
                            inputs,
                            generation_config=genai.types.GenerationConfig(temperature=self.temperature)
                        )
                        return response.text
            except Exception as fallback_error:
                return f"Error: {error_str}"
            
            return f"Error: {error_str}"
    
    def generate(self, prompt: str) -> str:
        return self(prompt)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    from langchain_text_splitters import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        # Try with the lightweight model first
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e1:
            # Fallback to default model
            st.warning("Using default embeddings model...")
            embeddings = HuggingFaceEmbeddings()
        
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except ImportError as e:
        st.error(f"Import Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def get_conversation_chain(vectorstore):
    llm = GeminiLLM()
    # Return the vectorstore directly so we can use similarity_search
    return {
        "llm": llm,
        "vectorstore": vectorstore
    }


def handle_userinput(user_question):
    # Get context from the vectorstore
    conversation_obj = st.session_state.conversation
    llm = conversation_obj["llm"]
    vectorstore = conversation_obj["vectorstore"]
    
    try:
        # Retrieve relevant documents using similarity search
        docs = vectorstore.similarity_search(user_question, k=3)
        context = "\n".join([doc.page_content for doc in docs if doc])
        
        # Create prompt with context
        prompt = f"""Based on the following context, answer the question clearly and concisely.

Context:
{context}

Question: {user_question}

Answer:"""
        
        # Generate response
        response_text = llm.generate(prompt)
        
        # Store in chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "bot", "content": response_text})
        
        # Display all messages
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(user_template.replace(
                    "{{MSG}}", message["content"]), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message["content"]), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")


def main():
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display header
    st.header("Chat with multiple PDFs :books:")
    
    # Main content area
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation is not None:
        handle_userinput(user_question)

    # Sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF")
            else:
                with st.spinner("Processing PDFs..."):
                    try:
                        # Extract text from PDFs
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text:
                            st.error("Could not extract text from PDFs")
                        else:
                            # Process text
                            st.info("Splitting text into chunks...")
                            text_chunks = get_text_chunks(raw_text)
                            
                            st.info("Creating vector store (this may take a minute)...")
                            vectorstore = get_vectorstore(text_chunks)
                            
                            if vectorstore is None:
                                st.error("Failed to create vector store")
                            else:
                                st.session_state.conversation = get_conversation_chain(vectorstore)
                                st.success("✅ PDFs processed successfully! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


if __name__ == '__main__':
    main()
