import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urlparse
import time
import socket
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

def check_internet_connection():
    """Check if there's an active internet connection."""
    try:
        # Try to connect to Google's DNS server
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,))
)
def create_embeddings_with_retry(api_key):
    """Create embeddings with retry logic."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
        task_type="retrieval_document"
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,))
)
def create_vector_store_with_retry(text_chunks, embeddings):
    """Create vector store with retry logic."""
    return FAISS.from_texts(text_chunks, embedding=embeddings)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,))
)
def search_similar_docs_with_retry(vector_store, query, k=4):
    """Search for similar documents with retry logic."""
    return vector_store.similarity_search(query, k=k)

def validate_url(url):
    """Validates and normalizes the URL."""
    try:
        # Add https:// if no scheme is provided
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Parse the URL to validate it
        parsed = urlparse(url)
        if not parsed.netloc:
            return None, "Invalid URL format"
        
        return url, None
    except Exception as e:
        return None, f"URL validation error: {str(e)}"

def get_website_content(url, use_playwright=False):
    """Fetches and parses website content, either statically or with JS rendering."""
    # Validate URL first
    validated_url, error = validate_url(url)
    if error:
        st.error(error)
        return ""
    
    if use_playwright:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                # Set user agent to avoid blocking
                page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                })
                
                try:
                    st.info(f"Loading website with JavaScript rendering: {validated_url}")
                    page.goto(validated_url, timeout=30000, wait_until="domcontentloaded")
                    
                    # Wait for any dynamic content to load
                    page.wait_for_timeout(3000)
                    
                    content = page.content()
                    st.success("Successfully loaded website content with JavaScript rendering")
                    return content
                    
                except Exception as e:
                    st.error(f"Error loading page with Playwright: {str(e)}")
                    return ""
                finally:
                    browser.close()
                    
        except Exception as e:
            st.error(f"Error initializing Playwright: {str(e)}")
            return ""
    else:
        try:
            st.info(f"Loading website: {validated_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(validated_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            st.success("Successfully loaded website content")
            return response.text
            
        except requests.exceptions.Timeout:
            st.error("Request timed out. The website might be slow to respond.")
            return ""
        except requests.exceptions.ConnectionError:
            st.error("Connection error. Please check your internet connection and the URL.")
            return ""
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error {e.response.status_code}: {e.response.reason}")
            return ""
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching website: {str(e)}")
            return ""

def get_text_chunks(html_content):
    """Extracts text from HTML and splits it into chunks."""
    try:
        st.info("Extracting text from website content...")
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text with better formatting
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up the text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)
        
        if not text or len(text) < 100:
            st.warning("Very little text content found on the website.")
            return []
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        
        st.success(f"Successfully extracted {len(chunks)} text chunks from website")
        return chunks
        
    except Exception as e:
        st.error(f"Error extracting text from HTML: {str(e)}")
        return []

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
        return None
        
    if not text_chunks:
        st.error("No text chunks provided for vector store creation.")
        return None
    
    # Check internet connection first
    if not check_internet_connection():
        st.error("❌ No internet connection detected. Please check your network connection.")
        return None
    
    try:
        st.info("Creating vector store from text chunks...")
        
        # Initialize embeddings with retry logic
        with st.spinner("Initializing embeddings..."):
            embeddings = create_embeddings_with_retry(api_key)
        
        # Create vector store with retry logic
        with st.spinner("Creating vector embeddings..."):
            vector_store = create_vector_store_with_retry(text_chunks, embeddings)
        
        # Save the vector store
        with st.spinner("Saving vector store..."):
            vector_store.save_local("faiss_index")
        
        st.success("Vector store created and saved successfully!")
        return vector_store
        
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "timeout" in error_msg.lower():
            st.error("❌ Google API servers are temporarily unavailable or your connection is slow.")
            st.info("💡 **Solutions to try:**")
            st.info("1. Check your internet connection")
            st.info("2. Wait a few minutes and try again")
            st.info("3. Try processing a smaller website")
            st.info("4. Verify your API key is correct")
        elif "404" in error_msg:
            st.error("❌ API model not found. Please check if your API key has access to the required models.")
        else:
            st.error(f"❌ Error creating vector store: {error_msg}")
        return None

def get_conversational_chain():
    """Creates a question-answering chain with a custom prompt."""
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please configure it.")
        return None
        
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
        If the answer is not in the provided context, just say "answer is not available in the context".
        Don't provide the wrong answer.

        Context:\n {context}\n
        Question: \n{question}\n

        Answer:
        """
        
        model = GoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.3,
            google_api_key=api_key
        )
        
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
        
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

def user_input(user_question):
    """Handles user questions, retrieves relevant documents, and generates a response."""
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please configure it.")
        return

    if not user_question.strip():
        st.warning("Please enter a question.")
        return

    # Check internet connection first
    if not check_internet_connection():
        st.error("❌ No internet connection detected. Please check your network connection.")
        return

    try:
        # Check if FAISS index exists
        if not os.path.exists("faiss_index"):
            st.error("No website has been processed yet. Please process a website first.")
            return
        
        st.info("Searching for relevant information...")
        
        # Initialize embeddings with retry logic
        with st.spinner("Loading embeddings..."):
            embeddings = create_embeddings_with_retry(api_key)
        
        # Load the vector store
        with st.spinner("Loading vector store..."):
            new_db = FAISS.load_local(
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        
        # Search for relevant documents with retry logic
        with st.spinner("Finding relevant content..."):
            docs = search_similar_docs_with_retry(new_db, user_question, k=4)
        
        if not docs:
            st.warning("No relevant information found in the processed website.")
            return
        
        # Get the conversational chain
        chain = get_conversational_chain()
        if not chain:
            return
        
        st.info("Generating answer...")
        
        # Generate response
        with st.spinner("Thinking..."):
            response = chain(
                {"input_documents": docs, "question": user_question}, 
                return_only_outputs=True
            )
        
        st.success("Answer generated successfully!")
        st.write("**Reply:**", response["output_text"])
        
    except FileNotFoundError:
        st.error("Vector store not found. Please process a website first.")
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "timeout" in error_msg.lower():
            st.error("❌ Google API servers are temporarily unavailable or your connection is slow.")
            st.info("💡 **Solutions to try:**")
            st.info("1. Check your internet connection")
            st.info("2. Wait a few minutes and try again")
            st.info("3. Ask a simpler question")
            st.info("4. Try processing the website again")
        elif "embedding" in error_msg.lower():
            st.error("❌ Error with text embeddings. This might be a temporary API issue.")
            st.info("💡 Try asking your question again in a few moments.")
        else:
            st.error(f"❌ An error occurred: {error_msg}")
            st.info("Please try processing the website again or check your API key configuration.")


def set_event_loop():
    """Sets up the event loop for asyncio operations."""
    try:
        import asyncio
        # Try to get the current event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Set Windows-specific event loop policy if on Windows
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
    except Exception as e:
        st.error(f"Failed to set event loop policy: {str(e)}")

def main():
    # Set up event loop first
    set_event_loop()
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Website Q&A Bot", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.header("Website Q&A Bot 🤖")
    
    # Check for API key
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found!")
        st.info("Please create a `.env` file in your project directory and add your Gemini API key:")
        st.code("GEMINI_API_KEY=your_api_key_here")
        st.stop()

    with st.sidebar:
        st.title("Menu")
        st.markdown("---")
        
        website_url = st.text_input(
            "Enter Website URL", 
            placeholder="e.g., https://example.com or example.com",
            help="Enter a valid website URL to analyze"
        )
        
        use_js_rendering = st.checkbox(
            "Enable JavaScript Rendering (Slower)", 
            help="Use this for websites that require JavaScript to load content"
        )
        
        if st.button("Process Website", type="primary"):
            if website_url:
                if website_url.strip():
                    with st.spinner("Processing website..."):
                        try:
                            # Step 1: Fetch website content
                            html_content = get_website_content(website_url.strip(), use_js_rendering)
                            
                            if html_content:
                                # Step 2: Extract text chunks
                                text_chunks = get_text_chunks(html_content)
                                
                                if text_chunks:
                                    # Step 3: Create vector store
                                    vector_store = get_vector_store(text_chunks)
                                    
                                    if vector_store:
                                        st.success("✅ Website processed successfully!")
                                        st.balloons()
                                    else:
                                        st.error("❌ Failed to create vector store.")
                                else:
                                    st.error("❌ Could not extract meaningful text from the website.")
                            else:
                                st.error("❌ Failed to fetch website content.")
                                
                        except Exception as e:
                            st.error(f"❌ An unexpected error occurred: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a valid website URL.")
            else:
                st.warning("⚠️ Please enter a website URL.")
        
        st.markdown("---")
        st.markdown("### Instructions:")
        st.markdown("1. Enter a website URL")
        st.markdown("2. Click 'Process Website'")
        st.markdown("3. Wait for processing to complete")
        st.markdown("4. Ask questions about the website")
        
        st.markdown("---")
        st.markdown("### 🔧 Network Status:")
        if check_internet_connection():
            st.success("✅ Internet connection active")
        else:
            st.error("❌ No internet connection detected")
            st.info("Please check your network connection before processing websites.")

    # Main content area
    st.title("Ask a Question")
    
    # Check if website has been processed
    if os.path.exists("faiss_index"):
        st.success("✅ Website is ready for questions!")
    else:
        st.info("ℹ️ Please process a website first before asking questions.")
    
    user_question = st.text_input(
        "What do you want to know?",
        placeholder="Ask anything about the processed website...",
        help="Type your question about the website content"
    )
    
    if user_question:
        with st.spinner("Thinking..."):
            user_input(user_question)

if __name__ == "__main__":
    main()
