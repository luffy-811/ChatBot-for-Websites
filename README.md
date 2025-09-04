# 🤖 Website Q&A Bot

A powerful AI-powered chatbot that can analyze any website and answer questions about its content using Google's Gemini AI and advanced text processing techniques.

## ✨ Features

- **🌐 Website Analysis**: Process any website URL and extract meaningful content
- **🔍 Intelligent Q&A**: Ask questions about the website content and get detailed answers
- **⚡ JavaScript Support**: Optional JavaScript rendering for dynamic websites
- **🧠 AI-Powered**: Uses Google's Gemini 1.5 Flash model for accurate responses
- **🔄 Retry Logic**: Robust error handling with automatic retries for network issues
- **📱 User-Friendly Interface**: Clean Streamlit web interface
- **🌍 Network Monitoring**: Real-time internet connection status

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Internet connection

### Installation

1. **Clone or download the project**
   ```bash
   git clone <your-repo-url>
   cd "chat bot for websites"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers** (for JavaScript rendering)
   ```bash
   playwright install chromium
   ```

4. **Set up your API key**
   
   Create a `.env` file in the project directory:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
   
   Get your API key from: [Google AI Studio](https://makersuite.google.com/app/apikey)

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** and go to: `http://localhost:8501`

## 📖 How to Use

### Step 1: Process a Website
1. Enter a website URL in the sidebar (e.g., `https://www.example.com`)
2. Optionally enable JavaScript rendering for dynamic websites
3. Click "Process Website" and wait for completion

### Step 2: Ask Questions
1. Once processing is complete, type your question in the main input field
2. Press Enter to get AI-generated answers based on the website content

### Example Questions
- "What is this website about?"
- "What are the main features mentioned?"
- "Can you summarize the key points?"
- "What services does this company offer?"

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Web Scraping**: Requests + BeautifulSoup (static) / Playwright (JavaScript)
- **Text Processing**: LangChain RecursiveCharacterTextSplitter
- **Embeddings**: Google Text Embedding 004 model
- **Vector Store**: FAISS for similarity search
- **AI Model**: Google Gemini 1.5 Flash for question answering

### Supported Website Types
- ✅ Static HTML websites
- ✅ Dynamic JavaScript websites (with JS rendering enabled)
- ✅ Most public websites
- ❌ Password-protected sites
- ❌ Sites that block automated access

## 🔧 Configuration

### Environment Variables
```env
GEMINI_API_KEY=your_gemini_api_key
```

### Model Configuration
The app uses these Google AI models:
- **Text Generation**: `gemini-1.5-flash`
- **Text Embeddings**: `models/text-embedding-004`

## 🚨 Troubleshooting

### Common Issues

**1. "GEMINI_API_KEY not found"**
- Ensure your `.env` file is in the project root directory
- Check that your API key is correctly formatted
- Verify your API key is active at [Google AI Studio](https://makersuite.google.com/)

**2. "No internet connection detected"**
- Check your internet connection
- Verify firewall settings aren't blocking the application
- Try using a VPN if you're in a restricted network

**3. "503 Service Unavailable" or Timeout Errors**
- Google's API servers might be experiencing high traffic
- Wait 5-10 minutes and try again
- The app will automatically retry failed requests

**4. "Failed to fetch website content"**
- Check if the website URL is correct and accessible
- Try enabling JavaScript rendering for dynamic sites
- Some websites block automated access

**5. Website Processing Fails**
- Ensure the website is publicly accessible
- Try a simpler website first (e.g., `https://example.com`)
- Check if the website requires JavaScript (enable JS rendering)

### Network Issues
The app includes automatic retry logic for:
- Connection timeouts
- API server unavailability
- Network connectivity issues

## 📋 Requirements

### System Requirements
- Windows 10/11, macOS, or Linux
- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- Internet connection

### Python Dependencies
```
streamlit>=1.49.1
python-dotenv>=1.1.1
requests>=2.32.5
beautifulsoup4>=4.13.5
playwright>=1.55.0
google-generativeai>=0.8.5
faiss-cpu>=1.12.0
langchain-google-genai>=2.0.10
langchain>=0.3.27
langchain-community>=0.3.29
tenacity>=9.1.2
```

## 🔒 Privacy & Security

- Your API key is stored locally in the `.env` file
- Website content is processed locally and temporarily
- No data is permanently stored or shared
- Vector embeddings are saved locally in the `faiss_index` folder

## 📝 Limitations

- **Content Size**: Large websites may take longer to process
- **API Limits**: Subject to Google Gemini API rate limits
- **Language**: Works best with English content
- **Accuracy**: Answers are based on processed website content only

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google AI for the Gemini API
- Streamlit for the web framework
- LangChain for AI orchestration
- FAISS for vector similarity search
- Playwright for web scraping

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify your API key configuration
4. Check the terminal output for detailed error messages

---

**Made with ❤️ for seamless website analysis and Q&A**