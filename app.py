# Streamlit app UI 
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# [data-testid="stMain"]{
# background-color: #e5e5f7;
# opacity: 0.8;
# background-image:  radial-gradient(#444cf7 0.5px, transparent 0.5px), radial-gradient(#444cf7 0.5px, #e5e5f7 0.5px);
# background-size: 20px 20px;
# background-position: 0 0,10px 10px;
# }

# background-color: #e5e5f7;
# opacity: 0.8;
# background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #e5e5f7 10px ), repeating-linear-gradient( #444cf755, #444cf7 );

# page_bg_img = """
# <style>
# [data-testid="stMain"]{
# background-color: #e5e5f7;
# opacity: 0.8;
# background-image:  linear-gradient(#444cf7 0.7000000000000001px, transparent 0.7000000000000001px), linear-gradient(to right, #444cf7 0.7000000000000001px, #e5e5f7 0.7000000000000001px);
# background-size: 14px 14px;
# </style>
# """

page_bg_img = """
<style>
[data-testid="stMain"]{
background-color: #ffffff;
opacity: 1;
background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #ffffff 40px ), repeating-linear-gradient( #adb1f655, #adb1f6 );
}
</style>
"""

an_bg_img = """
<style>
[data-testid="stWidgetLabel"]{
color: #000000;
font-size:55px !important;
font-weight:600;
opacity: 1.5;
}
</style>
"""

nav_bg_img = """
<style>
[data-testid="stToolbar"]{
color: #000000;
opacity: 1.5;
}
</style>
"""

# Streamlit config
st.set_page_config(page_title="Tequila Trends Agent", page_icon="🍹", layout="centered")
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(an_bg_img,unsafe_allow_html=True)
st.markdown(nav_bg_img,unsafe_allow_html=True)
 
# Title
st.markdown(
    """
    <h1 style='text-align: center; color: blue;'>Tequila Trends Agent</h1>
    <p style='text-align: center; font-size: 18px;color:black'>Ask questions about the tequila market, and get instant answers powered by RAG and Groq's Llama-3!</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Step 1: Cached Function to Create or Load VectorStore
@st.cache_resource(show_spinner="Indexing tequila market sources…")
def load_vectorstore():
    persist_dir = "./tequila_db"

    # If persisted directory exists, just load it
    if os.path.exists(persist_dir):
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)

    # Otherwise load from scratch
    urls = [
        "https://www.grandviewresearch.com/industry-analysis/tequila-market-report",
        "https://www.statista.com/outlook/cmo/alcoholic-drinks/spirits/tequila/worldwide",
        "https://www.ohbev.com/blog/tequila-market-2025-forecasts-and-trends",
        'https://www.fortunebusinessinsights.com/tequila-market-104172',
        'https://www.mordorintelligence.com/industry-reports/tequila-market',
        'https://www.expertmarketresearch.com/reports/tequila-market',
        'https://www.cognitivemarketresearch.com/tequila-market-report',
        'https://www.thebusinessresearchcompany.com/report/tequila-global-market-report',
        'https://www.credenceresearch.com/report/tequila-market',
        'https://www.euromonitor.com/from-shots-to-sips-understanding-tequila/report',
        'https://www.marketresearchfuture.com/reports/tequila-market-11972',
        'https://www.imarcgroup.com/india-tequila-market',
        'https://www.linkedin.com/pulse/2025-defining-year-tequila-industry-ricardo-cortizo-oog2e/',
        'https://www.linkedin.com/pulse/tequila-market-size-share-trends-growth-analysis-btrvf/',
        'https://www.imarcgroup.com/prefeasibility-report-tequila-manufacturing-plant',
        'https://www.marknteladvisors.com/research-library/tequila-market.html',
        'https://www.forbes.com/sites/rachelking/2025/01/05/the-tequila-boom-how-premium-and-craft-spirits-are-changing-the-industry/',
        'https://bartenderspiritsawards.com/en/blog/insights-1/nine-spirits-trends-you-can-expect-in-the-2025-u-s-drinks-market-874.htm'
    ]

    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding=embedding, persist_directory=persist_dir)
    vectorstore.persist()
    return vectorstore

# Load Vector DB once (outside the button click)
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# UI: Input box
st.markdown(
    """
    <style>
    label[data-testid="stMarkdownContainer"] > div {
        color: #000000;  
        font-weight: 600;
        font-size: 16px;
    }
    </style>
"""
    ,
    unsafe_allow_html=True

)

query = st.text_input("Enter your question below:", placeholder="e.g. What are the trends in the tequila market?")


# Handle button click
if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        # Load LLM
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192",
            temperature=0.4
        )

        # QA Chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

        # Run Query
        response = qa_chain.run(query)

        # Display Answer
        st.markdown(
            f"""
            <div style="background-color: #000000; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h4 style="color:#8B0000;">📌 Answer:</h4>
                <p style="font-size: 16px;">{response}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


