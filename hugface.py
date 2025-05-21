# app.py

import pandas as pd
import streamlit as st
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 📄 Load and prepare dataset
df = pd.read_csv("C:/Users/Asus/Desktop/test_csv/countries of the world.csv")
df.fillna("", inplace=True)

df["content"] = df.apply(lambda row: f"""
Country: {row['Country']}
Region: {row['Region']}
Population: {row['Population']}
Area (sq. mi.): {row['Area (sq. mi.)']}
GDP ($ per capita): {row['GDP ($ per capita)']}
Literacy (%): {row['Literacy (%)']}
Phones (per 1000): {row['Phones (per 1000)']}
Birthrate: {row['Birthrate']}
Deathrate: {row['Deathrate']}
""", axis=1)

# 🔄 Convert DataFrame to LangChain documents
loader = DataFrameLoader(df, page_content_column="content")
documents = loader.load()

# 🔹 Split text into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 🔎 Generate vector embeddings with HuggingFace
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embedding)

# 🤖 Load CPU-friendly LLM from Hugging Face
model_name = "google/flan-t5-base"

llm_pipeline = pipeline(
    "text2text-generation",
    model=AutoModelForSeq2SeqLM.from_pretrained(model_name),
    tokenizer=AutoTokenizer.from_pretrained(model_name),
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 🔗 RAG: Retrieval-Augmented QA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# 🌍 Streamlit UI
st.title("🌍 Geography RAG Chatbot (Hugging Face)")

user_input = st.text_input("Ask about countries:")

if user_input:
    try:
        response = qa_chain.run(user_input)
        st.write("🧠 Answer:", response)
    except Exception as e:
        st.error(f"❌ Error: {e}")

