import streamlit as sl
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

OpenAI_API_KEY="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"
sl.header("hitesh's bot")

with sl.sidebar:
    sl.title("MY PDFs")
    file = sl.file_uploader("Upload notes pdf and can start asking questions", type="pdf")

if file is not None:
    my_pdf=PdfReader(file)
    text=""
    for page in my_pdf.pages:
        text +=page.extract_text()
        #sl.write(text)

    text_splitter=RecursiveCharacterTextSplitter(separators=["\n"],chunk_size=200,chunk_overlap=50)
    chunks=text_splitter.split_text(text)
    #sl.write(chunks)

    embeddings=OpenAIEmbeddings(ap1_key=OpenAI_API_KEY)

    vector_store=FAISS.from_texts(chunks,embeddings)

user_query=sl.text_input("TYPE YOUR QUERY HERE")

if user_query:
    matching_chunks=vector_store.similarity_search(user_query)

    llm=ChatOpenAI(
        api_key=OpenAI_API_KEY,
        max_token=300,
        temperature=0,
        model="gpt-3.5-turbo"
    )

    chain=load_qa_chain(llm,chain_type="stuff")
    output=chain.run(question=user_query,input_documents=matching_chunks)
    st.write(output)
