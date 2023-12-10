from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

import os


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def process_pdfs(pdf_files, save_path=None):
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    if save_path:
        vectorstore.save_local(folder_path=save_path)
    return vectorstore


def add_new_pdfs(vectorstore, pdf_files, save_path=None):
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    vectorstore.add_texts(text_chunks)
    if save_path:
        vectorstore.save_local(folder_path=save_path)
    return vectorstore


def get_base_knowledge(pdf_folder_path='data', knowledge_folder_path='base_knowledge'):
    if knowledge_folder_path:
        try:
            print('loading existing knowledge base')
            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
            vectorstore = FAISS.load_local(folder_path=knowledge_folder_path, embeddings=embeddings)
        except:
            print('fetching the pdfs')
            pdf_files = []
            for filename in os.listdir(pdf_folder_path):
                if filename.endswith('.pdf'):
                    pdf_files.append(os.path.join(pdf_folder_path, filename))
            vectorstore = process_pdfs(pdf_files, save_path=knowledge_folder_path)
    else:
        print('fetching the pdfs')
        pdf_files = []
        for filename in os.listdir(pdf_folder_path):
            if filename.endswith('.pdf'):
                pdf_files.append(filename)
        vectorstore = process_pdfs(pdf_files, save_path=knowledge_folder_path)
    return vectorstore
