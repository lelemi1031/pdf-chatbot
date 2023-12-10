import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub, Cohere
from langchain import PromptTemplate

from src.utils import (
    get_base_knowledge, process_pdfs,
    get_pdf_text, get_text_chunks, add_new_pdfs
)


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(
    #     repo_id="google/flan-t5-base",
    #     model_kwargs={
    #         "temperature": 0.5,
    #         "max_length": 2048,
    #     }
    # )
    prompt = """
    You are an expert data scientist with an expertise in building deep learning models.
    Answer the question in a few lines.
    """

    llm = Cohere(temperature=0.8, max_tokens=2048)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    print("memory loaded")

    prompt_template = """
    You are an data expert with an expertise in CREA data. 
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 

    {context} 

    Question: {question} 
    Helpful Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    conversation_chain = ConversationalRetrievalChain.from_llm(
        combine_docs_chain_kwargs={"prompt": prompt},
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        # max_tokens_limit=1024,
    )

    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="QA ChatBot",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        vectorstore = get_base_knowledge()
        print("vectorstore loaded")
        st.session_state.conversation = get_conversation_chain(
            vectorstore)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("QA ChatBot:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Add' or 'Replace'. \n"
            "\nNOTE: If you click 'Replace', you will replace the existing knowledge base.", accept_multiple_files=True)
        if st.button("Add"):
            with st.spinner("Processing"):
                vectorstore = add_new_pdfs(vectorstore=vectorstore, pdf_files=pdf_docs, save_path='base_knowledge')
                print('new vectorstore added')

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
        if st.button("Replace"):
            with st.spinner("Processing"):
                vectorstore = process_pdfs(pdf_docs, save_path='base_knowledge')
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
