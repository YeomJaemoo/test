# langchain_huggingface 설치 명령어:
# !pip install langchain-huggingface

# langchain_openai 설치 명령어:
# !pip install langchain-openai

import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
import tiktoken
import json
import base64
from google.cloud import speech
import tempfile
import os


def main():
    st.set_page_config(page_title="에너지", page_icon="🌻")
    st.image('knowhow.png')
    st.title("_:red[에너지 학습 도움이]_ 🏫")
    st.header("😶주의! 이 챗봇은 참고용으로 사용하세요!", divider='rainbow')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "voice_input" not in st.session_state:
        st.session_state.voice_input = ""

    with st.sidebar:
        folder_path = Path()
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        model_name = 'gpt-4o-mini'

        st.text("아래의 'Process'를 누르고\n아래 채팅창이 활성화 될 때까지\n잠시 기다리세요!😊")
        process = st.button("Process", key="process_button")

        if process:
            files_text = get_text_from_folder(folder_path)
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key, model_name)
            st.session_state.processComplete = True

        if st.button("말하기", key="speak_button"):
            with st.spinner("음성을 인식하는 중..."):
                try:
                    client = speech.SpeechClient()
                    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
                        temp_audio_file_name = temp_audio_file.name
                        st.audio(temp_audio_file_name, format='audio/wav')

                        with open(temp_audio_file_name, "rb") as audio_file:
                            audio_content = audio_file.read()

                    audio = speech.RecognitionAudio(content=audio_content)
                    config = speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=16000,
                        language_code="ko-KR",
                    )
                    response = client.recognize(config=config, audio=audio)

                    if response.results:
                        st.session_state.voice_input = response.results[0].alternatives[0].transcript
                        st.success(f"인식된 음성: {st.session_state.voice_input}")
                    else:
                        st.warning("음성을 인식하지 못했습니다. 다시 시도하세요!")
                except Exception as e:
                    st.warning(f"음성을 인식하는 동안 오류가 발생했습니다: {e}")

        save_button = st.button("대화 저장", key="save_button")
        if save_button:
            if st.session_state.chat_history:
                save_conversation_as_txt(st.session_state.chat_history)
            else:
                st.warning("질문을 입력받고 응답을 확인하세요!")

        clear_button = st.button("대화 내용 삭제", key="clear_button")
        if clear_button:
            st.session_state.chat_history = []
            st.session_state.messages = [{"role": "assistant", "content": "에너지 학습에 대해 물어보세요!😊"}]
            st.experimental_rerun()

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "에너지 학습에 대해 물어보세요!😊"}]

    if st.session_state.voice_input:
        query = st.session_state.voice_input
        st.session_state.voice_input = ""
        st.experimental_rerun()
    else:
        query = st.chat_input("질문을 입력해주세요.")

    if query:
        st.session_state.messages.insert(0, {"role": "user", "content": query})
        chain = st.session_state.conversation
        with st.spinner("생각 중..."):
            result = chain({"question": query})
            with get_openai_callback() as cb:
                st.session_state.chat_history = result['chat_history']
            response = result['answer']
            source_documents = result['source_documents']

        st.session_state.messages.insert(1, {"role": "assistant", "content": response})

    for message_pair in (list(zip(st.session_state.messages[::2], st.session_state.messages[1::2]))):
        with st.chat_message(message_pair[0]["role"]):
            st.markdown(message_pair[0]["content"])
        with st.chat_message(message_pair[1]["role"]):
            st.markdown(message_pair[1]["content"])
        with st.expander("참고 문서 확인"):
            for doc in source_documents:
                st.markdown(doc.metadata['source'], help=doc.page_content)


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text_from_folder(folder_path):
    doc_list = []
    folder = Path(folder_path)
    files = folder.iterdir()

    for file in files:
        if file.is_file():
            if file.suffix == '.pdf':
                loader = PyPDFLoader(str(file))
                documents = loader.load_and_split()
            elif file.suffix == '.docx':
                loader = Docx2txtLoader(str(file))
                documents = loader.load_and_split()
            elif file.suffix == '.pptx':
                loader = UnstructuredPowerPointLoader(str(file))
                documents = loader.load_and_split()
            else:
                documents = []
            doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_conversation_chain(vectorstore, openai_api_key, model_name):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type='mmr'),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer', input_key='question'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain


def save_conversation_as_txt(chat_history):
    conversation = ""
    for message in chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content
        conversation += f"역할: {role}\n내용: {content}\n\n"

    b64 = base64.b64encode(conversation.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="대화.txt">대화 다운로드</a>'
    st.markdown(href, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
