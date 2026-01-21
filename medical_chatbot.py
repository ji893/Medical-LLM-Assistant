import os
import streamlit as st
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import time

# 🔑 환경변수 로드
load_dotenv()

# 🔒 PDF 폴더 해시 생성
def get_pdf_folder_hash(pdf_folder="./data/") -> str:
    md5 = hashlib.md5()
    pdf_files = sorted([f for f in os.listdir(pdf_folder) if f.endswith(".pdf")])
    for fname in pdf_files:
        with open(os.path.join(pdf_folder, fname), "rb") as f:
            md5.update(f.read())
    return md5.hexdigest()

# 🔄 PDF 로드 및 분할
@st.cache_resource
def load_and_split_all_pdfs(pdf_folder, pdf_hash):
    all_pages = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        try:
            file_path = os.path.join(pdf_folder, pdf_file)
            pages = None
            
            # PyMuPDFLoader를 먼저 시도 (가장 안정적, 폰트 문제 처리 우수)
            try:
                loader = PyMuPDFLoader(file_path)
                pages = loader.load()
            except:
                # PyMuPDFLoader 실패 시 PDFPlumberLoader 시도
                try:
                    loader = PDFPlumberLoader(file_path)
                    pages = loader.load()
                except:
                    # PDFPlumberLoader 실패 시 PyPDFLoader 시도
                    try:
                        loader = PyPDFLoader(file_path)
                        pages = loader.load()
                    except:
                        raise Exception("모든 PDF 로더 실패")
            
            if pages:
                all_pages.extend(pages)
        except Exception as e:
            st.warning(f"⚠️ PDF 파일 로드 실패: {pdf_file} - {str(e)}")
            continue

    if not all_pages:
        st.error("❌ 로드된 PDF 페이지가 없습니다. PDF 파일을 확인해주세요.")
        return []
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    split_docs = splitter.split_documents(all_pages)
    return split_docs

@st.cache_resource
def load_and_split_all_pdfs_cached(pdf_folder="./data/"):
    pdf_hash = get_pdf_folder_hash(pdf_folder)
    return load_and_split_all_pdfs(pdf_folder, pdf_hash), pdf_hash

# 💾 벡터 저장소 생성
@st.cache_resource
def get_vectorstore(_split_docs, pdf_hash):
    persist_directory = f"./FAISS_dB"
    print(f"시작 시간: {time.time()}")

    if os.path.exists(os.path.join(persist_directory, "index.faiss")):
        print("로컬 FAISS 인덱스 로드 중...")
        start = time.time()
        vectorstore = FAISS.load_local(
            persist_directory,
            OpenAIEmbeddings(model="text-embedding-3-small"),
            allow_dangerous_deserialization=True
        )
        print(f"로드 완료, 걸린 시간: {time.time() - start:.2f}초")
    else:
        print("새 FAISS 인덱스 생성 중...")
        start = time.time()
        vectorstore = FAISS.from_documents(
            _split_docs,
            OpenAIEmbeddings(model="text-embedding-3-small")
        )
        vectorstore.save_local(persist_directory)
        print(f"생성 완료, 걸린 시간: {time.time() - start:.2f}초")

    print(f"종료 시간: {time.time()}")
    return vectorstore

# 📄 문서 포맷
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 💬 대화 기록 포맷
def format_chat_history(chat_history):
    return "\n".join(
        f"User: {msg[1]}" if msg[0] == "human" else f"Assistant: {msg[1]}"
        for msg in chat_history
    )

# 🔁 체인 구성
@st.cache_resource
def chaining(pdf_folder="./data/"):
    split_docs, pdf_hash = load_and_split_all_pdfs_cached(pdf_folder)
    vectorstore = get_vectorstore(split_docs, pdf_hash=pdf_hash)
    retriever = vectorstore.as_retriever()

    qa_system_prompt = """
당신은 환자의 증상을 듣고 가능한 병명을 유추하거나, 정보가 부족한 경우 추가 질문을 통해 더 정확한 진단에 가까워지도록 돕는 의학 전문가입니다. \
초기 설명만으로 병명을 단정하기 어려운 경우, 추가 질문을 통해 정보를 수집하되, 한 번에 1~3개의 질문을 묶어 진행하십시오. \
명확한 진단이 어려워도 3번째 추가 질문때 유추 가능한 병명을 무조건 제시하고,  후속 질문을 이어가십시오. \
항상 조심스럽고 정중한 어조를 유지하되, 반복적인 인사나 불필요한 표현은 생략하고, 핵심적인 의료 정보 전달에 집중하십시오. \
사용자에게 해당 진단은 참고용이라는 점을 인식시키고, 불필요한 불안감을 주지 않도록 따뜻하게 안내해 주십시오. \
단, 호흡곤란, 의식 저하, 심한 통증 등의 **응급 증상**이 나타날 경우, 지체하지 말고 **즉시 119에 신고하거나 가까운 응급실을 방문하도록** 안내하십시오. \
답변은 명확하게 구성하며, 필요한 경우 이해를 돕기 위해 이모지를 함께 사용하십시오. \
모든 응답은 한국어로 작성해 주십시오.

{context}
"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("human", "다음 정보를 참고해 주세요:\n\n{context}\n\n{chat_history}\n\n{input}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini")
    retriever_chain = RunnableLambda(lambda x: x["input"]) | retriever | format_docs

    rag_chain = (
        {
            "context": retriever_chain,
            "input": RunnableLambda(lambda x: x["input"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"])
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# 🌐 Streamlit UI
st.header("의학 Q&A 챗봇 🩺📄")
rag_chain = chaining()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "어떤 증상이 있으신가요? 😊"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt_message := st.chat_input("질문을 입력해주세요. 성별과 나이를 입력하면 큰 도움이 됩니다 :)"):
    st.chat_message("user").write(prompt_message)
    st.session_state.messages.append({"role": "user", "content": prompt_message})

    # 대화 히스토리 구성
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(("human", msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(("ai", msg["content"]))
    chat_history_str = format_chat_history(chat_history)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 🔍 검색된 문서 중 가장 낮은 score 하나만 사용
            split_docs, pdf_hash = load_and_split_all_pdfs_cached()
            db = get_vectorstore(split_docs, pdf_hash)
            docs_with_scores = db.similarity_search_with_score(prompt_message, k=3)

            min_score = min(score for _, score in docs_with_scores)

            # 🧠 GPT 응답 생성
            response = rag_chain.invoke({
                "input": prompt_message,
                "chat_history": chat_history_str
            })
            st.session_state.messages.append({"role": "assistant", "content": response})

            st.write(response)
            st.markdown(f"<div style='color: gray;'>score: {min_score:.4f}</div>", unsafe_allow_html=True)

            # 📂 참고 문서 확인 영역
            with st.expander("📂 참고 문서 확인"):
                for doc, score in docs_with_scores:
                    filename = os.path.basename(doc.metadata["source"])
                    st.markdown(f"📄 **{filename}** &nbsp;&nbsp;&nbsp; *(score: {score:.4f})*", help=doc.page_content)


