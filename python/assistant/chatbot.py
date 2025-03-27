import streamlit as st
import irisnative
import sentence_transformers
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
import numpy as np
#from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

#llm = ChatOpenAI()
llm = ChatMistralAI(model="mistral-large-latest")

connection_string = "localhost:51774/LLMRAG"
username = "superuser"
password = "SYS"

connectionIRIS = irisnative.createConnection(connection_string, username, password)
cursorIRIS = connectionIRIS.cursor()

if not os.path.isdir("C:\Code\workshop-meetup\python\model"):
    model = sentence_transformers.SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')            
    model.save("C:\Code\workshop-meetup\python\model")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap  = 50,
)
path = "C:\Code\workshop-meetup\python\context"
loader = PyPDFDirectoryLoader(path)
docs_before_split = loader.load()
docs_after_split = text_splitter.split_documents(docs_before_split)
model = sentence_transformers.SentenceTransformer("C:\Code\workshop-meetup\python\model")
for doc in docs_after_split:
    embeddings = model.encode(doc.page_content, normalize_embeddings=True)
    array = np.array(embeddings)
    formatted_array = np.vectorize('{:.12f}'.format)(array)
    parameters = []
    parameters.append(doc.metadata['source'])
    parameters.append(str(doc.page_content))
    parameters.append(str(','.join(formatted_array)))
    cursorIRIS.execute("INSERT INTO LLMRAG.DOCUMENTCHUNK (Document, Phrase, VectorizedPhrase) VALUES (?, ?, TO_VECTOR(?,DECIMAL))", parameters)
connectionIRIS.commit()

# app config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Chatbot")


def get_response(user_query, chat_history):

    question = model.encode(user_query, normalize_embeddings=True)
    array = np.array(question)
    formatted_array = np.vectorize('{:.12f}'.format)(array)
    parameterQuery = []
    parameterQuery.append(str(','.join(formatted_array)))
    cursorIRIS.execute("SELECT distinct(Document), similarity FROM (SELECT VECTOR_DOT_PRODUCT(VectorizedPhrase, TO_VECTOR(?, DECIMAL)) AS similarity, Document FROM LLMRAG.DOCUMENTCHUNK) WHERE similarity > 0.6", parameterQuery)
    similarityRows = cursorIRIS.fetchall()
    context = ''
    for similarityRow in similarityRows:
        print(similarityRow[0]+" "+str(similarityRow[1]))
        for doc in docs_before_split:
            if similarityRow[0] == doc.metadata['source'].upper():
                context = context +"".join(doc.page_content)

    template = """
    Eres un asistente de farmacia. Responde las siguientes preguntas considerando la historia de la conversación y utilizando el contexto asociado:

    Chat history: {chat_history}

    Context: {context}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "user_question": user_query,
    })

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="¡Hola! Soy tu asistente de farmacia. ¿En qué puedo ayudarte?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Escriba su mensaje aquí...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.write(response)

    st.session_state.chat_history.append(AIMessage(content=response))
