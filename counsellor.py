import os
import dotenv
import re
import unicodedata
import cohere 
import streamlit as st 
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# Keys
dotenv.load_dotenv()
os.environ['COHERE_API_KEY'] = os.getenv('COHERE')
co =  cohere.Client(os.environ['COHERE_API_KEY'])

st.title('Personal Guidance Counsellor')
st.write('''
Students often face a lot of problems in their academic life. They are often confused about their career choices, the subjects they should choose, the colleges they should apply to, etc.
*We solve them!*         
''')
st.divider()
t1 = st.text_input('Enter your academic problems here',placeholder='Enter you background and the problem you are facing')
st.button('What do I do?')
chat = ChatGroq(temperature=0,groq_api_key= os.getenv('GROQ'),model_name = 'llama3-70b-8192')

# Brooms and Mops
def remove_control_characters(text):
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
def clean_text(text):
    # Remove leading/trailing whitespace and empty strings
    cleaned_text = re.sub(r'^\s*$\n', '', text, flags=re.MULTILINE)
    # Remove strings containing only whitespace
    cleaned_text = ''.join([line for line in cleaned_text.splitlines() if line.strip()])
    return cleaned_text

# Prompt
SYSTEM_TEMPLATE = """
Answer you're questions solely based on the context below and use the persona stated. If the question canot be answered by the context, just answer 'I don't know'.
You should give proper answers and provide almost all the information ,for example do not tell the student to search on the internet.

<context>
{context}
</context>
"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="input"),
    ]
)
# Instantiate Chroma
chroma_db = Chroma(collection_name="counsellor",embedding_function=GoogleGenerativeAIEmbeddings(model='models/embedding-001'), persist_directory="./db3")

# Answering
if t1:
    document_chain = create_stuff_documents_chain(chat,question_answering_prompt)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 25})
    docs = retriever.invoke(t1,k=25)
    cleaned_documents = [remove_control_characters(clean_text(doc.page_content)) for doc in docs]
    rerank_docs  = co.rerank(query=t1,documents=cleaned_documents,top_n=25,model="rerank-english-v2.0")
    response = document_chain.invoke(
    {
        "context": [Document(page_content=cleaned_documents[i]) for i,d in enumerate(rerank_docs)],
        "input": [
            HumanMessage(content=t1+' Please guide me and explain thoroughly on what should I do.')
        ],
    }
)
    st.write(response)