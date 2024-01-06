
import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

from youtubesearchpython import VideosSearch
from process_data import *
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI

from operator import itemgetter
import pinecone

# =============================================================================
# Retrieval Chain
# =============================================================================
def load_llm():
  llm = ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0.0,
    )
  return llm


def load_vectorstore():

    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )

    #index = pinecone.GRPCIndex("youtube-index")
    index = pinecone.Index("youtube-index")
    store = LocalFileStore("./cache/")
    core_embeddings_model = OpenAIEmbeddings()

    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model,
        store,
        namespace=core_embeddings_model.model
    )

    text_field = "text"

    vectorstore = Pinecone(
        index,
        embedder,  
        text_field
    )

    return vectorstore


def qa_chain():

    vectorstore = load_vectorstore()
    
    llm = load_llm()
    retriever = vectorstore.as_retriever()

    template = """You are a helpful assistant that answers questions on the provided context, if its not answered within the context respond with "This query is not directly mentioned in the provided video" then respond to the best to your ability. 
                  Additionally, the context includes a specific integer formatted as <int>, representing a timestamp. 
                  In your response, include this integer as a citation, formatted as a YouTube video link: "https://www.youtube.com/watch?v=[video_id]&t=<int>s" and text of link be the title of video.


    ### CONTEXT
    {context}

    ### QUESTION
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever,
        "question": itemgetter("question")
        }
        | RunnablePassthrough.assign(
            context=itemgetter("context")
        )
        | {
            "response": prompt  | llm,
            "context": itemgetter("context"),
        }
    )

    return retrieval_augmented_qa_chain

# =============================================================================
# Streamlit
# =============================================================================
st.set_page_config(page_title="🦜🔗 Chat with  ▶️")
st.title('🦜🔗 Ask YouTube ▶️')

with st.sidebar:
    video_url = st.text_input("Video URL", key="video_url", type="default")
    if st.button('Process Video'):
        with st.spinner('Processing video...'):
            index_video(video_url)

    st.markdown("---")

    if 'video_data' not in st.session_state:
        st.session_state['video_data'] = []

    search = st.text_input("Search", key="search", type="default")
    n_videos = st.slider('Select number of videos to search', 1, 5, 1)

    if st.button('Process Search'):
        # Clear previous data
        st.session_state['video_data'].clear()

        # Process videos
        with st.spinner('Processing videos...'):
            videosSearch = VideosSearch(search, limit=n_videos)
            ids = [video["id"] for video in videosSearch.result()["result"]]
            
            for id in ids:
                title, author, url = get_youtube_data(id)
                index_video(url)
                # Store the video data in session state
                st.session_state['video_data'].append((author, title, url))

    # Display stored video data
    for author, title, url in st.session_state['video_data']:
        st.write(author + ": ")
        st.markdown(f"[{title}]({url})", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask question about your video"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    messages=st.session_state.messages
    chain = qa_chain()
    response = chain.invoke({"question" :  prompt})
    msg = response['response'].content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)