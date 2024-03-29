
import os
import time
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
                  In your response, include this integer as a citation, formatted as a YouTube video link: "https://www.youtube.com/watch?v=<video_id>&t=<int>s". 
                  have the hyperlink text of the video link be the title of the video if you dont know the title simply put "source" as the text.
                  Even if there are multiple steps answered please only include one link at bottom of response.

                  here is an example:

                  User: What is RAG?

                  Chatbot: RAG stands for Retrieval Augmented Generation. 
                  It is a system that combines dense vector retrieval and in-context learning to perform question answering over documents. 
                  RAG is used to build language model applications and is an important component in the field of AI engineering. 
                  You can learn more about RAG in this video: [Retrieval Augmented Generation (RAG), An Overview](https://www.youtube.com/watch?v=fIDxnTe4mBA&t=128s)

                  -End of example, notice for this example there was relavent information in the context to answer. 
                  specifically in the video "Retrieval Augmented Generation (RAG), An Overview" which was hyperlinked to the correct timestamp.
                  so the included hyperlink format for this example would be "https://www.youtube.com/watch?v=fIDxnTe4mBA&t=128s" assuming <video_id> = 'fIDxnTe4mBA&t' and <int> = '128s'.
                  even if you feel tempted to dont put for example '(timestamp: 0s)' after the link.

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
st.set_page_config(page_title="🦜🔗 Video Assistant ▶️")
st.title('🦜🔗 Video Assistant ▶️')

with st.sidebar:
    video_url = st.text_input("Video URL", key="video_url", type="default")
    if st.button('Process Video'):
        start_time = time.time()
        with st.spinner('Processing video...'):
            index_video(video_url)
         
        processing_time = time.time() - start_time  # Calculate the processing time
        st.success(f'Video processing completed successfully in {processing_time:.2f} seconds!')
        video_id = video_url.split('=')[-1]
        title, author, url = get_youtube_data(video_id)
        st.write(author + ": ")
        st.markdown(f"[{title}]({video_url})", unsafe_allow_html=True)

    st.markdown("---")

    if 'video_data' not in st.session_state:
        st.session_state['video_data'] = []

    search = st.text_input("Search", key="search", type="default")
    n_videos = st.slider('Select number of videos to search', 1, 5, 1)

    if st.button('Process Search'):
        # Clear previous data
        st.session_state['video_data'].clear()

        # Initialize progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()  # This will be used to display the progress text

        # Process videos
        with st.spinner('Processing videos...'):
            videosSearch = VideosSearch(search, limit=n_videos)
            ids = [video["id"] for video in videosSearch.result()["result"]]
            
            for i, video_id in enumerate(ids):
                title, author, url = get_youtube_data(video_id)
                index_video(url)
                # Store the video data in session state
                st.session_state['video_data'].append((author, title, url))

                # Update the progress bar and text
                progress = int((i + 1) / n_videos * 100)
                progress_bar.progress(progress)
                progress_text.text(f' Completed video {i + 1}/{n_videos}...')

        # Remove progress bar and text after processing is complete
        progress_bar.empty()
        progress_text.empty()

    # Display stored video data
    #st.success('Video processing completed successfully!')
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