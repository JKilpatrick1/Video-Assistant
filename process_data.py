from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import requests
import json

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from tqdm.auto import tqdm
from uuid import uuid4
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

import pinecone
from tqdm.auto import tqdm
from uuid import uuid4
import scrapetube

# =============================================================================
# DATA PREPARATION
# =============================================================================
def get_youtube_transcript(video_id):

    url = "https://www.youtube.com/watch?v=" + video_id

    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id)
    except:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for transcript in transcript_list:
                raw = transcript.translate('en').fetch()
                break
        except:
            print(f"No transcript found for {url}") 
            return False

    df = pd.DataFrame(raw)

    df["transcript"] = df["text"] + '<' + df['start'].astype(str) + '>'
    transcript = df['transcript'].str.cat(sep=' ')

    return transcript

def get_youtube_data(video_id):

    url = "https://www.youtube.com/watch?v=" + video_id

    response = requests.get(f"https://noembed.com/embed?dataType=json&url={url}")
    data = json.loads(response.content)

    title, author = data["title"], data["author_name"]

    print(title, author)

    # ' is a reserved character
    title = title.replace("'", "")
    author = author.replace("'", "")

    return title, author, url

def create_index(video_id):

    transcript = get_youtube_transcript(video_id)
    
    if transcript == False:
        return False
    
    title, author, url = get_youtube_data(video_id)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, # the character length of the chunk
        chunk_overlap = 100, # the character length of the overlap between chunks
        length_function = len, # the length function - in this case, character length (aka the python len() fn.)
        separators=["\n\n", "\n", " ", ""]
    )

    index_name = 'youtube-index'

    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )

    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536
        )

    index = pinecone.Index(index_name)
    store = LocalFileStore("./cache/")
    core_embeddings_model = OpenAIEmbeddings()

    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model,
        store,
        namespace=core_embeddings_model.model
    )

    BATCH_LIMIT = 100

    texts = []
    metadatas = []

    metadata = {
        'source_document' : title,
        'link' : url,
        'author' : author
    }

    record_texts = text_splitter.split_text(transcript)

    record_metadatas = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]
    
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    
    if len(texts) >= BATCH_LIMIT:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embedder.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embedder.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        

    text_field = "text"

    Pinecone(
        index,
        embedder,  
        text_field
    )

# =============================================================================
# 
# =============================================================================
def index_channel(channel_id):

    videos = scrapetube.get_channel(channel_id)
    for video in videos:
        create_index(video['videoId'])

    live_streams = scrapetube.get_channel(channel_id, content_type='streams')
    for stream in live_streams:
        create_index(stream['videoId'])



def index_video(video_url):
    video_id = video_url.split('=')[-1]
    print(video_id)
    create_index(video_id)
    