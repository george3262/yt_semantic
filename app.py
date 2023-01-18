from curses import meta
from unittest import result
import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import pandas as pd

@st.experimental_singleton
def init_pinecone():
    pinecone.init(api_key="071d15be-9309-41fc-a16e-c3d7b2b0ddfb", environment="us-west1-gcp")
    return pinecone.Index('ytbc2')
    
@st.experimental_singleton
def init_retriever():
    return SentenceTransformer('multi-qa-mpnet-base-dot-v1')

index = init_pinecone()
model = init_retriever()

def card(thumbnail, title, url, context, start, score, published):
    url1 = url.split("=")[-1]
    start1 = "{:.0f}".format(start)
    url_final = (f"https://youtu.be/{url1}?t={start1}")
    return st.markdown(f"""
    <div class="container-fluid">
        <div class="row align-items-start">
            <div class="col-md-4 col-sm-4">
                 <div class="position-relative">
                     <a href={url}><img src={thumbnail} class="img-fluid" style="width: 192px; height: 106px"></a>
                 </div>
             </div>
             <div  class="col-md-8 col-sm-8">
                 <a href={url_final}>{title}</a>
                 <br>
                 <span style="color: white;">
                     <small>{context[:200].capitalize()+"...."} {published}</small>
                 </span>
             </div>
        </div>
     </div>
     <br>
        """, unsafe_allow_html=True)

st.write("""
Ask Bonnie a question
""")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)

query = st.text_input("Search!", "")

values = st.slider(
    'Select number of results',
    1, 50)

if query != "":
    xq = model.encode(query).tolist()
    xc = index.query(xq, top_k=values, include_metadata=True)

    results = []
    for context in xc['matches']:
        check = {
            "thumbnail": context['metadata']['thumbnail'],
            "title": context['metadata']['title'],
            "url": context['metadata']['url'],
            "text": context['metadata']['text'],
            "start": context['metadata']['start'],
            "score": context['score'],
            "published": context['metadata']['published']
        }
        results.append(check)
    df = pd.DataFrame(results).sort_values(by='published', ascending=True)

    for index, row in df.iterrows():
            card(
            row['thumbnail'],
            row['title'],
            row['url'],
            row['text'],
            row['start'],
            row['score'],
            row['published']
        ) 