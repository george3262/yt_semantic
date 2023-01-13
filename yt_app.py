from tracemalloc import start
import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer

st.write("hello")

@st.experimental_singleton
def init_pinecone():
    pinecone.init(api_key="071d15be-9309-41fc-a16e-c3d7b2b0ddfb", environment="us-west1-gcp")
    return pinecone.Index('youtube-search')
    
@st.experimental_singleton
def init_retriever():
    return SentenceTransformer('multi-qa-mpnet-base-dot-v1')

index = init_pinecone()
model = init_retriever()

def card(title, url, context, start):
    url1 = url.split("=")[-1]
    start1 = "{:.0f}".format(start)
    url_final = (f"https://youtu.be/{url1}?t={start1}")
    # print(url_final)
    return st.markdown(f"""
    <div class="container-fluid">
        <div class="row align-items-start">
             <div  class="col-md-8 col-sm-8">
                 <a href={url_final}>{title}</a>
                 <br>
                 <span style="color: #808080;">
                     <small>{context[:200].capitalize()+"...."}</small>
                 </span>
             </div>
        </div>
     </div>
        """, unsafe_allow_html=True)

    
st.write("""
# YouTube Q&A
Ask me a question!
""")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)

query = st.text_input("Search!", "")

if query != "":
    xq = model.encode(query).tolist()
    xc = index.query(xq, top_k=10, include_metadata=True)
    
    for context in xc['matches']:
        card(
            # context['metadata']['thumbnail'],
            context['metadata']['title'],
            context['metadata']['url'],
            context['metadata']['text'],
            context['metadata']['start']
        ) 