import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer

st.write("hello")

pinecone.init(
    api_key="071d15be-9309-41fc-a16e-c3d7b2b0ddfb",  # app.pinecone.io
    environment="us-west1-gcp"
)
index = pinecone.Index("youtube-search")
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

def card(title, url, context):
    return st.markdown(f"""
    <div class="container-fluid">
        <div class="row align-items-start">
             <div  class="col-md-8 col-sm-8">
                 <a href={url}>{title}</a>
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
    xc = index.query(xq, top_k=20, include_metadata=True)
    
    for context in xc['matches']:
        card(
            # context['metadata']['thumbnail'],
            context['metadata']['title'],
            context['metadata']['url'],
            context['metadata']['text']
        )