import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Simple Search Engine", layout="wide")


def truncate_text(text, max_length=50):
    return text[:max_length] + "..." if len(text) > max_length else text


def display_document(docs, selected_index, doc_type):
    st.write(f"{doc_type} Document:")
    st.write(docs[selected_index])


st.title("Simple Search Engine")

col1, col2 = st.columns(2)

with col1:
    query = st.text_input("Enter your search query", max_chars=200)

search_button = st.button("Search")

# Initialize session state
if "search_performed" not in st.session_state:
    st.session_state.search_performed = False

if search_button and query:
    response = requests.post("http://localhost:8000/search", json={"query": query})

    if response.status_code == 200:
        st.session_state.search_performed = True
        data = response.json()
        st.session_state.rel_docs = data.get("rel_docs", [])
        st.session_state.rel_docs_sim = data.get("rel_docs_sim", [])
        st.session_state.irel_docs = data.get("irel_docs", [])
        st.session_state.irel_docs_sim = data.get("irel_docs_sim", [])
    else:
        st.error(
            f"Error: Unable to retrieve documents. Status code: {response.status_code}"
        )
        st.write(f"Response content: {response.text}")

if st.session_state.search_performed:
    df_similar = pd.DataFrame(
        {
            "Document": [truncate_text(doc) for doc in st.session_state.rel_docs],
            "Cosine Similarity": st.session_state.rel_docs_sim,
        }
    ).reset_index(drop=True)

    df_dissimilar = pd.DataFrame(
        {
            "Document": [truncate_text(doc) for doc in st.session_state.irel_docs],
            "Cosine Similarity": st.session_state.irel_docs_sim,
        }
    ).reset_index(drop=True)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Most Similar Results:")
        selected_similar = st.selectbox(
            "Select a similar document to view full text:",
            options=list(range(len(df_similar))),
            format_func=lambda x: df_similar.loc[x, "Document"],
            key="similar_select",
        )
        st.table(df_similar.style.format({"Cosine Similarity": "{:.4f}"}))

        st.subheader("Least Similar Results:")
        selected_dissimilar = st.selectbox(
            "Select a dissimilar document to view full text:",
            options=list(range(len(df_dissimilar))),
            format_func=lambda x: df_dissimilar.loc[x, "Document"],
            key="dissimilar_select",
        )
        st.table(df_dissimilar.style.format({"Cosine Similarity": "{:.4f}"}))

    with col2:
        st.subheader("Selected Document:")
        doc_display_similar = st.empty()
        doc_display_dissimilar = st.empty()

        if selected_similar is not None:
            doc_display_similar.empty()
            display_document(st.session_state.rel_docs, selected_similar, "Similar")

        # Add vertical spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        if selected_dissimilar is not None:
            doc_display_dissimilar.empty()
            display_document(
                st.session_state.irel_docs, selected_dissimilar, "Dissimilar"
            )

else:
    st.write("Enter a query and click 'Search' to see results.")
