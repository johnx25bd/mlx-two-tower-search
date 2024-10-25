import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Simple Search Engine")

st.title("Simple Search Engine")

query = st.text_input("Enter your search query", max_chars=200)
search_button = st.button("Search")

if search_button and query:
    response = requests.post("http://localhost:8000/search", json={"query": query})

    if response.status_code == 200:
        data = response.json()
        rel_docs = data.get("rel_docs", [])
        rel_docs_sim = data.get("rel_docs_sim", [])
        irel_docs = data.get("irel_docs", [])
        irel_docs_sim = data.get("irel_docs_sim", [])

        st.subheader(f"Search Results: ")

        # Create dataframe for similar results (top 5)
        df_similar = pd.DataFrame(
            {"Document": rel_docs, "Cosine Similarity": rel_docs_sim}
        ).reset_index(drop=True)

        # Create dataframe for dissimilar results (bottom 5)
        df_dissimilar = pd.DataFrame(
            {"Document": irel_docs, "Cosine Similarity": irel_docs_sim}
        ).reset_index(drop=True)

        st.subheader("Most Similar Results:")
        st.table(df_similar.style.format({"Cosine Similarity": "{:.4f}"}))

        st.subheader("Least Similar Results:")
        st.table(df_dissimilar.style.format({"Cosine Similarity": "{:.4f}"}))
    else:
        st.error("Error: Unable to retrieve documents. Please try again.")
else:
    st.write("Enter a query and click 'Search' to see results.")
