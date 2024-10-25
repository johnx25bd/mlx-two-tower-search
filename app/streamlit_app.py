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
        documents = data.get("documents", [])
        similarity = data.get("similarity", [])

        st.subheader(f"Search Results: ")

        df = pd.DataFrame(
            {
                "Document": documents,
                "Similarity": similarity,
            }
        )
        df = df.reset_index(drop=True)

        st.table(df.style.format({"Similarity": "{:.4f}"}))
    else:
        st.error("Error: Unable to retrieve documents. Please try again.")
else:
    st.write("Enter a query and click 'Search' to see results.")
