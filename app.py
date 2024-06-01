import asyncio

import streamlit as st

from langchain_helper import user_input, get_pdf_text, get_youtube_transcript, get_vector_store, get_text_chunks



async def main():
    if 'is_doc_loaded' not in st.session_state:
        st.session_state['is_doc_loaded'] = True
        
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        if not st.session_state.is_doc_loaded:
            st.write("Enter the documents")
        else:
            response = user_input(user_question)
            st.write(response["output_text"])

    with st.sidebar:
        st.title("Enter all applicable:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        youtube_url = st.text_input("Enter the link to a youtube video")
        if st.button("Submit & Process"):
            st.session_state.is_doc_loaded=False
            with st.spinner("Processing..."):
                texts = ""
                if pdf_docs:
                    texts += get_pdf_text(pdf_docs)
                if youtube_url:
                    texts += get_youtube_transcript(youtube_url)
                
                if texts:
                    text_chunks = get_text_chunks(texts)
                    get_vector_store(text_chunks)
                    st.session_state.is_doc_loaded=True
                    st.success("Done")
                else: st.success("No text found")



if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())