import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
# Import the key directly from your existing secret_api_keys.py
from secret_api_keys import huggingface_api_key

# --- Page Configuration ---
st.set_page_config(page_title="AI Blog Studio", layout="centered")

# --- Model Initialization ---
@st.cache_resource
def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="conversational", 
        huggingfacehub_api_token=huggingface_api_key,
        max_new_tokens=1200, # Increased for full blog posts
        temperature=0.7
    )
    return ChatHuggingFace(llm=llm)

chat_model = load_llm()

# --- User Interface ---
st.title("AI Content Studio")
st.markdown("Generate catchy titles and professional blog posts in one place.")

# --- PART 1: Title Generator ---
st.header("1. Brainstorm Titles")
topic = st.text_input("Enter your blog topic:", placeholder="e.g., Deep Learning in Healthcare")

if st.button("Suggest 5 Titles"):
    if topic:
        with st.spinner("Brainstorming..."):
            prompt_template = PromptTemplate(
                input_variables=['topic'],
                template="Suggest 5 catchy titles for a blog post about {topic}. Provide only the list."
            )
            formatted_prompt = prompt_template.format(topic=topic)
            response = chat_model.invoke([HumanMessage(content=formatted_prompt)])
            st.success("Title Suggestions:")
            st.write(response.content)
    else:
        st.warning("Please enter a topic.")

st.divider() # Visual separation

# --- PART 2: Blog Generator (Now directly below) ---
st.header("2. Write the Full Blog Post")
st.info("Choose your favorite title from above or enter your own.")

col1, col2 = st.columns([2, 1])
with col1:
    blog_title = st.text_input("Final Blog Title", placeholder="Paste the title you like here")
with col2:
    word_count = st.select_slider("Word Count", options=[200, 400, 600, 800, 1000], value=400)

keywords = st.text_input("Keywords (comma separated)", placeholder="e.g., AI, neural networks, beginners")

if st.button("Generate Full Blog"):
    if blog_title:
        with st.spinner("Writing your masterpiece..."):
            blog_prompt = f"""
            You are a professional tech blogger. Write an engaging blog post.
            Title: {blog_title}
            Approximate Length: {word_count} words.
            Keywords to include: {keywords}.
            
            Format the post with:
            - An engaging Introduction
            - Clear subheadings for the body
            - A strong Conclusion
            """
            blog_response = chat_model.invoke([HumanMessage(content=blog_prompt)])
            st.markdown("---")
            st.markdown(f"## {blog_title}")
            st.write(blog_response.content)
    else:
        st.error("Please provide a title to generate the blog.")