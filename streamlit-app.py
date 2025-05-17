import os
import logging
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict, Any


load_dotenv()

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="RAG Assistant", page_icon="🤖")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set.")
    st.stop()

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY not set.")
    st.stop()
if not INDEX_NAME:
    st.error("PINECONE_INDEX_NAME not set.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    st.error(f"Error: Index '{INDEX_NAME}' does not exist in your Pinecone account.")
    st.stop()

index = pc.Index(INDEX_NAME)

CUSTOM_PROMPT = """You are a legal contract review assistant for business operations at Spring Care, Inc.

You have access to a database table named contracts_insights_llm_parsed, which contains metadata and OCRed contract texts for our customer agreements. Each row in this table represents a single contract with columns that include key details extracted from the contract, such as customer name, contract type, dates, pricing terms, and more. In addition, there is a column named merged_text_chunk that contains the full OCRed and concatenated text of the original contract.

Your job is to answer user questions about our contracts and contracting history with customers, using the data from this table as your primary source of truth.

**How to answer questions:**

1. **Parsed Columns First:**  
   When possible, answer questions directly using the structured, parsed columns in the contracts_insights_llm_parsed table (such as contract_name, contract_type, customer_name, effective_date, pricing_details, logo_usage_permission, etc.).

2. **Cross-check With Full Text:**  
   For every answer you give, **analyze and cross-check your answer using the merged_text_chunk column** (the full contract text). If you notice any discrepancy, always flag or explain it.

3. **Fallback to Text Search:**  
   If the parsed columns do not contain the requested information, or the answer is incomplete, carefully analyze the merged_text_chunk for the relevant contract(s) to find the information directly in the text. If you find the answer there, include the relevant direct quote from the contract text in your answer and reference the contract by filename.

4. **Be Precise and Factual:**  
   Never speculate or infer beyond what is present in the table or the contract text. If the answer cannot be found in either the parsed columns or the merged_text_chunk, state that clearly.

5. **References:**  
   Always cite the source—reference the contract's filename and, where possible, the clause, section, or page in the text.

**Additional guidelines:**
- Only answer using the uploaded contracts available in the contracts_insights_llm_parsed table.
- If multiple contracts or amendments exist for a customer, ensure your answer accounts for all relevant documents and their dates.
- For questions about contract types, use the contract_type column for classification (e.g., "Amendment", "Services Agreement", "Order Form", etc.).
- If asked for a summary or detailed policy, synthesize information from both structured columns and the full text for accuracy.

You are thorough, precise, and never make assumptions not supported by the data. If a question cannot be answered from the available sources, say so.

When possible, provide both a summary answer and direct supporting quotes from the contract text.

---

You may now begin answering user questions about contracts.

Based on the following retrieved documents, please answer the query.
Query: {query}
Retrieved Documents:
{context}
Answer:
"""

def create_embedding(text: str) -> List[float]:
    """
    Create a vector embedding from text using OpenAI's embedding model.
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",  # Using large model for 3072 dimensions
        input=text
    )
    return response.data[0].embedding

def condense_question(chat_history, question):
    """
    Condense a follow-up question based on chat history
    """
    chat_history_text = "\n".join([
        f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in chat_history
    ])
    
    prompt = f"""Given a conversation (between Human and Assistant) and a follow up message from Human, 
rewrite the message to be a standalone question that captures all relevant context 
from the conversation.

<Chat History> 
{chat_history_text}

<Follow Up Message>
{question}

<Standalone question>"""
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=200,
        temperature=0.1
    )
    return response.choices[0].text.strip()

def query_similar_documents(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find documents similar to the query text using Pinecone vector store.
    """
    query_embedding = create_embedding(query_text)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    similar_docs = []
    for match in results["matches"]:
        metadata = match.get("metadata", {})        
        text_content = metadata.get("text", "")
        
        similar_docs.append({
            "id": match["id"],
            "score": match["score"],
            "text_preview": text_content,
            "full_text": text_content,
            "metadata": {k: v for k, v in metadata.items() if k != "text"}
        })
    
    return similar_docs

def rag_query(query: str, max_tokens: int = 1000) -> Dict:
    """
    Perform a RAG query - retrieve relevant documents and generate a response.
    """
    similar_docs = query_similar_documents(query, top_k=3)
    
    context = "\n\n".join([f"Document {i+1}:\n{doc['full_text']}" 
                         for i, doc in enumerate(similar_docs)])
    
    prompt = CUSTOM_PROMPT.format(
        query=query,
        context=context
    )
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7
    )
    
    return {
        "response": response.choices[0].text.strip(),
        "source_nodes": similar_docs
    }

if "messages" not in st.session_state:
    st.session_state.messages = []
    ai_intro = "Hello, I'm your Legal Contract Review Assistant. What can I help you with?"
    st.session_state.messages.append({"role": "assistant", "content": ai_intro})

# Display title
st.title("Legal Contract Review Assistant")

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get input
user_input = st.chat_input("Ask me about your contracts...")

if user_input:
    # Add user message to the chat
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show assistant response with typing indicator
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Process the question with history if needed
        if len(st.session_state.messages) > 2:  # If we have chat history
            standalone_question = condense_question(
                st.session_state.messages[:-1],  # All messages except current
                user_input
            )
            logging.info(f"Original question: {user_input}")
            logging.info(f"Condensed question: {standalone_question}")
        else:
            standalone_question = user_input
        
        # Get response from RAG engine
        response = rag_query(standalone_question)
        
        # Update placeholder with response
        message_placeholder.markdown(response["response"])
        
        # Show sources
        with st.expander("Sources"):
            for i, source_node in enumerate(response["source_nodes"]):
                st.markdown(f"**Source {i+1}**")
                st.markdown(source_node["full_text"])
                st.markdown("---")
                st.markdown("**Metadata:**")
                for key, value in source_node["metadata"].items():
                    st.markdown(f"- **{key}**: {value}")
    
    # Add response to session state
    st.session_state.messages.append({"role": "assistant", "content": response["response"]})

with st.container():    
    last_output_message = []
    last_user_message = []

    for message in reversed(st.session_state.messages):
        if message["role"] == "assistant":
            last_output_message = message
            break
    for message in reversed(st.session_state.messages):
        if message["role"] == "user":
            last_user_message = message
            break 