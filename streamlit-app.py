import os
import logging
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional


load_dotenv()

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

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
- Remember to consider ALL relevant contracts in the database that match the query criteria, not just the first few or ones mentioned earlier in the conversation.
- Each question should be treated independently of previous questions, unless it explicitly references a previous question.

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

IMPORTANT: If the follow-up question seems to be asking about a completely new topic or doesn't explicitly reference the previous conversation,
treat it as an independent question and return it as is.

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

def extract_metadata_filters(query: str) -> Dict[str, str]:
    """
    Analyze the query to extract potential metadata filters.
    """
    prompt = f"""
Analyze the following query for potential metadata filters for a contract database search.
The metadata fields available include: customer_name, contract_type, effective_date, expiration_date.

If the query mentions a specific customer, contract type, or date range, extract these as filters.
Return ONLY a JSON object with the filters, nothing else. If no filters are found, return an empty JSON object.

Example query: "What marketing rights do we have in the contract with Acme Corp?"
Example response: {{"customer_name": "Acme Corp"}}

Example query: "Which MSAs were signed in 2023?"
Example response: {{"contract_type": "MSA", "effective_date": "2023"}}

Query: {query}
"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You extract structured metadata filters from natural language queries about contracts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    try:
        import json
        filters = json.loads(response.choices[0].message.content)
        return filters
    except:
        return {}

def query_similar_documents(query_text: str, top_k: int = 30, metadata_filters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Find documents similar to the query text using Pinecone vector store with optional metadata filtering.
    """
    query_embedding = create_embedding(query_text)
    
    # Prepare filter dict for Pinecone if metadata filters are provided
    filter_dict = {}
    if metadata_filters:
        for key, value in metadata_filters.items():
            if value:  # Only add non-empty filters
                filter_dict[key] = {"$eq": value}
    
    # Execute query with or without filters
    if filter_dict:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
    else:
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

def analyze_query_type(query: str) -> str:
    """
    Analyze if query is about all contracts or specific ones.
    Returns: "all_scan", "targeted", or "general"
    """
    prompt = f"""
Analyze the following query about contracts and classify it as one of these types:
1. "all_scan" - Requires scanning ALL contracts to answer (e.g., "Which customers allow marketing?", "List all contracts that permit us to use their logo")
2. "targeted" - Asks about specific customer(s) or specific contract(s) (e.g., "What are the termination clauses in our contract with Acme Corp?")
3. "general" - General question that doesn't fall into above categories

Query: {query}
Classification (just return the classification word):
"""
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=20,
        temperature=0.1
    )
    
    return response.choices[0].text.strip().lower()

def rag_query(query: str, max_tokens: int = 1000) -> Dict:
    """
    Perform a RAG query - retrieve relevant documents and generate a response.
    """
    # Analyze query type to determine search strategy
    query_type = analyze_query_type(query)
    logging.info(f"Query type detected: {query_type}")
    
    # Extract potential metadata filters
    metadata_filters = extract_metadata_filters(query)
    logging.info(f"Extracted metadata filters: {metadata_filters}")
    
    # Adjust retrieval strategy based on query type
    if query_type == "all_scan":
        # For all_scan queries, we need to retrieve many more documents
        similar_docs = query_similar_documents(query, top_k=100, metadata_filters=metadata_filters)
    elif query_type == "targeted" and metadata_filters:
        # For targeted queries with metadata, use metadata filtering
        similar_docs = query_similar_documents(query, top_k=20, metadata_filters=metadata_filters)
    else:
        # For general queries, use standard semantic search
        similar_docs = query_similar_documents(query, top_k=20)
    
    # Format the context for the prompt
    context = "\n\n".join([f"Document {i+1}:\n{doc['full_text']}\nMetadata: {doc['metadata']}" 
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
        "source_nodes": similar_docs,
        "query_type": query_type,
        "metadata_filters": metadata_filters
    }

def reset_conversation():
    """Reset the conversation history"""
    st.session_state.messages = []
    ai_intro = "Hello, I'm your Legal Contract Review Assistant. What can I help you with?"
    st.session_state.messages.append({"role": "assistant", "content": ai_intro})

if "messages" not in st.session_state:
    reset_conversation()

# Display title and reset button in the same row
col1, col2 = st.columns([5, 1])
with col1:
    st.title("Legal Contract Review Assistant")
with col2:
    if st.button("Reset Chat"):
        reset_conversation()
        st.rerun()

# Add a sidebar with search options
with st.sidebar:
    st.header("Search Options")
    st.info("For questions about all contracts (like 'Which customers allow marketing?'), the assistant will scan more documents to find comprehensive answers.")
    
    # Customer filter for direct search
    st.subheader("Direct Customer Search")
    customers = ["All Customers", "Acme Corp", "XYZ Health", "Sunrise Medical", "TechCare Solutions"]  # Replace with actual customer list
    selected_customer = st.selectbox("Select a customer to focus on:", customers)
    
    if st.button("Search All Contracts for Selected Customer"):
        if selected_customer != "All Customers":
            user_input = f"Show me all contracts with {selected_customer}"
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()

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
        
        # Show query analysis and sources
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Query Analysis"):
                st.markdown(f"**Query Type:** {response['query_type']}")
                st.markdown(f"**Metadata Filters:** {response['metadata_filters']}")
                st.markdown(f"**Documents Retrieved:** {len(response['source_nodes'])}")
        
        with col2:
            with st.expander("Sources"):
                for i, source_node in enumerate(response['source_nodes']):
                    st.markdown(f"**Source {i+1}**")
                    # Display a compact metadata summary first
                    meta_summary = ", ".join([f"**{k}**: {v}" for k, v in source_node['metadata'].items()])
                    st.markdown(meta_summary)
                    
                    # Show the document text in a collapsible section
                    with st.expander(f"View Document {i+1} Text"):
                        st.markdown(source_node['full_text'])
                    
                    st.markdown("---")
    
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