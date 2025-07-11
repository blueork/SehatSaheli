## for accessing the LLM deployed on Hugging Face Hub
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

## for making prompts to pass in to the LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

## for langraph message persistance
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from langchain_core.utils.function_calling import convert_to_openai_function

from embed_load_data import get_vector_store

import json
import re
from typing import List, Dict, Any, Union
import uuid # Import uuid for generating unique IDs

## for UI
import streamlit as st
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<YOUR HUGGING FACE TOKEN>"

# Check if the Hugging Face API token is set
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    st.error("HUGGINGFACEHUB_API_TOKEN environment variable not set. Please set it to your Hugging Face API token.")
    st.stop() # Stop execution if token is missing

## returns the specifed LLM
@st.cache_resource
def get_llm():
    """Initializes and returns the Hugging Face LLM."""
    st.spinner("Initializing Hugging Face LLM...")
    print("Initializing Hugging Face LLM...")

    chat = HuggingFaceEndpoint(
        repo_id="NousResearch/Hermes-2-Pro-Llama-3-8B",
        task="text-generation",
        temperature=0.7, 
        max_new_tokens=512
        # do_sample=False,
        # repetition_penalty=1.03,
    )

    llm = ChatHuggingFace(llm=chat)

    print("Hugging Face LLM ready.")
    return llm

llm = get_llm()
print('done')
vector_store = get_vector_store()
print('done')
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
print('done')

@tool(response_format="content_and_artifact")
def retrieve_food_info(query: str):
    """
    Retrieves detailed information about food and nutrition based on a specific query.
    Use this tool for any questions related to specific food types, dietary information,
    or nutrition facts.
    """
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

retrieve_food_info_schema = convert_to_openai_function(retrieve_food_info)
# The model description wants a JSON *string* inside <tools>
tools_xml_content = f"<tools> {json.dumps(retrieve_food_info_schema)} </tools>"

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are Sehat Saheli, an empathetic and helpful AI assistant for pregnant women in Pakistan. "
        "Maintain a warm, caring, and encouraging tone throughout. Use simple, direct language and avoid jargon. "
        "Do not offer medical diagnoses or complex medical advice, only general nutrition guidance. "
        "You are also a function calling AI model. You are provided with function signatures "
        "For any query regarding food, nutrition and dietary advice you must call one of the provided tools. "
        # "You are also provided with functions. Call them whenever applicable. Try to answer a query using function. "
        "within <tools></tools> XML tags. You may call one or more functions to assist "
        "with the user query. Don't make assumptions about what values to plug into functions. "
        "Here are the available tools: "
        f"{tools_xml_content}\n\n" # Inject the formatted tool schema here
        "Use the following pydantic model json schema for each tool call you will make: "
        # This is the Pydantic schema for the FunctionCall object itself, not for arguments
        "{\"properties\": {\"arguments\": {\"title\": \"Arguments\", \"type\": \"object\"}, \"name\": {\"title\": \"Name\", \"type\": \"string\"}}, \"required\": [\"arguments\", \"name\"], \"title\": \"FunctionCall\", \"type\": \"object\"}\n"
        "For each function call return a json object with function name and arguments within "
        "<tool_call></tool_call> XML tags as follows:\n"
        "<tool_call>\n"
        "{\"arguments\": <args-dict>, \"name\": <function-name>}\n"
        "</tool_call>"
    )),
    MessagesPlaceholder(variable_name="messages"), # Placeholder for user's query
])

# 3. Custom parser to extract the tool call from the LLM's raw string output.
class CustomToolCallParser:
    def parse(self, text: str) -> List[Dict[str, Any]]:
        tool_calls = []
        # Regex to find content between <tool_call> and </tool_call> (non-greedy)
        matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        for match in matches:
            try:
                # Attempt to parse the JSON inside the tag
                call_json = json.loads(match.strip())
                # Validate against the expected structure (name and arguments)
                if isinstance(call_json, dict) and "name" in call_json and "arguments" in call_json:
                    tool_calls.append({
                        "name": call_json["name"],
                        "args": call_json["arguments"], # LangChain's ToolCall expects 'args' not 'arguments'
                        "id": str(uuid.uuid4()) # <-- ADD THIS LINE: Generate a unique ID
                    })
                else:
                    print(f"Warning: Malformed tool call JSON inside <tool_call> tag: {match}")
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON inside <tool_call> tag: {match}")
        return tool_calls

# 4. Modified query_or_respond function
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""

    # IMPORTANT: DO NOT use llm.bind_tools() here.
    # Instead, chain your custom prompt template with the raw LLM.
    llm_with_custom_prompt_chain = prompt_template | llm

    print("\n--- Debugging query_or_respond LLM Call ---")
    print("Input messages to LLM (for prompt template):", state['messages'])

    try:
        # Invoke the chained runnable, passing messages to the MessagesPlaceholder
        raw_ai_message = llm_with_custom_prompt_chain.invoke({"messages": state["messages"]})

        print("SUCCESS: Raw AIMessage from LLM:", raw_ai_message)
        print("Raw AIMessage content (might contain <tool_call> tags):", raw_ai_message.content)

        # Parse the raw content for tool calls using your custom parser
        parser = CustomToolCallParser()
        tool_calls_parsed = parser.parse(raw_ai_message.content)

        # Construct the final AIMessage with parsed tool_calls
        response_message = AIMessage(
            content=raw_ai_message.content, # Keep the original content
            tool_calls=[{"name": tc["name"], "args": tc["args"], "id": tc["id"]} for tc in tool_calls_parsed]
        )

        print("Constructed AIMessage for graph:", response_message)
        print("AIMessage tool_calls for graph:", response_message.tool_calls)

    except Exception as e:
        print(f"ERROR: LLM failed with: {e}")
        raise
    print("--- End Debugging LLM Call ---")

    return {"messages": [response_message]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve_food_info])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    system_message_content = (
        "You are Sehat Saheli, an empathetic and helpful AI assistant for pregnant women in Pakistan. "
        "Maintain a warm, caring, and encouraging tone throughout. Use simple, direct language and avoid jargon. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use five sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

def tools_condition(state: MessagesState) -> str:
    # Check if the last message is an AI message and contains tool calls
    if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls:
        return "tools" # Go to the 'tools' node
    return END # Otherwise, end the conversation


# Define the function that calls the model
def call_model(state: MessagesState):
    print(state['messages'])
    print()
    prompt = prompt_template.invoke(state)
    response = llm.invoke(prompt)
    # Update message history with response:
    return {"messages": response}

@st.cache_resource
def get_app(): 
    
    # # Define a new graph
    # workflow = StateGraph(state_schema=MessagesState)

    # # Define the (single) node in the graph
    # workflow.add_edge(START, "model")
    # workflow.add_node("model", call_model)

    # # Add memory
    # memory = MemorySaver()
    # app = workflow.compile(checkpointer=memory)

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    app = graph_builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "1"}}

    return app, config

app, config = get_app()

# --- Streamlit UI ---
st.set_page_config(page_title="Sehat Saheli Chatbot MVP", layout="centered")
st.title("🤰 Sehat Saheli Chatbot (MVP)")
st.markdown("""
Your AI companion for Pregnancy Advice!
""")
st.info("Disclaimer: This is an MVP for demonstration purposes. The advice provided is general and not a substitute for professional medical consultation.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add an initial welcome message from the assistant
    st.session_state.messages.append({"role": "assistant", "content": "Assalamu Alaikum! I am Sehat Saheli, your friendly guide for healthy eating during pregnancy. How can I help you today?"})

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input for user
if prompt := st.chat_input("Ask about nutrition during pregnancy..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_response_text = ""
            try:
                # # Step 1: Detect language and translate to English for RAG/LLM
                # translated_query_en, detected_lang = detect_and_translate(prompt, target_lang='en')
                # st.info(f"Detected language: {detected_lang}. Query translated to English for processing.")

                # # --- DEBUGGING RETRIEVAL (Added) ---
                # # Let's explicitly see what the retriever is finding
                # retrieved_docs_for_debug = retriever.invoke(translated_query_en)
                # st.info(f"Retrieved {len(retrieved_docs_for_debug)} documents from vector store.")
                # if retrieved_docs_for_debug:
                #     for i, doc in enumerate(retrieved_docs_for_debug):
                #         st.info(f"  Doc {i+1} content (first 50 chars): {doc.page_content[:50]}...")
                #         st.info(f"  Doc {i+1} metadata: {doc.metadata}")
                # else:
                #     st.warning("No relevant documents found by the retriever.")
                # # --- END DEBUGGING RETRIEVAL ---

                input_messages = [HumanMessage(prompt)]
                # Step 2: Invoke the RAG chain with the translated English query
                llm_generated_text_en = app.invoke({"messages": input_messages}, config)
                # st.info(f"LLM generated (English): '{llm_generated_text_en}'") # Optional: for debugging

                print(f"Raw LLM generated (English): '{llm_generated_text_en}'")

                # Step 3: Translate LLM response back to detected local language
                # if detected_lang and detected_lang != 'en':
                #     final_response_text = translate_back(llm_generated_text_en, detected_lang)
                #     # st.info(f"Translated back to {detected_lang}: '{final_response_text}'") # Optional: for debugging
                # else:
                final_response_text = llm_generated_text_en['messages'][-1].content

            except Exception as e:
                st.error(f"An error occurred: {e}. Please try again.")
                error_msg_en = "I'm sorry, I'm having trouble processing your request right now. Please try again."
                # if detected_lang and detected_lang != 'en':
                #     final_response_text = translate_back(error_msg_en, detected_lang)
                # else:
                #     final_response_text = error_msg_en

            st.markdown(final_response_text)
        st.session_state.messages.append({"role": "assistant", "content": final_response_text})


