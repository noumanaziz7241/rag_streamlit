import os
import warnings
import logging

# Suppress warnings before importing other modules
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*pydantic.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Suppress ALTS warnings
logging.getLogger('grpc').setLevel(logging.ERROR)

import uuid
import sqlite3
import tiktoken
import streamlit as st
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()



# Fixed namespace for all operations
NAMESPACE = "polaris"
# Fixed user and session IDs
DEFAULT_USER_ID = "default_user"
DEFAULT_SESSION_ID = "main_session"


@dataclass
class ChatRequest:
    """Request model for chat API"""
    message: str
    user_id: str
    session_id: str


@dataclass
class ChatResponse:
    """Response model for chat API"""
    response: str
    user_id: str
    session_id: str
    success: bool
    error: Optional[str] = None


class VectorStoreManager:
    """Manages Pinecone vector stores for domain knowledge and memory"""
    
    def __init__(self):
        # os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

        os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]


        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        # self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        self.idx_domain = st.secrets["PINECONE_INDEX_NAME"]
        self.idx_domain = st.secrets["PINECONE_INDEX_NAME"]

        self.idx_memory = st.secrets["PINECONE_MEMORY_INDEX_NAME"]

    def get_domain_vectorstore(self) -> PineconeVectorStore:
        """Get domain knowledge vector store"""
        domain_index = self.pc.Index(self.idx_domain)
        return PineconeVectorStore(
            index=domain_index,
            embedding=self.embeddings,
            namespace=NAMESPACE
        )
    
    def get_memory_vectorstore(self) -> PineconeVectorStore:
        """Get or create memory vector store"""
        if not self.pc.has_index(self.idx_memory):
            example_embedding = self.embeddings.embed_query("test")
            embedding_dimension = len(example_embedding)
            
            self.pc.create_index(
                name=self.idx_memory,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        
        memory_index = self.pc.Index(self.idx_memory)
        return PineconeVectorStore(
            index=memory_index,
            embedding=self.embeddings,
            namespace=NAMESPACE
        )


class MemoryTools:
    """Tools for memory operations"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vs_manager = vector_store_manager
    
    @staticmethod
    def get_user_thread(config: RunnableConfig) -> tuple[str, str]:
        """Extract user_id and thread_id from config"""
        cfg = config["configurable"]
        user_id = cfg.get("user_id")
        thread_id = cfg.get("thread_id")
        if user_id is None or thread_id is None:
            raise ValueError("Need both user_id and thread_id in config.")
        return user_id, thread_id
    
    def create_tools(self):
        """Create tool instances with access to vector store manager"""
        
        @tool
        def save_memory(memory_text: str, config: RunnableConfig) -> str:
            """Save a memory (e.g. user fact) into memory vector store."""
            user_id, thread_id = self.get_user_thread(config)
            doc = Document(
                page_content=memory_text,
                id=str(uuid.uuid4()),
                metadata={"user_id": user_id, "thread_id": thread_id},
            )
            memory_vs = self.vs_manager.get_memory_vectorstore()
            memory_vs.add_documents([doc])
            return memory_text
        
        @tool
        def recall_memory(query: str, config: RunnableConfig) -> List[str]:
            """Recall relevant user-specific memories from the memory vector store."""
            user_id, thread_id = self.get_user_thread(config)
            memory_vs = self.vs_manager.get_memory_vectorstore()
            docs = memory_vs.similarity_search(query, k=5)
            filtered = [
                d for d in docs
                if d.metadata.get("user_id") == user_id 
                and d.metadata.get("thread_id") == thread_id
            ]
            return [d.page_content for d in filtered]
        
        @tool(response_format="content_and_artifact")
        def retrieve_domain(query: str):
            """Retrieve domain / knowledge documents for a query."""
            domain_vs = self.vs_manager.get_domain_vectorstore()
            docs = domain_vs.similarity_search(query=query, k=15, namespace=NAMESPACE)
            serialized = "\n\n".join(
                f"Source: {d.metadata}\nContent: {d.page_content}" for d in docs
            )
            return serialized, docs
        
        return [save_memory, recall_memory, retrieve_domain]


class State(MessagesState):
    """Extended state with recall memories"""
    recall_memories: List[str]


class MemoryAgent:
    """Main agent class handling conversation flow"""
    
    def __init__(self, db_path: str = "chat_memory.db"):
        self.vector_store_manager = VectorStoreManager()
        self.memory_tools = MemoryTools(self.vector_store_manager)
        self.tools = self.memory_tools.create_tools()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        
        # Initialize model
        self.model = ChatDeepSeek(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0,
        )
        self.model_with_tools = self.model.bind_tools(self.tools)
        
        # Initialize prompt
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an assistant with memory + retrieval. You have tools:\n"
                "- save_memory: store personal/user-specific memory\n"
                "- recall_memory: fetch memories for current user/thread\n"
                "- retrieve_domain: fetch knowledge from your domain database\n\n"
                "In your answers, use memory and retrieved docs to respond best. \n"
                "Memories: {recall_memories}\n\n"
                "Guidelines:\n"
                " * Use recall_memory when incorporating personal info\n"
                " * Use retrieve_domain for factual or domain knowledge questions\n"
                " * After using tools, respond using returned data"
            ),
            ("placeholder", "{messages}"),
        ])
        
        # Build graph
        self.graph = self._build_graph(db_path)
    
    def _agent_fn(self, state: State) -> State:
        """Agent function to process messages"""
        bound = self.prompt | self.model_with_tools
        recall_str = "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
        prediction = bound.invoke({
            "messages": state["messages"],
            "recall_memories": recall_str,
        })
        return {"messages": [prediction]}
    
    def _load_recalls(self, state: State, config: RunnableConfig) -> State:
        """Load relevant memories for current conversation"""
        convo = get_buffer_string(state["messages"])
        convo_trim = self.tokenizer.decode(self.tokenizer.encode(convo)[:2048])
        recs = self.tools[1].invoke(convo_trim, config)
        return {"recall_memories": recs}
    
    @staticmethod
    def _route_tools(state: State):
        """Route to tools if tool calls exist"""
        msg = state["messages"][-1]
        if msg.tool_calls:
            return "tools"
        return END
    
    def _build_graph(self, db_path: str):
        """Build and compile the LangGraph"""
        builder = StateGraph(State)
        builder.add_node(self._load_recalls)
        builder.add_node(self._agent_fn)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "_load_recalls")
        builder.add_edge("_load_recalls", "_agent_fn")
        builder.add_conditional_edges("_agent_fn", self._route_tools, ["tools", END])
        builder.add_edge("tools", "_agent_fn")
        
        # Setup checkpointer
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
        return builder.compile(checkpointer=checkpointer)
    
    def process_message(
        self,
        message: str,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID
    ) -> Dict[str, Any]:
        """
        Process a single message and return response
        
        Args:
            message: User's message
            user_id: Unique user identifier
            session_id: Unique session/thread identifier
            
        Returns:
            Dictionary containing response and metadata
        """
        config = {
            "configurable": {
                "user_id": user_id,
                "thread_id": session_id
            }
        }
        
        state = {"messages": [("user", message)]}
        
        # Stream through graph
        final_response = None
        for chunk in self.graph.stream(state, config=config):
            for node, updates in chunk.items():
                if "messages" in updates and updates["messages"]:
                    last_msg = updates["messages"][-1]
                    if hasattr(last_msg, 'content'):
                        final_response = last_msg.content
        
        return {
            "response": final_response or "No response generated",
            "user_id": user_id,
            "session_id": session_id
        }


class ChatAPI:
    """API interface for the memory agent"""
    
    def __init__(self, db_path: str = "chat_memory.db"):
        self.agent = MemoryAgent(db_path=db_path)
    
    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Main chat endpoint
        
        Args:
            request: ChatRequest with message, user_id, and session_id
            
        Returns:
            ChatResponse with result
        """
        try:
            result = self.agent.process_message(
                message=request.message,
                user_id=request.user_id,
                session_id=request.session_id
            )
            
            return ChatResponse(
                response=result["response"],
                user_id=result["user_id"],
                session_id=result["session_id"],
                success=True
            )
        except Exception as e:
            return ChatResponse(
                response="",
                user_id=request.user_id,
                session_id=request.session_id,
                success=False,
                error=str(e)
            )
    
    def chat_dict(
        self, 
        message: str, 
        user_id: str = DEFAULT_USER_ID, 
        session_id: str = DEFAULT_SESSION_ID
    ) -> Dict[str, Any]:
        """
        Convenience method that accepts direct parameters
        
        Args:
            message: User's message
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Dictionary with response data
        """
        request = ChatRequest(
            message=message,
            user_id=user_id,
            session_id=session_id
        )
        response = self.chat(request)
        
        return {
            "response": response.response,
            "user_id": response.user_id,
            "session_id": response.session_id,
            "success": response.success,
            "error": response.error
        }


# ==================== STREAMLIT APP ====================

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chat_api' not in st.session_state:
        st.session_state.chat_api = None


def main():
    """Main Streamlit app entry point"""
    # Page config
    st.set_page_config(
        page_title="Memory Agent Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize API if not already done
    if st.session_state.chat_api is None:
        with st.spinner("Initializing chat agent..."):
            st.session_state.chat_api = ChatAPI(db_path="chat_memory.db")
    
    # Sidebar
    with st.sidebar:
        st.title("Chat Info")
        st.markdown(f"**Namespace:** `{NAMESPACE}`")
        
        st.divider()
        
        # Clear chat button
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Info section
        st.markdown("### Agent Capabilities")
        st.markdown("""
        - **Memory**: Remembers information across conversations
        - **Retrieval**: Accesses domain knowledge from documents
        - **Persistence**: Maintains conversation history
        """)
        
        st.markdown("### Tips")
        st.markdown("""
        - Tell the agent facts you want it to remember
        - Ask questions about your uploaded documents
        - The agent will recall relevant information automatically
        """)
    
    # Main chat area
    st.title("Memory Agent Chat")
    st.caption("Chat with an AI assistant that has memory and knowledge retrieval")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(message["timestamp"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(timestamp)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_api.chat_dict(
                    message=prompt,
                    user_id=DEFAULT_USER_ID,
                    session_id=DEFAULT_SESSION_ID
                )
            
            if response["success"]:
                st.markdown(response["response"])
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.caption(timestamp)
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["response"],
                    "timestamp": timestamp
                })
            else:
                st.error(f"Error: {response['error']}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {response['error']}",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })


if __name__ == "__main__":
    main()