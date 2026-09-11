from fastapi import FastAPI, HTTPException, Depends
from core.llm_client import LLMClient
from memory.long_term import LongTermMemory
from core.tool_registry import ToolRegistry
from tools import register_agent_tools
from memory.short_term import ShortTermMemory
from core.agent_loop import AgentLoop
from core.logging import get_logger

app = FastAPI()

#Initialize the Logger
logger = get_logger(name="main")

#Initialize the LLMClient
llm_client = LLMClient(model_name="meta/muse-glimmer-30b")

# Initialize Long Term Memory
long_memory = LongTermMemory(table_name="knowledge_base", persist_dir="./lancedb_data")

#Register the tools
registry = ToolRegistry()
register_agent_tools(registry=registry, long_memory=long_memory)

def get_agent_engine(session_id: str = None):
    try:
        #Initialize the short term memory
        short_memory = ShortTermMemory(llm_client=llm_client, db_path="./chat_history.db", max_tokens=32000)

        #Load the session
        short_memory.load_session(session_id=session_id)

        #Initialize the Agent Loop
        agent = AgentLoop(llm_client=llm_client,
                               tool_registry= registry,
                               short_term_memory=short_memory,
                               system_prompt=(
                                   "You are an allround helping expert. Use tools whenever necessary to answer accurately. "
                               ),
                               max_steps=7,
                               loop_threshold=3)
        
        logger.info("Agent Engine Started.")

        return agent
    
    except Exception as e: 
        logger.error("Can't Start the Agent Engine")
        raise HTTPException(status_code=500, detail="Unable start the Agent Engine")


@app.get("/")
def root():
    return {"message" : "Vada venna"}

@app.get("/test-engine")
def test_engine(agent = Depends(get_agent_engine)):
    return {"message" : "Agent Eninge is Running"}