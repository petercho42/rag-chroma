from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from langchain_community.tools import DuckDuckGoSearchRun

model = ChatOllama(
    model="glm-4.7-flash:latest", 
    base_url="http://localhost:11434",
)


# Initialize the tool
web_search = DuckDuckGoSearchRun()

# Define it as a tool for your agent
@tool
def search_internet(query: str):
    """
    Searches the internet for current news, general facts, or AI research 
    not found in your local PDF. Use this for 'latest' information.
    """
    print(f"--- DEBUG: Agent is searching the web for: {query} ---")
    return web_search.run(query)

class SearchInput(BaseModel):
    query: str = Field(description="The technical term to look up in the PDF")

@tool(args_schema=SearchInput)
def search_pdf(query: str):
    """Searches the Transformer paper for specific values."""
    from query3 import pdf_search_tool
    print(f"--- DEBUG: Agent is searching the PDF for: {query} ---")
    return pdf_search_tool(query)

class CalculatorInput(BaseModel):
    # We rename 'expression' to 'query' because LLMs are 
    # extremely biased toward using 'query' as a default key.
    query: str = Field(description="The math expression to calculate, e.g. '512 * 5'")

@tool(args_schema=CalculatorInput)
def calculate(query: str):
    """Useful for doing math. Provide the math problem as the 'query'."""
    from query3 import calculator_tool
    # We still pass it to your original function
    print(f"--- DEBUG: Agent is calculating: {query} ---")
    return calculator_tool(query)

tools = [search_pdf, calculate, search_internet]

instructions = (
    "You are a methodical research assistant. "
    "To use a tool, you MUST output a JSON object with 'name' and 'arguments'. "
    "First, use search_pdf. Then, once you get the result, use calculate."
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=instructions
)

print("--- Starting Agent Execution ---")
result = agent.invoke({"messages": [("user", "What is d_model in my PDF, and how does that compare to the dimension used in the newest Llama 4 model?")]})

for msg in result["messages"]:
    print(f"{type(msg).__name__}: {msg.content}")