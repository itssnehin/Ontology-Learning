import logging
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

# --- THIS IS THE FIX ---
# Use '..' to go up from 'evaluation' to the 'src' package
from ..config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    OPENAI_API_KEY, LLM_MODEL
)

logger = logging.getLogger(__name__)

def run_qa_system():
    """
    Initializes and runs a Question-Answering system based on the Neo4j knowledge graph.
    """
    logger.info("--- Initializing Knowledge Graph QA System ---")

    # --- 1. Connect to the Neo4j Database ---
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        # Refresh the schema to ensure the QA chain has the latest graph structure
        graph.refresh_schema()
        logger.info("✅ Successfully connected to Neo4j and refreshed schema.")
        logger.info(f"   Graph Schema:\n{graph.schema}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j. Please ensure the database is running. Error: {e}", exc_info=True)
        return

    # --- 2. Set up the Language Model and QA Chain ---
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
        
        # This chain combines the LLM and the graph to answer questions
        chain = GraphCypherQAChain.from_llm(
            graph=graph,
            llm=llm,
            verbose=True, # Set to True to see the generated Cypher queries
            allow_dangerous_requests=True
        )
        logger.info("✅ QA Chain initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LangChain components. Check OpenAI API key. Error: {e}", exc_info=True)
        return

    # --- 3. Start the Interactive Q&A Loop ---
    print("\n--- Knowledge Graph QA System is Ready ---")
    print("Ask questions about your electronic components ontology.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            question = input("\n> ")
            if question.lower() in ['exit', 'quit']:
                print("Exiting QA system. Goodbye!")
                break
            
            if not question:
                continue

            print("Thinking...")
            
            # --- 4. Invoke the Chain and Get the Answer ---
            result = chain.invoke({"query": question})
            
            print("\n---")
            print(f"Question: {question}")
            print(f"Answer: {result['result']}")
            print("---")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            logger.error(f"Error during QA invocation: {e}", exc_info=True)

if __name__ == "__main__":
    run_qa_system()