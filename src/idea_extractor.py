from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.runnables import RunnableSequence
from config import LLM_MODEL, OPENAI_API_KEY
from typing import List
"""
    Ideas become the nodes of an ontology graph
    Will be used for similarity search later    
"""
def extract_ideas(chunks: List) -> List[str]:
    """
    Extract high-level themes or concepts from document chunks using OpenAI LLM.
    
    Args:
        chunks: List of document chunks (LangChain Document objects) with Markdown text and metadata.
    
    Returns:
        List of themes (e.g., "high-gain antenna design", "WiFi 6E compatibility").
    """
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.7, # Lower temp to focus on words in the corpus
        openai_api_key=OPENAI_API_KEY,
        max_tokens=300  # Smaller limit for concise themes
    )
    
    # Define prompt for theme extraction
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""You are an expert in identifying high-level themes or concepts from Markdown text for IoT and antenna specifications. Extract key themes or ideas, focusing on technical concepts, features, or applications. Use Markdown headers, bullet points, and tables as context.

Examples:
- Text: "## Features\nWiFi 6, 2.4 GHz support, U.FL connector" -> Themes: ["WiFi 6 compatibility", "U.FL connector usage"]
- Text: "Ground plane independent dipole antenna" -> Themes: ["ground plane independence", "dipole antenna design"]

Text:
```
{text}
```

Output exactly in this format, one theme per line starting with '-':
- {{theme}}
If no themes are found, output exactly:
- None
"""
    )
    
    # Create RunnableSequence
    chain = prompt | llm
    
    themes = []
    
    # Process each chunk with cost tracking
    with get_openai_callback() as cb:
        for chunk in chunks:
            try:
                # Skip non-technical chunks
                if any(keyword in chunk.page_content.lower() for keyword in ["warranty", "©", "temperature"]):
                    print(f"Skipping non-technical chunk: {chunk.page_content[:100]}...")
                    continue
                
                # Log chunk content and metadata
                print(f"Processing chunk: {chunk.page_content[:200]}...")
                if hasattr(chunk, 'metadata'):
                    print(f"Chunk metadata: {chunk.metadata}")
                
                # Ensure text is properly formatted
                chunk_text = chunk.page_content.strip()
                if not chunk_text:
                    print("Skipping empty chunk")
                    continue
                
                # Run chain with invoke
                result = chain.invoke({"text": chunk_text})
                
                # Extract content from LLM response
                result_text = result.content if hasattr(result, 'content') else str(result)
                
                # Parse output
                found_themes = False
                for line in result_text.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("-"):
                        theme = line[2:].strip()
                        if theme and theme != "None":
                            themes.append(theme)
                            found_themes = True
                
                # Log results
                print(f"Raw LLM output: {result_text}")
                if not found_themes:
                    print(f"No themes extracted from chunk: {chunk_text[:100]}...")
            
            except Exception as e:
                print(f"Error extracting themes from chunk: {e}")
        
        print(f"OpenAI API Usage: Tokens={cb.total_tokens}, Cost=${cb.total_cost:.4f}")
    
    return themes

if __name__ == "__main__":
    from data_loader import load_and_split_data
    
    # Load sample chunks
    chunks = load_and_split_data()
    
    # Extract themes (limit to 10 chunks for testing)
    themes = extract_ideas(chunks[:10])
    print("Extracted Themes:", themes)