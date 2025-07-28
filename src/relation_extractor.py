from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.runnables import RunnableSequence
from config import LLM_MODEL, OPENAI_API_KEY
from typing import List

def extract_relations(chunks: List) -> List[str]:
    """
    Extract implicit relations from document chunks using OpenAI LLM.
    
    Args:
        chunks: List of document chunks (LangChain Document objects) with Markdown text and metadata.
    
    Returns:
        List of relation triples in the format "concept1 -> relation -> concept2".
    """
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=500
    )
    
    # Define prompt with IoT-specific examples and Markdown handling
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""You are an expert in extracting implicit relations from Markdown text for IoT and antenna specifications. Extract relations in the format:
{{concept1}} -> {{relation}} -> {{concept2}}
Interpret Markdown headers (##), bullet points (·), and tables to infer technical relations. For introductory text, extract high-level relations. For feature lists or specs, infer feature-based relations.

Examples:
- WiFi 6E -> supports -> 6 GHz Band
- FPC Antenna -> has-feature -> U.FL Connector
- Dipole Antenna -> has-property -> Ground Plane Independent
- Antenna -> has-VSWR -> 1.4
- ANT-W63-FPC-LH -> operates-in -> 2.4 GHz Band

Text:
```
{text}
```

Output exactly in this format, one relation per line starting with '-':
- {{concept1}} -> {{relation}} -> {{concept2}}
If no relations are found, output exactly:
- None -> none -> None
"""
    )
    
    # Create RunnableSequence
    chain = prompt | llm
    
    relations = []
    
    # Process each chunk with cost tracking
    with get_openai_callback() as cb:
        for chunk in chunks:
            try:
                # Skip non-technical chunks (e.g., warranty, temperature)
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
                found_relations = False
                for line in result_text.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("-"):
                        triple = line[2:].strip()
                        if " -> " in triple:
                            parts = triple.split(" -> ")
                            if len(parts) == 3 and all(parts) and parts != ["None", "none", "None"]:
                                relations.append(triple)
                                found_relations = True
                
                # Log results
                print(f"Raw LLM output: {result_text}")
                if not found_relations:
                    print(f"No relations extracted from chunk: {chunk_text[:100]}...")
            
            except Exception as e:
                print(f"Error extracting relations from chunk: {e}")
        
        print(f"OpenAI API Usage: Tokens={cb.total_tokens}, Cost=${cb.total_cost:.4f}")
    
    return relations

if __name__ == "__main__":
    from data_loader import load_and_split_data
    
    # Load sample chunks
    chunks = load_and_split_data()
    
    # Extract relations (limit to 10 chunks for testing)
    relations = extract_relations(chunks[:10])
    print("Extracted Relations:", relations)