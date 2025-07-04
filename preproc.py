import os
import io
import re
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup
import pandas as pd

def load_file(filepath):
    """Loads content from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def save_file(filepath, content):
    """Saves content to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def preprocess_datasheet(markdown_text):
    """
    Intelligently parses markdown, extracting key-value pairs, lists, and tables
    to create a highly structured text output for an LLM.
    """
    md = MarkdownIt('gfm-like') # 'gfm-like' handles tables well
    html_content = md.render(markdown_text)

    # Parse the Markdown with BeautifulSoup
    soup = BeautifulSoup(html_content, 'lxml')
    
    output_lines = []

    #Iterate through all top-level elements
    for element in soup.find('body').find_all(recursive=False):
        # Handle Headings
        if element.name in ['h1', 'h2', 'h3', 'h4']:
            output_lines.append(f"\n## {element.get_text(strip=True)}\n")
            
        # Handle Key-Value Pairs (often in <p> tags with <strong>)
        elif element.name == 'p' and element.find('strong'):

            lines = element.prettify().split('<br/>')
            for line in lines:
                line_soup = BeautifulSoup(line, 'lxml')
                strong_tag = line_soup.find('strong')
                if strong_tag and ':' in strong_tag.get_text():
                    # Extract key and the text that follows it
                    key = strong_tag.get_text(strip=True).replace(':', '').strip()
                    value = strong_tag.next_sibling
                    if value:
                        value_text = value.strip().replace(':', '').strip()
                        if key and value_text:
                            output_lines.append(f"[KEY_VALUE] {key}: {value_text}")

                # Fallback for simple paragraphs
                elif line_soup.get_text(strip=True):
                    output_lines.append(line_soup.get_text(strip=True))
        
        # -- Handle Lists --
        elif element.name in ['ul', 'ol']:
            output_lines.append("\n[LIST_START]")
            for item in element.find_all('li'):
                output_lines.append(f"- {item.get_text(strip=True)}")
            output_lines.append("[LIST_END]\n")

        # -- Handle Tables --
        elif element.name == 'table':
            try:
                table_df = pd.read_html(io.StringIO(str(element)))[0]
                output_lines.append("\n[TABLE_START]")
                output_lines.append(table_df.to_string(index=False))
                output_lines.append("[TABLE_END]\n")
            except Exception:
                # If pandas fails, just add the raw text as a fallback
                output_lines.append(element.get_text(strip=True, separator='\n'))

        # -- Handle regular paragraphs and other text --
        else:
            text = element.get_text(strip=True)
            if text:
                output_lines.append(text)

    return "\n".join(output_lines)

def main():
    """
    Main function to find and process all markdown files in the directory.
    """
    # Find all markdown files in the current directory
    markdown_files = [f for f in os.listdir('.') if f.endswith('.md')]

    if not markdown_files:
        print("No Markdown (.md) files found. Place datasheets in this directory.")
        return

    print(f"Found {len(markdown_files)} markdown files to process.")
    
    # Process each file
    for md_file in markdown_files:
        print(f"\n--- Processing '{md_file}' ---")
        
        raw_content = load_file(md_file)
        if not raw_content:
            continue
            
        cleaned_content = preprocess_datasheet(raw_content)
        
        # Create a new filename for the output
        output_filename = os.path.splitext(md_file)[0] + "_preprocessed.txt"
        
        # Save the cleaned content
        save_file(output_filename, cleaned_content)
        
        print(f"Preprocessing complete. Cleaned data saved to '{output_filename}'")

if __name__ == "__main__":
    main()