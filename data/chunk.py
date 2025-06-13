import os
import re
from typing import List
from nltk.tokenize import sent_tokenize


def semantic_chunk_by_sentences(text: str, target_size: int = 500, max_size: int = 800) -> List[str]:
    # Dividi in frasi
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_word_count + sentence_words > max_size and current_chunk:
            # Salva il chunk corrente
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            # Aggiungi la frase al chunk corrente
            current_chunk.append(sentence)
            current_word_count += sentence_words
            
            # Se abbiamo raggiunto la dimensione target, chiudo chunk
            if current_word_count >= target_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
    
    # Aggiungo un altro se non è vuoto
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def semantic_chunk_by_paragraphs(text: str, target_size: int = 500, max_size: int = 800) -> List[str]:

    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for paragraph in paragraphs:
        paragraph_words = len(paragraph.split())
        
        # Se il paragrafo da solo è troppo grande, dividilo per frasi
        if paragraph_words > max_size:
            # Salva il chunk corrente se non è vuoto
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            # Dividi il paragrafo lungo in chunk per frasi
            para_chunks = semantic_chunk_by_sentences(paragraph, target_size, max_size)
            chunks.extend(para_chunks)
            continue
        
        # Se aggiungere questo paragrafo supera la dimensione massima
        if current_word_count + paragraph_words > max_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_word_count = paragraph_words
        else:
            current_chunk.append(paragraph)
            current_word_count += paragraph_words
            
            # Se abbiamo raggiunto la dimensione target
            if current_word_count >= target_size:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_word_count = 0
    
    # Aggiungi l'ultimo chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def smart_chunk_by_structure(text: str, target_size: int = 500, max_size: int = 800) -> List[str]:
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_words = len(line.split())
        
        # Rileva possibili titoli o sezioni (linee brevi, maiuscole, o con pattern specifici)
        is_header = (
            line_words < 10 and 
            (line.isupper() or 
             line.startswith('---') or 
             re.match(r'^(Chapter|Section|\d+\.)', line, re.IGNORECASE))
        )
        
        # Se è un header e abbiamo già contenuto, chiudi il chunk precedente
        if is_header and current_chunk and current_word_count > 100:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_word_count = line_words
        else:
            # Controlla se aggiungere questa linea supera la dimensione massima
            if current_word_count + line_words > max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_word_count = line_words
            else:
                current_chunk.append(line)
                current_word_count += line_words
                
                # Se abbiamo raggiunto la dimensione target
                if current_word_count >= target_size:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
    
    # Aggiungi l'ultimo chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def chunk_text_semantic(text: str, method: str = "paragraphs", target_size: int = 500, max_size: int = 800) -> List[str]:
    if method == "sentences":
        return semantic_chunk_by_sentences(text, target_size, max_size)
    elif method == "paragraphs":
        return semantic_chunk_by_paragraphs(text, target_size, max_size)
    elif method == "structure":
        return smart_chunk_by_structure(text, target_size, max_size)
    else:
        raise ValueError("Method deve essere 'sentences', 'paragraphs', o 'structure'")

def chunk_text_file_semantic(input_path: str, output_dir: str = "output_chunks", 
                           method: str = "paragraphs", target_size: int = 500, max_size: int = 800):
    """
    Applica chunking semantico a un file di testo.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = chunk_text_semantic(text, method, target_size, max_size)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    print(f"Creati {len(chunks)} chunk semantici con metodo '{method}'")
    
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(output_dir, f"{base_name}_semantic_{method}_{i:03d}.txt")
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(chunk)
            
        # Info sul chunk creato
        word_count = len(chunk.split())
        print(f"  Chunk {i+1}: {word_count} parole")


def test_semantic_chunking():
    sample_text = """
    Chapter 1: Introduction to RAG
    
    Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of retrieval-based and generation-based approaches in natural language processing.
    
    The main idea behind RAG is to enhance the generation process by incorporating relevant information retrieved from a knowledge base or document collection. This approach addresses some of the limitations of pure generative models, such as hallucination and outdated information.
    
    Chapter 2: How RAG Works
    
    RAG operates in two main phases: retrieval and generation. In the retrieval phase, relevant documents or passages are identified based on the input query. In the generation phase, these retrieved documents are used to inform and guide the text generation process.
    
    The retrieval component typically uses dense vector representations (embeddings) to find semantically similar content. These embeddings are created using pre-trained language models and stored in vector databases for efficient similarity search.
    """    
    print("CHUNKING PER FRASI")
    chunks_sentences = chunk_text_semantic(sample_text, method="sentences", target_size=100, max_size=150)
    for i, chunk in enumerate(chunks_sentences):
        print(f"Chunk {i+1} ({len(chunk.split())} parole): {chunk[:100]}")
    
    print("\nCHUNKING PER PARAGRAFI")
    chunks_paragraphs = chunk_text_semantic(sample_text, method="paragraphs", target_size=100, max_size=200)
    for i, chunk in enumerate(chunks_paragraphs):
        print(f"Chunk {i+1} ({len(chunk.split())} parole): {chunk[:100]}")
    
    print("\nCHUNKING PER STRUTTURA")
    chunks_structure = chunk_text_semantic(sample_text, method="structure", target_size=100, max_size=200)
    for i, chunk in enumerate(chunks_structure):
        print(f"Chunk {i+1} ({len(chunk.split())} parole): {chunk[:100]}...")

if __name__ == "__main__":
    test_semantic_chunking()