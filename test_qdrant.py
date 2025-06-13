from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import os

def test_qdrant_simple():
    # Setup
    client = QdrantClient(":memory:")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection_name = "test_rag"
    
    # Crea collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # CORREZIONE 1: Leggi il contenuto del file
    file_path = 'test_data/raw/rag.txt'
    if not os.path.exists(file_path):
        print(f"[ERROR] File    non trovato: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    # CORREZIONE 2: Dividi il testo in chunk logici
    # Per semplicità, dividi per paragrafi o frasi
    texts = [paragraph.strip() for paragraph in file_content.split('\n\n') if paragraph.strip()]
    
    # Se non ci sono paragrafi, dividi per frasi
    if len(texts) <= 1:
        import re
        texts = [sentence.strip() for sentence in re.split(r'[.!?]+', file_content) if sentence.strip()]
    
    print(f"[INFO] Trovati {len(texts)} chunk di testo")
    
    # CORREZIONE 3: Genera embeddings per tutti i testi
    embeddings = model.encode(texts)
    
    # CORREZIONE 4: Crea i punti correttamente
    points = []
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),  # embedding è già un array numpy
            payload={"text": text, "chunk_id": i}
        ))
    
    print(f"[INFO] Creati {len(points)} punti per Qdrant")
    
    # Inserisci i dati
    client.upsert(collection_name=collection_name, points=points)
    
    # Test ricerca
    query = "What is RAG?"
    query_embedding = model.encode([query])[0]  # Encode ritorna un array, prendi il primo
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=3,
        with_payload=True
    )
    
    print(f"\n[RISULTATI] Query: '{query}'")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.score:.4f}")
        print(f"   Chunk ID: {result.payload['chunk_id']}")
        print(f"   Testo: {result.payload['text'][:100]}...\n")
    
    print("[SUCCESS] Test Qdrant completato!")

def test_qdrant_with_sample_data():
    """
    Versione di test con dati di esempio se il file non esiste
    """
    # Setup
    client = QdrantClient(":memory:")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection_name = "test_rag"
    
    # Crea collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # Dati di esempio
    sample_texts = [
        "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval and generation.",
        "RAG systems first retrieve relevant documents from a knowledge base.",
        "The retrieved documents are then used to generate more accurate and informed responses.",
        "Vector embeddings are used to find semantically similar content in RAG systems.",
        "RAG helps reduce hallucination in large language models by grounding responses in factual data."
    ]
    
    print(f"[INFO] Usando {len(sample_texts)} testi di esempio")
    
    # Genera embeddings
    embeddings = model.encode(sample_texts)
    
    # Crea punti
    points = []
    for i, (text, embedding) in enumerate(zip(sample_texts, embeddings)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={"text": text, "chunk_id": i}
        ))
    
    # Inserisci i dati
    client.upsert(collection_name=collection_name, points=points)
    
    # Test multiple query
    queries = ["What is RAG?", "How does retrieval work?", "What are embeddings?"]
    
    for query in queries:
        query_embedding = model.encode([query])[0]
        
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=2,
            with_payload=True
        )
        
        print(f"\n[RISULTATI] Query: '{query}'")
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result.score:.4f}")
            print(f"   Testo: {result.payload['text']}")
        print("-" * 50)
    
    print("[SUCCESS] Test Qdrant con dati di esempio completato!")

if __name__ == "__main__":
    # Prova prima con il file reale
    try:
        test_qdrant_simple()
    except FileNotFoundError:
        print("[INFO] File non trovato, usando dati di esempio...")
        test_qdrant_with_sample_data()