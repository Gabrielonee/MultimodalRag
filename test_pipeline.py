import os
import time
from data.clean_text import clean_text_file
from data.extract_text import extract_text_from_pdf, save_text_to_file, process_pdf
from embeddings.qdrant_embedding import create_embeddings_and_index
def main():
    RAW_DIR = "test_data/raw"
    CLEAN_DIR = "test_data/clean"
    CHUNK_DIR = "test_data/chunks"
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)
    
    print("PIPELINE DI ELABORAZIONE DOCUMENTI")
    
    #Step 1: Pulizia del testo
    print("\n1. PULIZIA DEL TESTO")
    print("-" * 40)
    try:
        pdf = os.path.join(RAW_DIR, "rag.pdf")
        pdf_proc = process_pdf(pdf, output_dir=RAW_DIR)
        clean_file = os.path.join(CLEAN_DIR, "sample_clean.txt")
        
        if os.path.exists(pdf_proc):
            clean_text_file(pdf_proc, clean_file)
        else:
            print(f"[WARNING] File non trovato: {pdf_proc}")
            
    except Exception as e:
        print(f"[ERROR] Errore nella pulizia: {e}")
        return False
        
    #Step 2: Preparazione chunk
    print("\n2. PREPARAZIONE CHUNK SEMANTICI")
    print("-" * 40)
    try:
        from data.chunk import chunk_text_file_semantic
        
        clean_file = os.path.join(CLEAN_DIR, "sample_clean.txt")
        
        if os.path.exists(clean_file):
            chunk_text_file_semantic(
                clean_file, 
                output_dir=CHUNK_DIR,
                method="paragraphs",
                target_size=400,      
                max_size=600 
            )
            print(f"[DONE] Chunk semantici creati in: {CHUNK_DIR}")
        else:
            print(f"[ERROR] File pulito non trovato: {clean_file}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Errore nella creazione chunk: {e}")
        return False
    
    print("\n3. EMBEDDING E INDICIZZAZIONE")
    print("-" * 40)
    try:
        manager = create_embeddings_and_index(CHUNK_DIR, collection_name="rag_chunks")
        
        if not manager:
            print("[ERROR] Impossibile creare il manager per gli embedding")
            return False
            
        # Piccola pausa per assicurare la sincronizzazione
        time.sleep(1)
        
        # Step 4: Test di ricerca
        print("\n4. TEST DI RICERCA")
        print("-" * 40)
        
        test_queries = [
            "What is RAG",
            "What is embedding", 
        ]
        
        for query in test_queries:
            print(f"\n🔍 Ricerca: '{query}'")
            results = manager.search(query, limit=3, min_score=0.1)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Score: {result['score']:.3f}")
                    print(f"     File: {result['filename']}")
                    print(f"     Contenuto: {result['content'][:100]}...")
                    if i < len(results):
                        print()
            else:
                print("Nessun risultato trovato")
                
        return True
        
    except Exception as e:
        print(f"[ERROR] Errore nell'embedding o ricerca: {e}")
        return False
    

if __name__ == "__main__":
    success = main()
    
    print("\n" + "="*50)
    if success:
        print("PIPELINE COMPLETATA CON SUCCESSO!")
        
    else:
        print("PIPELINE FALLITA")
    print("="*50)