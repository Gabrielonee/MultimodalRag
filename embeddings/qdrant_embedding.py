import os
import time
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff, CollectionConfig, PointStruct
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_DIR = 'test_data/chunks'  
class QdrantServerManager:

    def __init__(self, 
                 collection_name: str = "rag_chunks", 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 host: str = "localhost",
                 port: int = 6333,
                 timeout: int = 30):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self.client = self._initialize_client()
        
        self.model = None
        self.model_name = model_name
        self._ensure_model_loaded()
        
    def _initialize_client(self) -> QdrantClient:
        logger.info(f"Connessione a Qdrant server su {self.host}:{self.port}...")
        try:
            client = QdrantClient(
                host=self.host, 
                port=self.port,
                timeout=self.timeout
            )
            health = client.get_collections()
            logger.info(f"✅ Connesso a Qdrant server - {len(health.collections)} collezioni esistenti")
            return client
            
        except Exception as e:
            logger.error(f"Impossibile connettersi al server Qdrant: {e}")
            logger.info("Docker: docker run -p 6333:6333 qdrant/qdrant")
            raise ConnectionError(f"Server Qdrant non raggiungibile: {e}")
        
    def _ensure_model_loaded(self):
        if self.model is None:
            logger.info(f"Caricamento modello {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Modello {self.model_name.split('/')[-1]} caricato")
        
    def create_collection(self, force_recreate: bool = False):
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                if force_recreate:
                    logger.info(f"Eliminazione collezione esistente '{self.collection_name}'...")
                    self.client.delete_collection(self.collection_name)
                    logger.info("Collezione eliminata")
                else:
                    logger.info(f"Collezione '{self.collection_name}' già esistente")
                    return
                    
            vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Creazione collezione con vettori di dimensione {vector_size}...")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
                optimizers_config = OptimizersConfigDiff(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=1,
                    memmap_threshold=50000,
                    indexing_threshold=20000,
                    flush_interval_sec=5,
                    max_optimization_threads=None,
                ),
                
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                    max_indexing_threads=0,
                    on_disk=False
                )
            )
            logger.info(f"Collezione '{self.collection_name}' creata")
            
        except Exception as e:
            logger.error(f"Errore nella gestione della collezione: {e}")
            raise
          
            
    def load_chunks(self, CHUNK_DIR: str) -> List[Dict[str, Any]]:
        chunks = []
        
        if not os.path.exists(CHUNK_DIR):
            logger.warning(f"Directory non trovata: {CHUNK_DIR}")
            return chunks
            
        txt_files = [f for f in os.listdir(CHUNK_DIR) if f.endswith('.txt')]
        logger.info(f"Trovati {len(txt_files)} file .txt in {CHUNK_DIR}")
        
        for filename in txt_files:
            file_path = os.path.join(CHUNK_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if content and len(content) > 10:  # Solo chunk con contenuto significativo
                    chunks.append({
                        'id': len(chunks),
                        'filename': filename,
                        'content': content,
                        'filepath': file_path
                    })
                    
            except Exception as e:
                logger.warning(f"Errore nel caricamento di {filename}: {e}")
                
        logger.info(f"✅ Caricati {len(chunks)} chunk validi")
        return chunks

    def create_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> List[List[float]]:
        if not chunks:
            return []
        logger.info(f"Generazione embedding per {len(chunks)} chunk...")
        texts = [chunk['content'] for chunk in chunks]
        
        # Processing a batch per efficienza
        embeddings = self.model.encode(
            texts, 
            convert_to_tensor=False, 
            show_progress_bar=True,
            batch_size=batch_size
        )
        
        logger.info("Embedding generati")
        return embeddings.tolist()
        
    def index_documents(self, chunk_dir: str, force_recreate: bool = True, batch_size: int = 100) -> bool:
        try:
            # Crea la collezione
            self.create_collection(force_recreate=force_recreate)
            
            # Carica i chunk
            chunks = self.load_chunks(chunk_dir)
            if not chunks:
                logger.warning("Nessun chunk trovato per l'indicizzazione")
                return False
                
            # Crea gli embedding
            embeddings = self.create_embeddings(chunks)
            if not embeddings:
                logger.error("Impossibile creare gli embedding")
                return False
                
            # Upload a batch per migliori performance
            logger.info(f"Upload {len(chunks)} punti in batch di {batch_size}...")
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                
                points = []
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    point = PointStruct(
                        id=i + j,  # ID univoco globale
                        vector=embedding,
                        payload={
                            'filename': chunk['filename'],
                            'content': chunk['content'],
                            'filepath': chunk['filepath'],
                            'chunk_id': i + j,
                            'indexed_at': time.time()
                        }
                    )
                    points.append(point)
                    
                # Upload del batch
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                logger.info(f"Upload batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} completato")
            
            # Forza il commit per assicurare la persistenza
            logger.info("Finalizzazione indicizzazione...")
            time.sleep(1)  # Breve pausa per il commit
            
            # Verifica l'inserimento
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Indicizzazione completata: {collection_info.points_count} punti totali")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore durante l'indicizzazione: {e}")
            return False
            
    def search(self, query: str, limit: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        try:
            # Verifica che la collezione esista
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.warning(f"Collezione '{self.collection_name}' non trovata")
                return []
                
            # Crea l'embedding della query
            query_embedding = self.model.encode([query])[0].tolist()
            
            # Esegui la ricerca con parametri ottimizzati
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_score,
                with_payload=True,
                with_vectors=False  # Non serve il vettore nel risultato
            )
            
            # Formatta i risultati
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'score': result.score,
                    'filename': result.payload.get('filename', ''),
                    'content': result.payload.get('content', ''),
                    'filepath': result.payload.get('filepath', ''),
                    'chunk_id': result.payload.get('chunk_id', 0),
                    'indexed_at': result.payload.get('indexed_at', 0)
                })
                
            logger.info(f"Trovati {len(formatted_results)} risultati (score >= {min_score})")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Errore durante la ricerca: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': info.points_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'config': info.config
            }
        except Exception as e:
            logger.error(f"Errore nel recupero info collezione: {e}")
            return {}

    def delete_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collezione '{self.collection_name}' eliminata")
        except Exception as e:
            logger.error(f"Errore nell'eliminazione: {e}")

def create_embeddings_and_index(chunk_dir: str, 
                                collection_name: str = "rag_chunks",
                                host: str = "localhost",
                                port: int = 6333) -> Optional[QdrantServerManager]:
        try:
            manager = QdrantServerManager(
                collection_name=collection_name,
                host=host,
                port=port
            )
            success = manager.index_documents(chunk_dir, force_recreate=True)
            
            if success:
                logger.info(f"✅ Embedding e indicizzazione completati per {chunk_dir}")
                return manager
            else:
                logger.error(f"❌ Fallimento nell'embedding e indicizzazione per {chunk_dir}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Errore nella creazione del manager: {e}")
            return None

if __name__ == "__main__":
    print("=== TEST QDRANT SERVER ===")
    
    try:
        # Test connessione
        manager = QdrantServerManager(collection_name="test_collection")
        
        # Crea directory e file di test
        os.makedirs("test_chunks", exist_ok=True)
        
        test_docs = [
            "Questo è un documento di test per il sistema RAG con Qdrant server.",
            "Python è un linguaggio di programmazione versatile e potente.",
            "Il machine learning sta rivoluzionando molti settori dell'industria."
        ]
        
        for i, doc in enumerate(test_docs):
            with open(f"test_chunks/test_{i}.txt", "w", encoding="utf-8") as f:
                f.write(doc)
        
        # Test del sistema
        success = manager.index_documents("test_chunks", force_recreate=True)
        
        if success:
            # Test ricerca
            results = manager.search("documento test", limit=2)
            if results:
                print(f"\n✅ Test riuscito! Trovati {len(results)} risultati")
                for r in results:
                    print(f"   Score: {r['score']:.3f} - {r['content'][:50]}...")
                    
                # Info collezione
                info = manager.get_collection_info()
                print(f"\n📊 Info collezione: {info['points_count']} documenti indicizzati")
            else:
                print("\n❌ Nessun risultato trovato")
        
        # Pulizia
        for i in range(len(test_docs)):
            if os.path.exists(f"test_chunks/test_{i}.txt"):
                os.remove(f"test_chunks/test_{i}.txt")
        if os.path.exists("test_chunks"):
            os.rmdir("test_chunks")
            
        manager.delete_collection()  
    except ConnectionError as e:
        print(f"\n❌ {e}")
        print("\n💡 Per avviare Qdrant server:")
        print("   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")