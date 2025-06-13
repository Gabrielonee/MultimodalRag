from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    CollectionConfig,
    HnswConfigDiff,
)
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_chunks"

class QdrantRetriever:
    def __init__(self, use_memory=True):
        if use_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(host="localhost", port=6333)

        self.model = SentenceTransformer(MODEL_NAME)

        # Verifica che la collection esista, altrimenti la crea
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if COLLECTION_NAME not in collection_names:
                print(f"[INFO] Collection '{COLLECTION_NAME}' non trovata. Creazione in corso...")
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000
                    )
                )
                print(f"[INFO] ✅ Collection '{COLLECTION_NAME}' creata con successo.")
            else:
                print(f"[INFO] Collection '{COLLECTION_NAME}' trovata.")
        except Exception as e:
            print(f"[ERROR] Errore nel controllo/creazione della collection: {e}")

    def search(self, query: str, top_k: int = 5):
        try:
            # Genera embedding per la query
            query_embedding = self.model.encode([query])[0]

            # Esegue la ricerca
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                with_payload=True
            )

            # Formatta i risultati per compatibilità
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'text': result.payload.get('text', ''),
                    'score': result.score,
                    'id': result.id
                })

            return formatted_results

        except Exception as e:
            print(f"[ERROR] Errore durante la ricerca: {e}")
            return []
