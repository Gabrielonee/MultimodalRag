import os
import shutil
import time
import hashlib
from typing import Dict
from dataclasses import dataclass, field
import json
from pathlib import Path

# Import dei moduli esistenti
from data.clean_text import clean_text_file
from data.extract_text import process_pdf
from data.chunk import chunk_text_file_semantic
from embeddings.qdrant_embedding import QdrantServerManager
from retrieval.retriever import QdrantRetriever

try:
    from generation.generator import RAGGenerator
    GENERATION_AVAILABLE = True
except ImportError:
    print("[WARNING] Modulo generazione non disponibile. Installa transformers e torch.")
    GENERATION_AVAILABLE = False


@dataclass
class DocumentInfo:
    """Informazioni su un documento processato."""
    pdf_path: str
    pdf_hash: str
    processed_at: str
    chunk_count: int
    file_size: int
    status: str = "processed"


@dataclass
class RAGConfig:
    # Chunking
    chunk_method: str = "paragraphs"
    target_size: int = 400
    max_size: int = 600
    
    # Retrieval
    top_k: int = 5
    min_score: float = 0.3
    
    # Generation
    model_name: str = "phi-3-mini"
    max_length: int = 512
    temperature: float = 0.7
    max_context_length: int = 2000
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "dynamic_rag_docs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Directories
    base_dir: str = "rag_workspace"
    
    # Document tracking
    processed_docs: Dict[str, DocumentInfo] = field(default_factory=dict)


class DynamicPDFRAGSystem:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.retriever = None
        self.generator = None
        self.qdrant_manager = None
        self.initialized = False
        
        # Setup directories
        self._setup_workspace()
        
        # Carica lo stato dei documenti processati
        self._load_document_state()
    
    def _setup_workspace(self):
        base_path = Path(self.config.base_dir)
        
        self.paths = {
            'base': base_path,
            'uploads': base_path / "uploads",
            'raw': base_path / "raw",
            'clean': base_path / "clean", 
            'chunks': base_path / "chunks",
            'metadata': base_path / "metadata"
        }
        
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        print(f"Workspace configurato in: {base_path.absolute()}")
    
    def _get_pdf_hash(self, pdf_path: str) -> str:
        with open(pdf_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _save_document_state(self):
        state_file = self.paths['metadata'] / "processed_docs.json"
        # Converti DocumentInfo in dict per serializzazione
        serializable_docs = {}
        for doc_id, doc_info in self.config.processed_docs.items():
            serializable_docs[doc_id] = {
                'pdf_path': doc_info.pdf_path,
                'pdf_hash': doc_info.pdf_hash,
                'processed_at': doc_info.processed_at,
                'chunk_count': doc_info.chunk_count,
                'file_size': doc_info.file_size,
                'status': doc_info.status
            }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
    
    def _load_document_state(self):
        state_file = self.paths['metadata'] / "processed_docs.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                
                # Ricostruisci DocumentInfo objects
                for doc_id, doc_data in docs_data.items():
                    self.config.processed_docs[doc_id] = DocumentInfo(**doc_data)
                
                print(f"Caricati {len(self.config.processed_docs)} documenti precedenti")
            except Exception as e:
                print(f"Errore nel caricamento stato documenti: {e}")
    
    #DA AGGIORNARE, QUESTA PRENDERA IN INPUT IL PDF COME ALLEGATO
    def add_pdf_from_path(self, pdf_path: str, force_reprocess: bool = False) -> bool:
        if not os.path.exists(pdf_path):
            print(f"File non trovato: {pdf_path}")
            return False
        if not pdf_path.lower().endswith('.pdf'):
            print(f"Il file deve essere un PDF: {pdf_path}")
            return False
        
        try:
            # Calcola hash del PDF
            pdf_hash = self._get_pdf_hash(pdf_path)
            pdf_name = os.path.basename(pdf_path)
            
            # Controlla se già processato
            if pdf_hash in self.config.processed_docs and not force_reprocess:
                print(f"PDF già processato: {pdf_name}")
                return True
            
            print(f"Processamento PDF: {pdf_name}")
            
            # Copia il PDF nella directory uploads
            upload_path = self.paths['uploads'] / pdf_name
            shutil.copy2(pdf_path, upload_path)
            
            # Processa il documento
            success = self._process_single_pdf(str(upload_path), pdf_hash)
            
            if success:
                # Aggiorna lo stato
                doc_info = DocumentInfo(
                    pdf_path=str(upload_path),
                    pdf_hash=pdf_hash,
                    processed_at=str(time.time()),
                    chunk_count=self._count_chunks_for_pdf(pdf_name),
                    file_size=os.path.getsize(pdf_path)
                )
                
                self.config.processed_docs[pdf_hash] = doc_info
                self._save_document_state()
                
                print(f" PDF processato con successo: {pdf_name}")
                return True
            else:
                print(f" Errore nel processamento: {pdf_name}")
                return False
                
        except Exception as e:
            print(f"Errore nell'aggiunta del PDF: {e}")
            import traceback
            traceback.print_exc()
            return False
    


    def add_pdfs_from_directory(self, directory_path: str, force_reprocess: bool = False) -> Dict[str, bool]:
        if not os.path.exists(directory_path):
            print(f" Directory non trovata: {directory_path}")
            return {}
        
        results = {}
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"📂 Nessun PDF trovato in: {directory_path}")
            return {}
        
        print(f"📚 Trovati {len(pdf_files)} PDF da processare")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            success = self.add_pdf_from_path(pdf_path, force_reprocess)
            results[pdf_file] = success
        
        return results
    
    #DA AGGIORNARE CON UI
    def upload_pdf_interactive(self) -> bool:
        print("\n📁 CARICAMENTO PDF INTERATTIVO")
        print("=" * 40)
        
        while True:
            print("\nOpzioni:")
            print("1. Aggiungi PDF singolo (percorso)")
            print("2. Aggiungi tutti i PDF da una directory")
            print("3. Mostra PDF caricati")
            print("4. Continua con il sistema RAG")
            
            choice = input("\nScelta (1-4): ").strip()
            
            if choice == "1":
                pdf_path = input("Inserisci il percorso del PDF: ").strip()
                if pdf_path:
                    self.add_pdf_from_path(pdf_path)
            
            elif choice == "2":
                dir_path = input("Inserisci il percorso della directory: ").strip()
                if dir_path:
                    results = self.add_pdfs_from_directory(dir_path)
                    successful = sum(1 for success in results.values() if success)
                    print(f"📊 Risultato: {successful}/{len(results)} PDF processati con successo")
            
            elif choice == "3":
                self.show_loaded_documents()
            
            elif choice == "4":
                break
            
            else:
                print(" Scelta non valida")
        
        return len(self.config.processed_docs) > 0
    
    def _process_single_pdf(self, pdf_path: str, pdf_hash: str) -> bool:
        try:
            # Step 1: Estrazione testo
            raw_text_path = process_pdf(pdf_path, output_dir=str(self.paths['raw']))
            
            # Step 2: Pulizia testo
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            clean_path = self.paths['clean'] / f"{base_name}_clean.txt"
            clean_text_file(raw_text_path, str(clean_path))
            
            # Step 3: Chunking
            chunk_text_file_semantic(
                str(clean_path),
                output_dir=str(self.paths['chunks']),
                method=self.config.chunk_method,
                target_size=self.config.target_size,
                max_size=self.config.max_size
            )
            return True
        except Exception as e:
            print(f" Errore nel processamento del PDF: {e}")
            return False
    
    def _count_chunks_for_pdf(self, pdf_name: str) -> int:
        base_name = os.path.splitext(pdf_name)[0]
        chunk_files = list(self.paths['chunks'].glob(f"{base_name}_clean_semantic_*"))
        return len(chunk_files)
    
    def initialize_rag_components(self) -> bool:
        if not self.config.processed_docs:
            print(" Nessun documento caricato. Carica almeno un PDF prima di inizializzare.")
            return False
        
        try:
            print("🔄 Inizializzazione componenti RAG con Qdrant...")
            
            # Step 1: Inizializza Qdrant Manager
            print("1️⃣  Connessione a Qdrant...")
            self.qdrant_manager = QdrantServerManager(
                collection_name=self.config.collection_name,
                model_name=self.config.embedding_model,
                host=self.config.qdrant_host,
                port=self.config.qdrant_port
            )
            
            # Step 2: Indicizza i documenti
            print("2️⃣  Indicizzazione documenti...")
            success = self.qdrant_manager.index_documents(
                str(self.paths['chunks']),
                force_recreate=True
            )
            
            if not success:
                raise RuntimeError("Errore nell'indicizzazione dei documenti")
            
            # Step 3: NON inizializzare QdrantRetriever - usa direttamente il manager
            print("3️⃣  Configurazione retriever...")
            # Il retriever sarà il qdrant_manager stesso
            self.retriever = self.qdrant_manager
            
            # Step 4: Inizializza generatore (se disponibile)
            if GENERATION_AVAILABLE:
                print("4️⃣  Inizializzazione generatore...")
                self.generator = RAGGenerator(
                    model_name=self.config.model_name,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature
                )
            else:
                print("⚠️  Generatore non disponibile - solo retrieval")
            
            self.initialized = True
            
            # Mostra statistiche Qdrant
            info = self.qdrant_manager.get_collection_info()
            print(f"✅ Sistema RAG inizializzato con successo!")
            print(f"📊 Collezione Qdrant: {info.get('points_count', 0)} documenti indicizzati")
            return True
            
        except Exception as e:
            print(f" Errore nell'inizializzazione: {e}")
            import traceback
            traceback.print_exc()
            return False


    def ask_question(self, question: str) -> Dict[str, any]:
        if not self.initialized:
            return {
                'question': question,
                'answer': "Sistema non inizializzato. Carica dei documenti e inizializza i componenti RAG.",
                'sources': [],
                'confidence': 0.0
            }
        try:
            # Retrieval usando direttamente Qdrant Manager
            print("🔍 Ricerca documenti rilevanti...")
            retrieved_docs = self.qdrant_manager.search(
                question, 
                limit=self.config.top_k,
                min_score=self.config.min_score
            )
            
            if not retrieved_docs:
                return {
                    'question': question,
                    'answer': "Non ho trovato informazioni rilevanti nei documenti caricati per rispondere alla tua domanda.",
                    'sources': [],
                    'confidence': 0.0,
                    'retrieved_docs': 0
                }
            
            # Generation (se disponibile)
            if self.generator:
                # Adatta il formato per il generatore
                docs_for_generation = []
                for doc in retrieved_docs:
                    docs_for_generation.append({
                        'text': doc.get('content', ''),
                        'filename': doc.get('filename', ''),
                        'score': doc.get('score', 0)
                    })
                
                response_data = self.generator.generate_response(
                    question, 
                    docs_for_generation,
                    max_context_length=self.config.max_context_length
                )
                
                avg_score = sum(doc.get('score', 0) for doc in retrieved_docs) / len(retrieved_docs)
                
                return {
                    'question': question,
                    'answer': response_data['response'],
                    'sources': response_data['sources'],
                    'confidence': avg_score,
                    'retrieved_docs': len(retrieved_docs),
                    'model_used': response_data.get('model', 'N/A')
                }
            else:
                # Solo retrieval - restituisci i documenti più rilevanti
                context_parts = []
                for i, doc in enumerate(retrieved_docs[:3], 1):
                    content = doc.get('content', '')[:200]
                    score = doc.get('score', 0)
                    filename = doc.get('filename', 'Unknown')
                    context_parts.append(f"[{i}] {filename} (Score: {score:.3f})\n{content}...")
                
                return {
                    'question': question,
                    'answer': f"Documenti rilevanti trovati:\n\n" + "\n\n".join(context_parts),
                    'sources': [doc.get('filename', 'Unknown') for doc in retrieved_docs[:3]],
                    'confidence': sum(doc.get('score', 0) for doc in retrieved_docs) / len(retrieved_docs),
                    'retrieved_docs': len(retrieved_docs)
                }
                
        except Exception as e:
            print(f" Errore nella ricerca: {e}")
            return {
                'question': question,
                'answer': f"Errore nell'elaborazione della domanda: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'retrieved_docs': 0
            }
    
    def interactive_session(self):
        """Avvia una sessione interattiva completa."""
        print("\n🤖 SISTEMA RAG DINAMICO CON QDRANT")
        print("=" * 50)
        
        # Verifica connessione Qdrant
        self._test_qdrant_connection()
        
        # Step 1: Caricamento PDF
        if not self.config.processed_docs:
            print("📚 Nessun documento caricato.")
            if not self.upload_pdf_interactive():
                print(" Nessun documento caricato. Uscita.")
                return
        else:
            print(f"📋 {len(self.config.processed_docs)} documenti già caricati")
            self.show_loaded_documents()
            
            load_more = input("\nVuoi caricare altri PDF? (y/n): ").strip().lower()
            if load_more == 'y':
                self.upload_pdf_interactive()
        
        # Step 2: Inizializzazione RAG
        if not self.initialized:
            print("\n🔧 Inizializzazione sistema RAG...")
            if not self.initialize_rag_components():
                print(" Errore nell'inizializzazione. Uscita.")
                return
        
        # Step 3: Chat interattiva
        self._start_chat()
    
    def _test_qdrant_connection(self):
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=self.config.qdrant_host, port=self.config.qdrant_port)
            collections = client.get_collections()
            print(f" Connesso a Qdrant su {self.config.qdrant_host}:{self.config.qdrant_port}")
            print(f" Collezioni esistenti: {len(collections.collections)}")
        except Exception as e:
            print(f" Errore connessione Qdrant: {e}")
            print("\n Per avviare Qdrant:")
            print("   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
            raise
    
    def _start_chat(self):
        """Avvia la chat interattiva."""
        print("\n CHAT RAG - Poni le tue domande sui documenti!")
        print("Comandi speciali:")
        print("- 'docs' - Mostra documenti caricati")
        print("- 'stats' - Mostra statistiche sistema")
        print("- 'qdrant' - Mostra info Qdrant")
        print("- 'quit' - Esci")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n Domanda: ").strip()
                
                if not question:
                    continue
                
                if question.lower() == 'quit':
                    print(" Arrivederci!")
                    break
                
                if question.lower() == 'docs':
                    self.show_loaded_documents()
                    continue
                
                if question.lower() == 'stats':
                    self.show_system_stats()
                    continue
                
                if question.lower() == 'qdrant':
                    self.show_qdrant_stats()
                    continue
                
                # Processa la domanda
                print("🔍 Elaborazione...")
                result = self.ask_question(question)
                
                # Mostra la risposta
                print(f"\n Risposta:")
                print(result['answer'])
                
                if result['sources']:
                    print(f"\n Fonti: {', '.join(result['sources'])}")
                
                print(f" Confidenza: {result['confidence']:.2f}")
                print(f" Documenti utilizzati: {result.get('retrieved_docs', 0)}")
                
            except KeyboardInterrupt:
                print("\nChat interrotta")
                break
            except Exception as e:
                print(f" Errore: {e}")
    
    def show_loaded_documents(self):
        """Mostra i documenti caricati."""
        if not self.config.processed_docs:
            print("📂 Nessun documento caricato")
            return
        
        print(f"\n📋 DOCUMENTI CARICATI ({len(self.config.processed_docs)}):")
        print("-" * 50)
        
        for i, (doc_hash, doc_info) in enumerate(self.config.processed_docs.items(), 1):
            pdf_name = os.path.basename(doc_info.pdf_path)
            size_mb = doc_info.file_size / (1024 * 1024)
            
            print(f"{i}. {pdf_name}")
            print(f"    Chunk: {doc_info.chunk_count}")
            print(f"    Dimensione: {size_mb:.1f} MB")
            print(f"    Stato: {doc_info.status}")
    
    def show_system_stats(self):
        total_chunks = sum(doc.chunk_count for doc in self.config.processed_docs.values())
        total_size = sum(doc.file_size for doc in self.config.processed_docs.values())
        
        print(f"\n STATISTICHE SISTEMA:")
        print("-" * 30)
        print(f" Documenti: {len(self.config.processed_docs)}")
        print(f" Chunk totali: {total_chunks}")
        print(f" Dimensione totale: {total_size / (1024*1024):.1f} MB")
        print(f" Modello generazione: {self.config.model_name}")
        print(f" Metodo chunking: {self.config.chunk_method}")
        print(f" Target size: {self.config.target_size} parole")
        print(f"  Database: Qdrant ({self.config.qdrant_host}:{self.config.qdrant_port})")
    
    def show_qdrant_stats(self):
        """Mostra statistiche Qdrant."""
        if self.qdrant_manager:
            info = self.qdrant_manager.get_collection_info()
            print(f"\n  STATISTICHE QDRANT:")
            print("-" * 30)
            print(f"Collezione: {info.get('name', 'N/A')}")
            print(f"Punti indicizzati: {info.get('points_count', 0)}")
            print(f"Vettori indicizzati: {info.get('indexed_vectors_count', 0)}")
            print(f"Host: {self.config.qdrant_host}:{self.config.qdrant_port}")
            print(f"Modello embedding: {self.config.embedding_model}")
        else:
            print(" Qdrant non inizializzato")


def main():
    #Crea il sistema
    rag_system = DynamicPDFRAGSystem(config)
    # Avvia sessione interattiva
    rag_system.interactive_session()


if __name__ == "__main__":
    config = RAGConfig(
    qdrant_host="localhost",    # Indirizzo Qdrant
    qdrant_port=6333,          # Porta Qdrant
    collection_name="my_docs",  # Nome collezione
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    main()