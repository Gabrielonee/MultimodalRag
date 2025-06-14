from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time
from typing import List, Dict, Optional
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelli open source consigliati per RAG
AVAILABLE_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "zephyr-7b": "HuggingFace/zephyr-7b-beta", 
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "flan-t5-large": "google/flan-t5-large"
}

class RAGGenerator:
    def __init__(self, model_name: str = "phi-3-mini", max_length: int = 512, temperature: float = 0.7):
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Inizializzazione generatore con modello: {model_name}")
        logger.info(f"Device utilizzato: {self.device}")
        
        # Carica il modello
        self._load_model()
    
    def _load_model(self):
        """Carica il modello e il tokenizer."""
        try:
            if self.model_name not in AVAILABLE_MODELS:
                raise ValueError(f"Modello non supportato. Scegli tra: {list(AVAILABLE_MODELS.keys())}")
            
            model_path = AVAILABLE_MODELS[self.model_name]
            
            # Carica tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Imposta pad_token se non presente
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Carica il modello con configurazioni ottimizzate
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Crea pipeline di generazione
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("Modello caricato con successo!")
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {e}")
            raise
    
    def _format_context(self, retrieved_chunks: List[Dict], max_context_length: int = 2000) -> str:

        if not retrieved_chunks:
            return "Nessun contesto disponibile."
        
        context_parts = []
        total_length = 0
        
        for i, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk.get('text', chunk.get('content', ''))
            score = chunk.get('score', 0.0)
            
            # Aggiungi informazioni sul chunk
            formatted_chunk = f"[Documento {i+1} - Rilevanza: {score:.3f}]\n{chunk_text}\n"
            
            # Controlla se supera la lunghezza massima
            if total_length + len(formatted_chunk) > max_context_length:
                break
            
            context_parts.append(formatted_chunk)
            total_length += len(formatted_chunk)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        # Template del prompt ottimizzato per RAG
        prompt_template = """Sei un assistente AI che risponde alle domande basandosi esclusivamente sui documenti forniti nel contesto.

CONTESTO:
{context}

DOMANDA: {query}

ISTRUZIONI:
- Rispondi basandoti SOLO sulle informazioni presenti nel contesto
- Se l'informazione non è presente nel contesto, specifica che non hai informazioni sufficienti
- Sii preciso e conciso
- Cita elementi specifici dal contesto quando possibile

RISPOSTA:"""
        
        return prompt_template.format(context=context, query=query)
    
    def generate_response(self, query: str, retrieved_chunks: List[Dict], 
                         max_context_length: int = 2000) -> Dict[str, any]:
        try:
            # Formatta il contesto
            context = self._format_context(retrieved_chunks, max_context_length)
            
            # Crea il prompt
            prompt = self._create_prompt(query, context)
            
            logger.info(f"Generazione risposta per query: {query[:50]}...")
            
            # Genera la risposta
            response = self.generator(
                prompt,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Estrai il testo generato
            generated_text = response[0]['generated_text'].strip()
            
            # Pulisci la risposta da eventuali artefatti
            generated_text = self._clean_response(generated_text)
            
            return {
                'response': generated_text,
                'query': query,
                'context_used': len(retrieved_chunks),
                'model': self.model_name,
                'sources': [chunk.get('filename', 'Unknown') for chunk in retrieved_chunks[:3]]
            }
            
        except Exception as e:
            logger.error(f"Errore nella generazione: {e}")
            return {
                'response': f"Errore nella generazione della risposta: {str(e)}",
                'query': query,
                'context_used': 0,
                'model': self.model_name,
                'sources': []
            }
    
    def _clean_response(self, response: str) -> str:
        # Rimuovi ripetizioni del prompt
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Salta linee vuote all'inizio
            if not line and not cleaned_lines:
                continue
            # Salta linee che sembrano essere parti del prompt
            if line.startswith(('CONTESTO:', 'DOMANDA:', 'ISTRUZIONI:', 'RISPOSTA:')):
                continue
            cleaned_lines.append(line)
        
        cleaned_response = '\n'.join(cleaned_lines).strip()
        
        # Rimuovi caratteri speciali di fine generazione
        for token in ['<|endoftext|>', '<|end|>', '[END]']:
            cleaned_response = cleaned_response.replace(token, '')
        
        return cleaned_response
    
    def batch_generate(self, queries_and_contexts: List[tuple], batch_size: int = 4) -> List[Dict]:
        """
        Genera risposte per multiple query in batch.
        
        Args:
            queries_and_contexts: Lista di tuple (query, retrieved_chunks)
            batch_size: Dimensione del batch per la generazione
        
        Returns:
            Lista di risposte generate
        """
        results = []
        
        for i in range(0, len(queries_and_contexts), batch_size):
            batch = queries_and_contexts[i:i+batch_size]
            
            for query, chunks in batch:
                result = self.generate_response(query, chunks)
                results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, any]:
        """Restituisce informazioni sul modello corrente."""
        return {
            'model_name': self.model_name,
            'model_path': AVAILABLE_MODELS[self.model_name],
            'device': self.device,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'parameters': self.model.num_parameters() if hasattr(self.model, 'num_parameters') else 'Unknown'
        }


# Utility functions
def create_generator(model_name: str = "phi-3-mini", **kwargs) -> RAGGenerator:
    return RAGGenerator(model_name=model_name, **kwargs)


def list_available_models() -> Dict[str, str]:
    return AVAILABLE_MODELS.copy()


# Esempio di utilizzo
if __name__ == "__main__":
    print("Inizializzazione del generatore RAG...")
    
    # Crea il generatore (usa un modello più leggero per test)
    generator = create_generator(model_name="phi-3-mini", max_length=256, temperature=0.7)
    
    # Simula alcuni chunk recuperati
    mock_chunks = [
        {
            'text': 'RAG (Retrieval-Augmented Generation) è una tecnica che combina il recupero di informazioni con la generazione di testo.',
            'score': 0.95,
            'filename': 'rag_intro.txt'
        },
        {
            'text': 'Il sistema RAG funziona in due fasi: prima recupera documenti rilevanti, poi genera una risposta basata su questi documenti.',
            'score': 0.87,
            'filename': 'rag_process.txt'
        }
    ]
    
    # Test di generazione
    test_query = "Cos'è RAG e come funziona?"
    
    print(f"Test query: {test_query}")
    print("Generazione in corso...")
    
    result = generator.generate_response(test_query, mock_chunks)
    
    print("\n" + "="*50)
    print("RISULTATO:")
    print(f"Query: {result['query']}")
    print(f"Risposta: {result['response']}")
    print(f"Fonti utilizzate: {result['sources']}")
    print(f"Modello: {result['model']}")
    
    # Mostra info del modello
    print("\n" + "="*50)
    print("INFO MODELLO:")
    model_info = generator.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")