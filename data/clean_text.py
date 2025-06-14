import re
import os

def cleaning(text: str) -> str:
    # Rimozione sezione References
    text = re.split(r'(?i)\n+references\n+', text)[0]   
    
    # Rimozione piè di pagina
    text = re.sub(r'\n\s*Page \d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Rimozione righe con solo simboli o molto corte
    text = re.sub(r'^\s*[^a-zA-Z\n\s]{2,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.{1,10}$', '', text, flags=re.MULTILINE)
    
    # Rimozione righe vuote multiple
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def clean_text_file(input_path: str, output_path: str) -> None:
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File non trovato: {input_path}")
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    
        cleaned = cleaning(raw_text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        print(f"[DONE] Testo pulito salvato in: {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Errore durante la pulizia del file: {e}")


if __name__ == "__main__":
    # Test della funzione
    sample_text = """
    Questo è un testo di esempio.    
    Page 1

    Contenuto principale del documento.
    Altra riga importante.
    
    123
    
    !!!
    
    References
    [1] Riferimento da rimuovere
    """    
    cleaned = cleaning(sample_text)
    print("Testo pulito:")
    print(cleaned)