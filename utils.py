import wikipedia as wiki

def load_wiki_documents(question):
    """
    Carica documenti da Wikipedia in base alla query.
    """
    results = wiki.search(question)
    docs = []
    for page in results[:3]:
        try:
            docs.append(wiki.page(page, auto_suggest=False).content)
        except Exception as e:
            print(f"Errore nel caricamento della pagina '{page}': {e}")
    return docs

def read_uploaded_files(files):
    """
    Legge i file caricati e restituisce una lista di testi.
    """
    docs = []
    if files == None:
        return docs
    
    for file in files:
        try:
            with open(file.name, "r", encoding="utf-8") as f:
                doc = f.read()
            docs.append(doc)
        except Exception as e:
            raise Exception(f"Errore durante la lettura del file {file.name}: {str(e)}")
    return docs
