import configparser
import threading
import gradio as gr
from qa_pipeline import DocumentQuestionAnsweringPipeline
from utils import load_wiki_documents, read_uploaded_files
from huggingface_hub import login

# Load Hugging Face Token
config = configparser.ConfigParser()
config.read('./config.ini')
HF_TOKEN = config['hf_token']['access_token'] # For Hugging Face
login(HF_TOKEN)

# Inizializzazione della pipeline
qa_pipeline = DocumentQuestionAnsweringPipeline()

# Funzione di risposta che integra i documenti caricati e quelli da Wikipedia (opzionali)
def answer_question(files, question, use_wiki_docs=False, use_keyllm=False):
    docs = []
    try:
        docs = read_uploaded_files(files)
    except Exception as e:
        return f"Errore durante la lettura del file: {str(e)}", ""
    
    if use_wiki_docs:
        docs.extend(load_wiki_documents(question))

    if not docs:
        return "Nessun documento caricato o disponibile.", ""
    
    try:
        qa_pipeline.add_documents(
            docs=docs, 
            max_tokens=500, # set
            overlap_percentage=0.25, # set
            use_keyllm=use_keyllm, 
            diversity=0.3, # set
            top_n_kw=10 # set
        )
        answer, summary = qa_pipeline.answer_question(
            question=question, 
            use_keyllm=use_keyllm, 
            diversity=0.3, # set
            top_n_chunk=3, # set 
            top_n_kw=10, # set
            sum_max_length=130, # set in base a top chunks e a token per chunk
            sum_min_length=50 # set in base a top chunks e a token per chunk
        )
        clean_answer = answer['answer'].strip()
        return clean_answer, summary
    except Exception as e:
        return f"Errore durante l'elaborazione della domanda: {str(e)}", ""

# Funzione per gestire l'interruzione del processo (se necessario)
stop_event = threading.Event()
def stop_processing():
    stop_event.set()
    return "Elaborazione interrotta dall'utente.", "Elaborazione interrotta dall'utente."

# Creazione dell'interfaccia con Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Sistema di Domanda e Risposta su Documenti")
    with gr.Row():
        file_input = gr.File(file_count="multiple", label="Carica i tuoi documenti")
        question_input = gr.Textbox(lines=2, placeholder="Inserisci la tua domanda qui...", label="Domanda")
    with gr.Row():
        use_wiki_checkbox = gr.Checkbox(label="Usa Wikipedia", value=False, interactive=True)
        use_keyllm_checkbox = gr.Checkbox(label="Abilita KeyLLM", value=False, interactive=True)
    with gr.Row():
        submit_button = gr.Button("Invia")
        stop_button = gr.Button("Interrompi")
    output_answer = gr.Textbox(label="Risposta breve")
    output_summary = gr.Textbox(label="Riassunto")
    
    submit_button.click(
        fn=lambda files, question, use_wiki_docs, use_keyllm: answer_question(
            files, question, use_wiki_docs, use_keyllm
        ),
        inputs=[file_input, question_input, use_wiki_checkbox, use_keyllm_checkbox],
        outputs=[output_answer, output_summary]
    )
    stop_button.click(fn=stop_processing, inputs=None, outputs=[output_answer, output_summary])

if __name__ == "__main__":
    demo.launch(share=False)
