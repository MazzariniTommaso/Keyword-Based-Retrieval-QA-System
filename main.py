import configparser
import threading
import gradio as gr
from qa_pipeline import DocumentQuestionAnsweringPipeline
from utils import load_wiki_documents, read_uploaded_files
from huggingface_hub import login

# Load Hugging Face Token
config = configparser.ConfigParser()
config.read('./config.ini')
HF_TOKEN = config['hf_token']['access_token']  # For Hugging Face
login(HF_TOKEN)

# Initialize the pipeline
qa_pipeline = DocumentQuestionAnsweringPipeline()

# Response function that integrates uploaded documents and those from Wikipedia (optional)
def answer_question(files=None, question=None, use_wiki_docs=False, use_keyllm=False):
    docs = []
    try:
        docs = read_uploaded_files(files)
    except Exception as e:
        return f"Error reading file: {str(e)}", ""
    
    if use_wiki_docs:
        docs.extend(load_wiki_documents(question))

    if not docs:
        return "No documents uploaded or available.", ""
    
    try:
        qa_pipeline.add_documents(
            docs=docs, 
            max_tokens=1000,
            overlap_percentage=0.5,
            use_keyllm=use_keyllm, 
            diversity=0.3,
            top_n_kw=10
        )
        answer, summary = qa_pipeline.answer_question(
            question=question, 
            use_keyllm=use_keyllm, 
            diversity=0.3,
            top_n_chunk=10, 
            top_n_kw=10,
            sum_max_length=130,  # set based on top chunks and tokens per chunk
            sum_min_length=50  # set based on top chunks and tokens per chunk
        )
        clean_answer = answer['answer'].strip()
        return clean_answer, summary
    except Exception as e:
        return f"Error processing the question: {str(e)}", ""

# Function to handle stopping the process (if needed)
stop_event = threading.Event()
def stop_processing():
    stop_event.set()
    return "Processing interrupted by the user.", "Processing interrupted by the user."

# Create the interface with Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Document Question and Answering System")
    with gr.Row():
        file_input = gr.File(file_count="multiple", label="Upload your documents")
        question_input = gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question")
    with gr.Row():
        use_wiki_checkbox = gr.Checkbox(label="Use Wikipedia", value=False, interactive=True)
        use_keyllm_checkbox = gr.Checkbox(label="Enable KeyLLM", value=False, interactive=True)
    with gr.Row():
        submit_button = gr.Button("Submit")
        stop_button = gr.Button("Stop")
    output_answer = gr.Textbox(label="Short Answer")
    output_summary = gr.Textbox(label="Summary")
    
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
