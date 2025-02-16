# DNLP_project

## Introduction
This project addresses the task of enhancing retrieval-based Q&A systems through keyword extraction. We leverage KeyBERT, an unsupervised method using BERT embeddings, and integrate KeyLLM, a lightweight approach using large language models, to refine keyword extraction. Our system supports tasks like document summarization and retrieval-based question answering (QA).

You can find the detailed report of the project [here](https://github.com/yourusername/DNLP_project/blob/main/Mazzarini_Merelli_Pisanu_Stinà_Report_DNLP.pdf).

## Features
- **Keyword Extraction**: Utilizes KeyBERT and KeyLLM for efficient keyword and keyphrase identification from documents.
- **Question Answering**: Implements a retrieval-based QA system capable of processing large corpora using fine-tuned BERT.
- **Summarization**: Provides abstractive summarization with the BART model for concise content summaries.
- **Interactive Interface**: Includes a graphical interface built with Gradio, enabling users to upload `.txt` files and interact with the system for QA and summarization tasks.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```
## Usage
1. Run the project:
```bash
python main.py
```
2. Use the graphical interface to upload documents and ask questions. Optional features include:
   - Integrating a Wikipedia search.
   - Enhancing results with KeyLLM.

## Visual Example
Here is an example of the system's graphical interface:

<img src="images/gradio_ui.png" alt="System Graphical Interface" width="800">


## Experimental Results

### Keyword Extraction
We compared KeyBERT alone and KeyBERT combined with KeyLLM on the SemEval2017 Task 10 dataset:
- KeyBERT-only: Achieved an F1-score of 21.06% (using all-MiniLM-L6-v2 embeddings).
- KeyBERT + KeyLLM: Showed slight improvements in diversity but increased inference times due to computational overhead.

### QA System
Using the SQuAD 1.1 dataset, the QA system achieved:
- KeyBERT-only: F1-score of 83.88% and Exact Match (EM) of 81.20%.
- KeyBERT + KeyLLM: F1-score of 82.84% and EM of 80.40%.
- Compared to a fine-tuned BERT model (F1: 93.24%, EM: 87.26%), our retrieval-based approach offers robust yet computationally efficient results.

### Summarization
The system employs a BART model to generate summaries (no eavluation is performed).

## Future Work
- Experiment with various parameters to evaluate their impact on the pipeline’s overall performance and effectiveness.  
- Explore larger LLMs (e.g., Gemma, Mistral) and advanced embeddings with greater representational capacity to enhance keyword extraction accuracy and efficiency.  
- Investigate alternative question answering and summarization models to improve system robustness and output quality.  


## Contributors
- Tommaso Mazzarini - tommaso.mazzarini@studenti.polito.it
- Leonardo Merelli - leonardo.merelli@studenti.polito.it
- Riccardo Pisanu - s328202@studenti.polito.it
- Giovanni Stina - giovanni.stina@studenti.polito.it

## License
This project is licensed under the MIT License.
