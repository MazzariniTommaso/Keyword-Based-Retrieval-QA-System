import re
from nltk.tokenize import word_tokenize 
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from keybert import KeyBERT, KeyLLM
from keybert.llm import TextGeneration
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sentence_transformers import SentenceTransformer

KEY_LLM_PROMPT = """
<s>[INST] <<SYS>>

You are a helpful assistant specialized in extracting comma-separated keywords.
You are to the point and only give the answer in isolation without any chat-based fluff.

<</SYS>>
I have the following document:
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST] meat, beef, eat, eating, emissions, steak, food, health, processed, chicken [INST]

I have the following document:
- [DOCUMENT]

With the following candidate keywords:
- [CANDIDATES]

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST]
"""

def initialize_models(embedding_model, llm_model, qa_model, sum_model):
    """
    Loads and initializes machine learning models for embeddings, keyword extraction, text generation, 
    question answering, and summarization, using GPU if available.

    Args:
        embedding_model (str): Model for sentence embeddings. Defaults to 'all-MiniLM-L6-v2'.
        llm_model (str): Model for text generation. Defaults to 'gpt-2'.
        qa_model (str): Model for question answering. Defaults to 'distilbert-base-cased-distilled-squad'.
        sum_model (str): Model for summarization. Defaults to 'facebook/bart-large-cnn'.

    Returns:
        tuple: Initialized models:
            - SentenceTransformer for embeddings.
            - KeyBERT for keyword extraction.
            - KeyLLM for LLM-based keyword extraction.
            - HuggingFace pipelines for question answering and summarization.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Initialize the sentence transformer model
        print("Loading Sentence Transformer model...")
        model = SentenceTransformer(embedding_model, device=device)
        
        # Initialize the KeyBERT model
        print("Loading KeyBERT model...")
        kw_bert_model = KeyBERT(model)
        
        # Initialize the KeyLLM model
        print("Loading KeyLLM model...")
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            trust_remote_code=True,
            device_map='auto'
        )
        generator = pipeline(
            model=llm_model, tokenizer=tokenizer,
            task='text-generation',
            max_new_tokens=50,
            repetition_penalty=1.1,
            model_kwargs={"load_in_4bit": True}
        )
        llm = TextGeneration(generator, prompt=KEY_LLM_PROMPT)  # KEYLLM_PROMPT global variable
        kw_llm_model = KeyLLM(llm)
            
        # Initialize the question answering model
        print("Loading Question Answering model...")
        question_answerer = pipeline("question-answering", model=qa_model, device=device)
        
        # Initialize the summarization model
        print("Loading Summarization model...")
        summarizer = pipeline("summarization", model=sum_model, device=device)
        
        print("Models loaded successfully!")
    except Exception as e:
        print(f"An error occurred while loading the models: {e}")
    
    return model, kw_bert_model, kw_llm_model, question_answerer, summarizer

def clean_document(document):
    """
    Cleans the text of a document by removing unwanted characters and extra spaces.

    Args:
        document (str): The text of the document.

    Returns:
        str: The cleaned text.
    """
    # Ensure the input is a string, raise an error if it's not
    if not isinstance(document, str):
        raise ValueError("The document must be a string.")

    return re.sub(r"\s+", " ", document).strip()

    
def split_document(document, max_tokens, overlap_percentage):
    """
    Splits a document into overlapping chunks of a specified maximum token length.

    Args:
        document (str): The input text to be split.
        max_tokens (int): The maximum number of tokens allowed in each chunk.
        overlap_percentage (float): The percentage of overlap between consecutive chunks (e.g., 0.1 for 10% overlap).

    Returns:
        list: A list of text chunks, each containing up to max_tokens tokens, with the specified overlap.
    """
    if not isinstance(document, str):
        raise ValueError("Input document must be a string.")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0.")
    if not (0 <= overlap_percentage < 1):
        raise ValueError("overlap_percentage must be between 0 and 1.")

    # Tokenize and clean the document
    tokens = word_tokenize(clean_document(document))
    overlap = int(max_tokens * overlap_percentage)

    # Ensure at least one token overlaps when overlap_percentage > 0
    step = max(max_tokens - overlap, 1)
    
    # Generate chunks with overlap
    chunks = [' '.join(tokens[i:i + max_tokens]) for i in range(0, len(tokens), step)]
    return chunks

def get_top_kw (doc, candidates, model, top_n):
    """
    Get top keywords based on similarity to the document.
    
    Args:
        doc: Input document text
        candidates: List of candidate keywords
        model: SentenceTransformer model for encoding
        top_n: Number of top keywords to return
    
    Returns:
        List of tuples containing (keyword, similarity_score)
    """
    # Encode document and candidates
    doc_embedding = model.encode(doc).reshape(1, -1)
    candidate_embeddings = model.encode(candidates)
    
    # Calculate cosine similarities using sklearn
    similarities = cosine_similarity(candidate_embeddings, doc_embedding).flatten()
    
    # Get top keywords with their scores
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(candidates[idx], float(similarities[idx])) for idx in top_indices]
    
def extract_keywords_from_text(doc, kw_bert_model, kw_llm_model, model, use_keyllm, diversity, top_n):
    """
    Extract keywords from text using KeyBERT and optionally KeyLLM.
    
    Args:
        doc: Input text
        kw_bert_model: KeyBERT model instance
        kw_llm_model: KeyLLM model instance (optional)
        model: SentenceTransformer model for encoding
        use_keyllm: Whether to use KeyLLM for refinement
        diversity: Diversity parameter for MMR
        top_n: Number of keywords to extract
    
    Returns:
        List of tuples containing (keyword, score)
    """
    def extract_with_keybert(top_n):
        """Helper function for KeyBERT extraction with error handling"""
        try:
            return kw_bert_model.extract_keywords(
                docs=doc,
                vectorizer=KeyphraseCountVectorizer(),
                use_mmr=True,
                diversity=diversity,
                top_n=top_n
            )
        except ValueError as e:
            print(f"KeyphraseCountVectorizer failed, falling back to default: {e}")
            return kw_bert_model.extract_keywords(
                docs=doc,
                use_mmr=True,
                diversity=diversity,
                top_n=top_n
            )

    if use_keyllm:
        # Get initial keywords from KeyBERT (limited to 20 for LLM processing)
        initial_keywords = extract_with_keybert(20)
        initial_keyword_texts = [kw[0] for kw in initial_keywords]
        
        # Refine using KeyLLM
        refined_keywords = kw_llm_model.extract_keywords(
            docs=doc,
            candidate_keywords=initial_keywords
        )[0]
        
        # Combine and deduplicate candidates
        all_candidates = list(set(initial_keyword_texts) | set(refined_keywords))
        all_candidates = [c for c in all_candidates if c]  # Remove empty strings
        
        # Get final keywords based on similarity
        return get_top_kw(doc, all_candidates, model, top_n)
    
    # Use KeyBERT only
    return extract_with_keybert(top_n)

def create_vector_db(chunks, model, kw_bert_model, kw_llm_model, use_keyllm, diversity, top_n):
    """
    Create a vector database from text chunks using keyword extraction.
    
    Args:
        chunks: List of text chunks to process
        model: SentenceTransformer model for encoding
        kw_bert_model: KeyBERT model instance
        kw_llm_model: Optional KeyLLM model instance
        use_keyllm: Whether to use KeyLLM
        diversity: Diversity parameter for MMR
        top_n: Number of keywords per chunk
    
    Returns:
        Dictionary mapping keyword embeddings to associated chunks
    """
    chunk_keywords_dict = {}
    
    for chunk in chunks:
        # Extract keywords for current chunk
        keywords = extract_keywords_from_text(
            doc=chunk,
            kw_bert_model=kw_bert_model,
            kw_llm_model=kw_llm_model,
            model=model,
            use_keyllm=use_keyllm,
            diversity=diversity,
            top_n=top_n
        )
        
        # Create sorted keyword string and generate embedding
        keywords_str = " ".join(sorted(kw[0] for kw in keywords))
        keywords_emb = tuple(model.encode(keywords_str).tolist())
        
        # Store chunk with its keyword embedding
        chunk_keywords_dict.setdefault(keywords_emb, []).append(chunk)
    
    return chunk_keywords_dict

def retrieve_documents(chunk_keywords_dict, question_emb, top_n):
    """
    Retrieves the most relevant document chunks based on the similarity between 
    a given question embedding and the keyword embeddings of the chunks.

    Args:
        chunk_keywords_dict (dict): A dictionary where keys are keyword embeddings 
                                     (tuples), and values are lists of document chunks.
        question_emb (array-like): The embedding of the question or query to compare against.
        top_n (int, optional): The number of top relevant chunks to return. Defaults to 3.

    Returns:
        list: A list of the top_n most relevant document chunks based on the cosine similarity 
              between the question embedding and the chunk keyword embeddings.
    """
    
    keys = list(chunk_keywords_dict.keys())
    similarity_scores = cosine_similarity([list(question_emb)], [list(key) for key in keys])[0]
    top_indices = np.argsort(similarity_scores)[-top_n:][::-1]
    top_keys = [keys[i] for i in top_indices]
    retrieved_chunks = [chunk for key in top_keys for chunk in chunk_keywords_dict[key]]
    return retrieved_chunks[:top_n]

def get_answer_with_summary(question_answerer, summarizer, question, context, sum_max_length, sum_min_length):
    """
    Retrieves an answer to the question from the context and generates a summary.

    Args:
        question_answerer (callable): Function to get an answer from the context.
        summarizer (callable): Function to generate a summary of the context.
        question (str): The question to answer.
        context (str): The context from which the answer is derived.
        sum_max_length (int): Max length of the summary.
        sum_min_length (int): Min length of the summary.

    Returns:
        tuple: The answer and the generated summary.
    """
    
    answer = question_answerer(question=question, context=context)
    summary = summarizer(context[:4500], max_length=sum_max_length, min_length=sum_min_length, do_sample=False)[0]['summary_text']
    summary = 'None'
    return answer, summary

class DocumentQuestionAnsweringPipeline:
    """
    A pipeline for processing documents, extracting keywords, and answering questions 
    by leveraging embeddings, keyword extraction, and summarization.
    """
    def __init__(self, embedding_model='all-MiniLM-L6-v2', llm_model='Qwen/Qwen2.5-3B', qa_model='google-bert/bert-large-cased-whole-word-masking-finetuned-squad', sum_model='facebook/bart-large-cnn'):
        """
        Initializes the pipeline with the required models for keyword extraction, 
        question answering, and summarization.
        
        Args:
            embedding_model (str): Model used for generating document embeddings.
            llm_model (str): Large Language Model used for keyword extraction.
            qa_model (str): Model used for question answering.
            sum_model (str): Model used for summarization.
        """
        # Initialize models
        self.model, self.kw_bert_model, self.kw_llm_model, self.question_answerer, self.summarizer = initialize_models(
            embedding_model, llm_model, qa_model, sum_model)
        self.chunk_keywords_dict = {}

    def add_documents(self, docs, max_tokens=1000, overlap_percentage=0.5, use_keyllm=False, diversity=0.3, top_n_kw=10):
        """
        Splits documents into chunks and extracts keyword embeddings for each chunk.
        
        Args:
            docs (list): List of documents to process.
            max_tokens (int): Maximum number of tokens per chunk.
            overlap_percentage (float): Overlap between consecutive chunks.
            use_keyllm (bool): Whether to use KeyLLM for keyword extraction.
            diversity (float): Controls the diversity of keyword selection.
            top_n_kw (int): Number of top keywords to extract from each chunk.
        """
        all_chunks = []
        # Split documents into chunks and extract keyword embeddings
        for doc in docs:
            chunks = split_document(doc, max_tokens, overlap_percentage)
            all_chunks.extend(chunks)

        # Create a vector database of chunk keywords and update the chunk_keywords_dict
        new_chunk_keywords = create_vector_db(all_chunks, self.model, self.kw_bert_model, self.kw_llm_model, 
                                              use_keyllm, diversity, top_n_kw)
        self.chunk_keywords_dict = new_chunk_keywords

    def answer_question(self, question, use_keyllm=False, diversity=0.3, top_n_chunk=10, top_n_kw=10, 
                        sum_max_length=130, sum_min_length=30):
        """
        Answers a question based on the context retrieved from relevant document chunks.
        
        Args:
            question (str): The question to answer.
            use_keyllm (bool): Whether to use KeyLLM for keyword extraction.
            diversity (float): Controls the diversity of keyword selection.
            top_n_chunk (int): Number of top chunks to retrieve.
            top_n_kw (int): Number of top keywords to extract from the question.
            sum_max_length (int): Maximum length of the summary.
            sum_min_length (int): Minimum length of the summary.
        
        Returns:
            tuple: A tuple containing the answer to the question and the generated summary.
        """
        # Extract keywords from the question
        question_kw = extract_keywords_from_text(question, self.kw_bert_model, self.kw_llm_model, self.model, use_keyllm, 
                                                 diversity, top_n_kw)
        question_kw_str = ", ".join(sorted([key[0] for key in question_kw]))
        question_emb = tuple(self.model.encode(question_kw_str))
        
        # Retrieve relevant document chunks and build context
        retrieved_chunks = retrieve_documents(self.chunk_keywords_dict, question_emb, top_n_chunk)
        context = " ".join(retrieved_chunks)
        # Generate answer and summary
        answer, summary = get_answer_with_summary(self.question_answerer, self.summarizer, question, context, 
                                                  sum_max_length, sum_min_length)
        return answer, summary