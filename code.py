!pip install pymupdf  # For fitz (PyMuPDF)
!pip install regex    # For re (though Python's built-in re may suffice)
!pip install spacy
!pip install tiktoken
!pip install torch
!pip install chromadb
!pip install numpy
!pip install google-generativeai
!pip install sentence-transformers
!pip install scikit-learn  # For sklearn.metrics.pairwise
!pip install transformers
!pip install -U google-generativeai

pip install pytesseract pillow

!pip install bert-score
!pip install google-generativeai

!pip install datasets bitsandbytes accelerate peft transformers

!pip install -U datasets bitsandbytes peft transformers accelerate

import fitz  # PyMuPDF
import re
import spacy
import os
import tiktoken
import torch
import chromadb
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    pipeline,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import bitsandbytes as bnb
from typing import List, Dict, Any, Tuple, Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from PIL import Image
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MultiAgentLegalQA")

# Configure Gemini API
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

# Load NLP & Embedding Models
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Efficient embeddings
re_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")  # Re-ranking model

class FineTunedQAModel:
    """Wrapper class for QA model with LoRA/QLoRA fine-tuning support"""
    def __init__(self, model_name="deepset/roberta-base-squad2", use_lora=True, use_qlora=False, fine_tuned_path=None):
        self.model_name = model_name
        self.use_lora = use_lora
        self.use_qlora = use_qlora
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._load_model(fine_tuned_path)
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def _load_model(self, fine_tuned_path=None):
        """Load the model with optional LoRA/QLoRA configuration"""
        if fine_tuned_path:
            model = AutoModelForQuestionAnswering.from_pretrained(fine_tuned_path)
            if (self.use_lora or self.use_qlora) and os.path.exists(os.path.join(fine_tuned_path, "adapter_config.json")):
                model = get_peft_model(model, LoraConfig.from_pretrained(fine_tuned_path))
            return model

        if self.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)

        if self.use_lora or self.use_qlora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["query", "key", "value"],
                lora_dropout=0.05,
                bias="none",
                task_type="QUESTION_ANS"
            )
            model = get_peft_model(model, lora_config)

        return model

    def train(self, train_dataset, eval_dataset=None, output_dir="./fine_tuned_model"):
        """Fine-tune the model with the given dataset"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=3e-4,
            per_device_train_batch_size=8,
            num_train_epochs=3,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            logging_dir='./logs',
            report_to="none",
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def create_dataset(self, examples):
        """Create training dataset from examples"""
        questions = [ex['question'] for ex in examples]
        contexts = [ex['context'] for ex in examples]
        answers = [ex['answer'] for ex in examples]

        encodings = self.tokenizer(
            questions,
            contexts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        start_positions = []
        end_positions = []

        for i in range(len(answers)):
            answer = answers[i]
            context = contexts[i]
            start_char = context.find(answer)
            end_char = start_char + len(answer)

            if start_char == -1:
                logger.warning(f"Answer not found in context for example {i}")
                start_positions.append(0)
                end_positions.append(0)
                continue

            tokenized = self.tokenizer(
                questions[i],
                context,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True
            )

            start_token = None
            end_token = None
            for idx, (start, end) in enumerate(tokenized['offset_mapping']):
                if start <= start_char < end:
                    start_token = idx
                if start < end_char <= end:
                    end_token = idx
                    break

            start_positions.append(start_token if start_token is not None else 0)
            end_positions.append(end_token if end_token is not None else 0)

        encodings.update({
            'start_positions': torch.tensor(start_positions),
            'end_positions': torch.tensor(end_positions)
        })

        return Dataset.from_dict(encodings)

class Agent:
    """Base class for all agents in the system"""
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")

    def process(self, input_data: Any) -> Any:
        """Process input data according to the agent's responsibility"""
        self.logger.info(f"Processing data in {self.name}")
        return input_data

class MemoryAgent(Agent):
    """Agent responsible for maintaining conversation history and context"""
    def __init__(self, max_history: int = 20):
        super().__init__("MemoryAgent")
        self.conversation_history = deque(maxlen=max_history)
        self.context_window = []
        self.max_context_length = 4000  # Approximate token limit for context window

    def add_interaction(self, user_input: str, system_response: str):
        """Add a user-system interaction to memory"""
        self.conversation_history.append({
            'user': user_input,
            'system': system_response,
            'timestamp': time.time()
        })

    def get_recent_history(self, num_interactions: int = 3) -> List[Dict[str, str]]:
        """Get the most recent interactions"""
        return list(self.conversation_history)[-num_interactions:]

    def update_context_window(self, new_context: str):
        """Update the context window with new information"""
        tokens = new_context.split()
        if len(tokens) > self.max_context_length:
            new_context = ' '.join(tokens[-self.max_context_length:])

        self.context_window.append(new_context)
        if len(self.context_window) > 5:
            self.context_window = self.context_window[-5:]

    def get_context_summary(self) -> str:
        """Generate a summary of the current context"""
        if not self.conversation_history:
            return "No conversation history available."

        recent = self.get_recent_history(3)
        summary = "Recent conversation history:\n"
        for i, interaction in enumerate(recent, 1):
            summary += f"{i}. User: {interaction['user']}\n   System: {interaction['system']}\n"

        if self.context_window:
            summary += "\nAdditional context:\n" + "\n".join(f"- {ctx}" for ctx in self.context_window)

        return summary

    def process(self, interaction: Dict[str, str]) -> str:
        """Process an interaction and return context summary"""
        self.add_interaction(interaction['user'], interaction['system'])
        return self.get_context_summary()

class DocumentExtractorAgent(Agent):
    """Agent responsible for extracting text from documents"""
    def __init__(self):
        super().__init__("DocumentExtractor")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        self.logger.info(f"Extracting text from {pdf_path}")
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text("text") + "\n" for page in doc)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process(self, pdf_files: List[str]) -> Dict[str, str]:
        """Process a list of PDF files and extract text"""
        documents = {}
        for pdf in pdf_files:
            try:
                documents[os.path.basename(pdf)] = self.extract_text_from_pdf(pdf)
                self.logger.info(f"Successfully extracted text from {pdf}")
            except Exception as e:
                self.logger.error(f"Failed to extract text from {pdf}: {e}")
        return documents

class TextProcessingAgent(Agent):
    """Agent responsible for processing and chunking text"""
    def __init__(self):
        super().__init__("TextProcessor")

    def sent_tokenize_spacy(self, text: str) -> List[str]:
        """Tokenize text into sentences using spaCy"""
        return [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]

    def chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
        """Chunk text with adaptive size and overlap"""
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        sentences = self.sent_tokenize_spacy(text)
        chunks, current_chunk, current_length = [], [], 0

        sentence_indices = []

        for i, sentence in enumerate(sentences):
            token_count = len(tokenizer.encode(sentence))

            if current_length + token_count > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                overlap_token_count = 0
                overlap_sentences = []
                for j in range(len(current_chunk) - 1, -1, -1):
                    sent = current_chunk[j]
                    sent_tokens = len(tokenizer.encode(sent))
                    if overlap_token_count + sent_tokens <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_token_count += sent_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_token_count

            current_chunk.append(sentence)
            current_length += token_count
            sentence_indices.append(i)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def create_document_structure(self, documents: Dict[str, str]) -> Dict[str, Any]:
        """Create semantic document structure"""
        structured_docs = {}

        for doc_name, text in documents.items():
            base_doc = doc_name.split('_')[0].split('-')[0]

            if base_doc not in structured_docs:
                structured_docs[base_doc] = {'parts': {}}

            structured_docs[base_doc]['parts'][doc_name] = {
                'text': text,
                'chunks': self.chunk_text(text)
            }

        return structured_docs

    def flatten_document_structure(self, structured_documents: Dict[str, Any]) -> Dict[str, List[str]]:
        """Flatten document structure for embedding"""
        chunked_documents = {}
        for base_doc, content in structured_documents.items():
            for part_name, part_content in content['parts'].items():
                chunked_documents[part_name] = part_content['chunks']
        return chunked_documents

    def process(self, documents: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """Process documents to create structured and chunked representations"""
        self.logger.info("Creating document structure")
        structured_documents = self.create_document_structure(documents)
        self.logger.info("Flattening document structure for embedding")
        chunked_documents = self.flatten_document_structure(structured_documents)
        return structured_documents, chunked_documents

class EmbeddingAgent(Agent):
    """Agent responsible for creating and managing embeddings"""
    def __init__(self):
        super().__init__("Embedder")

    def get_embedding(self, text_list: List[str]) -> torch.Tensor:
        """Get embeddings for a list of text chunks"""
        return embedding_model.encode(text_list, convert_to_tensor=True)

    def process(self, chunked_documents: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Process chunked documents to create embeddings"""
        embedded_documents = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_doc = {
                executor.submit(self.get_embedding, chunks): doc_name
                for doc_name, chunks in chunked_documents.items()
            }

            for future in as_completed(future_to_doc):
                doc_name = future_to_doc[future]
                try:
                    embedded_documents[doc_name] = future.result()
                    self.logger.info(f"Created embeddings for {doc_name}")
                except Exception as e:
                    self.logger.error(f"Failed to create embeddings for {doc_name}: {e}")

        return embedded_documents

class DatabaseAgent(Agent):
    """Agent responsible for managing the vector database"""
    def __init__(self, db_path: str = "./chroma_db"):
        super().__init__("DatabaseManager")
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None

    def setup_collection(self):
        """Set up the ChromaDB collection"""
        try:
            self.client.delete_collection("document_embeddings")
            self.logger.info("Deleted existing collection")
        except:
            self.logger.info("No existing collection to delete")

        self.collection = self.client.create_collection(name="document_embeddings")
        self.logger.info("Created new collection: document_embeddings")
        return self.collection

    def store_embeddings(self, embedded_documents: Dict[str, Any], chunked_documents: Dict[str, List[str]]):
        """Store embeddings in the database with enhanced metadata"""
        if self.collection is None:
            self.setup_collection()

        total_chunks = 0
        for doc_name, embeddings in embedded_documents.items():
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy().tolist()

            doc_parts = doc_name.split('_')
            base_doc = doc_name
            doc_type = "MAIN"

            if len(doc_parts) > 1:
                base_doc = '_'.join(doc_parts[:2])
                if len(doc_parts) > 2:
                    doc_type = '_'.join(doc_parts[2:])

            batch_ids = []
            batch_embeddings = []
            batch_metadatas = []
            batch_documents = []

            for i, (embedding, chunk) in enumerate(zip(embeddings, chunked_documents[doc_name])):
                batch_ids.append(f"{doc_name}_chunk{i+1}")
                batch_embeddings.append(embedding)
                batch_metadatas.append({
                    "document": doc_name,
                    "base_document": base_doc,
                    "document_type": doc_type,
                    "chunk_number": i+1,
                    "total_chunks": len(chunked_documents[doc_name])
                })
                batch_documents.append(chunk)
                total_chunks += 1

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )

            self.logger.info(f"Stored {len(batch_ids)} chunks for {doc_name}")

        self.logger.info(f"Total of {total_chunks} chunks stored in ChromaDB")
        return total_chunks

    def process(self, data: Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]) -> chromadb.Collection:
        """Set up database and store embeddings"""
        embedded_documents, chunked_documents = data
        self.setup_collection()
        self.store_embeddings(embedded_documents, chunked_documents)
        return self.collection

class QueryAgent(Agent):
    """Agent responsible for processing queries and retrieving information"""
    def __init__(self, few_shot_examples: Dict[str, str] = None, use_lora=True, use_qlora=False):
        super().__init__("QueryProcessor")
        self.few_shot_examples = few_shot_examples or {}
        self.collection = None
        self.SIMILARITY_THRESHOLD = 0.15
        self.TOP_K = 15
        self.memory_agent = None
        self.qa_model = FineTunedQAModel(use_lora=use_lora, use_qlora=use_qlora)

    def set_collection(self, collection: chromadb.Collection):
        """Set the ChromaDB collection for querying"""
        self.collection = collection

    def set_memory_agent(self, memory_agent: 'MemoryAgent'):
        """Set the memory agent for conversation context"""
        self.memory_agent = memory_agent

    def retrieve_few_shot_answer(self, query: str) -> str:
        """Check if query is in few-shot examples, otherwise use advanced retrieval."""
        if query in self.few_shot_examples:
            self.logger.info(f"Found few-shot example for query: {query}")
            return self.few_shot_examples[query]
        self.logger.info(f"No few-shot example for query: {query}")
        return self.retrieve_advanced_rag_answer(query)

    def merge_overlapping_answers(self, answers: List[Dict[str, Any]]) -> str:
        """Merge overlapping or adjacent answer spans to create a more comprehensive answer."""
        if not answers:
            return ""

        sorted_answers = sorted(answers, key=lambda x: x['score'], reverse=True)
        primary_answer = sorted_answers[0]['answer']

        additional_info = []
        for answer in sorted_answers[1:4]:
            if answer['score'] > 0.6:
                if answer['answer'] not in primary_answer and primary_answer not in answer['answer']:
                    additional_info.append(answer['answer'])

        if additional_info:
            return primary_answer + " " + " ".join(additional_info)
        return primary_answer

    def retrieve_advanced_rag_answer(self, query: str) -> str:
        """Enhanced retrieval using multi-document context and answer synthesis."""
        if self.collection is None:
            self.logger.error("No collection set for querying")
            return "Database not initialized. Please set up the document collection first."

        conversation_context = ""
        if self.memory_agent:
            conversation_context = self.memory_agent.get_context_summary()
            self.logger.info(f"Using conversation context: {conversation_context[:200]}...")

        query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy().tolist()

        count_info = self.collection.count()
        if count_info == 0:
            return "No documents found in the database."

        actual_top_k = min(self.TOP_K, count_info)
        results = self.collection.query(query_embeddings=query_embedding, n_results=actual_top_k)

        if not results or not results["documents"] or len(results["documents"][0]) == 0:
            return "Answer not found in the documents."

        context_chunks = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]

            if "base_document" not in metadata:
                doc_id = results["ids"][0][i]
                metadata["base_document"] = doc_id.split("_chunk")[0]

            context_chunks.append({
                "text": doc,
                "metadata": metadata,
                "id": results["ids"][0][i]
            })

        rerank_inputs = [(query, chunk["text"]) for chunk in context_chunks]
        rerank_scores = re_ranker.predict(rerank_inputs)

        for i, score in enumerate(rerank_scores):
            context_chunks[i]["score"] = float(score)

        context_chunks = sorted(context_chunks, key=lambda x: x["score"], reverse=True)
        context_window_size = min(5, len(context_chunks))
        top_chunks = context_chunks[:context_window_size]

        document_groups = {}
        for chunk in top_chunks:
            base_doc = chunk["metadata"]["base_document"]
            if base_doc not in document_groups:
                document_groups[base_doc] = []
            document_groups[base_doc].append(chunk)

        combined_context = ""
        if conversation_context:
            combined_context += f"\nConversation Context:\n{conversation_context}\n\n"

        for base_doc, chunks in document_groups.items():
            doc_text = "\n".join([f"{chunk['text']}" for chunk in chunks])
            combined_context += f"\n{doc_text}\n"

        answer_candidates = []

        for chunk in top_chunks:
            try:
                answer = self.qa_model.qa_pipeline(question=query, context=chunk["text"])
                answer["source_chunk"] = chunk["id"]
                answer["source_doc"] = chunk["metadata"]["document"]
                answer_candidates.append(answer)
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk['id']}: {e}")
                continue

        try:
            combined_answer = self.qa_model.qa_pipeline(question=query, context=combined_context)
            combined_answer["source_chunk"] = "combined_context"
            combined_answer["source_doc"] = "multiple_documents"
            answer_candidates.append(combined_answer)
        except Exception as e:
            self.logger.error(f"Error processing combined context: {e}")

        if not answer_candidates:
            return "Unable to extract an answer from the documents."

        final_answer = self.merge_overlapping_answers(answer_candidates)

        try:
            prompt = f"""
            Question: {query}

            Context from relevant documents:
            {combined_context}

            Based solely on the information provided in the context, provide a comprehensive answer to the question.
            """

            model = genai.GenerativeModel('gemini-1.5-pro')
            gemini_response = model.generate_content(prompt)

            if gemini_response.text and len(gemini_response.text) > len(final_answer):
                final_answer = gemini_response.text
        except Exception as e:
            self.logger.error(f"Error using Gemini API: {e}")

        if len(final_answer.split()) < 5:
            final_answer = f"{final_answer} (Note: This answer may be incomplete. The most relevant text from the document states: '{top_chunks[0]['text'][:200]}...')"

        return final_answer

    def set_summary_agent(self, summary_agent: 'SummaryGeneratorAgent'):
        self.summary_agent = summary_agent

    def process(self, query: str) -> str:
        """Process a query and return an answer"""
        self.logger.info(f"Processing query: {query}")
        return self.retrieve_few_shot_answer(query)

class AnswerValidatorAgent(Agent):
    """Agent responsible for validating and refining answers with semantic similarity evaluation"""
    def __init__(self):
        super().__init__("AnswerValidator")
        self.nlp = nlp
        self.similarity_model = embedding_model

    def calculate_semantic_similarity(self, question: str, answer: str) -> float:
        """Calculate semantic similarity between question and answer using embeddings"""
        question_embedding = self.similarity_model.encode(question, convert_to_tensor=True)
        answer_embedding = self.similarity_model.encode(answer, convert_to_tensor=True)
        similarity = cosine_similarity(
            question_embedding.unsqueeze(0).cpu(),
            answer_embedding.unsqueeze(0).cpu()
        )[0][0]
        return similarity

    def validate_answer(self, answer: str, query: str) -> Dict[str, Any]:
        """Validate and score the answer quality with semantic similarity"""
        if len(answer.split()) < 5:
            return {
                "valid": False,
                "score": 0.3,
                "reason": "Answer is too short",
                "refined_answer": answer,
                "semantic_similarity": 0.0
            }

        generic_phrases = ["not found", "unable to extract", "not available"]
        if any(phrase in answer.lower() for phrase in generic_phrases):
            return {
                "valid": False,
                "score": 0.4,
                "reason": "Answer is too generic",
                "refined_answer": answer,
                "semantic_similarity": 0.0
            }

        # Calculate semantic similarity
        semantic_sim = self.calculate_semantic_similarity(query, answer)

        query_doc = self.nlp(query)
        answer_doc = self.nlp(answer)

        query_entities = set([ent.text.lower() for ent in query_doc.ents])
        answer_entities = set([ent.text.lower() for ent in answer_doc.ents])

        entity_overlap = query_entities.intersection(answer_entities)
        validation_score = min(1.0, 0.5 + (len(entity_overlap) / max(1, len(query_entities))) * 0.5)

        legal_terms = ["agreement", "contract", "party", "parties", "section", "clause", "terms"]
        has_legal_terms = any(term in answer.lower() for term in legal_terms)

        if has_legal_terms:
            validation_score = min(1.0, validation_score + 0.2)

        # Incorporate semantic similarity into final score (weighted average)
        final_score = (validation_score * 0.6) + (semantic_sim * 0.4)

        return {
            "valid": final_score > 0.5,
            "score": final_score,
            "reason": "Answer validation complete",
            "refined_answer": answer,
            "semantic_similarity": semantic_sim,
            "entity_overlap": len(entity_overlap),
            "has_legal_terms": has_legal_terms
        }

    def refine_answer(self, answer: str, validation_result: Dict[str, Any]) -> str:
        """Refine the answer based on validation results"""
        if validation_result["valid"]:
            return answer

        if validation_result["score"] < 0.5:
            feedback = []
            if validation_result["semantic_similarity"] < 0.4:
                feedback.append("The answer may not be semantically relevant to the question.")
            if validation_result["entity_overlap"] == 0:
                feedback.append("The answer doesn't mention key entities from the question.")
            if not validation_result["has_legal_terms"]:
                feedback.append("The answer may lack legal specificity.")

            feedback_str = "\n".join(f"- {f}" for f in feedback)
            return f"{answer}\n\n(Note: This answer may not fully address your question. {feedback_str})"

        return answer

    def process(self, data: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
        """Process and validate an answer, returning both refined answer and validation metrics"""
        query, answer = data
        self.logger.info("Validating answer")
        validation_result = self.validate_answer(answer, query)

        if not validation_result["valid"]:
            self.logger.warning(f"Answer validation failed: {validation_result['reason']}")

        refined_answer = self.refine_answer(answer, validation_result)
        return refined_answer, validation_result

class SummaryGeneratorAgent(Agent):
    """Agent responsible for generating summaries from documents or answers"""
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        super().__init__("SummaryGenerator")
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Generate a summary for the given text"""
        if not text or len(text.split()) < 50:
            return "Text too short for summarization."
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]["summary_text"]
        except Exception as e:
            self.logger.error(f"Error during summarization: {e}")
            return "Summarization failed."

    def process(self, text: str) -> str:
        """Process incoming text and return a summary"""
        self.logger.info("Generating summary...")
        return self.summarize_text(text)

class CoordinatorAgent(Agent):
    """Agent responsible for coordinating the multi-agent workflow"""
    def __init__(self, use_lora=True, use_qlora=False):
        super().__init__("Coordinator")
        self.document_extractor = DocumentExtractorAgent()
        self.text_processor = TextProcessingAgent()
        self.embedder = EmbeddingAgent()
        self.db_agent = DatabaseAgent()
        self.query_agent = QueryAgent(use_lora=use_lora, use_qlora=use_qlora)
        self.validator = AnswerValidatorAgent()
        self.memory_agent = MemoryAgent()
        self.summary_agent = SummaryGeneratorAgent()

        self.few_shot_examples = {
            "Who is the agreement between?": "The agreement is between FEDERATED INVESTMENT MANAGEMENT COMPANY and FEDERATED ADVISORY SERVICE COMPANY",
            "What is the title of John B. Fisher?": "President",
            "What is the title of J. Christopher Donahue?": "Chairman"
        }

        self.query_agent.few_shot_examples = self.few_shot_examples
        self.query_agent.set_memory_agent(self.memory_agent)

    def initialize_system(self, pdf_files: List[str]):
        """Initialize the system with PDF files"""
        self.logger.info("Starting system initialization")
        start_time = time.time()

        self.logger.info("Step 1: Extracting text from documents")
        documents = self.document_extractor.process(pdf_files)

        self.logger.info("Step 2: Processing document text")
        structured_docs, chunked_docs = self.text_processor.process(documents)

        self.logger.info("Step 3: Creating document embeddings")
        embedded_docs = self.embedder.process(chunked_docs)

        self.logger.info("Step 4: Storing embeddings in database")
        collection = self.db_agent.process((embedded_docs, chunked_docs))

        self.query_agent.set_collection(collection)

        end_time = time.time()
        self.logger.info(f"System initialization completed in {end_time - start_time:.2f} seconds")

    def fine_tune_qa_model(self, training_examples, eval_examples=None, output_dir="./fine_tuned_qa"):
        """Fine-tune the QA model with provided examples"""
        logger.info("Starting QA model fine-tuning")

        train_dataset = self.query_agent.qa_model.create_dataset(training_examples)
        eval_dataset = self.query_agent.qa_model.create_dataset(eval_examples) if eval_examples else None

        self.query_agent.qa_model.train(train_dataset, eval_dataset, output_dir)

        self.query_agent.qa_model = FineTunedQAModel(
            fine_tuned_path=output_dir,
            use_lora=self.query_agent.qa_model.use_lora,
            use_qlora=self.query_agent.qa_model.use_qlora
        )

        logger.info("QA model fine-tuning completed successfully")

    def process_query(self, query: str, context: str = None) -> Tuple[str, Dict[str, Any]]:
        """Process a query through the multi-agent system and return answer with metrics"""
        self.logger.info(f"Processing query: {query}")

        final_query = query
        if context:
            final_query = query + "\n\n" + context

        raw_answer = self.query_agent.process(final_query)
        refined_answer, validation_metrics = self.validator.process((final_query, raw_answer))
        self.memory_agent.add_interaction(final_query, refined_answer)

        return refined_answer, validation_metrics

    def process(self, input_data: Any) -> Any:
        """Process input based on type"""
        if isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            return self.initialize_system(input_data)
        elif isinstance(input_data, str):
            answer, metrics = self.process_query(input_data)
            return answer
        else:
            self.logger.error(f"Unsupported input type: {type(input_data)}")
            return "Unsupported input type"

def main():
    from PIL import Image
    import pytesseract

    # Create coordinator agent with LoRA enabled (set use_qlora=True for QLoRA)
    coordinator = CoordinatorAgent(use_lora=True, use_qlora=False)

    # Initialize system with PDF files
    pdf_files = [
        "/content/FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT.pdf",
        "/content/FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_AMENDMENT.pdf",
        "/content/FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_POWEROF.pdf",
        "/content/FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_SECONDAMENDMENT.pdf",
    ]
    coordinator.initialize_system(pdf_files)
    logger.info("✅ Multi-agent system initialized successfully!")

    #Example of how to fine-tune (uncomment and provide your training data)
    training_examples = [
        {
            "question": "What is consideration in contract law?",
            "context": "Consideration is value exchanged between parties to a contract...",
            "answer": "value exchanged between parties to a contract"
        },
        # More examples...
    ]
    coordinator.fine_tune_qa_model(training_examples)

    print("\n===== Welcome to the Legal Document Question Answering System =====")
    print("This system uses a multi-agent approach to answer questions about legal documents.")
    print("Documents have been processed and indexed. You can now ask questions about them.\n")

    def extract_text_from_image(image_path):
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print("❌ Error in OCR:", e)
            return ""

    while True:
        user_query = input("Enter your question (or 'exit' to quit): ").strip()
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("👋 Exiting the system. Goodbye!")
            break

        upload_image = input("Do you want to upload an image with your query? (yes/no): ").strip().lower()
        query_to_process = user_query

        if upload_image == 'yes':
            image_path = input("Enter the image file path (e.g., /path/to/image.png): ").strip()
            ocr_text = extract_text_from_image(image_path)

            if ocr_text:
                print("\n📝 OCR Extracted Text:")
                print(ocr_text)
                query_to_process = ocr_text
            else:
                print("⚠ No text found in the image. Proceeding with your typed query.")

        answer, metrics = coordinator.process_query(query_to_process)

        print(f"\n🧠 Answer: {answer}\n")
        print("📊 Evaluation Metrics:")
        print(f"- Semantic Similarity: {metrics['semantic_similarity']:.2f}")
        print(f"- Validation Score: {metrics['score']:.2f}")
        print(f"- Entity Overlap: {metrics['entity_overlap']} entities")
        print(f"- Contains Legal Terms: {'Yes' if metrics['has_legal_terms'] else 'No'}")
        print(f"- Validation Status: {'Valid' if metrics['valid'] else 'Invalid'}")
        print(f"- Reason: {metrics['reason']}\n")

if __name__ == "__main__":
    main()

