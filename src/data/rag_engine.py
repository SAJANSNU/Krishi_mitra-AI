import os
import json
import pickle
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL
from src.utils.logger import log

class RAGEngine:
    def __init__(self):
        self.pc = None
        self.embeddings = None
        self.scheme_embedder = None
        self.pinecone_index = None
        self.scheme_contents = {}
        self.scheme_summaries = {}
        self.scheme_names = []
        self.scheme_embeddings = None
        self.rag_system_save_path = "krishi_mitra_rag_system.pkl"
        
        self._initialize_components()

    def _initialize_components(self):
        """Initialize RAG components dynamically."""
        try:
            # Initialize Pinecone
            if PINECONE_API_KEY:
                self.pc = Pinecone(api_key=PINECONE_API_KEY)
                log.info("Pinecone initialized")

            # Initialize embeddings
            if GOOGLE_API_KEY:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=EMBEDDING_MODEL, 
                    google_api_key=GOOGLE_API_KEY
                )
                log.info("Google embeddings initialized")

            # Initialize sentence transformer
            try:
                self.scheme_embedder = SentenceTransformer('all-mpnet-base-v2')
                log.info("Sentence transformer initialized")
            except Exception as e:
                log.warning(f"Failed to load sentence transformer: {e}")

        except Exception as e:
            log.error(f"Failed to initialize RAG components: {e}")

    def build_knowledge_base(self, pdf_path: str):
        """Build knowledge base from PDF with government schemes."""
        try:
            if not os.path.exists(pdf_path):
                log.error(f"PDF file not found: {pdf_path}")
                return False

            log.info("Building RAG: Extracting PDF content...")
            loader = PyPDFLoader(pdf_path)
            full_text = "\n".join([page.page_content for page in loader.load()])

            # Parse schemes
            self.known_schemes = [
                "Pradhan Mantri Kisan Samman Nidhi", "PM Kisan Maan Dhan Yojana", 
                "Credit facility for farmers", "Crop insurance schemes", 
                "Pradhan Mantri Krishi Sinchai Yojana", "Interest subvention for dairy sector",
                "National Scheme of Welfare of Fishermen", "Agriculture Infrastructure Fund", 
                "KCC for animal husbandry and fisheries", "National Mission on Edible Oils",
                "Krishi UDAN scheme", "Group Accident Insurance scheme for Fishermen",
                "Primary Agricultural Credit Societies", "Vibrant Villages Programme",
                "Mission Amrit Sarovar", "National Mission on Natural Farming",
                "Animal Husbandry Infrastructure Development Fund", "National Beekeeping and Honey Mission",
                "Unique package for farmers", "Tea Development & Promotion Scheme",
                "Financial assistance to organic farmers", "Prime Minister Dhan-Dhaanya Krishi Yojana"
            ]

            schemes = {}
            for scheme_name in self.known_schemes:
                start_index = full_text.lower().find(scheme_name.lower())
                if start_index != -1:
                    end_index = len(full_text)
                    for other_scheme in self.known_schemes:
                        if other_scheme != scheme_name:
                            other_start = full_text.lower().find(other_scheme.lower(), start_index + 1)
                            if other_start != -1:
                                end_index = min(end_index, other_start)
                    
                    content = full_text[start_index:end_index].strip()
                    if len(content) > 100:
                        schemes[scheme_name] = content

            self.scheme_contents = schemes
            log.info(f"Building RAG: Parsed {len(schemes)} schemes.")

            # Create summaries and chunks
            log.info("Building RAG: Creating summaries and chunks...")
            summaries = {name: ' '.join(content.split()[:100]) for name, content in self.scheme_contents.items()}
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = []
            
            for scheme_name, content in self.scheme_contents.items():
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata={"scheme": scheme_name}))

            # Setup Pinecone and VectorStores
            if self.pc and self.embeddings:
                log.info("Building RAG: Setting up Pinecone index...")
                if PINECONE_INDEX_NAME not in [index.name for index in self.pc.list_indexes()]:
                    self.pc.create_index(
                        name=PINECONE_INDEX_NAME, 
                        dimension=768, 
                        metric="cosine", 
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                
                self.pinecone_index = self.pc.Index(PINECONE_INDEX_NAME)
                
                log.info("Building RAG: Ingesting documents into Pinecone...")
                PineconeVectorStore.from_documents(
                    documents=documents, 
                    embedding=self.embeddings, 
                    index_name=PINECONE_INDEX_NAME
                )

            self.scheme_summaries = summaries
            self.scheme_names = list(summaries.keys())
            
            if self.scheme_embedder:
                self.scheme_embeddings = self.scheme_embedder.encode(list(summaries.values()))

            self._save_system()
            log.info("RAG System build complete!")
            return True

        except Exception as e:
            log.error(f"Failed to build knowledge base: {e}")
            return False

    def retrieve(self, query: str, context_type: Optional[str] = None, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Dynamic retrieval method for context."""
        try:
            if context_type == "government_scheme" or "scheme" in query.lower():
                return self._query_schemes(query, top_k)
            else:
                return self._query_general(query, top_k)
        except Exception as e:
            log.error(f"Retrieval failed: {e}")
            return {"results": [], "total_results": 0, "categories": []}

    def search_knowledge(self, query: str, context_type: Optional[str] = None, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Alternative method name for retrieval."""
        return self.retrieve(query, context_type, top_k, **kwargs)

    def _query_schemes(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Query government schemes specifically."""
        try:
            if not self.scheme_embedder or not hasattr(self, 'scheme_embeddings'):
                return {"results": [], "total_results": 0, "categories": ["schemes"]}

            # Find relevant schemes
            relevant_schemes = self._find_relevant_schemes(query, top_k)
            
            if not relevant_schemes:
                return {"results": [], "total_results": 0, "categories": ["schemes"]}

            # Get detailed content
            results = []
            for scheme_name, score in relevant_schemes:
                content = self.scheme_summaries.get(scheme_name, "No details available")
                results.append({
                    "content": content,
                    "scheme": scheme_name,
                    "score": score
                })

            return {
                "results": results,
                "total_results": len(results),
                "categories": ["schemes"]
            }

        except Exception as e:
            log.error(f"Scheme query failed: {e}")
            return {"results": [], "total_results": 0, "categories": ["schemes"]}

    def _query_general(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Query general agricultural knowledge."""
        try:
            if not self.pinecone_index or not self.embeddings:
                return {"results": [], "total_results": 0, "categories": ["general"]}

            vectorstore = PineconeVectorStore(index=self.pinecone_index, embedding=self.embeddings)
            retrieved_docs = vectorstore.similarity_search(query, k=top_k)

            results = []
            for doc in retrieved_docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

            return {
                "results": results,
                "total_results": len(results),
                "categories": ["agricultural_knowledge"]
            }

        except Exception as e:
            log.error(f"General query failed: {e}")
            return {"results": [], "total_results": 0, "categories": ["general"]}

    def _find_relevant_schemes(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find relevant schemes using semantic similarity."""
        try:
            if not self.scheme_embedder or not hasattr(self, 'scheme_embeddings'):
                return []

            query_embedding = self.scheme_embedder.encode([query])
            similarities = cosine_similarity(query_embedding, self.scheme_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            return [(self.scheme_names[idx], similarities[idx]) for idx in top_indices if similarities[idx] > 0.3]

        except Exception as e:
            log.error(f"Scheme finding failed: {e}")
            return []

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "schemes_loaded": len(self.scheme_contents),
            "has_pinecone": self.pinecone_index is not None,
            "has_embeddings": self.embeddings is not None,
            "has_sentence_transformer": self.scheme_embedder is not None
        }

    def test_rag_retrieval(self, test_queries: List[str]) -> Dict[str, Any]:
        """Test RAG retrieval with sample queries."""
        results = {}
        for query in test_queries:
            try:
                result = self.retrieve(query, top_k=2)
                results[query] = {
                    "success": result["total_results"] > 0,
                    "count": result["total_results"]
                }
            except Exception as e:
                results[query] = {"success": False, "error": str(e)}
        return results

    def _save_system(self):
        """Save RAG system state."""
        try:
            save_data = {
                'scheme_names': self.scheme_names,
                'scheme_summaries': self.scheme_summaries,
                'scheme_embeddings': self.scheme_embeddings,
                'scheme_contents': self.scheme_contents,
            }
            with open(self.rag_system_save_path, 'wb') as f:
                pickle.dump(save_data, f)
            log.info(f"RAG system saved to {self.rag_system_save_path}")
        except Exception as e:
            log.error(f"Failed to save RAG system: {e}")

    def _load_system(self) -> bool:
        """Load RAG system state."""
        try:
            if not os.path.exists(self.rag_system_save_path):
                return False

            with open(self.rag_system_save_path, 'rb') as f:
                data = pickle.load(f)

            self.scheme_names = data['scheme_names']
            self.scheme_summaries = data['scheme_summaries']
            self.scheme_embeddings = data['scheme_embeddings']
            self.scheme_contents = data['scheme_contents']

            if self.pc:
                self.pinecone_index = self.pc.Index(PINECONE_INDEX_NAME)

            log.info(f"RAG system loaded from {self.rag_system_save_path}")
            return True

        except Exception as e:
            log.error(f"Failed to load RAG system: {e}")
            return False
