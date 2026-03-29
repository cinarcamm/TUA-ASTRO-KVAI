from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

@dataclass
class RetrievalResult:
    error_type: str
    chunks: List[str]

class RAGEngine:
    """
    RadShield için LangChain tabanlı %100 YEREL (Offline) RAG motoru.
    """
    def __init__(
        self,
        docs_dir: str = os.path.join(os.path.dirname(__file__), "..", "docs"),
        persist_dir: str = ".chroma_rag",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gemini-1.5-pro", 
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self.docs_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs")))
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._vectorstore: Optional[Chroma] = None
        self._indexed: bool = False

    def index_documents(self) -> bool:
        print(f"[RAG][index_documents] Scanning docs directory: {self.docs_dir}")
        if not self.docs_dir.exists():
            print(f"[RAG][index_documents][ERROR] docs directory not found: {self.docs_dir}")
            return False

        pdf_files = list(self.docs_dir.glob("*.pdf"))
        txt_files = list(self.docs_dir.glob("*.txt"))
        print(f"[RAG][index_documents] Found files -> pdf: {len(pdf_files)}, txt: {len(txt_files)}, total: {len(pdf_files) + len(txt_files)}")

        raw_docs = self._load_docs()
        print(f"[RAG][index_documents] Loaded raw docs/chunks before split: {len(raw_docs)}")
        if not raw_docs:
            return False

        chunks = self._split_docs(raw_docs)
        print(f"[RAG][index_documents] Split into chunks: {len(chunks)}")
        if not chunks:
            return False

        embeddings = self._safe_embeddings()
        if embeddings is None:
            return False

        try:
            self._vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.persist_dir,
            )
            self._vectorstore.persist()
            self._indexed = True
            print(f"[RAG][index_documents] Index persisted successfully at: {self.persist_dir}")
            return True
        except Exception as e:
            print(f"[RAG][index_documents][ERROR] Failed to build Chroma index: {str(e)}")
            self._vectorstore = None
            self._indexed = False
            return False

    def generate_expert_report(self, analysis_results: Dict[str, Any]) -> str:
        parsed = self._parse_analysis_results(analysis_results)
        error_type = self._dominant_error_type(parsed["stuck_count"], parsed["seu_count"])

        retrieval = self.retrieve_technical_context(error_type=error_type, top_k=3)
        technical_chunks = retrieval.chunks

        if not technical_chunks:
            # HATA BURADAYDI! _summary yerine _fallback_summary yapıldı.
            return self._fallback_summary(
                parsed,
                technical_chunks,
                reason="retrieval returned 0 chunks (index/db/embedding issue)",
            )

        llm = self._safe_llm()
        if llm is None:
            return self._fallback_summary(parsed, technical_chunks, reason="Ollama initialization failed")

        prompt = self._build_prompt(
            raw_analysis=parsed["raw_report"],
            stuck_count=parsed["stuck_count"],
            seu_count=parsed["seu_count"],
            technical_references=technical_chunks,
            dominant_error_type=error_type,
        )

        try:
            response = llm.invoke(prompt)
            if hasattr(response, "content") and response.content:
                return str(response.content).strip()
            return self._fallback_summary(parsed, technical_chunks, reason="LLM returned empty content")
        except Exception as e:
            return self._fallback_summary(parsed, technical_chunks, reason=f"LLM invoke failed: {str(e)}")

    def retrieve_technical_context(self, error_type: str, top_k: int = 3) -> RetrievalResult:
        if not self._indexed:
            self._load_existing_db_if_possible()

        if self._vectorstore is None:
            return RetrievalResult(error_type=error_type, chunks=[])

        query = self._query_for_error_type(error_type)
        try:
            docs = self._vectorstore.similarity_search(query, k=top_k)
            chunks = [d.page_content.strip() for d in docs if getattr(d, "page_content", "").strip()]
            return RetrievalResult(error_type=error_type, chunks=chunks[:top_k])
        except Exception as e:
            print(f"[RAG][retrieve][ERROR] similarity_search failed: {str(e)}")
            return RetrievalResult(error_type=error_type, chunks=[])

    def _load_docs(self) -> List[Any]:
        docs: List[Any] = []
        pdf_files = list(self.docs_dir.glob("*.pdf"))
        txt_files = list(self.docs_dir.glob("*.txt"))

        for p in pdf_files:
            try:
                loader = PyPDFLoader(str(p))
                docs.extend(loader.load())
            except Exception as e:
                print(f"[RAG][_load_docs][WARN] PDF load failed ({p.name}): {str(e)}")

        for t in txt_files:
            try:
                loader = TextLoader(str(t), encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                print(f"[RAG][_load_docs][WARN] TXT load failed ({t.name}): {str(e)}")
        return docs

    def _split_docs(self, docs: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(docs)

    def _safe_embeddings(self):
        try:
            emb = HuggingFaceEmbeddings(model_name=self.embedding_model)
            print(f"[RAG][_safe_embeddings] Local embedding active: {self.embedding_model}")
            return emb
        except Exception as e:
            print(f"[RAG][_safe_embeddings][ERROR] Local embedding failed: {str(e)}")
            return None

    def _safe_llm(self) -> Optional[ChatOllama]:
        # GOOGLE API KONTROLLERİ TAMAMEN SİLİNDİ. DİREKT OLLAMA'YA BAĞLANIYOR.
        try:
            llm = ChatOllama(
                model=self.llm_model, 
                temperature=0.1
            )
            print(f"[RAG][_safe_llm] Local Ollama LLM active: {self.llm_model}")
            return llm
        except Exception as e:
            print(f"[RAG][_safe_llm][ERROR] Ollama handshake failed. Is Ollama running? Error: {str(e)}")
            return None

    def _load_existing_db_if_possible(self) -> None:
        embeddings = self._safe_embeddings()
        if embeddings is None:
            return
        try:
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
            )
            self._indexed = True
        except Exception:
            self._vectorstore = None
            self._indexed = False

    @staticmethod
    def _query_for_error_type(error_type: str) -> str:
        e = (error_type or "").lower()
        if "seu" in e:
            return "SEU single event upset kozmik radyasyon kaynaklı bit flip hataları, neden analizi, etkiler, KURAL-XX ve SOP-X önlemleri"
        return "stuck-at fault donma hatası, sabitlenen telemetri, sensör kilitlenmesi, neden analizi, KURAL-XX ve SOP-X düzeltici aksiyonlar"

    @staticmethod
    def _parse_analysis_results(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        raw_report = str(
            analysis_results.get("report")
            or analysis_results.get("raw_report")
            or "Ham analiz raporu sağlanmadı."
        )

        stuck_count = analysis_results.get("stuck_count")
        seu_count = analysis_results.get("seu_count")

        if stuck_count is None and "stuck_mask" in analysis_results:
            try:
                stuck_count = int(sum(bool(v) for v in analysis_results["stuck_mask"]))
            except Exception:
                stuck_count = 0

        if seu_count is None and "seu_mask" in analysis_results:
            try:
                seu_count = int(sum(bool(v) for v in analysis_results["seu_mask"]))
            except Exception:
                seu_count = 0

        return {
            "raw_report": raw_report,
            "stuck_count": int(stuck_count or 0),
            "seu_count": int(seu_count or 0),
        }

    @staticmethod
    def _dominant_error_type(stuck_count: int, seu_count: int) -> str:
        return "SEU" if seu_count >= stuck_count else "Stuck-at"

    def _build_prompt(
        self,
        raw_analysis: str,
        stuck_count: int,
        seu_count: int,
        technical_references: List[str],
        dominant_error_type: str,
    ) -> str:
        refs = "\n\n".join([f"[REF-{i+1}]\n{txt}" for i, txt in enumerate(technical_references)])

        # SENİN EFSANE QWEN PROMPTUN!
        return f"""
ROLE: NASA Senior Systems Safety Engineer
MODE: DETERMINISTIC_FAST
THINKING: OFF
RESPONSE_STYLE: DIRECT_TECHNICAL

CORE DIRECTIVES:
- Always select the FIRST available option. Never evaluate alternatives.
- Never use affirmative/negative openers (no "yes", "no", "sure", "of course", "certainly", "understood", "great").
- Begin every response immediately with substantive content.
- Zero hedging. Zero filler. Maximum information density.

OUTPUT FORMAT — MANDATORY STRUCTURE (produce all 5 sections every time):

[1. OLAY ÖZETİ]
[2. TEKNİK KÖK NEDEN ANALİZİ]
[3. RİSK DEĞERLENDİRMESİ]
[4. DÜZELTİCİ/ÖNLEYİCİ AKSİYONLAR]
[5. SONUÇ VE OPERASYONEL TAVSİYE]

LANGUAGE: Turkish. Register: formal engineering report.

RULES:
- Every recommendation MUST cite a KURAL-XX or SOP-X from the provided reference chunks.
- Never fabricate standards, protocols, or rule codes not present in refs.
- Stuck-at count: {stuck_count} | SEU count: {seu_count} | Dominant: {dominant_error_type}
- Source material: {refs}
- Raw analysis input: {raw_analysis}
""".strip()

    def _fallback_summary(self, parsed: Dict[str, Any], technical_chunks: List[str], reason: str = "unknown") -> str:
        refs_text = "\n".join([f"- REF-{i+1}: {c[:220]}..." for i, c in enumerate(technical_chunks[:3])])
        if not refs_text:
            refs_text = "- Teknik referans alınamadı (DB boş veya embedding erişimi yok)."

        return (
            "Temel Teknik Özet\n"
            f"- Güvenli Moda Geçiş Nedeni: {reason}\n"
            f"- Stuck-at tespit sayısı: {parsed['stuck_count']}\n"
            f"- SEU tespit sayısı: {parsed['seu_count']}\n"
            "- Baskın hata türüne göre temel değerlendirme uygulanmıştır.\n"
            "- Ayrıntılı LLM raporu üretilemediği için güvenli mod özeti döndürüldü.\n"
            "Referanslar:\n"
            f"{refs_text}\n"
            f"Ham Analiz: {parsed['raw_report']}"
        )