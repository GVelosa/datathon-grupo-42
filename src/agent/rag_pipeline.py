"""Pipeline RAG com FAISS para recuperação de contexto do knowledge base."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
DEFAULT_TOP_K: int = 3
EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RAGPipeline:
    """Pipeline RAG: carrega documentos, cria embeddings e serve via FAISS.

    Attributes:
        chunk_size: Tamanho máximo de cada chunk em caracteres.
        chunk_overlap: Overlap entre chunks consecutivos.
        top_k: Número de chunks a recuperar por query.
        index: Índice FAISS (carregado após build_index ou load_index).
        chunks: Lista de chunks com metadados.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        """Inicializa o pipeline RAG.

        Args:
            chunk_size: Tamanho máximo de cada chunk em caracteres.
            chunk_overlap: Overlap entre chunks para preservar contexto.
            top_k: Número de chunks a recuperar por query.
            embedding_model: Identificador do modelo sentence-transformers.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        self.index: Any = None
        self.chunks: list[dict[str, str]] = []
        self._encoder: Any = None

    def _get_encoder(self) -> Any:
        """Carrega o modelo de embeddings (lazy loading).

        Returns:
            Instância do SentenceTransformer carregada.
        """
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model_name)
            logger.info("Encoder carregado", extra={"model": self.embedding_model_name})
        return self._encoder

    def _split_text(self, text: str, source: str) -> list[dict[str, str]]:
        """Divide texto em chunks com overlap.

        Args:
            text: Texto completo a dividir.
            source: Identificador da fonte (nome do arquivo).

        Returns:
            Lista de dicionários com 'content' e 'source'.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"content": chunk, "source": source})
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def load_documents(self, docs_dir: str | Path) -> int:
        """Carrega documentos Markdown e TXT de um diretório.

        Args:
            docs_dir: Diretório com arquivos .md e .txt.

        Returns:
            Número total de chunks gerados.
        """
        docs_dir = Path(docs_dir)
        self.chunks = []

        for pattern in ("**/*.md", "**/*.txt"):
            for file_path in sorted(docs_dir.glob(pattern)):
                text = file_path.read_text(encoding="utf-8")
                file_chunks = self._split_text(text, source=file_path.name)
                self.chunks.extend(file_chunks)

        logger.info("Documentos carregados", extra={"chunks": len(self.chunks), "dir": str(docs_dir)})
        return len(self.chunks)

    def build_index(self) -> None:
        """Constrói o índice FAISS a partir dos chunks carregados.

        Raises:
            RuntimeError: Se nenhum chunk foi carregado antes de chamar este método.
        """
        if not self.chunks:
            raise RuntimeError("Nenhum chunk carregado. Chame load_documents() primeiro.")

        import faiss
        import numpy as np

        encoder = self._get_encoder()
        texts = [c["content"] for c in self.chunks]
        embeddings = encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity (após normalização)
        self.index.add(embeddings.astype("float32"))

        logger.info("Índice FAISS construído", extra={"vectors": len(self.chunks), "dim": dim})

    def save_index(self, index_path: str | Path) -> None:
        """Persiste o índice FAISS em disco.

        Args:
            index_path: Caminho base para salvar index.faiss e chunks.json.
        """
        import json

        import faiss

        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path / "index.faiss"))
        (index_path / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False))
        logger.info("Índice FAISS salvo", extra={"path": str(index_path)})

    def load_index(self, index_path: str | Path) -> None:
        """Carrega índice FAISS previamente salvo.

        Args:
            index_path: Caminho base com index.faiss e chunks.json.
        """
        import json

        import faiss

        index_path = Path(index_path)
        self.index = faiss.read_index(str(index_path / "index.faiss"))
        self.chunks = json.loads((index_path / "chunks.json").read_text())
        logger.info("Índice FAISS carregado", extra={"vectors": len(self.chunks)})

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, str | float]]:
        """Recupera os chunks mais relevantes para uma query.

        Args:
            query: Texto da consulta em linguagem natural.
            top_k: Número de chunks a retornar. Usa self.top_k se None.

        Returns:
            Lista de dicionários com 'content', 'source' e 'relevance_score'.

        Raises:
            RuntimeError: Se o índice não foi construído ou carregado.
        """
        if self.index is None:
            raise RuntimeError("Índice não disponível. Chame build_index() ou load_index().")

        import numpy as np

        k = top_k or self.top_k
        encoder = self._get_encoder()
        query_emb = encoder.encode([query], convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        scores, indices = self.index.search(query_emb.astype("float32"), k)

        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["relevance_score"] = float(score)
                results.append(chunk)

        logger.info("RAG retrieval concluído", extra={"query_len": len(query), "results": len(results)})
        return results

    def get_context(self, query: str, top_k: int | None = None) -> str:
        """Retorna contexto concatenado para injeção no prompt do LLM.

        Args:
            query: Texto da consulta.
            top_k: Número de chunks a incluir no contexto.

        Returns:
            String com chunks concatenados e separados por delimitadores.
        """
        chunks = self.retrieve(query, top_k=top_k)
        context_parts = [f"[Fonte: {c['source']}]\n{c['content']}" for c in chunks]
        return "\n\n---\n\n".join(context_parts)
