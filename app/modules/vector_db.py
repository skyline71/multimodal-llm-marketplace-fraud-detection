# app/modules/vector_db.py

import chromadb
from sentence_transformers import SentenceTransformer
import os

class VectorDB:
    def __init__(self, persist_dir="./data/chroma_db"):
        os.makedirs(persist_dir, exist_ok=True)
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="fraud_lots",
            metadata={"hnsw:space": "cosine"}
        )

    def add_lot(self, lot_id: str, text: str, detected_objects: list, risk_level: str, verdict: str):
        doc = f"Описание: {text}. Объекты: {', '.join(detected_objects)}. Уровень риска: {risk_level}. Вердикт: {verdict}"
        emb = self.model.encode(doc).tolist()
        try:
            self.collection.add(
                ids=[lot_id],
                embeddings=[emb],
                documents=[doc],
                metadatas=[{"risk_level": risk_level, "text": text, "objects": str(detected_objects)}]
            )
        except Exception as e:
            # Лот уже существует — пропускаем
            pass

    def query_similar(self, query_text: str, top_k=2):
        query_emb = self.model.encode(query_text).tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        cases = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            cases.append({
                "description": doc,
                "risk_level": meta["risk_level"],
                "recommendation": "Проверьте историю продавца" if meta["risk_level"] != "низкий" else "Лот безопасен"
            })
        return cases