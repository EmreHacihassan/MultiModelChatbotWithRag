#!/usr/bin/env python
"""Pending dökümanları manuel olarak işle."""

import json
from pathlib import Path
from rag.pipelines import RAGPipeline, get_embedding_model

# Index'i oku
index_file = Path('rag/index/document_index.json')
with open(index_file, 'r', encoding='utf-8') as f:
    index = json.load(f)

print('=== DOCUMENT INDEX ===')
print(f'Documents: {len(index.get("documents", {}))}')
print(f'Pending: {len(index.get("pending", {}))}')

for h, info in index.get('pending', {}).items():
    print(f'  Pending: {info.get("file_name")} - {info.get("status")}')
    print(f'    Path: {info.get("file_path")}')

# Embedding model'i test et
print('\n=== EMBEDDING MODEL ===')
model = get_embedding_model()
test = model.encode(['test'], convert_to_numpy=True)
print(f'Model OK, embedding size: {len(test[0])}')

# RAG Pipeline'ı başlat
print('\n=== RAG PIPELINE ===')
rag = RAGPipeline()
stats = rag.get_stats()
print(f'Stats: {stats}')

# Pending dökümanları manuel işle
print('\n=== PROCESSING PENDING DOCUMENTS ===')
for file_hash, info in list(index.get('pending', {}).items()):
    file_path = info.get('file_path')
    print(f'Processing: {info.get("file_name")}')
    
    # Dosya var mı kontrol et
    if Path(file_path).exists():
        result = rag.add_document(file_path)
        print(f'  Result: {result}')
    else:
        print(f'  ERROR: File not found at {file_path}')

# Final stats
print('\n=== FINAL STATS ===')
final_stats = rag.get_stats()
print(f'Stats: {final_stats}')

docs = rag.list_documents()
print(f'Documents: {len(docs)}')
for doc in docs:
    print(f'  - {doc.get("file_name")} ({doc.get("status", "unknown")})')
