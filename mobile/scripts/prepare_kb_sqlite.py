#!/usr/bin/env python3
"""
Prepare knowledge base for mobile deployment with sqlite-vec.

This script:
1. Loads the knowledge base from data/kb/
2. Generates embeddings using TFLite sentence encoder
3. Creates SQLite database with sqlite-vec extension
4. Populates database with text chunks, embeddings, and metadata

Usage:
    python prepare_kb_sqlite.py --kb ../../data/kb --encoder ../assets/sentence_encoder.tflite --output ../assets/kb.db
"""

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_knowledge_base(kb_path: Path) -> pd.DataFrame:
    """Load knowledge base from manifest.parquet or fallback to individual files."""
    logger.info(f"Loading knowledge base from {kb_path}")
    
    manifest_path = kb_path / 'manifest.parquet'
    
    if manifest_path.exists():
        logger.info(f"Loading from manifest: {manifest_path}")
        df = pd.read_parquet(manifest_path)
        logger.info(f"Loaded {len(df)} chunks from manifest")
        return df
    
    # Fallback: load from individual text files
    logger.info("Manifest not found, loading from individual files...")
    chunks = []
    
    for txt_file in kb_path.glob('*.txt'):
        if txt_file.name in ['.gitkeep', 'failed.txt', 'classes_cleaned.txt']:
            continue
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Parse filename: Plant_Disease.txt
        filename = txt_file.stem
        parts = filename.split('_', 1)
        plant = parts[0] if len(parts) > 0 else ''
        disease = parts[1] if len(parts) > 1 else ''
        
        chunks.append({
            'doc_id': filename,
            'text': text,
            'plant': plant,
            'disease': disease,
            'url': '',
            'title': filename.replace('_', ' ')
        })
    
    df = pd.DataFrame(chunks)
    logger.info(f"Loaded {len(df)} chunks from individual files")
    return df


def generate_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Generate embeddings using sentence-transformers (for now, will use TFLite later)."""
    logger.info(f"Generating embeddings for {len(texts)} texts using {model_name}")
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def create_sqlite_db(db_path: Path, df: pd.DataFrame, embeddings: np.ndarray):
    """Create SQLite database with knowledge base and embeddings."""
    logger.info(f"Creating SQLite database: {db_path}")
    
    # Ensure output directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database
    if db_path.exists():
        logger.warning(f"Removing existing database: {db_path}")
        db_path.unlink()
    
    # Create connection
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create main table for KB chunks
    cursor.execute('''
        CREATE TABLE kb_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            plant TEXT,
            disease TEXT,
            url TEXT,
            title TEXT,
            n_tokens INTEGER,
            embedding BLOB
        )
    ''')
    
    # Create indices for faster filtering
    cursor.execute('CREATE INDEX idx_plant ON kb_chunks(plant)')
    cursor.execute('CREATE INDEX idx_disease ON kb_chunks(disease)')
    cursor.execute('CREATE INDEX idx_doc_id ON kb_chunks(doc_id)')
    
    # Insert data
    logger.info("Inserting chunks into database...")
    for idx, row in df.iterrows():
        embedding_blob = embeddings[idx].astype(np.float32).tobytes()
        
        cursor.execute('''
            INSERT INTO kb_chunks (doc_id, text, plant, disease, url, title, n_tokens, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row.get('doc_id', ''),
            row.get('text', ''),
            row.get('plant', ''),
            row.get('disease', ''),
            row.get('url', ''),
            row.get('title', ''),
            row.get('n_tokens', 0),
            embedding_blob
        ))
    
    # Create metadata table
    cursor.execute('''
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Store metadata
    metadata = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dim': str(embeddings.shape[1]),
        'num_chunks': str(len(df)),
        'version': '1.0'
    }
    
    for key, value in metadata.items():
        cursor.execute('INSERT INTO metadata (key, value) VALUES (?, ?)', (key, value))
    
    conn.commit()
    conn.close()
    
    # Get database size
    size_mb = db_path.stat().st_size / (1024 * 1024)
    logger.info(f"Database created successfully: {db_path} ({size_mb:.2f} MB)")


def validate_database(db_path: Path):
    """Validate the created database."""
    logger.info("Validating database...")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check chunk count
    cursor.execute('SELECT COUNT(*) FROM kb_chunks')
    chunk_count = cursor.fetchone()[0]
    logger.info(f"Total chunks: {chunk_count}")
    
    # Check metadata
    cursor.execute('SELECT key, value FROM metadata')
    metadata = dict(cursor.fetchall())
    logger.info(f"Metadata: {metadata}")
    
    # Sample a few chunks
    cursor.execute('SELECT id, doc_id, plant, disease, LENGTH(text), LENGTH(embedding) FROM kb_chunks LIMIT 5')
    samples = cursor.fetchall()
    
    logger.info("Sample chunks:")
    for sample in samples:
        logger.info(f"  ID={sample[0]}, doc_id={sample[1]}, plant={sample[2]}, disease={sample[3]}, "
                   f"text_len={sample[4]}, embedding_bytes={sample[5]}")
    
    conn.close()
    logger.info("✅ Database validation successful")


def test_vector_search(db_path: Path, query: str = "What are the symptoms of tomato blight?"):
    """Test vector search functionality (basic cosine similarity without sqlite-vec)."""
    logger.info(f"Testing vector search with query: '{query}'")
    
    # Generate query embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0].astype(np.float32)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Fetch all embeddings (for small KB, this is acceptable)
    cursor.execute('SELECT id, doc_id, plant, disease, text, embedding FROM kb_chunks')
    rows = cursor.fetchall()
    
    # Compute cosine similarity
    similarities = []
    for row in rows:
        chunk_id, doc_id, plant, disease, text, embedding_blob = row
        chunk_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        
        # Cosine similarity
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        
        similarities.append((similarity, chunk_id, doc_id, plant, disease, text[:100]))
    
    # Sort by similarity
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Show top 3
    logger.info("Top 3 results:")
    for i, (score, chunk_id, doc_id, plant, disease, text_preview) in enumerate(similarities[:3], 1):
        logger.info(f"  {i}. Score={score:.4f}, Plant={plant}, Disease={disease}")
        logger.info(f"     Text: {text_preview}...")
    
    conn.close()
    logger.info("✅ Vector search test successful")


def main():
    parser = argparse.ArgumentParser(description='Prepare KB for mobile deployment')
    parser.add_argument('--kb', type=str, default='../../data/kb',
                        help='Path to knowledge base directory')
    parser.add_argument('--encoder', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence encoder model name (will use TFLite later)')
    parser.add_argument('--output', type=str, default='../assets/kb.db',
                        help='Path to output SQLite database')
    parser.add_argument('--test', action='store_true',
                        help='Run vector search test after creation')
    
    args = parser.parse_args()
    
    kb_path = Path(args.kb).resolve()
    db_path = Path(args.output).resolve()
    
    logger.info("=" * 60)
    logger.info("Knowledge Base Preparation for Mobile")
    logger.info("=" * 60)
    
    # Step 1: Load KB
    df = load_knowledge_base(kb_path)
    
    # Step 2: Generate embeddings
    texts = df['text'].tolist()
    embeddings = generate_embeddings(texts, args.encoder)
    
    # Step 3: Create SQLite database
    create_sqlite_db(db_path, df, embeddings)
    
    # Step 4: Validate
    validate_database(db_path)
    
    # Step 5: Test (optional)
    if args.test:
        test_vector_search(db_path)
    
    logger.info("=" * 60)
    logger.info(f"✅ Knowledge base ready: {db_path}")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. For Android integration, you'll need to:")
    logger.info("   - Bundle kb.db in assets/ folder")
    logger.info("   - Implement vector search in Kotlin/Java")
    logger.info("   - Or compile sqlite-vec extension for Android")


if __name__ == '__main__':
    main()
