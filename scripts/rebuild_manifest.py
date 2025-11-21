"""
Quick script to create manifest.parquet from existing chunks/ directory.
This bypasses the full build_kb pipeline which is failing on Olive data.
"""

import json
from pathlib import Path
import pandas as pd

chunks_dir = Path("data/kb/chunks")
output_path = Path("data/kb/manifest.parquet")

print(f"Loading chunks from {chunks_dir}")

rows = []
for chunk_file in sorted(chunks_dir.glob("*.md")):
    # Read the chunk text
    with open(chunk_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Parse filename: {doc_id}_{index}.md
    stem = chunk_file.stem
    parts = stem.rsplit('_', 1)
    doc_id = parts[0] if len(parts) > 1 else stem
    
    # Try to extract plant/disease from text (look for ## headers)
    plant = ""
    disease = ""
    lines = text.split('\n')
    for line in lines:
        if line.startswith('## Plant:'):
            plant = line.replace('## Plant:', '').strip()
        elif line.startswith('## Disease:'):
            disease = line.replace('## Disease:', '').strip()
    
    rows.append({
        'doc_id': doc_id,
        'text': text,
        'plant': plant,
        'disease': disease,
        'url': '',
        'title': f"{plant} {disease}".strip(),
        'n_tokens': len(text.split())  # Rough estimate
    })

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} chunks")

# Save to parquet
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_path, index=False)

print(f"✅ Saved manifest to {output_path}")
print(f"   Total chunks: {len(df)}")
print(f"   Sample:")
print(df[['plant', 'disease', 'n_tokens']].head(10))
