import pickle
from pathlib import Path

chunks_path = Path("data/guidelines/chunks.pkl")

if chunks_path.exists():
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Total chunks: {len(chunks)}")
    print("--- Chunks ---")
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1}:\n{c}\n{'-'*20}")
else:
    print("chunks.pkl not found")
