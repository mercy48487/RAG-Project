from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["A binary search tree is balanced.", "AVL trees stay balanced automatically."]
embeddings = model.encode(sentences)

print(embeddings.shape)
print(embeddings[0][:5])

similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
print(f"Similarity between the two sentences: {similarity[0][0]:.4f}")