import numpy as np
import glob
import sys

# Assuming you have a list of embeddings for 10 faces, each with 10 embeddings
# Replace these with your actual embeddings
# 10 faces, 10 embeddings per face, each embedding has 128 dimensions
#  embeddings = np.random.randn(10, 10, 128)
voice_embs = sorted(glob.glob("embedings/*_voice.embs"))
face_embs = sorted(glob.glob("embedings/*_face.embs"))

voice_embs = np.array([np.loadtxt(voice_emb)[:7, :] for voice_emb in voice_embs])
print(voice_embs.shape)
embeddings = voice_embs.copy()

#  face_embs = np.array([np.loadtxt(face_emb) for face_emb in face_embs])





def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute the triplet loss.

    Parameters:
    - anchor: Embedding of the anchor face.
    - positive: Embedding of the positive face (same person as anchor).
    - negative: Embedding of the negative face (different person from anchor).
    - margin: Margin value to control the relative distance between
               positive and negative pairs.

    Returns:
    - loss: Triplet loss value.
    """
    # Calculate the Euclidean distances between embeddings
    print(anchor,anchor.shape)
    print(positive,positive.shape)
    d_pos = np.linalg.norm(anchor - positive, axis=1)
    d_neg = np.linalg.norm(anchor - negative, axis=1)

    # Calculate the triplet loss
    loss = np.maximum(d_pos - d_neg + margin, 0.0)

    # Compute the mean triplet loss
    mean_loss = np.mean(loss)

    return mean_loss


# Example: Calculate triplet loss for the entire dataset
num_faces, num_embeddings, embedding_dim = embeddings.shape

triplet_losses = []

# Loop through the dataset to generate and compute triplet losses
for i in range(num_faces):
    for j in range(num_embeddings):
        anchor = embeddings[i][j]
        # Use the next embedding of the same person as positive
        positive = embeddings[i][(j + 1) % num_embeddings]
        for k in range(num_faces):
            if k != i:  # Make sure negative is from a different person
                # Use the first embedding of a different person as negative
                negative = embeddings[k][0]
                loss = triplet_loss(anchor, positive, negative)
                triplet_losses.append(loss)

# Compute the mean triplet loss for the entire dataset
mean_triplet_loss = np.mean(triplet_losses)
print(f"Mean Triplet Loss: {mean_triplet_loss:.4f}")
