from itertools import product
import numpy as np
import glob
import sys
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Assuming you have a list of embeddings for 10 faces, each with 10 embeddings
# Replace these with your actual embeddings
# 10 faces, 10 embeddings per face, each embedding has 128 dimensions
#  embeddings = np.random.randn(10, 10, 128)
voice_embs = sorted(glob.glob("embedings/*_voice.embs"))
face_embs = sorted(glob.glob("embedings/*_face.embs"))

voice_embs = np.array([np.loadtxt(voice_emb)[:7, :]
                      for voice_emb in voice_embs])
face_embs = np.array([np.loadtxt(face_emb)[:50, :] for face_emb in face_embs])
print(voice_embs.shape)
#  embeddings = voice_embs.copy()

#  face_embs = np.array([np.loadtxt(face_emb) for face_emb in face_embs])


def calculate_eer(genuine_distances, impostor_distances):
    # Create histograms of distances
    genuine_hist, genuine_bins = np.histogram(
        genuine_distances, bins=100, density=True)
    impostor_hist, impostor_bins = np.histogram(
        impostor_distances, bins=100, density=True)

    # Calculate cumulative distributions
    genuine_cum = np.cumsum(genuine_hist) / np.sum(genuine_hist)
    impostor_cum = np.cumsum(impostor_hist) / np.sum(impostor_hist)

    # Interpolate the curves
    f_genuine = interp1d(
        genuine_bins[:-1], genuine_cum, bounds_error=False, fill_value=(0, 1))
    f_impostor = interp1d(
        impostor_bins[:-1], impostor_cum, bounds_error=False, fill_value=(0, 1))

    # Calculate the EER by finding the threshold where FAR equals FRR
    eer = brentq(lambda x: 1.0 - x - f_genuine(x) + f_impostor(x), 0.0, 1.0)

    return eer


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
    #  print(anchor,anchor.shape)
    #  print(positive,positive.shape)
    d_pos = np.linalg.norm(anchor - positive)
    d_neg = np.linalg.norm(anchor - negative)

    # Calculate the triplet loss
    loss = np.maximum(d_pos - d_neg + margin, 0.0)

    # Compute the mean triplet loss
    mean_loss = np.mean(loss)

    return mean_loss


# Example: Calculate triplet loss for the entire dataset
def compute_err_dataset(embeddings, tresh=0):
    num_faces, num_embeddings, embedding_dim = embeddings.shape
    same_person = []
    different_person = []
    for embs in embeddings:
        for emb1, emb2 in product(embs, embs):
            #  dist = np.linalg.norm(emb1-emb2)
            #  same_person.append(dist)
            cos_sim = np.dot(emb1, emb2) / \
                (np.linalg.norm(emb1)*np.linalg.norm(emb2))
            same_person.append(cos_sim)
    for i, embs1 in enumerate(embeddings):
        for j, embs2 in enumerate(embeddings):
            if i != j:
                for emb1, emb2 in product(embs1, embs2):
                    cos_sim = np.dot(emb1, emb2) / \
                        (np.linalg.norm(emb1)*np.linalg.norm(emb2))
                    different_person.append(cos_sim)
                    #  dist = np.linalg.norm(emb1-emb2)
                    #  different_person.append(dist)
    same_person = np.array(same_person)
    different_person = np.array(different_person)
    #  same_person -= tresh
    #  different_person -= tresh
    #  same_person[same_person < 0] = 0
    #  different_person[different_person < 0] = 0
    #  same_person
    same_person -= 1
    same_person *= -1

    different_person -= 1
    different_person *= -1

    err = calculate_eer(same_person, different_person)
    mean_same_person = np.mean(same_person)
    mean_different_person = np.mean(different_person)
    print(f"err : {err:.4f}")
    print(f"means : {mean_same_person:.4f} {mean_different_person:.4f}")


def compute_triple_loss_dataset(embeddings):
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


compute_triple_loss_dataset(face_embs)
compute_triple_loss_dataset(voice_embs)
#  compute_err_dataset(face_embs, 0.75)
#  compute_err_dataset(voice_embs, 250)
compute_err_dataset(face_embs)
compute_err_dataset(voice_embs)
