from nltk.cluster.util import cosine_distance
import numpy as np


def cosine_similarity(u, v):
	# cosine distance is defined as 1 - cosine similarity
	return cosine_distance(u, v) - 1

def compute_centroid(u):
	# calculate centroid of set of n vectors with dimension embedding_dim
	# u: (n, embedding_dim)
	# returns (1, embedding_dim)

	return np.mean(u, axis = 0)

def intracluster_distance(sentence, narrativeSet):
	# computes centroid of set of narratives
	# computes cosine distance of sentence from narrativeSet centroid
	# sentence: (1, embedding_dim)
	# narrativeSet: (n, embedding_dim)

	narrativeSet = np.array(narrativeSet) # ensure numpy array
	narrativeSet_centroid = compute_centroid(narrativeSet)
	return cosine_distance(sentence, narrativeSet_centroid)

def intercluster_distance(sentence, narrativeSet):
	# computes centroid of set of narratives
	# computes cosine distance of sentence from narrativeSet centroid
	# sentence: (embedding_dim,)
	# narrativeSet: (n, embedding_dim)

	narrativeSet = np.array(narrativeSet) # ensure numpy array
	narrativeSet_centroid = compute_centroid(narrativeSet)
	return cosine_distance(sentence, narrativeSet_centroid)

