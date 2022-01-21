import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from src.utils.embeddings import compute_embedding_similarities
import ipdb as pdb


def random_policy(state):
    _, query_embeddings, _, _ = state
    num_options = len(query_embeddings)
    return np.random.choice(num_options)


def greedy_policy(state, sim_type="bleu"):

    (
        narrative_embeddings,
        query_embeddings,
        narrative_sentences,
        query_sentences,
    ) = state

    if sim_type == "bleu":
        sims = np.array(
            [
                sentence_bleu(references=narrative_sentences, hypothesis=query)
                for query in query_sentences
            ]
        )

    elif sim_type == "embedding_sim":
        paired_sims = compute_embedding_similarities(
            narrative_embeddings, query_embeddings
        )
        sims = paired_sims.mean(axis=1)
    # pdb.set_trace()
    return np.argmax(sims)

