import logging
import json
import random
import gym
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from src.utils.embeddings import (
    compute_embeddings,
    compute_embeddings_muse,
    index_embeddings,
    get_embeddings_from_index,
)
from src.utils.metrics import intercluster_distance, intracluster_distance
from src.utils.helper import get_logger
import ipdb as pdb
from annoy import AnnoyIndex

logger = get_logger()


class NarrativeEnv(gym.Env):
    def __init__(self, sentences, narrative2sents={}, alpha=0.5, **config_dict):
        super().__init__()
        self.seed_sentences = sentences
        self.narrative2sents = narrative2sents
        self.config = config_dict

        # Define the observation space
        self.observation_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.config.get("embed_size", 768),),
            dtype=np.float32,
        )

        # Define the action space
        self.action_space = gym.spaces.Discrete(2)

        # Define the embedding model
        self.embed_model_type = config_dict.get(
            "embed_model_type", "distiluse-base-multilingual-cased-v2"
        )
        self.embed_model = SentenceTransformer(
            self.embed_model_type,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Define the index to store embeddings
        self.ann_index = AnnoyIndex(
            self.embed_model.get_sentence_embedding_dimension(), "angular"
        )
        self.idx_to_all_sents = []
        idx_st = 0
        # Compute and store all narrative embeddings
        # pdb.set_trace()
        self.narrative2emb = {}

        logger.info("Computing and Indexing Embeddings for Narrative Sentences")
        for narrative in self.narrative2sents.keys():
            self.narrative2emb[narrative] = compute_embeddings_muse(
                self.narrative2sents[narrative],
                self.embed_model,
                convert_to_numpy=True,
                progress_bar=True,
            )
            sent_ids = [i + idx_st for i in range(len(self.narrative2sents[narrative]))]
            idx_st = sent_ids[-1] + 1
            index_embeddings(self.narrative2emb[narrative], sent_ids, self.ann_index)
            self.idx_to_all_sents += self.narrative2sents[narrative]

        # Compute and store all seed sentence embeddings
        logger.info("Computing and Indexing Embeddings for Seed Sentences")
        self.seed_embeddings = compute_embeddings_muse(
            self.seed_sentences,
            self.embed_model,
            convert_to_numpy=True,
            progress_bar=True,
        )
        sent_ids = [i + idx_st for i in range(len(self.seed_sentences))]
        index_embeddings(self.seed_embeddings, sent_ids, self.ann_index)
        self.idx_to_all_sents += self.seed_sentences
        self.sent_to_idx = {sent: idx for idx, sent in enumerate(self.idx_to_all_sents)}

        # Finally build the index
        self.ann_index.build(n_trees=self.config.get("n_trees", 10))

        # Other parameters that would be useful
        self.max_queries = self.config.get("max_queries", 100)
        self.resample_queries = self.config.get("resample_queries", False)
        self.max_steps = self.config.get("max_steps", 10)
        self.query_sample_strategy = self.config.get(
            "query_sample_strategy", "centroid_nn"
        )

    def step(self, action):
        """
		Takes a step in the environment resulting in a new state and reward
		Args:
			action (int): Index for the query sentence to be selected as part of the narrative
		"""
        # Compute the reward

        reward = self.reward_function(self.state, action)

        # Update the current state
        (
            narrative_embeddings,
            query_embeddings,
            narrative_sentences,
            query_sentences,
        ) = self.state
        selected_sentence = query_sentences[action]
        selected_sentence_embedding = query_embeddings[action]

        narrative_sentences += [selected_sentence]
        narrative_embeddings += [selected_sentence_embedding]

        if not self.resample_queries:
            query_embeddings = [
                embedding
                for i, embedding in enumerate(query_embeddings)
                if query_sentences[i] != selected_sentence
            ]

            query_sentences = [
                sent for sent in query_sentences if sent != selected_sentence
            ]
        else:
            query_sentences = self.sample_query_sentences()
            query_embeddings = get_embeddings_from_index(
                query_sentences, self.sent_to_idx, self.ann_index
            )

        self.state = (
            narrative_embeddings,
            query_embeddings,
            narrative_sentences,
            query_sentences,
        )

        # Update current step
        self.current_step += 1

        # Check if the episode is complete
        done = (self.current_step >= self.max_steps) or (len(query_sentences) == 0)

        return self.state, reward, done, {}

    def reward_function(self, state, action):

        # TODO: Modify this to something smarter, currently just measuring the bleu b/w selected and narrative sentences
        narrative_embeddings, query_embeddings, _, _ = state
        selected_sentence_embedding = query_embeddings[action]
        intracluster_distance_ = intracluster_distance(
            selected_sentence_embedding, narrative_embeddings
        )

        if self.narrative2sents == {}:
            return -intracluster_distance_

        intercluster_distance_ = 0

        for narrative in self.narrative2emb.keys():
            if self.curr_narrative != narrative:
                intercluster_distance_ += intercluster_distance(
                    selected_sentence_embedding, self.narrative2emb[narrative]
                )

        intercluster_distance_ /= len(self.narrative2emb.keys())

        return intercluster_distance_ - intracluster_distance_

    def reset(self):
        # Track the current step, this will be useful to determine the termination
        self.current_step = 0

        # Choose a narrative randomly if narrative2sents dict is provided
        if self.narrative2sents != {}:
            random_narrative = random.choice(self.narrative2sents.keys())
            narrative_sentences = self.narrative2sents[random_narrative]

            self.curr_narrative = random_narrative

        # If not then just sample a random seed sentence to used for narrative
        else:
            narrative_sentences = [random.choice(self.seed_sentences)]

        # Select the seed sentences that do not appear in the narrative yet as the query sentences which will be chosen by RL agent
        query_sentences = self.sample_query_sentences(narrative_sentences)

        # Get the embeddings for narrative sentences
        # logger.info("Computing Embeddings for Narrative Sentences")
        narrative_embeddings = get_embeddings_from_index(
            narrative_sentences, self.sent_to_idx, self.ann_index
        )

        # Get the embeddings for query sentences
        # logger.info("Computing Embeddings for Query Sentences")
        query_embeddings = get_embeddings_from_index(
            query_sentences, self.sent_to_idx, self.ann_index
        )

        # Return initial state which is a 4 tuple (narrative_embeddings, query_embeddings, narrative_sentences, query_sentences)
        self.state = (
            narrative_embeddings,
            query_embeddings,
            narrative_sentences,
            query_sentences,
        )

        return self.state

    def sample_query_sentences(self, narrative_sentences):
        def random_sample_queries():
            query_sentences = [
                sent for sent in self.seed_sentences if sent not in narrative_sentences
            ]

            # Since it might be expensive to operate over all the queries, prune the list to `self.max_queries`
            # TODO: Instead of randomly selecting max_queries maybe select the queries most similar to narrative sentences using some metric like BLEU or embedding similarity
            random.shuffle(query_sentences)
            query_sentences = query_sentences[: self.max_queries]

            return query_sentences

        def closest_to_centroid_sample_queries():
            narrative_embeddings = [
                self.ann_index.get_item_vector(self.sent_to_idx[sentence])
                for sentence in narrative_sentences
            ]
            centroid = np.mean(np.array(narrative_embeddings), axis=0)
            query_sentence_ids = self.ann_index.get_nns_by_vector(
                centroid, n=int(1.5 * self.max_queries)
            )
            query_sentences = [
                self.idx_to_all_sents[qid]
                for qid in query_sentence_ids
                if self.idx_to_all_sents[qid] not in narrative_sentences
            ][: self.max_queries]

            return query_sentences

        def each_sent_closest_sample_queries():
            query_sentences = []
            for narr_sent in narrative_sentences:
                nid = self.sent_to_idx[narr_sent]
                query_sentence_ids = self.ann_index.get_nns_by_item(
                    nid, n=self.max_queries
                )
                query_sentences += [
                    self.idx_to_all_sents[qid]
                    for qid in query_sentence_ids
                    if self.idx_to_all_sents[qid] not in narrative_sentences
                ]

            query_sentences = list(set(query_sentences))
            random.shuffle(query_sentences)
            query_sentences = query_sentences[: self.max_queries]

            return query_sentences

        if self.query_sample_strategy == "random":
            return random_sample_queries()

        if self.query_sample_strategy == "centroid_nn":
            return closest_to_centroid_sample_queries()

        if self.query_sample_strategy == "each_sent_nn":
            return each_sent_closest_sample_queries()
