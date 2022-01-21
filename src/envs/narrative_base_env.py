import logging
import json
import random
import gym
import numpy as np
from stable_baselines3.common import logger
from transformers import AutoModel, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from src.utils.embeddings import compute_embeddings
import ipdb as pdb


class NarrativeEnv(gym.Env):
    def __init__(self, sentences, narrative2sents={}, **config_dict):
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
        self.embed_model = AutoModel.from_pretrained(
            self.config.get("embed_model_name_or_path", "roberta-base")
        )

        # Define tokenizer
        self.embed_tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("embed_model_name_or_path", "roberta-base")
        )

        # Other parameters that would be useful
        self.max_queries = self.config.get("max_queries", 100)
        self.resample_queries = self.config.get("resample_queries", False)
        self.embedding_pooling = self.config.get("embedding_pooling", "Avg")
        self.max_steps = self.config.get("max_steps", 10)

    def step(self, action):
        """
        Takes a step in the environment resulting in a new state and reward
        Args:
            action (int): Index for the query sentence to be selected as part of the narrative
        """
        # Compute the reward
        try:
            reward = self.reward_function(self.state, action)
        except:
            pdb.set_trace()
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
            # TODO: Computing embeddings again and again might not be efficient, instead compute the embeddings once and store them somewhere
            query_embeddings = compute_embeddings(
                query_sentences,
                self.embed_model,
                self.embed_tokenizer,
                pooling=self.embedding_pooling,
                convert_to_numpy=True,
                progress_bar=False,
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
        _, _, narrative_sentences, query_sentences = state
        selected_sentence = query_sentences[action]
        return sentence_bleu(
            references=narrative_sentences, hypothesis=selected_sentence
        )

    def reset(self):

        # Track the current step, this will be useful to determine the termination
        self.current_step = 0

        # Choose a narrative randomly if narrative2sents dict is provided
        if self.narrative2sents != {}:
            random_narrative = random.choice(self.narrative2sents.keys())
            narrative_sentences = self.narrative2sents[random_narrative]

        # If not then just sample a random seed sentence to used for narrative
        else:
            narrative_sentences = [random.choice(self.seed_sentences)]

        # Select the seed sentences that do not appear in the narrative yet as the query sentences which will be chosen by RL agent
        query_sentences = self.sample_query_sentences(narrative_sentences)

        # Get the embeddings for narrative sentences
        # logger.info("Computing Embeddings for Narrative Sentences")
        narrative_embeddings = compute_embeddings(
            narrative_sentences,
            self.embed_model,
            self.embed_tokenizer,
            pooling=self.embedding_pooling,
            convert_to_numpy=True,
            progress_bar=False,
        )

        # Get the embeddings for query sentences
        # logger.info("Computing Embeddings for Query Sentences")
        query_embeddings = compute_embeddings(
            query_sentences,
            self.embed_model,
            self.embed_tokenizer,
            pooling=self.embedding_pooling,
            convert_to_numpy=True,
            progress_bar=False,
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
        query_sentences = [
            sent for sent in self.seed_sentences if sent not in narrative_sentences
        ]

        # Since it might be expensive to operate over all the queries, prune the list to `self.max_queries`
        # TODO: Instead of randomly selecting max_queries maybe select the queries most similar to narrative sentences using some metric like BLEU or embedding similarity
        random.shuffle(query_sentences)
        query_sentences = query_sentences[: self.max_queries]

        return query_sentences
