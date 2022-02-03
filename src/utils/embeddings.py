import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def compute_embeddings_muse(
    texts, model, batch_size=4, convert_to_numpy=False, progress_bar=False
):
    """
    Computes embeddings for  a sequence of texts given by the list `texts`
    """
    embeddings = []
    with tqdm(total=len(texts) // batch_size, disable=not progress_bar) as pbar:
        for batch_st in range(0, len(texts), batch_size):
            batch_end = min(batch_st + batch_size, len(texts))
            texts_batch = texts[batch_st:batch_end]
            with torch.no_grad():
                embeddings_batch = model.encode(
                    texts_batch,
                    convert_to_numpy=convert_to_numpy,
                    convert_to_tensor=not convert_to_numpy,
                    show_progress_bar=False,
                )

            embeddings.append(embeddings_batch)
            pbar.update(1)

    if convert_to_numpy:
        embeddings = np.concatenate(embeddings, axis=0)

    else:
        embeddings = torch.cat(embeddings, dim=0)

    return embeddings


def compute_embeddings(
    texts,
    model,
    tokenizer,
    batch_size=4,
    pooling="Avg",
    convert_to_numpy=False,
    progress_bar=False,
):

    """
    Computes embeddings for  a sequence of texts given by the list `texts`
    """
    embeddings = []
    batch_itr = range(0, len(texts), batch_size)
    if progress_bar:
        batch_itr = tqdm(batch_itr)
    for batch_st in batch_itr:
        batch_end = min(batch_st + batch_size, len(texts))
        texts_batch = texts[batch_st:batch_end]
        with torch.no_grad():
            model_inputs = tokenizer(texts_batch, padding=True, return_tensors="pt")
            model_inputs = dict(
                map(lambda kv: (kv[0], kv[1].to(model.device)), model_inputs.items())
            )
            attention_mask = model_inputs["attention_mask"]
            out = model(**model_inputs)["last_hidden_state"]

            if pooling == "CLS":
                embeddings_batch = out[:, 0, :]

            elif pooling == "Avg":
                embeddings_batch = (out * attention_mask.unsqueeze(-1)).sum(
                    dim=1
                ) / attention_mask.sum(axis=-1).unsqueeze(-1)

            else:
                raise NotImplementedError

            if convert_to_numpy:
                embeddings.append(embeddings_batch.detach().cpu().numpy())

            else:
                embeddings.append(embeddings_batch.detach())

    if convert_to_numpy:
        embeddings = np.concatenate(embeddings, axis=0)

    else:
        embeddings = torch.cat(embeddings, dim=0)

    return embeddings


def compute_embedding_similarities(key_embeddings, query_embeddings):
    similarities = cosine_similarity(query_embeddings, key_embeddings)

    return similarities


def index_embeddings(embeddings, idxs, ann_index):
    for embedding, idx in zip(embeddings, idxs):
        ann_index.add_item(idx, embedding)


def get_embeddings_from_index(sentences, sent_to_idx, ann_index):
    embeddings = [ann_index.get_item_vector(sent_to_idx[sent]) for sent in sentences]
    return np.array(embeddings,)
