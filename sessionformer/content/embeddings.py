import numpy as np

from sessionformer.data.vocab import ItemVocab


def build_item_content_embeddings(
    item_text: dict[int, str],
    vocab: ItemVocab,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """Encodes each vocab item's pseudo-document with a sentence-transformer.
    Returns a (len(vocab), embedding_dim) array aligned to vocab indices;
    PAD/UNK and items with no property data get a zero vector."""
    from sentence_transformers import SentenceTransformer  # local import: heavy, torch-dependent

    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()

    vocab_size = len(vocab)
    texts, indices_with_text = [], []
    for idx in range(vocab_size):
        item_id = vocab.decode(idx)
        text = item_text.get(item_id) if item_id is not None else None
        if text:
            texts.append(text)
            indices_with_text.append(idx)

    embeddings = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    if texts:
        encoded = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        embeddings[indices_with_text] = encoded
    return embeddings
