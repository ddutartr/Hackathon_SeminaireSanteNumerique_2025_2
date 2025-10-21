import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Constants
MAX_LENGTH = 512   # Max sequence length for CamemBERT
STRIDE = 256       # Stride for sliding window

# Directory to save tensors
SAVE_DIR = 'data/processed/embed_CamemBio_sliding_note/'
os.makedirs(SAVE_DIR, exist_ok=True)


def load_model():
    """Load CamemBERT-Bio-Base tokenizer and model."""
    model_name = "almanach/camembert-bio-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def mean_max_aggregate(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Perform mean-max pooling on a 2D tensor of shape (num_chunks, hidden_dim).
    Returns concatenated tensor [mean; max].
    """
    mean_emb = embeddings.mean(dim=0)
    max_emb, _ = embeddings.max(dim=0)
    return torch.cat((mean_emb, max_emb), dim=0)


def process_sentence(sentence, tokenizer, model, device):
    """Encode a sentence (or chunk) and return its embedding."""
    encoded = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)

    # Aggregate across tokens → sentence-level embedding
    sentence_embedding = mean_max_aggregate(last_hidden)
    return sentence_embedding


def process_dataset(input_csv, tokenizer, model, device):
    """Process the dataset and save embeddings."""
    data = pd.read_csv(input_csv)

    for idx, note in enumerate(data['observationBlob']):
        if not isinstance(note, str) or note.strip() == "":
            continue  # skip empty/non-text rows

        note_embeddings = []

        # Simple sentence splitting
        sentences = note.split('. ')
        for sentence in sentences:
            if len(sentence) > MAX_LENGTH:
                # Sliding window for long text
                start = 0
                while start < len(sentence):
                    end = start + MAX_LENGTH
                    chunk = sentence[start:end]
                    emb = process_sentence(chunk, tokenizer, model, device)
                    note_embeddings.append(emb)
                    start += STRIDE
            else:
                emb = process_sentence(sentence, tokenizer, model, device)
                note_embeddings.append(emb)

        # Stack and aggregate to get document-level embedding
        note_tensor = torch.stack(note_embeddings)
        doc_embedding = mean_max_aggregate(note_tensor)

        # Save based on DataFrame index
        torch.save(doc_embedding.cpu(), os.path.join(SAVE_DIR, f"{idx}.pt"))
        print(f"Saved embedding for index {idx}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model()
    model.to(device)

    input_csv = 'data/raw/physician_dataset.csv'
    process_dataset(input_csv, tokenizer, model, device)