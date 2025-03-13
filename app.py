import streamlit as st
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset
from collections import Counter

dataset = load_dataset('Helsinki-NLP/tatoeba_mt', 'ara-eng', keep_in_memory=True)

# Use 'validation' as training data and 'test' as evaluation data.
train_data_raw = dataset['validation']
eval_data_raw = dataset['test']

print("Training samples:", len(train_data_raw))    # Expected: 19528
print("Evaluation samples:", len(eval_data_raw))     # Expected: 10304

# Use these keys from the dataset
src_key = 'sourceString'
tgt_key = 'targetString'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Build vocabulary with special tokens using a simple whitespace tokenizer.
def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        tokens = sentence.strip().split()
        counter.update(tokens)
    # Reserve special tokens with fixed indices.
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3
    }
    next_index = 4
    for token, count in counter.items():
        if count >= min_freq and token not in vocab:
            vocab[token] = next_index
            next_index += 1
    return vocab

# Extract sentences from training data.
src_sentences = [ex[src_key] for ex in train_data_raw]
tgt_sentences = [ex[tgt_key] for ex in train_data_raw]

src_vocab = build_vocab(src_sentences)
tgt_vocab = build_vocab(tgt_sentences)
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# For decoding outputs later.
inv_tgt_vocab = {idx: token for token, idx in tgt_vocab.items()}

# Helper: Convert text into a list of token IDs using the vocabulary.
def text_to_ids(text, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in text.strip().split()]

# Custom Dataset class for translation.
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_key, tgt_key):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_key = src_key
        self.tgt_key = tgt_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_ids = text_to_ids(item[self.src_key], self.src_vocab)
        # Add <sos> at the beginning and <eos> at the end of target sequence.
        tgt_ids = [self.tgt_vocab['<sos>']] + text_to_ids(item[self.tgt_key], self.tgt_vocab) + [self.tgt_vocab['<eos>']]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

# Positional Encoding as described in "Attention is All You Need".
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Transformer-based Seq2Seq model.
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=5, num_decoder_layers=5, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Embedding layers for source and target.
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encodings.
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        # Transformer module from PyTorch.
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        # Final linear layer to map to target vocabulary.
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # src and tgt shape: (batch_size, seq_len)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encodings.
        src_emb = self.pos_encoder(src_emb)  # (batch_size, src_seq_len, d_model)
        tgt_emb = self.pos_decoder(tgt_emb)  # (batch_size, tgt_seq_len, d_model)

        # Transformer expects inputs in shape (seq_len, batch_size, d_model)
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)

        output = self.transformer(src_emb, tgt_emb,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        # Transpose output back to (batch_size, tgt_seq_len, d_model)
        output = output.transpose(0, 1)
        logits = self.fc_out(output)  # (batch_size, tgt_seq_len, tgt_vocab_size)
        return logits

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def load_model():
    """Load the trained Transformer model."""
    try:
        model = TransformerModel(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
        model.load_state_dict(torch.load("transformer_model2.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

#beam search in main cell
def greedy_decode(model, src_sentence, max_len=20):
    model.eval()
    # Convert source sentence to token IDs.
    src_ids = text_to_ids(src_sentence, src_vocab)
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    # Encode the source sentence.
    memory = model.pos_encoder(model.src_embedding(src_tensor) * math.sqrt(model.d_model))
    memory = memory.transpose(0, 1)  # shape: (src_seq_len, batch_size, d_model)

    # Start with <sos> token.
    ys = torch.tensor([[tgt_vocab.get('<sos>')]], dtype=torch.long).to(device)

    for i in range(max_len):
        tgt_emb = model.pos_decoder(model.tgt_embedding(ys) * math.sqrt(model.d_model))
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model.transformer(memory, tgt_emb, tgt_mask=tgt_mask)
        out = out.transpose(0, 1)
        prob = model.fc_out(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
        # Stop if <eos> token is generated.
        if next_word == tgt_vocab.get('<eos>'):
            break

    # Convert token IDs back to words, skipping special tokens.
    token_ids = ys.squeeze().tolist()
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    if token_ids[0] == tgt_vocab.get('<sos>'):
        token_ids = token_ids[1:]
    if tgt_vocab.get('<eos>') in token_ids:
        token_ids = token_ids[: token_ids.index(tgt_vocab.get('<eos>'))]
    translation = ' '.join([inv_tgt_vocab.get(idx, '<unk>') for idx in token_ids])
    return translation

def main():
    st.title("Arabic to English Translation")
    st.write("Enter Arabic text below and get the English translation.")
    
    model = load_model()
    
    user_input = st.text_area("Enter Arabic Text:")
    if st.button("Translate"):
        if model and user_input.strip():
            translation = greedy_decode(model, user_input)
            st.subheader("Translation:")
            print(translation)
            st.write(translation)
        else:
            st.warning("Please enter some text to translate.")

if __name__ == "__main__":
    main()
    print("App started")  # Debug print
