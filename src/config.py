import pandas as pd


def build_vocab(annot_csv):
    df = pd.read_csv(annot_csv)
    # Đảm bảo utf8_string được xử lý như string
    df['utf8_string'] = df['utf8_string'].astype(str)
    # Lấy tất cả ký tự unique
    all_chars = set()
    for text in df['utf8_string']:
        all_chars.update(text)
    vocab = sorted(list(all_chars))
    # Thêm special tokens
    vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + vocab
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for i, c in enumerate(vocab)}
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample characters: {vocab[4:20]}")
    
    return vocab, char2idx, idx2char


def build_expanded_vocab(annot_csv, vietnamese_csv):
    # Load existing vocab
    orig_vocab, orig_char2idx, orig_idx2char = build_vocab(annot_csv)
    
    # Add Vietnamese characters
    df_vn = pd.read_csv(vietnamese_csv)
    vn_chars = set()
    for text in df_vn['utf8_string'].astype(str):
        vn_chars.update(text)
    # Find only new characters not in original vocab
    new_chars = [c for c in sorted(list(vn_chars)) if c not in orig_char2idx]
    
    # Create expanded vocab (keep special tokens at same indices)
    expanded_vocab = orig_vocab[:4] + orig_vocab[4:] + new_chars
    char2idx = {c: i for i, c in enumerate(expanded_vocab)}
    idx2char = {i: c for i, c in enumerate(expanded_vocab)}
    
    return expanded_vocab, char2idx, idx2char


# Initialize vocabulary
VOCAB, CHAR2IDX, IDX2CHAR = build_expanded_vocab("new_data\\annot.csv", "annot_viet.csv")
VOCAB_SIZE = len(VOCAB)
