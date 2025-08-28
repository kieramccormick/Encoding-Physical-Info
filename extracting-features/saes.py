# ADOPTED FROM ANTHROPIC

import transformer_lens
from transformer_lens import HookedTransformer, utils
import torch
import numpy as np
import gradio as gr
import pprint
import json
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from IPython.display import HTML
from functools import partial
import tqdm.notebook as tqdm
import plotly.express as px
import pandas as pd
from collections import defaultdict
import re

# Define the Autoencoder
cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 384,
    "lr": 1e-4,
    "num_tokens": int(2e9),
    "l1_coeff": 3e-4,
    "beta1": 0.9,
    "beta2": 0.99,
    "dict_mult": 8,
    "seq_len": 128,
    "d_mlp": 2048,
    "enc_dtype":"fp32",
    "remove_rare_dir": False,
}
cfg["model_batch_size"] = 64
cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_mlp"] * cfg["dict_mult"]
        d_mlp = cfg["d_mlp"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to("cuda")

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

    @classmethod
    def load_from_hf(cls, version):
        """
        Loads the saved autoencoder from HuggingFace.

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        if version=="run1":
            version = 25
        elif version=="run2":
            version = 47

        cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
        return self

# Top feature visualization tool that outputs the top words that fire for each feature

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
model = HookedTransformer.from_pretrained("gelu-1l").to(DTYPES["fp32"])
sae_model = AutoEncoder.load_from_hf("run1")

def get_word_token_mapping(text, model, context_window=3):
    word_pattern = r'\b\w+(?:\'\w+)?\b'
    words_with_pos = [(m.group(), m.start(), m.end()) for m in re.finditer(word_pattern, text)]

    tokens = model.to_tokens(text)
    token_strings = model.to_str_tokens(text)
    token_char_positions = []
    current_pos = 0

    for i, token_str in enumerate(token_strings):
        if i == 0:
            token_char_positions.append((0, 0))
            continue

        clean_token = token_str.replace("Ġ", " ").replace("▁", " ")
        remaining_text = text[current_pos:]
        token_pos = remaining_text.find(clean_token.strip())

        if token_pos >= 0:
            start_pos = current_pos + token_pos
            end_pos = start_pos + len(clean_token.strip())
            token_char_positions.append((start_pos, end_pos))
            current_pos = end_pos
        else:
            token_char_positions.append((current_pos, current_pos))

    token_to_word = []
    word_to_tokens = defaultdict(list)

    for token_idx, (token_start, token_end) in enumerate(token_char_positions):
        best_word_idx = None
        best_overlap = 0

        for word_idx, (word, word_start, word_end) in enumerate(words_with_pos):
            overlap_start = max(token_start, word_start)
            overlap_end = min(token_end, word_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_word_idx = word_idx

        token_to_word.append(best_word_idx)
        if best_word_idx is not None:
            word_to_tokens[best_word_idx].append(token_idx)

    return words_with_pos, token_to_word, word_to_tokens

def get_extended_context(words_with_pos, word_idx, context_window=3):
    total_words = len(words_with_pos)

    left_start = max(0, word_idx - context_window)
    left_context = [words_with_pos[i][0] for i in range(left_start, word_idx)]
    target_word = words_with_pos[word_idx][0]
    right_end = min(total_words, word_idx + context_window + 1)
    right_context = [words_with_pos[i][0] for i in range(word_idx + 1, right_end)]
    return left_context, target_word, right_context

def analyze_text_for_word_features(text, model, sae_model, layer_name="blocks.0.mlp.hook_post", context_window=3):
    words_with_pos, token_to_word, word_to_tokens = get_word_token_mapping(text, model, context_window)
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        activations = cache[layer_name].squeeze(0)

        if hasattr(sae_model, 'W_enc'):
            expected_dim = sae_model.W_enc.shape[0]
            print(f"Debug: activations shape: {activations.shape}, SAE expects: {expected_dim}")

    with torch.no_grad():
        sae_output = sae_model(activations)
        if isinstance(sae_output, tuple):
            sae_features = sae_output[1] if len(sae_output) >= 2 else sae_output[0]
        else:
            sae_features = sae_output

    word_features = {}
    word_contexts = {}

    for word_idx, (word_text, word_start, word_end) in enumerate(words_with_pos):
        token_indices = word_to_tokens.get(word_idx, [])

        if not token_indices:
            continue

        word_feature_vectors = []
        for token_idx in token_indices:
            if token_idx < sae_features.shape[0]:
                word_feature_vectors.append(sae_features[token_idx])

        if word_feature_vectors:
            word_features[word_text.lower()] = torch.stack(word_feature_vectors).mean(dim=0)

            left_context, target, right_context = get_extended_context(words_with_pos, word_idx, context_window)
            context_str = " ".join(left_context) + f" __{target}__ " + " ".join(right_context)
            word_contexts[word_text.lower()] = context_str

    return word_features, word_contexts

def get_phrases_from_text(text, max_phrase_length=3):
    words = re.findall(r'\b\w+(?:\'\w+)?\b', text.lower())
    phrases = {}

    for word in words:
        phrases[word] = [word]

    for n in range(2, max_phrase_length + 1):
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            phrases[phrase] = words[i:i+n]

    return phrases

def analyze_text_for_phrase_features(text, model, sae_model, layer_name="blocks.0.mlp.hook_post", max_phrase_length=3, context_window=3): 
    word_features, word_contexts = analyze_text_for_word_features(text, model, sae_model, layer_name, context_window)
    phrases = get_phrases_with_positions(text, max_phrase_length)

    phrase_features = {}
    phrase_contexts = {}

    for phrase_info in phrases:
        phrase_text = phrase_info['phrase']
        constituent_words = phrase_info['words']
        phrase_positions = phrase_info['positions']

        if len(constituent_words) == 1:
            word = constituent_words[0]
            if word in word_features:
                phrase_features[phrase_text] = word_features[word]
                phrase_contexts[phrase_text] = word_contexts.get(word, f"__{phrase_text}__")
        else:
            constituent_features = []
            for word in constituent_words:
                if word in word_features:
                    constituent_features.append(word_features[word])

            if constituent_features:
                phrase_features[phrase_text] = torch.stack(constituent_features).mean(dim=0)
                phrase_contexts[phrase_text] = get_phrase_context(text, phrase_text, context_window)

    return phrase_features, phrase_contexts

def get_phrases_with_positions(text, max_phrase_length=3):
    word_pattern = r'\b\w+(?:\'\w+)?\b'
    words_with_pos = [(m.group().lower(), m.start(), m.end()) for m in re.finditer(word_pattern, text)]

    phrases = []

    for word, start, end in words_with_pos:
        phrases.append({
            'phrase': word,
            'words': [word],
            'positions': [(start, end)]
        })

    for n in range(2, max_phrase_length + 1):
        for i in range(len(words_with_pos) - n + 1):
            phrase_words = [words_with_pos[j][0] for j in range(i, i + n)]
            phrase_text = ' '.join(phrase_words)
            start_pos = words_with_pos[i][1]
            end_pos = words_with_pos[i + n - 1][2]

            phrases.append({
                'phrase': phrase_text,
                'words': phrase_words,
                'positions': [(start_pos, end_pos)]
            })

    return phrases

def get_phrase_context(text, phrase, context_window=3):
    phrase_pattern = r'\b' + re.escape(phrase.replace(' ', r'\s+')) + r'\b'
    matches = list(re.finditer(phrase_pattern, text, re.IGNORECASE))

    if not matches:
        return f"__{phrase}__"

    match = matches[0]
    phrase_start = match.start()
    phrase_end = match.end()
    word_pattern = r'\b\w+(?:\'\w+)?\b'
    all_words = [(m.group(), m.start(), m.end()) for m in re.finditer(word_pattern, text)]
    left_words = []
    right_words = []

    for word, start, end in all_words:
        if end <= phrase_start:
            left_words.append(word)
        elif start >= phrase_end:
            right_words.append(word)

    left_context = left_words[-context_window:] if left_words else []
    right_context = right_words[:context_window] if right_words else []
    context_parts = []
    if left_context:
        context_parts.append(" ".join(left_context))
    context_parts.append(f"__{phrase}__")
    if right_context:
        context_parts.append(" ".join(right_context))

    return " ".join(context_parts)

def analyze_dataset_for_word_features(dataset_name, text_column="answer", max_samples=None, top_k=20, split="train", use_phrases=True, context_window=3):
    data = load_dataset(dataset_name, split=split)

    if max_samples and len(data) > max_samples:
        data = data.shuffle(seed=42)
        data = data.select(range(max_samples)) # took out an else statement here so need to make sure this still works
    feature_word_activations = defaultdict(lambda: defaultdict(list))

    for idx in range(len(data)):
        if idx % 50 == 0 or len(data) <= 10:
            print(f"Progress: {idx}/{len(data)} ({100*idx/len(data):.1f}%)")

        text = data[idx][text_column]
        if not text or not text.strip():
            continue

        try:
            if use_phrases:
                word_features, word_contexts = analyze_text_for_phrase_features(text, model, sae_model, context_window=context_window)
            else:
                word_features, word_contexts = analyze_text_for_word_features(text, model, sae_model, context_window=context_window)

            for word, feature_vector in word_features.items():
                context = word_contexts.get(word, f"__{word}__")
                for feature_idx in range(len(feature_vector)):
                    activation = feature_vector[feature_idx].item()
                    if activation > 1e-4:
                        feature_word_activations[feature_idx][(word, context)].append(activation)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            if idx == 0:
                print("Check SAE-model compatibility.")
            continue

    print("\nCalculating top words for each feature...")
    results = {}
    for feature_idx, word_activations in feature_word_activations.items():
        word_aggregated = defaultdict(lambda: {'activations': [], 'contexts': []})

        for (word, context), activations in word_activations.items():
            word_aggregated[word]['activations'].extend(activations)
            word_aggregated[word]['contexts'].append(context)

        word_scores = []
        for word, data in word_aggregated.items():
            avg_activation = np.mean(data['activations'])
            frequency = len(data['activations'])
            representative_context = data['contexts'][0]

            if len(word) >= 2 and not word.endswith('_fallback'):
                word_scores.append(((word, representative_context), avg_activation, frequency))

        word_scores.sort(key=lambda x: x[1], reverse=True)
        results[feature_idx] = word_scores[:top_k]

    return results

def display_results(results, num_features_to_show=10):
    if not results:
        print("No active features were found.")
        return

    print(f"Found {len(results)} active features!")
    feature_strength = [(feat_idx, data[0][1] if data else 0) for feat_idx, data in results.items()]
    feature_strength.sort(key=lambda x: x[1], reverse=True)

    for i, (feature_idx, max_activation) in enumerate(feature_strength[:num_features_to_show]):
        print(f"Feature {feature_idx:4d} (max activation: {max_activation:.4f})")
        print("-" * 60)

        for rank, ((word, context), avg_act, freq) in enumerate(results[feature_idx][:15]):
            print(f"{rank+1:2d}. {word:25s} | Avg: {avg_act:.4f} | Freq: {freq:3d} | Context: {context}")
        print()

def run_analysis(dataset_name="kieramccormick/Cluster2", text_column="answer", max_samples=None, use_phrases=True, context_window=3):
    print("Word-based analysis with complete words and phrases" if use_phrases else "Word-based analysis with complete words")
    print(f"Using context window of {context_window} words on each side\n")

    results = analyze_dataset_for_word_features(
        dataset_name=dataset_name,
        text_column=text_column,
        max_samples=max_samples,
        top_k=20,
        use_phrases=use_phrases,
        context_window=context_window
    )

    display_results(results, num_features_to_show=10)

    print(f"Analysis complete! Found {len(results)} active features.")
    return results

# To run in line:
results = run_analysis(context_window=7, use_phrases=False)
