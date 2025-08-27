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

"""## Defining the Autoencoder"""

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

    # def get_version(self):
    #     return 1+max([int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)])

    # def save(self):
    #     version = self.get_version()
    #     torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
    #     with open(SAVE_DIR/(str(version)+"_cfg.json"), "w") as f:
    #         json.dump(cfg, f)
    #     print("Saved as version", version)

    # def load(cls, version):
    #     cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
    #     pprint.pprint(cfg)
    #     self = cls(cfg=cfg)
    #     self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
    #     return self

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

"""## Utils

### Get Reconstruction Loss
"""

def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post

@torch.no_grad()
def get_recons_loss(num_batches=5, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    loss_list = []
    for i in range(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), partial(replacement_hook, encoder=local_encoder))])
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), zero_ablate_hook)])
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(f"loss: {loss:.4f}, recons_loss: {recons_loss:.4f}, zero_abl_loss: {zero_abl_loss:.4f}")
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    print(f"Reconstruction Score: {score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss

"""### Get Frequencies"""

# Frequency
@torch.no_grad()
def get_freqs(num_batches=25, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).cuda()
    total = 0
    for i in tqdm.trange(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]

        _, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
        mlp_acts = cache[utils.get_act_name("post", 0)]
        mlp_acts = mlp_acts.reshape(-1, d_mlp)

        hidden = local_encoder(mlp_acts)[2]

        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores

"""## Visualise Feature Utils"""

from html import escape
import colorsys

from IPython.display import display

SPACE = "·"
NEWLINE="↩"
TAB = "→"

def create_html(strings, values, max_value=None, saturation=0.5, allow_different_length=False, return_string=False):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [
        s.replace("\n", f"{NEWLINE}<br/>").replace("\t", f"{TAB}&emsp;").replace(" ", "&nbsp;")
        for s in escaped_strings
    ]

    if isinstance(values, torch.Tensor) and len(values.shape)>1:
        values = values.flatten().tolist()

    if not allow_different_length:
        assert len(processed_strings) == len(values)

    # scale values
    if max_value is None:
        max_value = max(max(values), -min(values))+1e-3
    scaled_values = [v / max_value * saturation for v in values]

    # create html
    html = ""
    for i, s in enumerate(processed_strings):
        if i<len(scaled_values):
            v = scaled_values[i]
        else:
            v = 0
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(
            hue, v, 1
        )  # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'
    if return_string:
        return html
    else:
        display(HTML(html))

def basic_feature_vis(text, feature_index, max_val=0):
    feature_in = encoder.W_enc[:, feature_index]
    feature_bias = encoder.b_enc[feature_index]
    _, cache = model.run_with_cache(text, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
    mlp_acts = cache[utils.get_act_name("post", 0)][0]
    feature_acts = F.relu((mlp_acts - encoder.b_dec) @ feature_in + feature_bias)
    if max_val==0:
        max_val = max(1e-7, feature_acts.max().item())
        # print(max_val)
    # if min_val==0:
    #     min_val = min(-1e-7, feature_acts.min().item())
    return basic_token_vis_make_str(text, feature_acts, max_val)
def basic_token_vis_make_str(strings, values, max_val=None):
    if not isinstance(strings, list):
        strings = model.to_str_tokens(strings)
    values = utils.to_numpy(values)
    if max_val is None:
        max_val = values.max()
    # if min_val is None:
    #     min_val = values.min()
    header_string = f"<h4>Max Range <b>{values.max():.4f}</b> Min Range: <b>{values.min():.4f}</b></h4>"
    header_string += f"<h4>Set Max Range <b>{max_val:.4f}</b></h4>"
    # values[values>0] = values[values>0]/ma|x_val
    # values[values<0] = values[values<0]/abs(min_val)
    body_string = create_html(strings, values, max_value=max_val, return_string=True)
    return header_string + body_string
# display(HTML(basic_token_vis_make_str(tokens[0, :10], mlp_acts[0, :10, 7], 0.1)))
# # %%
# The `with gr.Blocks() as demo:` syntax just creates a variable called demo containing all these components
import gradio as gr
try:
    demos[0].close()
except:
    pass
demos = [None]
def make_feature_vis_gradio(feature_id, starting_text=None, batch=None, pos=None):
    if starting_text is None:
        starting_text = model.to_string(all_tokens[batch, 1:pos+1])
    try:
        demos[0].close()
    except:
        pass
    with gr.Blocks() as demo:
        gr.HTML(value=f"Hacky Interactive Neuroscope for gelu-1l")
        # The input elements
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", value=starting_text)
                # Precision=0 makes it an int, otherwise it's a float
                # Value sets the initial default value
                feature_index = gr.Number(
                    label="Feature Index", value=feature_id, precision=0
                )
                # # If empty, these two map to None
                max_val = gr.Number(label="Max Value", value=None)
                # min_val = gr.Number(label="Min Value", value=None)
                inputs = [text, feature_index, max_val]
        with gr.Row():
            with gr.Column():
                # The output element
                out = gr.HTML(label="Neuron Acts", value=basic_feature_vis(starting_text, feature_id))
        for inp in inputs:
            inp.change(basic_feature_vis, inputs, out)
    demo.launch(share=True)
    demos[0] = demo

"""### Inspecting Top Logits"""

SPACE = "·"
NEWLINE="↩"
TAB = "→"
def process_token(s):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE+"\n")
    s = s.replace("\t", TAB)
    return s

def process_tokens(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [process_token(s) for s in l]

def process_tokens_index(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [f"{process_token(s)}/{i}" for i,s in enumerate(l)]

def create_vocab_df(logit_vec, make_probs=False, full_vocab=None):
    if full_vocab is None:
        full_vocab = process_tokens(model.to_str_tokens(torch.arange(model.cfg.d_vocab)))
    vocab_df = pd.DataFrame({"token": full_vocab, "logit": utils.to_numpy(logit_vec)})
    if make_probs:
        vocab_df["log_prob"] = utils.to_numpy(logit_vec.log_softmax(dim=-1))
        vocab_df["prob"] = utils.to_numpy(logit_vec.softmax(dim=-1))
    return vocab_df.sort_values("logit", ascending=False)

"""### Make Token DataFrame"""

def list_flatten(nested_list):
    return [x for y in nested_list for x in y]
def make_token_df(tokens, len_prefix=10, len_suffix=10):
    str_tokens = [process_tokens(model.to_str_tokens(t)) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]

    context = []
    batch = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        # context.append([])
        # batch.append([])
        # pos.append([])
        # label.append([])
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        batch=batch,
        pos=pos,
        label=label,
    ))

"""FEATURES WITH WORDS (and phrases but thats broken rn) AND CONTEXT"""

import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
import json
from collections import defaultdict
import re
import pprint
from datasets import load_dataset

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

def analyze_text_for_phrase_features(text, model, sae_model, layer_name="blocks.0.mlp.hook_post", max_phrase_length=3, context_window=3): # need to clean this up in order to use the phrase option accurately
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

    print(f"Found {len(results)} active features!!!!!")
    feature_strength = [(feat_idx, data[0][1] if data else 0) for feat_idx, data in results.items()]
    feature_strength.sort(key=lambda x: x[1], reverse=True)

    for i, (feature_idx, max_activation) in enumerate(feature_strength[:num_features_to_show]):
        print(f"Feature {feature_idx:4d} (max activation: {max_activation:.4f})")
        print("-" * 60)

        for rank, ((word, context), avg_act, freq) in enumerate(results[feature_idx][:15]):
            print(f"{rank+1:2d}. {word:25s} | Avg: {avg_act:.4f} | Freq: {freq:3d} | Context: {context}")
        print()

def run_analysis(dataset_name="kieramccormick/Cluster3", text_column="answer", max_samples=None, use_phrases=True, context_window=3):
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

results = run_analysis(context_window=7, use_phrases=False)

"""WORD ATTEMPT BY MOST FREQUENT FEATURES"""

import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
import json
from collections import defaultdict
import re
import pprint
from datasets import load_dataset

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
        data = data.select(range(max_samples))

    feature_word_activations = defaultdict(lambda: defaultdict(list))
    feature_activation_counts = defaultdict(int)  # Track total activation count per feature

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
                        # Count each activation instance for frequency tracking
                        feature_activation_counts[feature_idx] += 1

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

        # Sort words within each feature by frequency
        word_scores.sort(key=lambda x: x[2], reverse=True)  # x[2] is frequency
        results[feature_idx] = word_scores[:top_k]

    return results, feature_activation_counts

def display_results(results, feature_activation_counts, num_features_to_show=10):
    if not results:
        print("No active features were found.")
        return

    print(f"Found {len(results)} active features!!!!!")

    # Sort features by their total activation frequency
    feature_frequency = [(feat_idx, feature_activation_counts[feat_idx]) for feat_idx in results.keys()]
    feature_frequency.sort(key=lambda x: x[1], reverse=True)

    for i, (feature_idx, total_activations) in enumerate(feature_frequency[:num_features_to_show]):
        print(f"Feature {feature_idx:4d} (total activations: {total_activations})")
        print("-" * 70)

        for rank, ((word, context), avg_act, freq) in enumerate(results[feature_idx][:15]):
            print(f"{rank+1:2d}. {word:25s} | Freq: {freq:3d} | Avg: {avg_act:.4f} | Context: {context}")
        print()

def run_analysis(dataset_name="kieramccormick/Cluster3", text_column="answer", max_samples=None, use_phrases=True, context_window=3):
    print("Word-based analysis with complete words and phrases" if use_phrases else "Word-based analysis with complete words")
    print(f"Using context window of {context_window} words on each side")
    print("Prioritizing features by FREQUENCY (most often activated)\n")

    results, feature_activation_counts = analyze_dataset_for_word_features(
        dataset_name=dataset_name,
        text_column=text_column,
        max_samples=max_samples,
        top_k=20,
        use_phrases=use_phrases,
        context_window=context_window
    )

    display_results(results, feature_activation_counts, num_features_to_show=10)

    print(f"Analysis complete! Found {len(results)} active features.")
    return results, feature_activation_counts

results = run_analysis(context_window=7, use_phrases=False)

"""WORD FEATURES WITH A (HOPEFULLY) BETTER VISUAL"""

import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
import json
from collections import defaultdict
import re
import pprint
from datasets import load_dataset

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
        data = data.select(range(max_samples))
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

def display_results_enhanced(results, num_features_to_show=10):
    """Enhanced display function matching the desired visual format"""
    if not results:
        print("No active features were found.")
        return

    print(f"Found {len(results)} active features!")
    feature_strength = [(feat_idx, data[0][1] if data else 0) for feat_idx, data in results.items()]
    feature_strength.sort(key=lambda x: x[1], reverse=True)

    for i, (feature_idx, max_activation) in enumerate(feature_strength[:num_features_to_show]):
        # Header line matching the format from image 1
        print(f"Feature {feature_idx:3d} (max activation: {max_activation:.4f})")
        print("-" * 120)

        # Display top words in the enhanced format
        for rank, ((word, context), avg_act, freq) in enumerate(results[feature_idx][:15]):
            # Format: rank. word | Avg: activation | Freq: frequency | Context: context
            print(f"{rank+1:2d}. {word:<15s} | Avg: {avg_act:.4f} | Freq: {freq:3d} | Context: {context}")

        print()  # Empty line between features

def display_results_table_format(results, num_features_to_show=10):
    """Alternative table format display matching the second image style"""
    if not results:
        print("No active features were found.")
        return

    print(f"Found {len(results)} active features!")

    # Create a comprehensive table across all features
    all_entries = []

    feature_strength = [(feat_idx, data[0][1] if data else 0) for feat_idx, data in results.items()]
    feature_strength.sort(key=lambda x: x[1], reverse=True)

    for feature_idx, max_activation in feature_strength[:num_features_to_show]:
        for rank, ((word, context), avg_act, freq) in enumerate(results[feature_idx][:10]):
            all_entries.append({
                'feature': feature_idx,
                'rank': rank + 1,
                'word': word,
                'avg_activation': avg_act,
                'frequency': freq,
                'context': context,
                'max_activation': max_activation
            })

    # Sort by activation strength across all features
    all_entries.sort(key=lambda x: x['avg_activation'], reverse=True)

    # Display header
    print(f"{'Word':<15} {'Feature':<8} {'Rank':<4} {'Avg Act':<8} {'Freq':<5} {'Context'}")
    print("-" * 120)

    # Display entries
    for entry in all_entries[:50]:  # Show top 50 across all features
        print(f"{entry['word']:<15} {entry['feature']:<8} {entry['rank']:<4} "
              f"{entry['avg_activation']:<8.4f} {entry['frequency']:<5} {entry['context']}")

# Keep the original display function for backward compatibility
def display_results(results, num_features_to_show=10):
    """Original display function"""
    display_results_enhanced(results, num_features_to_show)

def run_analysis(dataset_name="kieramccormick/Cluster1Refined", text_column="answer", max_samples=None, use_phrases=True, context_window=3, display_style="enhanced"):
    """
    Run analysis with enhanced visualization options

    Args:
        display_style: "enhanced" (default, matches image 1) or "table" (matches image 2)
    """
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

    if display_style == "table":
        display_results_table_format(results, num_features_to_show=10)
    else:
        display_results_enhanced(results, num_features_to_show=10)

    print(f"Analysis complete! Found {len(results)} active features.")
    return results

# Use the enhanced format (matches your first image)
#results = run_analysis(display_style="enhanced")

# Or use the table format (similar to your second image)
results = run_analysis(display_style="table")

"""## Loading the Model"""

model = HookedTransformer.from_pretrained("gelu-1l").to(DTYPES[cfg["enc_dtype"]])
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab

"""## Loading Data"""

data = load_dataset("kieramccormick/Cluster1", split="train") # make my own dataset
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128, column_name="answer")
tokenized_data = tokenized_data.shuffle(42)
all_tokens = tokenized_data["tokens"]

print({len(data)})

"""# Analysis

## Loading the Autoencoder

There are two runs on separate random seeds, along with a bunch of intermediate checkpoints
"""

auto_encoder_run = "run1" # @param ["run1", "run2"]
encoder = AutoEncoder.load_from_hf(auto_encoder_run)

"""## Using the Autoencoder

We run the model and replace the MLP activations with those reconstructed from the autoencoder, and get 91% loss recovered
"""

_ = get_recons_loss(num_batches=5, local_encoder=encoder)

"""## Rare Features Are All The Same

For each feature we can get the frequency at which it's non-zero (per token, averaged across a bunch of batches), and plot a histogram
"""

freqs = get_freqs(num_batches = 50, local_encoder = encoder)
print(len(freqs))

# Add 1e-6.5 so that dead features show up as log_freq -6.5
log_freq = (freqs + 10**-6.5).log10()
px.histogram(utils.to_numpy(log_freq), title="Log Frequency of Features", histnorm='percent')

"""We see that it's clearly bimodal! Let's define rare features as those with freq < 1e-4, and look at the cosine sim of each feature with the average rare feature - we see that almost all rare features correspond to this feature!"""

is_rare = freqs < 1e-4
rare_enc = encoder.W_enc[:, is_rare]
rare_mean = rare_enc.mean(-1)
px.histogram(utils.to_numpy(rare_mean @ encoder.W_enc / rare_mean.norm() / encoder.W_enc.norm(dim=0)), title="Cosine Sim with Ave Rare Feature", color=utils.to_numpy(is_rare), labels={"color": "is_rare", "count": "percent", "value": "cosine_sim"}, marginal="box", histnorm="percent", barmode='overlay')

"""## Interpreting A Feature

Let's go and investigate a non rare feature, feature 7
"""

feature_id = 1012 # @param {type:"number"}
batch_size = 128 # @param {type:"number"}

print(f"Feature freq: {freqs[25].item():.4f}")

"""Let's run the model on some text and then use the autoencoder to process the MLP activations"""

tokens = all_tokens[:batch_size]
_, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
mlp_acts = cache[utils.get_act_name("post", 0)]
mlp_acts_flattened = mlp_acts.reshape(-1, cfg["d_mlp"])
loss, x_reconstruct, hidden_acts, l2_loss, l1_loss = encoder(mlp_acts_flattened)
# This is equivalent to:
# hidden_acts = F.relu((mlp_acts_flattened - encoder.b_dec) @ encoder.W_enc + encoder.b_enc)
print("hidden_acts.shape", hidden_acts.shape)

"""
We can now sort and display the top tokens, and we see that this feature activates on text like " and I" (ditto for other connectives and pronouns)! It seems interpretable!

**Aside:** Note on how to read the context column:

A line like "·himself·as·democratic·socialist·and|·he|·favors" means that the preceding 5 tokens are " himself as democratic socialist and", the current token is " he" and the next token is " favors".  · are spaces, ↩ is a newline.

This gets a bit confusing for this feature, since the pipe separators look a lot like a capital I
"""

token_df = make_token_df(tokens)
token_df["feature"] = utils.to_numpy(hidden_acts[:, feature_id])
token_df.sort_values("feature", ascending=False).head(20).style.background_gradient("coolwarm")

feature_idx = 7918
word_token_df = analyze_feature_with_word_reconstruction(
    encoder=encoder,
    all_tokens=all_tokens,
    feature_idx=feature_idx,
    model=model
)

word_token_df = word_token_df.sort_values('feature', ascending=False).head(20)
#print(word_token_df[['str_tokens', 'unique_token', 'context', 'batch', 'pos', 'label', 'feature']])

# See all variables in your environment
print([var for var in dir() if not var.startswith('_')])

# Check your main data variables
print("Data shape:", data.shape if hasattr(data, 'shape') else type(data))
print("Hidden acts shape:", hidden_acts.shape if 'hidden_acts' in globals() else "Not found")
print("Tokens shape:", tokens.shape if hasattr(tokens, 'shape') else type(tokens))

# Check if you have feature activations
if hasattr(auto_encoder_run, 'feature_acts') or hasattr(auto_encoder_run, 'acts'):
    print("Feature activations available in auto_encoder_run")

# Check what's in your data variable
print("Data columns:", data.columns.tolist() if hasattr(data, 'columns') else "No columns")
print("Data head:")
print(data.head() if hasattr(data, 'head') else data[:5])

# Check if auto_encoder_run has feature activations
print("auto_encoder_run attributes:", dir(auto_encoder_run))

# The hidden_acts should be your SAE feature activations
print("Hidden acts shape:", hidden_acts.shape)  # Should be [n_summaries, n_features]

# want to double check about SAE activations being in hidden_acts

print("Hidden acts shape:", hidden_acts.shape)

# Calculate feature statistics
feature_activation_counts = {}
feature_mean_activations = {}

# For each feature (assuming they're columns in hidden_acts)
for feature_idx in range(hidden_acts.shape[1]):
    feature_column = hidden_acts[:, feature_idx]

    # Count non-zero activations
    activation_count = (feature_column > 0).sum().item()
    feature_activation_counts[feature_idx] = activation_count

    # Calculate mean of non-zero activations
    nonzero_activations = feature_column[feature_column > 0]
    if len(nonzero_activations) > 0:
        mean_activation = nonzero_activations.mean().item()
        feature_mean_activations[feature_idx] = mean_activation

# Now sort to get top features
top_by_count = sorted(feature_activation_counts.items(), key=lambda x: x[1], reverse=True)[:20]
top_by_strength = sorted(feature_mean_activations.items(), key=lambda x: x[1], reverse=True)[:20]

print("Top 20 features by activation count:")
for feature_id, count in top_by_count:
    print(f"Feature {feature_id}: {count} activations")

print("\nTop 20 features by mean activation strength:")
for feature_id, strength in top_by_strength:
    print(f"Feature {feature_id}: {strength:.4f} mean activation")

"""It's easy to misread evidence like the above, so it's useful to take some text and edit it and see how this changes the model's activations. Here's a hacky interactive tool to play around with some text."""

from torch.nn.utils.rnn import pad_sequence

# Convert token lists to padded tensor
token_lists = tokenized_data["tokens"]  # list of token ID lists from the new dataset
token_tensors = [torch.tensor(t, dtype=torch.long) for t in token_lists]
padded_tokens = pad_sequence(token_tensors, batch_first=True, padding_value=0)

from functools import partial

# 1. Choose the layer and hook name (usually after MLP layer)
layer = 0
act_name = utils.get_act_name("post", layer)  # e.g., 'blocks.0.hook_post'

# 2. Get a batch of tokens
batch_tokens = padded_tokens[:256]  # or however many fit in memory

# 3. Run with cache to extract hidden activations
_, cache = model.run_with_cache(batch_tokens, stop_at_layer=1, names_filter=act_name)

# 4. Extract activations (shape: [batch_size, seq_len, d_model])
mlp_acts = cache[act_name]

# 5. Average over sequence length (or use CLS token if applicable)
mlp_acts_mean = mlp_acts.mean(dim=1)  # shape: [batch_size, d_model]

# 6. Run through SAE encoder
with torch.no_grad():
    hidden_activations = encoder(mlp_acts_mean)[1]  # get latent activations

top_features = torch.argmax(hidden_activations, dim=1).cpu().numpy()

from collections import Counter
feature_counts = Counter(top_features)

# Display top 10
top_n = 20
print(f"\nTop {top_n} most activated SAE features across the dataset:")
for feature, count in feature_counts.most_common(top_n):
    print(f"Feature {feature}: {count} samples")

# Plot
import matplotlib.pyplot as plt
plt.bar(*zip(*feature_counts.most_common(top_n)))
plt.xlabel("Feature Index")
plt.ylabel("Activation Count")
plt.title("Top Activated SAE Features")
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import torch

# Convert to binary activation (1 if > 0)
binary_activations = (hidden_acts > 0).int()
activations = hidden_acts.cpu()

# Count features with the highest average activation
feature_means = activations.mean(dim=0)
top_feature_indices = torch.topk(feature_means, top_n).indices

# Slice to top features across samples
top_feature_activations = activations[:, top_feature_indices]

# Limit to first 100 samples
activation_matrix = top_feature_activations[:100].detach().numpy().T  # [features, samples] took out [:100]

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    activation_matrix,
    cmap="viridis",
    cbar=True,
    xticklabels=False,
    yticklabels=[f"F{idx.item()}" for idx in top_feature_indices]
)
plt.title("SAE Feature Activation Strengths (Top 20 features, first 100 samples)")
plt.xlabel("Sample Index")
plt.ylabel("Feature Index")
plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans

# zero variance and NaN values
def safe_correlation_analysis(activation_matrix):
    feature_vars = np.var(activation_matrix, axis=1)
    active_features = feature_vars > 1e-8

    if active_features.sum() < 2:
        print("Not enough active features for correlation analysis")
        return None, None, None

    # only active features
    active_activation_matrix = activation_matrix[active_features]
    active_feature_indices = top_feature_indices[active_features]

    sample_correlation = np.corrcoef(active_activation_matrix.T)
    feature_correlation = np.corrcoef(active_activation_matrix)

    # std = 0
    sample_correlation = np.nan_to_num(sample_correlation)
    feature_correlation = np.nan_to_num(feature_correlation)

    return sample_correlation, feature_correlation, active_feature_indices

sample_corr, feature_corr, active_indices = safe_correlation_analysis(activation_matrix)

if sample_corr is not None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(sample_corr, cmap='coolwarm', center=0)
    plt.title('Sample-Sample Correlation (based on feature activations)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.show()

    plt.figure(figsize=(8, 8))
    sns.heatmap(feature_corr, cmap='coolwarm', center=0,
                xticklabels=[f"F{idx.item()}" for idx in active_indices],
                yticklabels=[f"F{idx.item()}" for idx in active_indices])
    plt.title('Feature-Feature Correlation')
    plt.xlabel('Feature')
    plt.ylabel('Feature')
    plt.show()

    feature_corr_upper = np.triu(feature_corr, k=1)  # Upper triangle, excluding diagonal
    high_corr_indices = np.where(np.abs(feature_corr_upper) > 0.5)

    if len(high_corr_indices[0]) > 0:
        print("Highly correlated feature pairs:")
        for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
            feat1 = active_indices[i].item()
            feat2 = active_indices[j].item()
            corr_val = feature_corr[i, j]
            print(f"  F{feat1} ↔ F{feat2}: {corr_val:.3f}")
    else:
        print("No highly correlated feature pairs found")

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

feature_means = activations.mean(dim=0)
top_feature_indices = torch.topk(feature_means, top_n).indices

top_feature_activations = activations[:, top_feature_indices]

# play with these parameters
n_samples = min(500, activations.shape[0])  # Use up to 500 samples
sample_indices = torch.randperm(activations.shape[0])[:n_samples]
tsne_data = top_feature_activations[sample_indices].detach().numpy()

scaler = StandardScaler()
tsne_data_scaled = scaler.fit_transform(tsne_data)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    n_iter=1000,
    verbose=1
)

tsne_embedding = tsne.fit_transform(tsne_data_scaled)

plt.figure(figsize=(12, 8))

# sample index
colors = np.arange(len(tsne_embedding))
scatter = plt.scatter(
    tsne_embedding[:, 0],
    tsne_embedding[:, 1],
    c=colors,
    cmap='viridis',
    alpha=0.7,
    s=50
)
plt.colorbar(scatter, label='Sample Index')
plt.title('t-SNE of SAE Feature Activations')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.show()

# activation strength of top feature
if top_n > 0:
    plt.figure(figsize=(12, 8))
    feature_activations = tsne_data[:, 0]
    feature_idx = top_feature_indices[0].item()

    scatter = plt.scatter(
        tsne_embedding[:, 0],
        tsne_embedding[:, 1],
        c=feature_activations,
        cmap='RdYlBu_r',
        alpha=0.7,
        s=50
    )
    plt.colorbar(scatter, label=f'Feature {feature_idx} Activation')
    plt.title(f't-SNE of SAE Feature Activations (colored by Feature {feature_idx})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.show()

# total activation strength
plt.figure(figsize=(12, 8))
total_activations = np.sum(tsne_data, axis=1)
scatter = plt.scatter(
    tsne_embedding[:, 0],
    tsne_embedding[:, 1],
    c=total_activations,
    cmap='plasma',
    alpha=0.7,
    s=50
)
plt.colorbar(scatter, label='Total Activation Strength')
plt.title('t-SNE of SAE Feature Activations (colored by Total Activation)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.show()

'''
# Optional: Add clustering to the t-SNE plot
from sklearn.cluster import KMeans

# Perform clustering on the original feature space
n_clusters = 5  # Adjust as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(tsne_data_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    tsne_embedding[:, 0],
    tsne_embedding[:, 1],
    c=cluster_labels,
    cmap='tab10',
    alpha=0.7,
    s=50
)
plt.colorbar(scatter, label='Cluster', ticks=range(n_clusters))
plt.title('t-SNE of SAE Feature Activations with K-means Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.show()

print(f"t-SNE completed with {len(tsne_embedding)} samples and {tsne_data.shape[1]} features")
print(f"Original feature space: {tsne_data.shape}")
print(f"t-SNE embedding shape: {tsne_embedding.shape}")
'''

model.cfg

s = "The source in question is a transient X-ray binary (XB) that has been identified with significant variability"
t = model.to_tokens(s)
print(t)

starting_text = "[MENTIONED: NO]  ### A) X-ray Properties The text provided does not include any direct information about specific X-ray properties of the source classified as type WR*. Therefore, we cannot provide details regarding variability, spectral properties, flux measurements, or timing analysis. Nevertheless, in general, Wolf-Rayet (WR) stars are known to exhibit significant X-ray emissions due to their massive and hot stellar atmospheres. Limited studies have suggested that X-ray variability may arise from stellar pulsations or possibly interactions in binary systems, but specific data is not available in this context.  ### B) Use in Scientific Hypotheses In terms of scientific hypotheses, properties associated with WR stars, such as their strong stellar winds, high luminosities, and potential as progenitors of supernovae, are crucial for modeling stellar evolution and the environment in star-forming regions. Their feedback mechanisms, including the energy and momentum input from stellar winds and supernova explosions, play critical roles in regulating star formation and the dynamics within starburst galaxies like the Carina region. Such feedback processes are relevant for understanding the lifecycle of matter in the interstellar medium and the influence of massive stars on their surrounding environments. Nonetheless, without specific X-ray data or properties, a direct application to the model tests discussed in the text cannot be articulated." # @param {type:"string"}
make_feature_vis_gradio(feature_id, starting_text)

"""A final piece of evidence: This is a one layer model, so the neurons can only matter by directly impacting the final logits! We can directly look at how the decoder weights for this feature affect the logits, and see that it boosts `'ll`! This checks out, I and he'll etc is a common construction."""

logit_effect = encoder.W_dec[feature_id] @ model.W_out[0] @ model.W_U
create_vocab_df(logit_effect).head(40).style.background_gradient("coolwarm")

# Inspect the dataset structure to find the correct column name
print(data)
print(data.column_names)

