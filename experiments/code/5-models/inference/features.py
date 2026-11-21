"""Compute the linguistic features for ONE text (a paragraph, or a sentence at sentence level).

The functions are copied from the notebooks in code/2-feature-extraction/ so that a new text gets
exactly the same numbers as the texts in the master sheets. Every notebook uses one en_core_web_trf
parse (sentences, POS tags and dependencies), so this file does too.
"""
import math

import spacy
import torch
from lexicalrichness import LexicalRichness
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# en_core_web_trf for everything (same as kazmi-*, mean-mdl and mean-surprisal-gpt2 notebooks)
spacy.prefer_gpu()
parser_nlp = spacy.load("en_core_web_trf", disable=["ner", "lemmatizer"])

# GPT-2 for surprisal (uses the Mac GPU if available)
device = "mps" if torch.backends.mps.is_available() else "cpu"
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.eval()


# ---------- kazmi-raw.ipynb ----------
def raw_features(doc):
    sen_lens = []
    for sent in doc.sents:
        length = 0
        for token in sent:
            if token.pos_ != "SPACE":
                length += 1
        sen_lens.append(length)
    if not sen_lens:
        return {"avg_sen_len": 0.0, "std_sen_len": 0.0}
    avg = sum(sen_lens) / len(sen_lens)
    std = math.sqrt(sum((x - avg) ** 2 for x in sen_lens) / len(sen_lens))
    return {"avg_sen_len": avg, "std_sen_len": std}


# ---------- kazmi-lexical.ipynb ----------
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
FUNCTION_POS = {"DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART"}
CD_NAMES = ["ttr", "maas_ttr", "msttr", "mtld", "yulek"]  # the ones kept in final_features (vocd etc. dropped)


def lexical_diversity(tokens):
    result = {name: 0.0 for name in CD_NAMES}
    if len(tokens) < 2:
        return result
    lex = LexicalRichness(tokens, preprocessor=None, tokenizer=None)
    try:
        result["ttr"] = float(lex.ttr)
    except (ValueError, ZeroDivisionError):
        pass
    try:
        result["maas_ttr"] = float(lex.Maas)
    except (ValueError, ZeroDivisionError):
        pass
    try:
        result["msttr"] = float(lex.msttr(segment_window=25)) if len(tokens) > 25 else result["ttr"]
    except (ValueError, ZeroDivisionError):
        pass
    try:
        result["mtld"] = float(lex.mtld(threshold=0.72))
    except (ValueError, ZeroDivisionError):
        pass
    try:
        result["yulek"] = float(lex.yulek)
    except (ValueError, ZeroDivisionError):
        pass
    return result


def lexical_features(doc):
    char_tokens = [c for c in doc.text if c != " "]                      # character level, no prefix
    word_tokens = [t.text.lower() for t in doc if not (t.is_punct or t.is_space)]   # word level, word_ prefix
    result = lexical_diversity(char_tokens)
    for name, value in lexical_diversity(word_tokens).items():
        result["word_" + name] = value
    n_content = sum(1 for t in doc if t.pos_ in CONTENT_POS)
    n_function = sum(1 for t in doc if t.pos_ in FUNCTION_POS)
    result["lex_den"] = n_content / n_function if n_function else 0.0
    return result


# ---------- kazmi-pos.ipynb ----------
def pos_features(doc):
    n = {"ADV": 0, "NOUN": 0, "PRON": 0, "ADJ": 0, "VERB": 0}
    for token in doc:
        pos = "NOUN" if token.pos_ == "PROPN" else token.pos_
        if pos in n:
            n[pos] += 1
    result = {name: 0.0 for name in ["pos_adv_noun", "pos_adv_pron", "pos_adj_verb", "pos_noun_verb",
                                     "pos_verb_pron", "pos_adv_adj", "pos_adj_pron", "pos_noun_pron"]}
    if n["NOUN"]:
        result["pos_adv_noun"] = n["ADV"] / n["NOUN"]
    if n["PRON"]:
        result["pos_adv_pron"] = n["ADV"] / n["PRON"]
        result["pos_verb_pron"] = n["VERB"] / n["PRON"]
        result["pos_adj_pron"] = n["ADJ"] / n["PRON"]
        result["pos_noun_pron"] = n["NOUN"] / n["PRON"]
    if n["VERB"]:
        result["pos_adj_verb"] = n["ADJ"] / n["VERB"]
        result["pos_noun_verb"] = n["NOUN"] / n["VERB"]
    if n["ADJ"]:
        result["pos_adv_adj"] = n["ADV"] / n["ADJ"]
    return result


# ---------- kazmi-syntactic.ipynb ----------
ARG_DEPS = {"nsubj", "nsubj:pass", "obj", "iobj", "csubj", "csubj:pass", "ccomp", "xcomp", "expl"}
ADJUNCT_DEPS = {"obl", "obl:agent", "obl:tmod", "obl:npmod", "advmod", "advcl", "nmod", "nmod:poss",
                "nmod:tmod", "acl", "acl:relcl"}
DEP_REL_TYPES = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "aux:pass", "case", "cc",
                 "ccomp", "compound", "conj", "cop", "csubj", "dep", "det", "discourse", "expl", "fixed",
                 "flat", "iobj", "mark", "nmod", "nmod:poss", "nsubj", "nsubj:pass", "nummod", "obj", "obl",
                 "obl:npmod", "parataxis", "punct", "xcomp"]
DEP_BIGRAM_KEYS = [f"{pair}_{order}" for pair in
                   ["VERB_ADV", "VERB_PRON", "VERB_ADP", "VERB_PROPN", "VERB_SCONJ", "VERB_NUM", "PRON_NOUN",
                    "PRON_VERB", "PROPN_PROPN", "PROPN_NUM", "ADJ_SCONJ", "NOUN_DET", "NOUN_ADJ"]
                   for order in ["before", "after"]]


def tree_depth(token):
    max_depth = 1
    stack = [(token, 1)]
    seen = set()
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        if depth >= 256:
            continue
        for child in node.children:
            edge = (node.i, child.i)
            if edge in seen or child.i == node.i:
                continue
            seen.add(edge)
            stack.append((child, depth + 1))
    return max_depth


def syntactic_features(doc):
    result = {}
    # syn_depth and syn_isc: per sentence, then unweighted mean over sentences
    sentence_depths, sentence_iscs = [], []
    for sent in doc.sents:
        if not sent.text.strip():
            continue
        sentence_depths.append(tree_depth(sent.root))
        isc = 0.0
        for t in sent:
            if t.pos_ in ("SCONJ", "VERB", "AUX"):
                isc += 1
            if t.dep_ in ("acl:relcl", "advcl", "ccomp", "xcomp"):
                isc += 1
        sentence_iscs.append(isc)
    result["syn_depth"] = sum(sentence_depths) / len(sentence_depths) if sentence_depths else 0.0
    result["syn_isc"] = sum(sentence_iscs) / len(sentence_iscs) if sentence_iscs else 0.0

    n_arg = sum(1 for t in doc if t.dep_ in ARG_DEPS)
    n_adjunct = sum(1 for t in doc if t.dep_ in ADJUNCT_DEPS)
    result["arg_adjunct_ratio"] = n_arg / n_adjunct if n_adjunct else 0.0

    tokens = [t for t in doc if t.pos_ != "SPACE"]
    n_non_space = len(tokens) or 1
    for dep in DEP_REL_TYPES:
        result["dep_" + dep.replace(":", "_")] = sum(1 for t in tokens if t.dep_ == dep) / n_non_space

    big_counts = {key: 0 for key in DEP_BIGRAM_KEYS}
    n_non_punct = 0
    for t in doc:
        if t.pos_ in ("SPACE", "PUNCT"):
            continue
        n_non_punct += 1
        if t.dep_ == "ROOT" or t.head.pos_ in ("SPACE", "PUNCT"):
            continue
        order = "before" if t.i < t.head.i else "after"
        key = t.head.pos_ + "_" + t.pos_ + "_" + order
        if key in big_counts:
            big_counts[key] += 1
    n_non_punct = n_non_punct or 1
    for key in DEP_BIGRAM_KEYS:
        result["depbig_" + key] = big_counts[key] / n_non_punct
    return result


# ---------- mean-mdl.ipynb (mean dependency length) ----------
def dependency_arcs(doc):
    """(sentence index, dependent, head, length) for every arc that counts towards MDL."""
    punct_seen = []
    count = 0
    for token in doc:
        if token.pos_ == "PUNCT":
            count += 1
        punct_seen.append(count)
    arcs = []
    for s_index, sent in enumerate(doc.sents):
        for token in sent:
            if token.dep_ == "ROOT" or token.pos_ == "PUNCT":
                continue
            head = token.head
            while head.pos_ == "PUNCT" and head.i != head.head.i:  # punct head -> nearest non-punct ancestor
                head = head.head
            if head.pos_ == "PUNCT":
                continue
            raw_distance = abs(head.i - token.i)
            punct_between = abs(punct_seen[head.i] - punct_seen[token.i])
            arcs.append((s_index, token, head, raw_distance - punct_between))
    return arcs


def mdl_feature(doc):
    per_sentence = {}
    for s_index, _, _, length in dependency_arcs(doc):
        per_sentence.setdefault(s_index, []).append(length)
    sentence_mdls = [sum(v) / len(v) for v in per_sentence.values()]  # sentences with no arcs are skipped
    return {"mean_mdl": sum(sentence_mdls) / len(sentence_mdls) if sentence_mdls else 0.0}


def mdl_breakdown(doc):
    """Per sentence: the ROOT word and every arc behind the MDL (for showing / explaining, not a feature).

    The ROOT has no head, so it never gets an arc of its own - every other word is measured as its
    distance (punctuation not counted) to its head. Uses dependency_arcs, so the numbers are the MDL's own.
    """
    arcs_by_sentence = {}
    for s_index, token, head, length in dependency_arcs(doc):
        arcs_by_sentence.setdefault(s_index, []).append(
            {"word": token.text, "relation": token.dep_, "head": head.text, "length": length})
    sentences = []
    for s_index, sent in enumerate(doc.sents):
        arcs = arcs_by_sentence.get(s_index, [])
        sentences.append({
            "sentence": sent.text.strip(),
            "root": sent.root.text,
            "root_pos": sent.root.pos_,
            "arcs": arcs,
            "mdl": sum(a["length"] for a in arcs) / len(arcs) if arcs else None,  # None = no arc, skipped in MDL
        })
    return sentences


# ---------- mean-surprisal-gpt2.ipynb ----------
def token_surprisals(text, level="paragraph"):
    """[(char start, char end, surprisal in bits)] for every scored GPT-2 token, in order.

    The whole text goes to GPT-2 in one pass. At sentence level <|endoftext|> is put in front so the
    first token also gets a surprisal; at paragraph level the first token has no context and is not scored.
    """
    encoding = gpt2_tokenizer(text, truncation=True, max_length=1024, return_offsets_mapping=True)
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    if level == "sentence":
        token_ids = [gpt2_tokenizer.bos_token_id] + token_ids[:1023]
        offsets = [(0, 0)] + offsets[:1023]
    if len(token_ids) < 2:
        return []
    input_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        logits = gpt2_model(input_ids).logits[0]
    next_token_probs = torch.softmax(logits[:-1], dim=-1)
    next_token_ids = input_ids[0, 1:].unsqueeze(1)
    surprisals = (-torch.log2(next_token_probs.gather(1, next_token_ids).squeeze(1))).tolist()

    scored = []
    for index, surprisal in enumerate(surprisals, start=1):
        start, end = offsets[index]
        token_text = text[start:end]
        start += len(token_text) - len(token_text.lstrip())   # skip the leading space GPT-2 tokens carry
        scored.append((start, end, surprisal))
    return scored


def surprisal_feature(doc, scored):
    """Paragraph surprisal = unweighted mean of the sentence means (same as the notebook)."""
    sentences = list(doc.sents)
    sentence_values = []
    for index, sent in enumerate(sentences):
        # a sentence owns all characters up to the start of the next one
        start_char = 0 if index == 0 else sent.start_char
        end_char = sentences[index + 1].start_char if index + 1 < len(sentences) else len(doc.text)
        values = [s for char_start, _, s in scored if start_char <= char_start < end_char]
        if values:
            sentence_values.append(sum(values) / len(values))
    return {"mean_surprisal": sum(sentence_values) / len(sentence_values) if sentence_values else 0.0}


def analyse_text(text, level="paragraph"):
    """Everything the app needs for one text: features, the parse and per-token surprisal."""
    text = text or ""
    # thinc keeps the GPU choice per thread, and Streamlit runs every rerun in a new thread
    spacy.prefer_gpu()
    doc = parser_nlp(text)
    scored = token_surprisals(doc.text, level)
    features = {}
    features.update(raw_features(doc))
    features.update(lexical_features(doc))
    features.update(pos_features(doc))
    features.update(syntactic_features(doc))
    features.update(mdl_feature(doc))
    features.update(surprisal_feature(doc, scored))
    features = {name: round(value, 5) for name, value in features.items()}  # same 5 decimals as the CSVs
    return {"features": features, "doc": doc, "token_surprisals": scored, "mdl_breakdown": mdl_breakdown(doc)}


def compute_features(text, level="paragraph"):
    """Return a dict {feature_name: value} for one text."""
    return analyse_text(text, level)["features"]
