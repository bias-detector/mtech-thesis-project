"""Demo UI.   Run:  streamlit run app.py"""
import difflib
import html
import random

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Base or Edit?", layout="wide")

FEATURE_LABELS = {
    "mean_surprisal": "Mean surprisal (GPT-2, bits)",
    "mean_mdl": "Mean dependency length",
    "std_sen_len": "Sentence length spread",
    "lex_den": "Lexical density (content / function words)",
    "syn_depth": "Parse tree depth",
    "arg_adjunct_ratio": "Argument / adjunct ratio",
    "ttr": "Type-token ratio (chars)", "maas_ttr": "Maas TTR (chars)", "msttr": "MSTTR (chars)",
    "mtld": "MTLD (chars)", "yulek": "Yule's K (chars)",
    "word_ttr": "Type-token ratio (words)", "word_maas_ttr": "Maas TTR (words)", "word_msttr": "MSTTR (words)",
    "word_mtld": "MTLD (words)", "word_yulek": "Yule's K (words)",
}


def label(feature):
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("pos_"):
        a, b = feature[4:].split("_")
        return f"{a.upper()} / {b.upper()} ratio"
    if feature.startswith("depbig_"):
        head, dep, order = feature[7:].split("_")
        return f"{dep} {order} its {head} head"
    if feature.startswith("dep_"):
        return f"Share of '{feature[4:]}' relations"
    return feature


@st.cache_resource(show_spinner=False)
def load():
    import inference          # loads spaCy (trf), GPT-2 and the trained models once
    return inference


@st.cache_data(show_spinner=False)
def load_examples(level):
    # hand-picked pairs (extra-examples.csv, next to this file - kept when the models are retrained) come first,
    # then the held-out test pairs written by train_models.py
    from pathlib import Path
    frames = []
    extra = Path(__file__).with_name("extra-examples.csv")
    if extra.exists():
        held_out = set(pd.read_csv(load().BASE / "data" / "test-articles.csv")["article"].astype(str))
        df = pd.read_csv(extra)
        df["held_out"] = df["article"].astype(str).isin(held_out)
        frames.append(df)
    path = load().MODELS_DIR / f"demo-examples-{level}.csv"
    if path.exists():
        frames.append(pd.read_csv(path).assign(held_out=True))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


with st.spinner("Loading spaCy transformer parser, GPT-2 and trained models..."):
    inference = load()

state = st.session_state
for key, default in [("text_1", ""), ("text_2", ""), ("truth", None), ("result", None), ("revealed", False)]:
    state.setdefault(key, default)

# ---------------- sidebar ----------------
with st.sidebar:
    level = "paragraph"  # the demo is paragraph level only
    bundle = inference.load_bundle(level)
    st.markdown("**Pipeline**")
    st.caption("wikiHow edit history (scraped) → 8-stage cleaning (markup, vandalism, BLEU filter) → "
               "revision pairs → 71 linguistic + cognitive-load features → pair classifiers")
    st.markdown("**Held-out test accuracy** (step pairs)")
    acc = pd.DataFrame({"Model": bundle["model_names"],
                        "Accuracy": [bundle["test_accuracy"][n] for n in bundle["model_names"]]})
    st.dataframe(acc, hide_index=True, column_config={"Accuracy": st.column_config.NumberColumn(format="%.3f")})
    st.caption(f"Trained on {bundle['n_train']:,} pairs, tested on {bundle['n_test']:,} pairs from unseen articles. "
               "Chance = 0.5.")

# ---------------- header ----------------
st.title("Base or Edit?")
st.markdown("**Do edits reduce cognitive load?** Surprisal- and dependency-length-based modelling of revisions "
            "in wikiHow instructional texts.  \nGive two versions of an instruction; the toolkit decides which one "
            "is the **revision** and shows *why*, using surprisal, dependency length and 69 other linguistic features.")

# ---------------- input ----------------
examples = load_examples(level)
col_pick, col_buttons = st.columns([3, 2])
with col_pick:
    if len(examples):
        choice = st.selectbox("wikiHow pair (from held-out test articles the models never saw, unless marked)",
                              range(len(examples)),
                              format_func=lambda i: examples.loc[i, "article"].replace("-", " ")
                              + ("" if examples.loc[i, "held_out"] else " (training article)"))
with col_buttons:
    st.write("")
    b1, b2, b3 = st.columns(3)
    if len(examples) and b1.button("Load example", use_container_width=True):
        row = examples.loc[choice]
        edit_first = random.random() < 0.5          # random order, so nobody knows which box is the edit
        state.text_1, state.text_2 = (row.edit_text, row.base_text) if edit_first else (row.base_text, row.edit_text)
        state.truth = {"edit_box": 1 if edit_first else 2, "url": row.diff_url, "text_1": state.text_1, "text_2": state.text_2}
        state.result, state.revealed = None, False
    if b2.button("Swap ⇄", use_container_width=True):
        state.text_1, state.text_2 = state.text_2, state.text_1
        if state.truth:
            state.truth.update(edit_box=3 - state.truth["edit_box"], text_1=state.text_1, text_2=state.text_2)
        state.result = None
    if b3.button("Clear", use_container_width=True):
        state.text_1 = state.text_2 = ""
        state.truth, state.result, state.revealed = None, None, False

c1, c2 = st.columns(2)
c1.text_area("Text 1", key="text_1", height=170)
c2.text_area("Text 2", key="text_2", height=170)

if st.button("Classify", type="primary"):
    if not state.text_1.strip() or not state.text_2.strip():
        st.warning("Please fill in both texts.")
    else:
        with st.spinner("Parsing, scoring surprisal with GPT-2, computing 71 features, running 6 models..."):
            state.result = inference.analyse_pair(state.text_1, state.text_2, level)
            state.result["inputs"] = (state.text_1, state.text_2)

result = state.result
if result is None or result["inputs"] != (state.text_1, state.text_2):
    st.stop()

# ---------------- verdict ----------------
best = result["results"][0]
p2 = best["p_text_2_is_edit"]
st.divider()
st.subheader(f"Verdict — {best['model']}")
v1, v2, v3 = st.columns([2, 2, 3])
v1.metric("Text 1", "EDIT" if p2 < 0.5 else "BASE", f"P(edit) = {1 - p2:.2f}", delta_color="off")
v2.metric("Text 2", "EDIT" if p2 >= 0.5 else "BASE", f"P(edit) = {p2:.2f}", delta_color="off")
with v3:
    truth = state.truth
    if truth and (truth["text_1"], truth["text_2"]) == result["inputs"]:
        if st.button("Reveal true answer"):
            state.revealed = True
        if state.revealed:
            predicted_box = 2 if p2 >= 0.5 else 1
            ok = predicted_box == truth["edit_box"]
            (st.success if ok else st.error)(f"The real edit is **Text {truth['edit_box']}** — model was "
                                             f"{'right' if ok else 'wrong'}.")
            st.markdown(f"[See this edit on wikiHow]({truth['url']})")
    else:
        st.caption("Typed input: there is no ground truth, judge for yourself.")


# ---------------- diff ----------------
def word_diff(a, b):
    wa, wb = a.split(), b.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb).get_opcodes():
        if op in ("delete", "replace"):
            out.append(f"<del style='background:rgba(220,60,60,.25)'>{html.escape(' '.join(wa[i1:i2]))}</del>")
        if op in ("insert", "replace"):
            out.append(f"<ins style='background:rgba(40,170,90,.25);text-decoration:none'>{html.escape(' '.join(wb[j1:j2]))}</ins>")
        if op == "equal":
            out.append(html.escape(" ".join(wa[i1:i2])))
    return " ".join(out)


st.markdown("**What changed** (Text 1 → Text 2)")
st.markdown(f"<div style='line-height:1.9'>{word_diff(*result['inputs'])}</div>", unsafe_allow_html=True)


# ---------------- cognitive load ----------------
def heatmap(text, scored, cap=15.0):
    parts, pos = [], 0
    for start, end, s in scored:
        if start < pos:
            continue
        parts.append(html.escape(text[pos:start]))
        alpha = min(s, cap) / cap
        parts.append(f"<span title='{s:.1f} bits' style='background:rgba(255,140,0,{alpha:.2f});"
                     f"border-radius:3px;padding:1px 0'>{html.escape(text[start:end])}</span>")
        pos = end
    parts.append(html.escape(text[pos:]))
    return "<div style='line-height:2'>" + "".join(parts) + "</div>"


st.divider()
st.subheader("Cognitive load")
st.caption("Each token is shaded by its GPT-2 surprisal (how unexpected it is, in bits); darker = harder to predict. "
           "Hover a word for its value.")
f1, f2 = result["text_1"]["features"], result["text_2"]["features"]
h1, h2 = st.columns(2)
for col, name, a, other in [(h1, "Text 1", result["text_1"], f2), (h2, "Text 2", result["text_2"], f1)]:
    with col:
        m1, m2 = st.columns(2)
        fa = a["features"]
        m1.metric("Mean surprisal", f"{fa['mean_surprisal']:.2f}", f"{fa['mean_surprisal'] - other['mean_surprisal']:+.2f} vs other",
                  delta_color="inverse")
        m2.metric("Mean dependency length", f"{fa['mean_mdl']:.2f}", f"{fa['mean_mdl'] - other['mean_mdl']:+.2f} vs other",
                  delta_color="inverse")
        roots = ", ".join(f"**{s['root']}**" for s in a["mdl_breakdown"])
        st.caption(f"Root word{'s' if len(a['mdl_breakdown']) > 1 else ''} (one per sentence): {roots}")
        st.markdown(heatmap(a["doc"].text, a["token_surprisals"]), unsafe_allow_html=True)

with st.expander("Dependency parse and root words (what mean dependency length measures)"):
    from spacy import displacy
    st.caption("MDL = average distance between each word and its head, per sentence, then averaged over sentences. "
               "Punctuation is not counted. The **root** is the word the whole sentence hangs on (usually the main "
               "verb): it has no head, so it has no arc of its own and is not in the average. "
               "Long arcs mean the reader must hold words in memory longer.")
    for name, a in [("Text 1", result["text_1"]), ("Text 2", result["text_2"])]:
        st.markdown(f"**{name}** — MDL {a['features']['mean_mdl']:.2f}")
        breakdown = a["mdl_breakdown"]
        st.dataframe(pd.DataFrame([{"Sentence": s["sentence"], "Root word": s["root"], "Root POS": s["root_pos"],
                                    "Arcs": len(s["arcs"]),
                                    "Sentence MDL": "no arc (skipped)" if s["mdl"] is None else f"{s['mdl']:.2f}"}
                                   for s in breakdown]),
                     hide_index=True, use_container_width=True)
        arcs = [{"Sentence": i + 1, "Word": arc["word"], "Relation": arc["relation"], "Head": arc["head"],
                 "Length": arc["length"]} for i, s in enumerate(breakdown) for arc in s["arcs"]]
        if st.toggle(f"Show every arc of {name} ({len(arcs)})", key=f"arcs_{name}"):
            st.dataframe(pd.DataFrame(arcs), hide_index=True, use_container_width=True, height=min(400, 38 + 35 * len(arcs)))
        svg = displacy.render(list(a["doc"].sents), style="dep", page=False,
                              options={"compact": True, "distance": 95, "bg": "#ffffff"})
        components.html(f"<div style='overflow-x:auto;background:#fff'>{svg}</div>", height=330, scrolling=True)

# ---------------- why ----------------
st.divider()
exp = result["explanation"]
st.subheader("Why?")
st.caption(f"Contributions from the interpretable model *{exp['model']}* (averaged over both orderings). "
           "Bars to the right push towards **Text 2 is the edit**, bars to the left towards **Text 1**.")
w1, w2 = st.columns(2)


def bar(rows, key):
    df = pd.DataFrame(rows)
    if df.empty:
        st.caption("Nothing to show.")
        return
    df["towards"] = ["Text 2" if c > 0 else "Text 1" for c in df["contribution"]]
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("contribution:Q", title="push towards Text 2 is edit  →"),
        y=alt.Y(f"{key}:N", sort=None, title=None),
        color=alt.Color("towards:N", scale=alt.Scale(domain=["Text 1", "Text 2"], range=["#4C78A8", "#E45756"]),
                        legend=None),
        tooltip=list(df.columns))
    st.altair_chart(chart, use_container_width=True)


with w1:
    st.markdown(f"**Linguistic features** (total {exp['feature_total']:+.2f})")
    rows = [dict(r, label=label(r["feature"])) for r in exp["features"]]
    bar(rows, "label")
with w2:
    st.markdown(f"**Words / phrases** (total {exp['word_total']:+.2f})")
    bar([dict(r, term=f"{r['term']} ({r['in']})") for r in exp["words"]], "term")

# ---------------- all models ----------------
st.divider()
st.subheader("All models on this input")
rows = [{"Model": r["model"], "Test accuracy": r["test_accuracy"], "P(Text 2 is edit)": r["p_text_2_is_edit"],
         "Says the edit is": "Text 2" if r["p_text_2_is_edit"] >= 0.5 else "Text 1"} for r in result["results"]]
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
             column_config={"Test accuracy": st.column_config.NumberColumn(format="%.3f"),
                            "P(Text 2 is edit)": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")})
with st.expander(f"All {len(bundle['features'])} model feature values"):
    names = bundle["features"]
    st.dataframe(pd.DataFrame({"Feature": [label(n) for n in names], "Text 1": [f1[n] for n in names],
                               "Text 2": [f2[n] for n in names], "Δ (2 − 1)": [f2[n] - f1[n] for n in names]}),
                 hide_index=True, use_container_width=True)
