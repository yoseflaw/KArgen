import warnings
warnings.filterwarnings('ignore')
from spacy import displacy
from nltk import sent_tokenize
import stanza
import re

nlp = stanza.Pipeline(
    "en",
    processors="tokenize",
    package="gum",
    verbose=False,
    tokenize_no_ssplit=True
)

def reduce_duplicate_entities(entities):
    unique_entities = []
    for i in range(len(entities)):
        ent1 = entities[i]
        ent1_set = set(range(ent1["beginOffset"], ent1["endOffset"]))
        found = False
        for j in range(len(unique_entities)):
            ent2 = unique_entities[j]
            ent2_set = set(range(ent2["beginOffset"], ent2["endOffset"]))
            found = len(ent1_set & ent2_set) > 0
            if found: break
        if not found:
            unique_entities.append(ent1)
    return unique_entities


def analyze(sequence_model, paragraph):
    colors = {
        "TERM": "#ffacb7",
        "TERM+": "#ffacb7",
        "NE": "#ffe0ac",
        "NE+": "#ffe0ac",
        "REL": "#f9f9f9",
        "REL+": "#f9f9f9",
    }
    options = {"ents": colors.keys(), "colors": colors}
    sentences = sent_tokenize(re.sub("\n+", " ", paragraph))
    for sentence in sentences:
        analysis = sequence_model.analyze(sentence)
        words, nes, terms, rels = analysis["words"], analysis["entities"], analysis["terms"], analysis["head_rels"]
        offset = 0
        begin_offsets = []
        for word in words:
            begin_offsets.append(offset)
            offset += len(word) + 1
        prepared_rels = []
        has_rel = [0 for _ in range(len(words))]
        for rel in rels:
            curr_offset = rel["offset"]
            prepared_rel = {
                "text": rel["text"],
                "beginOffset": rel["offset"],
                "endOffset": rel["offset"] + 1,
                "type": "REL"
            }
            has_rel[curr_offset] = 1
            prepared_rels.append(prepared_rel)
        prepared_nes = []
        for ne in nes:
            if ne["type"] != "PERSON":
                news_ne = ne.copy()
                news_ne["type"] = "NE"
                prepared_nes.append(news_ne)
        unique_entities = reduce_duplicate_entities(terms + prepared_nes + prepared_rels)
        displacy_ents = []
        for entity in unique_entities:
            is_rel = 0
            i = entity["beginOffset"]
            while i < entity["endOffset"] and is_rel == 0:
                is_rel = has_rel[i]
                i += 1
            ent = {
                "start": begin_offsets[entity["beginOffset"]],
                "end": begin_offsets[entity["endOffset"]-1] + len(words[entity["endOffset"]-1]),
                "label": entity["type"] + ("+" if is_rel == 1 else "")
            }
            displacy_ents.append(ent)
        sorted_ents = sorted(displacy_ents, key=lambda k: k["start"])
        result = {
            "text": " ".join(words),
            "ents": sorted_ents
        }
        displacy.render(result, style="ent", manual=True, jupyter=True, options=options)