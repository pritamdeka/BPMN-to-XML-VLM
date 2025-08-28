#############   for a fairer evaluation for flows

import os
import pandas as pd
import difflib
import re
from lxml import etree
from collections import defaultdict
from difflib import SequenceMatcher
import argparse

NAMESPACE = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

# -------- Canonicalisation helpers --------
def canon(txt: str) -> str:
    """Lowercase, normalize whitespace, remove simple stopwords and punctuation."""
    txt = txt.lower()
    txt = re.sub(r'[\-_]', ' ', txt)
    txt = re.sub(r'\b(the|a|an)\b', '', txt)
    txt = re.sub(r'[^a-z0-9 ]', '', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

TAG_ALIASES = {
    # Tasks
    "task": "task",
    "usertask": "task",
    "manualtask": "task",
    "servicetask": "task",
    "scripttask": "task",
    "businesstask": "task",
    "sendtask": "task",
    "receivetask": "task",
    "callactivity": "activity",
    "subprocess": "activity",
    "transaction": "activity",
    "adhocsubprocess": "activity",

    # Events
    "event": "event",
    "startevent": "event",
    "intermediatecatchevent": "event",
    "intermediatethrowevent": "event",
    "boundaryevent": "event",
    "endevent": "event",
    "messageevent": "event",
    "timer": "event",
    "message": "event",
    "timerintermediatecatchevent": "event",
    "escalationevent": "event",
    "compensationevent": "event",
    "conditionalintermediatecatchevent": "event",
    "conditionalboundaryevent": "event",
    "signalintermediatecatchevent": "event",
    "signalboundaryevent": "event",
    "linkevent": "event",
    "terminateevent": "event",
    "errorevent": "event",
    "cancelevent": "event",

    # Gateways
    "gateway": "gateway",
    "exclusivegateway": "gateway",
    "inclusivegateway": "gateway",
    "parallelgateway": "gateway",
    "eventbasedgateway": "gateway",
    "complexgateway": "gateway",

    # Data
    "datastore": "data",
    "datastorereference": "data",
    "dataobject": "data",
    "dataobjectreference": "data",
    "datainput": "data",
    "dataoutput": "data",
    "dataassociation": "data",
    "inputset": "data",
    "outputset": "data",

    # Swimlanes/Participants
    "participant": "pool",
    "pool": "pool",
    "lane": "lane",

    # Flows
    "sequenceflow": "sequenceflow",
    "association": "association",
    "messageflow": "messageflow",

    # Artifacts
    "textannotation": "annotation",
    "annotation": "annotation",
    "group": "group",

    # Misc
    "process": "process",
    "collaboration": "collaboration",
    "choreography": "choreography",
    "conversation": "conversation",
    "conversationlink": "conversationlink",
    "callconversation": "conversation",
    "participantassociation": "participantassociation",

    # Other
    "lane": "lane",
    "flowelement": "flowelement",  # abstract
    "artifact": "artifact",
}

def canonical_tag(tag):
    return TAG_ALIASES.get(tag.lower(), tag.lower())


# ---- XML Extraction ----
def extract_elements(filepath):
    parser = etree.XMLParser(recover=True)
    try:
        with open(filepath, 'rb') as f:
            xml_bytes = f.read()
        try:
            tree = etree.parse(filepath, parser)
            root = tree.getroot()
        except Exception:
            root = etree.fromstring(xml_bytes, parser)
        if root is None:
            return {}, {}
        processes = root.findall('.//bpmn:process', namespaces=NAMESPACE)
        elements_by_type = defaultdict(list)
        id2name = {}
        for process in processes:
            for elem in process.iter():
                if not isinstance(elem.tag, str):
                    continue
                tag = canonical_tag(etree.QName(elem.tag).localname)
                elem_id = elem.attrib.get('id', '')
                elem_name = canon(elem.attrib.get('name', ''))
                if tag == 'sequenceflow':
                    elements_by_type['sequenceFlow'].append({
                        'id': elem_id,
                        'name': elem_name,
                        'sourceRef': elem.attrib.get('sourceRef', ''),
                        'targetRef': elem.attrib.get('targetRef', '')
                    })
                else:
                    elements_by_type[tag].append({'id': elem_id, 'name': elem_name})
                    id2name[elem_id] = elem_name

        # Message Flows (outside <process>)
        message_flows = root.findall('.//bpmn:messageFlow', namespaces=NAMESPACE)
        for elem in message_flows:
            tag = canonical_tag(etree.QName(elem.tag).localname)
            elem_id = elem.attrib.get('id', '')
            elem_name = canon(elem.attrib.get('name', ''))
            elements_by_type['messageFlow'].append({
                'id': elem_id,
                'name': elem_name,
                'sourceRef': elem.attrib.get('sourceRef', ''),
                'targetRef': elem.attrib.get('targetRef', '')
            })
            id2name[elem_id] = elem_name
        return elements_by_type, id2name
    except Exception as e:
        print(f"Parsing failed: {filepath} — {e}")
        return {}, {}

# ---- Set Extraction ----
def name_set(elements):
    return {e['name'] for e in elements if e['name']}

def type_set(elements_by_type):
    return set(elements_by_type.keys())

def flow_set(flows, id2name, directed=True):
    pairs = set()
    for f in flows:
        src = canon(id2name.get(f['sourceRef'], f['sourceRef']))
        tgt = canon(id2name.get(f['targetRef'], f['targetRef']))
        if src and tgt:
            edge = (src, tgt) if directed else tuple(sorted([src, tgt]))
            pairs.add(edge)
    return pairs

def name_type_set(elements_by_type):
    pairs = set()
    for typ, elements in elements_by_type.items():
        for e in elements:
            if e['name']:
                pairs.add((e['name'], typ))
    return pairs

# ---- Fuzzy Matching ----
def match_partial(set_a, set_b, cutoff=0.7):
    unmatched_b = set(set_b)
    matches = set()
    for a in set_a:
        best = difflib.get_close_matches(a, unmatched_b, n=1, cutoff=cutoff)
        if best:
            b = best[0]
            matches.add((a, b))
            unmatched_b.remove(b)
    return matches

def match_partial_name_type(gold_pairs, pred_pairs, cutoff=0.7):
    matches = set()
    pred_pairs_left = set(pred_pairs)
    for g_name, g_type in gold_pairs:
        candidates = [(p_name, p_type) for (p_name, p_type) in pred_pairs_left if p_type == g_type]
        best = difflib.get_close_matches(g_name, [p[0] for p in candidates], n=1, cutoff=cutoff)
        if best:
            chosen = next((p for p in candidates if p[0] == best[0]), None)
            if chosen:
                matches.add((g_name, chosen[0], g_type))
                pred_pairs_left.remove(chosen)
    return matches

def best_edge_match(gold_edge, pred_edges, cutoff=0.7):
    g_src, g_tgt = gold_edge
    best_edge = None
    best_score = 0.0
    for p_src, p_tgt in pred_edges:
        src_sim = SequenceMatcher(None, g_src, p_src).ratio()
        tgt_sim = SequenceMatcher(None, g_tgt, p_tgt).ratio()
        score = (src_sim + tgt_sim) / 2
        if score > best_score and score >= cutoff:
            best_score = score
            best_edge = (p_src, p_tgt)
    return best_edge

def match_partial_flows(gold_flows, pred_flows, cutoff=0.7):
    unmatched_pred = set(pred_flows)
    matches = set()
    for g_edge in gold_flows:
        best = best_edge_match(g_edge, unmatched_pred, cutoff=cutoff)
        if best:
            matches.add((g_edge, best))
            unmatched_pred.remove(best)
    return matches

# ---- PRF (strict and partial) ----
def prf(gold, pred):
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0
    recall = tp / len(gold) if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def prf_partial(gold, pred, cutoff=0.7):
    matches = match_partial(gold, pred, cutoff)
    tp = len(matches)
    precision = tp / len(pred) if pred else 0
    recall = tp / len(gold) if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def prf_pairs(gold_pairs, pred_pairs):
    tp = len(gold_pairs & pred_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0
    recall = tp / len(gold_pairs) if gold_pairs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def prf_pairs_partial(gold_pairs, pred_pairs, cutoff=0.7):
    matches = match_partial_name_type(gold_pairs, pred_pairs, cutoff)
    tp = len(matches)
    precision = tp / len(pred_pairs) if pred_pairs else 0
    recall = tp / len(gold_pairs) if gold_pairs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def prf_flows_partial(gold_flows, pred_flows, cutoff=0.7):
    matches = match_partial_flows(gold_flows, pred_flows, cutoff)
    tp = len(matches)
    precision = tp / len(pred_flows) if pred_flows else 0
    recall = tp / len(gold_flows) if gold_flows else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

# ---- Evaluation Main ----
def evaluate_folders_simple(gold_folder, gen_folder, partial_cutoff=0.7, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    name_rows, type_rows = [], []
    seq_rows, msg_rows, allrel_rows = [], [], []
    name_type_rows = []
    gold_files = {os.path.splitext(f)[0]: os.path.join(gold_folder, f) for f in os.listdir(gold_folder) if f.endswith('.bpmn')}
    gen_files = {os.path.splitext(f)[0]: os.path.join(gen_folder, f) for f in os.listdir(gen_folder) if f.endswith('.bpmn')}
    common = set(gold_files.keys()) & set(gen_files.keys())
    for fname in sorted(common):
        gold_elements, gold_id2name = extract_elements(gold_files[fname])
        gen_elements, gen_id2name = extract_elements(gen_files[fname])
        # Name only
        g_names = set()
        for v in gold_elements.values():
            g_names.update(name_set(v))
        m_names = set()
        for v in gen_elements.values():
            m_names.update(name_set(v))
        p, r, f = prf(g_names, m_names)
        pp, pr_, pf = prf_partial(g_names, m_names, cutoff=partial_cutoff)
        name_rows.append({'filename': fname, 'precision': p, 'recall': r, 'f1': f,
                          'precision_partial': pp, 'recall_partial': pr_, 'f1_partial': pf})

        # Type only
        g_types = type_set(gold_elements)
        m_types = type_set(gen_elements)
        p2, r2, f2 = prf(g_types, m_types)
        pp2, pr2_, pf2 = prf_partial(g_types, m_types, cutoff=partial_cutoff)
        type_rows.append({'filename': fname, 'precision': p2, 'recall': r2, 'f1': f2,
                          'precision_partial': pp2, 'recall_partial': pr2_, 'f1_partial': pf2})

        # Sequence Flows (directed)
        gold_seq = flow_set(gold_elements.get('sequenceFlow', []), gold_id2name)
        gen_seq = flow_set(gen_elements.get('sequenceFlow', []), gen_id2name)
        p3, r3, f3 = prf(gold_seq, gen_seq)
        pp3, pr3_, pf3 = prf_flows_partial(gold_seq, gen_seq, cutoff=partial_cutoff)
        seq_rows.append({'filename': fname, 'precision': p3, 'recall': r3, 'f1': f3,
                         'precision_partial': pp3, 'recall_partial': pr3_, 'f1_partial': pf3})

        # Message Flows (directed)
        gold_msg = flow_set(gold_elements.get('messageFlow', []), gold_id2name)
        gen_msg = flow_set(gen_elements.get('messageFlow', []), gen_id2name)
        p4, r4, f4 = prf(gold_msg, gen_msg)
        pp4, pr4_, pf4 = prf_flows_partial(gold_msg, gen_msg, cutoff=partial_cutoff)
        msg_rows.append({'filename': fname, 'precision': p4, 'recall': r4, 'f1': f4,
                         'precision_partial': pp4, 'recall_partial': pr4_, 'f1_partial': pf4})

        # All Flows: union of sequence and message flows
        gold_all = gold_seq | gold_msg
        gen_all = gen_seq | gen_msg
        p5, r5, f5 = prf(gold_all, gen_all)
        pp5, pr5_, pf5 = prf_flows_partial(gold_all, gen_all, cutoff=partial_cutoff)
        allrel_rows.append({'filename': fname, 'precision': p5, 'recall': r5, 'f1': f5,
                            'precision_partial': pp5, 'recall_partial': pr5_, 'f1_partial': pf5})

        # Name + Type
        gold_pairs = name_type_set(gold_elements)
        gen_pairs = name_type_set(gen_elements)
        p_nt, r_nt, f_nt = prf_pairs(gold_pairs, gen_pairs)
        pp_nt, pr_nt, pf_nt = prf_pairs_partial(gold_pairs, gen_pairs, cutoff=partial_cutoff)
        name_type_rows.append({'filename': fname, 'precision': p_nt, 'recall': r_nt, 'f1': f_nt,
                               'precision_partial': pp_nt, 'recall_partial': pr_nt, 'f1_partial': pf_nt})

    def save_csv(rows, path):
        df = pd.DataFrame(rows)
        if not df.empty:
            avg = {'filename': 'AVERAGE'}
            for col in df.columns:
                if col != 'filename':
                    avg[col] = df[col].mean()
            df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    save_csv(name_rows, os.path.join(out_dir, "bpmn_comparison_name_only.csv"))
    save_csv(type_rows, os.path.join(out_dir, "bpmn_comparison_type_only.csv"))
    #save_csv(seq_rows, os.path.join(out_dir, "bpmn_comparison_sequence_flows.csv"))
    #save_csv(msg_rows, os.path.join(out_dir, "bpmn_comparison_message_flows.csv"))
    save_csv(allrel_rows, os.path.join(out_dir, "bpmn_comparison_relations.csv"))
    save_csv(name_type_rows, os.path.join(out_dir, "bpmn_comparison_name_type.csv"))

def main():
    ap = argparse.ArgumentParser(description="Fairer evaluation for BPMN flows (strict + partial matching).")
    ap.add_argument("--gold-folder", required=True, help="Folder of gold .bpmn files")
    ap.add_argument("--pred-folder", required=True, help="Folder of predicted .bpmn files")
    ap.add_argument("--cutoff", type=float, default=0.7, help="Partial match cutoff (0..1)")
    ap.add_argument("--out-dir", default="outputs/eval_flows", help="Directory to write CSV reports")
    args = ap.parse_args()
    evaluate_folders_simple(args.gold_folder, args.pred_folder, partial_cutoff=args.cutoff, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
