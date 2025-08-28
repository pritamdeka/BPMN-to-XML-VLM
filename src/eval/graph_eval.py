
import os, re
import argparse
import networkx as nx
from lxml import etree
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ---- CONFIG ----
NAMESPACE = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

def robust_parse_bpmn(filepath):
    parser = etree.XMLParser(recover=True)
    try:
        with open(filepath, 'rb') as f:
            xml_bytes = f.read()
        try:
            tree = etree.parse(filepath, parser)
            root = tree.getroot()
        except Exception as e:
            print(f"etree.parse() failed: {filepath} — {e}")
            try:
                root = etree.fromstring(xml_bytes, parser)
            except Exception as e2:
                print(f"etree.fromstring() also failed: {filepath} — {e2}")
                return None
        return root
    except Exception as e:
        print(f"Opening/Parsing failed: {filepath} — {e}")
        return None

# ---- GRAPH & NAME UTILS ----
def bpmn_to_nx_graph(bpmn_file):
    G = nx.DiGraph()
    root = robust_parse_bpmn(bpmn_file)
    if root is None:
        return G
    processes = root.findall('.//bpmn:process', namespaces=NAMESPACE)
    for process in processes:
        for elem in process.iter():
            if not isinstance(elem.tag, str): continue
            tag = etree.QName(elem.tag).localname
            elem_id = elem.attrib.get('id', '')
            elem_name = elem.attrib.get('name', '').strip().lower()
            if tag != "sequenceFlow" and elem_id:
                G.add_node(elem_id, label=elem_name, type=tag)
        for elem in process.iter():
            if not isinstance(elem.tag, str): continue
            tag = etree.QName(elem.tag).localname
            if tag == 'sequenceFlow':
                src = elem.attrib.get('sourceRef', '')
                tgt = elem.attrib.get('targetRef', '')
                if src in G.nodes() and tgt in G.nodes():
                    G.add_edge(src, tgt)
    return G

def flatten_bpmn_elements(xml_path):
    root = robust_parse_bpmn(xml_path)
    elements = []
    if root is not None:
        processes = root.findall('.//bpmn:process', namespaces=NAMESPACE)
        for process in processes:
            for elem in process.iter():
                if not isinstance(elem.tag, str): continue
                tag = etree.QName(elem.tag).localname
                name = elem.attrib.get('name', '').strip().lower()
                elements.append(f"{tag}:{name}")
    return elements

def lcs(X, Y):
    m, n = len(X), len(Y)
    L = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i==0 or j==0: L[i][j]=0
            elif X[i-1]==Y[j-1]: L[i][j]=L[i-1][j-1]+1
            else: L[i][j]=max(L[i-1][j],L[i][j-1])
    return L[m][n]

def degree_histogram_similarity(g1, g2):
    hist1 = np.bincount([d for n, d in g1.degree()]) if g1.number_of_nodes() else np.array([0])
    hist2 = np.bincount([d for n, d in g2.degree()]) if g2.number_of_nodes() else np.array([0])
    maxlen = max(len(hist1), len(hist2))
    hist1 = np.pad(hist1, (0, maxlen - len(hist1)))
    hist2 = np.pad(hist2, (0, maxlen - len(hist2)))
    if np.linalg.norm(hist1) == 0 or np.linalg.norm(hist2) == 0:
        return 0.0
    sim = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
    return sim

# ---- SEMANTIC SIMILARITY ----
def semantic_similarity_bpmn(gold_G, gen_G, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    gold_labels = [d['label'] for n, d in gold_G.nodes(data=True) if d['label']]
    gen_labels = [d['label'] for n, d in gen_G.nodes(data=True) if d['label']]
    if not gold_labels or not gen_labels:
        return 0, 0
    gold_emb = model.encode(gold_labels)
    gen_emb = model.encode(gen_labels)
    sim_matrix = cosine_similarity(gold_emb, gen_emb)
    max_sim_gold = sim_matrix.max(axis=1)
    avg_sim_gold = np.mean(max_sim_gold)
    max_sim_gen = sim_matrix.max(axis=0)
    avg_sim_gen = np.mean(max_sim_gen)
    return avg_sim_gold, avg_sim_gen

def evaluate_graph_similarity(gold_folder, pred_folder, output_csv):
    gold_files = [f for f in os.listdir(gold_folder) if f.endswith('.bpmn')]
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.bpmn')]
    gold_stems = {os.path.splitext(f)[0]: f for f in gold_files}
    pred_stems = {os.path.splitext(f)[0]: f for f in pred_files}
    results = []

    for stem, gf in gold_stems.items():
        pf = pred_stems.get(stem)
        if pf:
            try:
                gold_path = os.path.join(gold_folder, gf)
                pred_path = os.path.join(pred_folder, pf)
                gold_G = bpmn_to_nx_graph(gold_path)
                pred_G = bpmn_to_nx_graph(pred_path)

                gold_nodes = gold_G.number_of_nodes()
                pred_nodes = pred_G.number_of_nodes()
                gold_edges = gold_G.number_of_edges()
                pred_edges = pred_G.number_of_edges()
                node_ratio = pred_nodes / gold_nodes if gold_nodes > 0 else 0
                edge_ratio = pred_edges / gold_edges if gold_edges > 0 else 0
                node_diff = pred_nodes - gold_nodes
                edge_diff = pred_edges - gold_edges

                deg_sim = degree_histogram_similarity(gold_G, pred_G)

                flat_gold = flatten_bpmn_elements(gold_path)
                flat_pred = flatten_bpmn_elements(pred_path)
                lcs_len = lcs(flat_gold, flat_pred)
                lcs_score = lcs_len / max(len(flat_gold), len(flat_pred)) if max(len(flat_gold), len(flat_pred)) > 0 else 0

                avg_sim_gold, avg_sim_pred = semantic_similarity_bpmn(gold_G, pred_G)

                results.append({
                    "Process ID": pf,
                    "Gold Nodes": gold_nodes,
                    "Pred Nodes": pred_nodes,
                    "Node Ratio": node_ratio,
                    "Node Diff": node_diff,
                    "Gold Edges": gold_edges,
                    "Pred Edges": pred_edges,
                    "Edge Ratio": edge_ratio,
                    "Edge Diff": edge_diff,
                    "Degree Hist Sim": deg_sim,
                    "LCS Score": lcs_score,
                    "Semantic Sim (gold2pred)": avg_sim_gold,
                    "Semantic Sim (pred2gold)": avg_sim_pred,
                })
            except Exception as e:
                print(f"❌ Error in {stem}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df.loc[len(df)] = {
            "Process ID": "AVERAGE",
            "Gold Nodes": df["Gold Nodes"].mean(),
            "Pred Nodes": df["Pred Nodes"].mean(),
            "Node Ratio": df["Node Ratio"].mean(),
            "Node Diff": df["Node Diff"].mean(),
            "Gold Edges": df["Gold Edges"].mean(),
            "Pred Edges": df["Pred Edges"].mean(),
            "Edge Ratio": df["Edge Ratio"].mean(),
            "Edge Diff": df["Edge Diff"].mean(),
            "Degree Hist Sim": df["Degree Hist Sim"].mean(),
            "LCS Score": df["LCS Score"].mean(),
            "Semantic Sim (gold2pred)": df["Semantic Sim (gold2pred)"].mean(),
            "Semantic Sim (pred2gold)": df["Semantic Sim (pred2gold)"].mean()
        }
        df.to_csv(output_csv, index=False)
        print(f"✅ Fair evaluation results saved to: {output_csv}")
        print(df.head().to_string(index=False))
    else:
        print("⚠️ No results to save: DataFrame is empty. Check input folders or parsing issues.")

def main():
    ap = argparse.ArgumentParser(description="Graph-based BPMN evaluation (structure + semantics).")
    ap.add_argument("--gold-folder", required=True, help="Folder of gold .bpmn files")
    ap.add_argument("--pred-folder", required=True, help="Folder of predicted .bpmn files")
    ap.add_argument("--out-csv", default="outputs/bpmn_eval_fair.csv", help="Output CSV path")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    evaluate_graph_similarity(args.gold_folder, args.pred_folder, args.out_csv)

if __name__ == "__main__":
    main()
