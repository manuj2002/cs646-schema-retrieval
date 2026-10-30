import argparse, csv, json, os, random, re, sqlite3, statistics, sys, time
from collections import defaultdict

import sqlglot
from sqlglot import exp

def norm(s):
    return re.sub(r"[`\"\[\]]", "", s).strip().lower()

def split_ident(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return re.sub(r"[_\-\.]+", " ", s).lower()

def tokenize(text):
    return re.findall(r"[a-z0-9]+", split_ident(text))

def gold_tables(sql, valid):
    names = set()
    try:
        for t in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Table):
            names.add(norm(t.name))
    except Exception:
        for m in re.finditer(r"(?:from|join)\s+([`\"\[]?[\w ]+?[`\"\]]?)(?:\s|$|,|\))", sql, re.I):
            names.add(norm(m.group(1)))
    return {n for n in names if n in valid}

def load_descriptions(db_root, db_id, table):
    if not db_root:
        return {}
    path = os.path.join(db_root, db_id, "database_description", f"{table}.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                for row in csv.DictReader(f):
                    row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
                    col = row.get("original_column_name", "")
                    desc = row.get("column_description", "") or row.get("column_name", "")
                    if col:
                        out[norm(col)] = desc
            return out
        except UnicodeDecodeError:
            continue
    return out

def sample_values(db_root, db_id, table, cols, n=3):
    if not db_root:
        return {}
    path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
    if not os.path.exists(path):
        return {}
    vals = {}
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode(errors="ignore")
        for c in cols:
            try:
                rows = con.execute(f'SELECT DISTINCT "{c}" FROM "{table}" WHERE "{c}" IS NOT NULL LIMIT {n}').fetchall()
                vals[norm(c)] = [str(r[0])[:40] for r in rows]
            except Exception:
                pass
        con.close()
    except Exception:
        pass
    return vals

def build_docs(schemas, db_root=None):
    docs = []
    for s in schemas:
        db = s["db_id"]
        tnames = s["table_names_original"]
        cols = defaultdict(list)
        for (ti, cname) in s["column_names_original"]:
            if ti >= 0:
                cols[ti].append(cname)
        fk_nb = defaultdict(set)
        for a, b in s.get("foreign_keys", []):
            ta, tb = s["column_names_original"][a][0], s["column_names_original"][b][0]
            if ta != tb:
                fk_nb[ta].add(tnames[tb]); fk_nb[tb].add(tnames[ta])
        for ti, t in enumerate(tnames):
            desc = load_descriptions(db_root, db, t)
            vals = sample_values(db_root, db, t, cols[ti])
            parts = []
            for c in cols[ti]:
                p = split_ident(c)
                d = desc.get(norm(c))
                if d and d.lower() != p:
                    p += f" ({d})"
                if vals.get(norm(c)):
                    p += " e.g. " + ", ".join(vals[norm(c)])
                parts.append(p)
            text = (f"table {split_ident(t)} ({t}) in database {split_ident(db)}. "
                    f"columns: {'; '.join(parts)}. "
                    f"joins with: {', '.join(sorted(fk_nb[ti])) or 'none'}.")
            docs.append({"id": f"{db}.{norm(t)}", "db": db, "table": norm(t), "text": text})
    return docs

class BM25:
    def __init__(self, docs):
        import bm25s
        self.docs = docs
        self.r = bm25s.BM25(k1=0.9, b=0.4, method="lucene")
        self.r.index([tokenize(d["text"]) for d in docs], show_progress=False)
    def scores(self, query):
        import numpy as np
        return np.asarray(self.r.get_scores(tokenize(query)))

def pick_device(arg):
    if arg != "auto":
        return arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

class Dense:
    def __init__(self, docs, model, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device=device)
        self.prefix = "Represent this sentence for searching relevant passages: " if "bge" in model.lower() and "-zh" not in model else ""
        if "e5" in model.lower():
            self.prefix, dp = "query: ", "passage: "
        else:
            dp = ""
        self.D = self.m.encode([dp + d["text"] for d in docs], batch_size=16,
                               normalize_embeddings=True, show_progress_bar=False)
    def scores_batch(self, queries):
        Q = self.m.encode([self.prefix + q for q in queries], batch_size=32,
                          normalize_embeddings=True, show_progress_bar=False)
        return Q @ self.D.T

class Reranker:
    def __init__(self, model, device="cpu"):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(model, max_length=512, device=device)
    def rerank(self, query, cand_idx, docs):
        s = self.m.predict([(query, docs[i]["text"]) for i in cand_idx], batch_size=16, show_progress_bar=False)
        return [cand_idx[j] for j in sorted(range(len(cand_idx)), key=lambda j: -s[j])]

def evaluate(rankings, golds, ks):
    res = {}
    for k in ks:
        rec, cov = [], []
        for r, g in zip(rankings, golds):
            hit = len(g & set(r[:k]))
            rec.append(hit / len(g)); cov.append(float(hit == len(g)))
        res[f"recall@{k}"] = 100 * statistics.mean(rec)
        res[f"coverage@{k}"] = 100 * statistics.mean(cov)
    mrr = []
    for r, g in zip(rankings, golds):
        rr = 0.0
        for i, d in enumerate(r):
            if d in g:
                rr = 1 / (i + 1); break
        mrr.append(rr)
    res["MRR"] = 100 * statistics.mean(mrr)
    return res

def bootstrap_ci(rankings, golds, k, n=1000, seed=0):
    rnd = random.Random(seed)
    cov = [float(len(g & set(r[:k])) == len(g)) for r, g in zip(rankings, golds)]
    means = sorted(statistics.mean(rnd.choices(cov, k=len(cov))) for _ in range(n))
    return 100 * means[int(0.025 * n)], 100 * means[int(0.975 * n)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--db_root", default=None)
    ap.add_argument("--methods", default="bm25,dense,rerank")
    ap.add_argument("--setting", default="pooled", choices=["pooled", "per_db"])
    ap.add_argument("--with_evidence", action="store_true")
    ap.add_argument("--dense_model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--rerank_model", default="BAAI/bge-reranker-base")
    ap.add_argument("--rerank_depth", type=int, default=50)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="baseline_results.json")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps (auto = cuda if available, else cpu)")
    ap.add_argument("--interactive", action="store_true",
                    help="type your own questions and see the top-10 tables (uses the first method in --methods)")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]

    dev = json.load(open(a.dev))
    schemas = json.load(open(a.tables))
    used = {q["db_id"] for q in dev}
    schemas = [s for s in schemas if s["db_id"] in used]
    docs = build_docs(schemas, a.db_root)
    id2i = {d["id"]: i for i, d in enumerate(docs)}
    tables_by_db = defaultdict(set)
    for d in docs:
        tables_by_db[d["db"]].add(d["table"])

    device = pick_device(a.device)
    if a.interactive:
        method = a.methods.split(",")[0]
        print(f"Indexing {len(docs)} tables from {len(schemas)} databases with {method} ...")
        if method == "bm25":
            bm = BM25(docs); score = lambda q: bm.scores(q)
        else:
            de = Dense(docs, a.dense_model, device); score = lambda q: de.scores_batch([q])[0]
            rr = Reranker(a.rerank_model, device) if method == "rerank" else None
        import numpy as np
        print("Type a question (empty line to quit).")
        while True:
            try:
                q = input("\nQuestion: ").strip()
            except EOFError:
                break
            if not q:
                break
            order = list(np.argsort(-score(q), kind="stable"))
            if method == "rerank":
                order = rr.rerank(q, order[:a.rerank_depth], docs) + order[a.rerank_depth:]
            for r, i in enumerate(order[:10], 1):
                print(f"  {r:2d}. {docs[i]['id']}")
        return

    qs, golds, ngold, dbs = [], [], [], []
    skipped = 0
    for q in dev:
        sql = q.get("SQL") or q.get("query")
        g = gold_tables(sql, tables_by_db[q["db_id"]])
        if not g:
            skipped += 1; continue
        text = q["question"]
        if a.with_evidence and q.get("evidence"):
            text += " " + q["evidence"]
        qs.append(text); dbs.append(q["db_id"])
        golds.append({f"{q['db_id']}.{t}" for t in g}); ngold.append(len(g))
    if a.limit:
        qs, golds, ngold, dbs = qs[:a.limit], golds[:a.limit], ngold[:a.limit], dbs[:a.limit]

    print(f"questions={len(qs)} (skipped {skipped} with no parsable gold table) | "
          f"databases={len(schemas)} | tables={len(docs)} | setting={a.setting} | "
          f"evidence={'yes' if a.with_evidence else 'no'}", file=sys.stderr)
    print(f"gold tables/question: mean={statistics.mean(ngold):.2f}  "
          f"single={sum(n==1 for n in ngold)}  two={sum(n==2 for n in ngold)}  three+={sum(n>=3 for n in ngold)}",
          file=sys.stderr)

    import numpy as np
    mask_db = np.array([d["db"] for d in docs])
    def rank_from_scores(sc, db):
        sc = sc.copy()
        if a.setting == "per_db":
            sc[mask_db != db] = -1e9
        order = np.argsort(-sc, kind="stable")
        if a.setting == "per_db":
            order = [i for i in order if mask_db[i] == db]
        return list(order)

    methods = a.methods.split(",")
    out = {"setting": a.setting, "evidence": a.with_evidence, "n_questions": len(qs),
           "n_tables": len(docs), "n_databases": len(schemas), "results": {}, "timing_s": {}}
    rankings = {}

    def show(name):
        m = evaluate(rankings[name], golds, ks)
        print(f"\n>>> {name}: " + "  ".join(f"{k}={v:.1f}" for k, v in m.items()), flush=True)

    if "bm25" in methods:
        t0 = time.time(); bm = BM25(docs)
        rankings["B1 BM25"] = [[docs[i]["id"] for i in rank_from_scores(bm.scores(q), db)] for q, db in zip(qs, dbs)]
        show("B1 BM25")
        out["timing_s"]["B1 BM25"] = round(time.time() - t0, 1)

    dense_rank_idx = None
    if "dense" in methods or "rerank" in methods:
        t0 = time.time(); de = Dense(docs, a.dense_model, device)
        S = de.scores_batch(qs)
        dense_rank_idx = [rank_from_scores(S[j], dbs[j]) for j in range(len(qs))]
        rankings[f"B2 Dense ({a.dense_model.split('/')[-1]})"] = [[docs[i]["id"] for i in r] for r in dense_rank_idx]
        show(f"B2 Dense ({a.dense_model.split('/')[-1]})")
        out["timing_s"]["B2 Dense"] = round(time.time() - t0, 1)

    if "rerank" in methods:
        t0 = time.time(); rr = Reranker(a.rerank_model, device)
        res = []
        for j, q in enumerate(qs):
            if j % 100 == 0:
                print(f"  rerank: {j}/{len(qs)} questions ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
            cand = dense_rank_idx[j][:a.rerank_depth]
            top = rr.rerank(q, cand, docs)
            rest = dense_rank_idx[j][a.rerank_depth:]
            res.append([docs[i]["id"] for i in top + rest])
        rankings[f"B3 Dense+Rerank ({a.rerank_model.split('/')[-1]})"] = res
        show(f"B3 Dense+Rerank ({a.rerank_model.split('/')[-1]})")
        out["timing_s"]["B3 Rerank"] = round(time.time() - t0, 1)

    kmid = ks[min(1, len(ks) - 1)]
    for name, r in rankings.items():
        m = evaluate(r, golds, ks)
        lo, hi = bootstrap_ci(r, golds, kmid)
        m[f"coverage@{kmid}_95CI"] = [round(lo, 1), round(hi, 1)]
        for bucket, sel in (("1 table", lambda n: n == 1), ("2 tables", lambda n: n == 2), ("3+ tables", lambda n: n >= 3)):
            idx = [i for i, n in enumerate(ngold) if sel(n)]
            if idx:
                m[f"coverage@{kmid} [{bucket}, n={len(idx)}]"] = round(
                    evaluate([r[i] for i in idx], [golds[i] for i in idx], [kmid])[f"coverage@{kmid}"], 1)
        out["results"][name] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in m.items()}

    json.dump(out, open(a.out, "w"), indent=2)
    cols = [f"recall@{k}" for k in ks] + [f"coverage@{k}" for k in ks] + ["MRR"]
    print("\n" + "method".ljust(42) + "".join(c.rjust(13) for c in cols))
    for name, m in out["results"].items():
        print(name.ljust(42) + "".join(f"{m[c]:13.1f}" for c in cols))
    print("\nLaTeX rows:")
    for name, m in out["results"].items():
        print(name.replace("_", r"\_") + " & " + " & ".join(f"{m[c]:.1f}" for c in cols) + r" \\")
    print(f"\nfull results (incl. breakdowns, CIs, timing) -> {a.out}")

if __name__ == "__main__":
    main()
