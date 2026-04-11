import argparse
import json
import os
import time
from pathlib import Path
from collections import defaultdict

from rag_mvp.pipeline import DefaultGenerator, GenerationOptions, RAGPipeline, RetrievalOptions

# --- Heuristic scoring ---
def keyword_score(answer, expected_keywords):
    if not expected_keywords:
        return None
    found = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
    return found / len(expected_keywords)

def citation_score(citations, expected):
    if expected is None:
        return None
    return 1.0 if citations else 0.0

def doc_match_score(citations, expected_docs):
    if not expected_docs:
        return None
    cited_docs = set(c.source for c in citations)
    found = sum(1 for doc in expected_docs if doc in cited_docs)
    return found / len(expected_docs)

def edge_case_score(answer, edge_case):
    if not edge_case:
        return None
    return 1.0 if ("i don't know" in answer.lower() or "no information" in answer.lower()) else 0.0

def total_score(scores, weights):
    total = 0.0
    total_weight = 0.0
    for k, v in scores.items():
        if v is not None:
            total += v * weights.get(k, 1.0)
            total_weight += weights.get(k, 1.0)
    return total / total_weight if total_weight > 0 else None

# --- Main evaluation ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-file', required=True)
    parser.add_argument('--retrieval', default='hybrid', choices=['bm25', 'vector', 'hybrid'])
    parser.add_argument('--answer-mode', default='extractive', choices=['extractive', 'llm'])
    parser.add_argument('--bm25-index', required=True)
    parser.add_argument('--chroma-dir')
    parser.add_argument('--embed-model')
    parser.add_argument(
        '--hybrid-fusion',
        default='rrf',
        choices=['legacy', 'rrf', 'weighted'],
        help='Fusion strategy for hybrid retrieval'
    )
    parser.add_argument('--hybrid-bm25-weight', type=float, default=0.5)
    parser.add_argument('--hybrid-vector-weight', type=float, default=0.5)
    parser.add_argument('--hybrid-rrf-k', type=int, default=60)
    parser.add_argument('--hybrid-candidate-k', type=int, default=20)
    parser.add_argument(
        '--rerank-mode',
        default='lexical',
        choices=['none', 'lexical', 'semantic', 'auto'],
        help='Reranking strategy applied after initial retrieval'
    )
    parser.add_argument('--rerank-candidate-k', type=int, default=20)
    parser.add_argument('--output-dir', default='evaluation/out')
    parser.add_argument('--weights', type=str, default=None, help='JSON string or file for weights')
    parser.add_argument('--max-questions', type=int, default=None)
    args = parser.parse_args()
    pipeline = RAGPipeline(generator=DefaultGenerator(fallback_to_extractive=False))
    retrieval_options = RetrievalOptions(
        retrieval_mode=args.retrieval,
        top_k=5,
        bm25_index_path=Path(args.bm25_index),
        chroma_dir=Path(args.chroma_dir) if args.chroma_dir else None,
        embed_model=args.embed_model,
        hybrid_fusion=args.hybrid_fusion,
        hybrid_bm25_weight=args.hybrid_bm25_weight,
        hybrid_vector_weight=args.hybrid_vector_weight,
        hybrid_rrf_k=args.hybrid_rrf_k,
        hybrid_candidate_k=args.hybrid_candidate_k,
        rerank_mode=args.rerank_mode,
        rerank_candidate_k=args.rerank_candidate_k,
    )
    generation_options = GenerationOptions(
        use_llm=args.answer_mode == 'llm',
        model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
        max_context_chars=3000,
        temperature=0.0,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.eval_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    if args.max_questions:
        questions = questions[:args.max_questions]

    weights = {
        'keyword_score': 0.4,
        'citation_score': 0.2,
        'doc_match_score': 0.2,
        'edge_case_score': 0.2,
    }
    if args.weights:
        if args.weights.strip().startswith('{'):
            weights.update(json.loads(args.weights))
        else:
            with open(args.weights, 'r') as wf:
                weights.update(json.load(wf))

    results = []
    summary_by_type = defaultdict(list)

    for q in questions:
        question = q['question']
        qtype = q.get('type', 'unknown')
        expected_keywords = q.get('expected_keywords')
        expected_docs = q.get('expected_docs')
        edge_case = q.get('edge_case', False)

        # Retrieval
        t0 = time.time()
        hits = pipeline.retrieve(question=question, options=retrieval_options)
        t1 = time.time()
        retrieval_time = t1 - t0

        # Answering
        answer_result = pipeline.generate(question=question, hits=hits, options=generation_options)
        t2 = time.time()
        answer_time = t2 - t1

        # Scoring
        scores = {
            'keyword_score': keyword_score(answer_result.answer, expected_keywords),
            'citation_score': citation_score(answer_result.citations, expected_docs is not None),
            'doc_match_score': doc_match_score(answer_result.citations, expected_docs),
            'edge_case_score': edge_case_score(answer_result.answer, edge_case),
        }
        scores['total_score'] = total_score(scores, weights)

        result = {
            'question': question,
            'type': qtype,
            'answer': answer_result.answer,
            'citations': [c.__dict__ for c in answer_result.citations],
            'retrieval_mode': args.retrieval,
            'answer_mode': args.answer_mode,
            'timing': {
                'retrieval': retrieval_time,
                'answer': answer_time,
                'total': t2 - t0,
            },
            'scores': scores,
            'expected_keywords': expected_keywords,
            'expected_docs': expected_docs,
            'edge_case': edge_case,
        }
        results.append(result)
        summary_by_type[qtype].append(scores)
        print(f"Q: {question}\n  Answer: {answer_result.answer[:80]}...\n  Scores: {scores}\n")

    # Write results
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Aggregate summary
    summary = {}
    for qtype, scores_list in summary_by_type.items():
        agg = {}
        for k in weights:
            vals = [s[k] for s in scores_list if s[k] is not None]
            agg[k] = sum(vals) / len(vals) if vals else None
        total_vals = [s['total_score'] for s in scores_list if s['total_score'] is not None]
        agg['total_score'] = sum(total_vals) / len(total_vals) if total_vals else None
        summary[qtype] = agg
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        for qtype, agg in summary.items():
            f.write(f"Type: {qtype}\n")
            for k, v in agg.items():
                f.write(f"  {k}: {v:.3f}\n" if v is not None else f"  {k}: n/a\n")
            f.write("\n")
    print("\n=== Aggregate Summary ===")
    for qtype, agg in summary.items():
        print(f"{qtype}:")
        for k, v in agg.items():
            print(f"  {k}: {v:.3f}" if v is not None else f"  {k}: n/a")
        print()

if __name__ == '__main__':
    main()
