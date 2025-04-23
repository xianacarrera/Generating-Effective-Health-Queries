from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking.models import CrossEncoder
from beir.reranking.models import MonoT5
from beir.reranking import Rerank

import beir_helper as bh
import logging
import time
from datetime import timedelta
import argparse


def order_results(
   results:dict
):
   sorted_results = {}
   for qid, res in results.items():
       sorted_results[qid] = dict(sorted(res.items(), key=lambda item: item[1],reverse=True))
   return sorted_results


def rerank(
    queries: object,
    corpus: object,
    results: object,
    top_k: int = 100,
    model: str = "cross-encoder",
    training: str = "cross-encoder/ms-marco-electra-base"
):
    retriever = EvaluateRetrieval()

    # Initialize Cross-Encoder
    if model == "cross-encoder":
        model = CrossEncoder(training)
    elif model == "MonoT5":
        model = MonoT5(training, token_false='▁false', token_true='▁true')
    else:
        raise ValueError("Invalid reranker model")
    reranker = Rerank(model, batch_size=128)

    # Re-rank the top 100 results
    print(f"Reranking top {top_k} results")
    reranked_results = reranker.rerank(corpus, queries, results, top_k=top_k)
    sorted_reranked_results = order_results(reranked_results)
    final_results = {}

    # Concatenate the reranked results with the full rank to get a list of 1000 results
    for qid, res in sorted_reranked_results.items():
        full_rank = list(results[qid].items())[top_k:]
        ll = list(res.items()) + full_rank
        dd = dict(ll)
        global_dd = {}
        count =1000
        for key in dd.keys():
            global_dd[key] = count
            count-=1
        final_results[qid] = global_dd

    # Evaluate the results
    logging.info("Retriever evaluation for k in: {}".format(
        retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, final_results, retriever.k_values)
    print("Longitud", len(final_results['1']))

    return ndcg, _map, recall, precision, final_results


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=str, help="Index to load configuration from")
    args = parser.parse_args()

    conf = bh.load_config(args.index, "RERANKER")
    rerank_model = conf["model_name"]
    rerank_model_training = conf["model_training"]    
    llm_model = conf["llm_model"]          # gpt, llama 
    dataset_name = conf["dataset_name"]    # misinfo-2020, C4-2021, C4-2022, CLEF
    method = conf["method"]                # e.g., "orig_RC1"

    start = time.time()

    # Load the custom data
    corpus, queries, qrels = bh.load_custom_data(
        query_path=conf["query_path"],
        qrels_path=conf["qrels_path"],
    )

    # Load the pre-built corpus and BM25 results
    corpus, results = bh.load_BM25_corpus(queries, conf["input_path"], conf["res_file"])

    if conf["clean"]:
        bh.clean_html(corpus)

    # Evaluate using a reranker on top of the BM25 results
    ndcg, _map, recall, precision, reranked_results = rerank(
        queries, corpus, results, 
        model=rerank_model, training=rerank_model_training
    )

    end = time.time()
    time_taken = end - start

    model_name = rerank_model_training
    if model_name.startswith("cross-encoder/ms-marco-"):
        model_name = model_name[len("cross-encoder/ms-marco-"):]

    if llm_model != "":
        full_name = f"{llm_model}_{dataset_name}_{model_name}_{method}"
    else:
        full_name = f"{dataset_name}_{model_name}_{method}"

    print(f"Logging results for {full_name}")
    print(f"Time taken: {timedelta(seconds=time_taken)}")

    # Results: full_model, abbrev, dataset, query type, qrels type, time taken 
    # (formatted in minutes and seconds), BEIR metrics
    bh.log_results(
        conf, full_name, time_taken, _map, precision, recall, ndcg, reranked_results
    )
