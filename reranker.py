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

    # Re-rank the top 100 results using the reranker
    print(f"Reranking top {top_k} results")
    reranked_results = reranker.rerank(corpus, queries, results, top_k=top_k)

    # Evaluate the results
    logging.info("Retriever evaluation for k in: {}".format(
        retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, reranked_results, retriever.k_values)
    return ndcg, _map, recall, precision, reranked_results



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

    full_name = f"rerank_bm25+{rerank_model}_{rerank_model_training}_top100"
    if conf["clean"]:
        full_name += "_cleanhtml"
        conf["abbrev"] += "-clean"

    print(f"Logging results for {full_name}")
    print(f"Time taken: {timedelta(seconds=time_taken)}")

    # Results: full_model, abbrev, dataset, query type, qrels type, time taken 
    # (formatted in minutes and seconds), BEIR metrics
    bh.log_results(
        conf, full_name, time_taken, _map, precision, recall, ndcg, reranked_results
    )
