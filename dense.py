from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import time
from datetime import timedelta
import beir_helper as bh
import argparse


def evaluate_dense(
    queries: object,
    corpus: object,
    results: object,
    model_name: str = "msmarco-distilbert-base-tas-b",
    score_function: str = "cosine"
):
    if model_name == "msmarco-distilbert-base-tas-b":
        beir_model = models.SentenceBERT(model_name)
        # Model fine-tuned on MS-MARCO using cosine-similarity
        model = DRES(beir_model, batch_size=256, corpus_chunk_size=512*9999)
    elif model_name == "msmarco-roberta-base-ance-firstp":
        beir_model = models.SentenceBERT(model_name)
        # Should always use dot-similarity
        # score_function = "dot"
        # Default batch_size=128, corpus_chunk_size=50000
        model = DRES(beir_model)
    elif model_name == "DPR":
        # Was fine-tuned using dot-product similarity
        model = DRES(models.SentenceBERT((
            "facebook-dpr-question_encoder-multiset-base",
            "facebook-dpr-ctx_encoder-multiset-base",
            " [SEP] "), batch_size=128))
    elif model_name == "use-qa":
        model = DRES(models.UseQA("https://tfhub.dev/google/universal-sentence-encoder-qa/3"))
    else:
        raise ValueError("Invalid model name")

    retriever = EvaluateRetrieval(model, score_function=score_function)

    # Retrieve the top 1000 results
    start_retrieval_time = time.time()
    results = retriever.retrieve(corpus, queries)
    end_retrieval_time = time.time()
    print(
        f"Retrieval time: {timedelta(seconds=end_retrieval_time - start_retrieval_time)}")

    # Evaluate the results
    logging.info("Retriever evaluation for k in: {}".format(
        retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values)

    mrr = retriever.evaluate_custom(
        qrels, results, retriever.k_values, metric="mrr")
    recall_cap = retriever.evaluate_custom(
        qrels, results, retriever.k_values, metric="r_cap")
    hole = retriever.evaluate_custom(
        qrels, results, retriever.k_values, metric="hole")

    return ndcg, _map, recall, precision, mrr, recall_cap, hole, results


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=str, help="Index to load configuration from")
    args = parser.parse_args()

    program = "dense"
    conf = bh.load_config(args.index, program)
    model_name = conf[program]["model_name"]
    score_function = conf[program]["score_function"]    

    start = time.time()

    # Load the custom data
    corpus, queries, qrels = bh.load_custom_data(
        query_path=conf["query_path"],
        qrels_path=conf["qrels_path"],
    )

    # Load the pre-built corpus and BM25 results
    corpus, results = bh.load_BM25_corpus(queries, conf["input_path"], conf["res_file"])

    start = time.time()

    # Evaluate using a dense model on top of the BM25 results
    ndcg, _map, recall, precision, mrr, recall_cap, hole, results = evaluate_dense(
        queries, corpus, results, model_name, score_function)

    end = time.time()
    time_taken = end - start

    full_name = f"dense_bm25+{model_name}_{score_function}"
    if conf["clean"]:
        full_name += "_cleanhtml"
        conf["abbrev"] += "-clean"

    print(f"Logging results for {full_name}")
    print(f"Time taken: {timedelta(seconds=time_taken)}")

    # Results: full_model, abbrev, dataset, query type, qrels type, time taken 
    # (formatted in minutes and seconds), BEIR metrics
    bh.log_results(
        conf, full_name, time_taken, _map, precision, recall, ndcg, results
    )
