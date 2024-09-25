from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pyserini.search import SimpleSearcher
from beir.reranking.models import CrossEncoder
from beir.reranking.models import MonoT5
from beir.reranking import Rerank

import logging
import csv
import configparser
import time
from datetime import timedelta


def load_custom_data(
    corpus_path: str = "",
    query_path: str = "",
    qrels_path: str = "",
):
    if corpus_path == "":
        use_corpus = False
    else:
        use_corpus = True

    print("Corpus used?: ", use_corpus)

    corpus, queries, qrels = GenericDataLoader(
        corpus_file=corpus_path,
        query_file=query_path,
        qrels_file=qrels_path,
        use_corpus=use_corpus).load_custom()

    return corpus, queries, qrels

def load_BM25_corpus(
    queries: object,
    input_path: str,
    res_file: str = "res"
):
    corpus = {}
    results = {}
    for qid in list(queries):
        # Load csv as a dictionary
        results[qid] = {}
        with open(f"{input_path}/query_{qid}/{res_file}", mode='r') as infile:
            reader = csv.reader(infile, delimiter=' ')

            for i, rows in enumerate(reader):
                docno = rows[2]              # 3rd column
                score = float(rows[4])       # 5th column (results are already sorted by score)
                results[qid][docno] = score

                #print(f"Query {qid}, doc {i+1}: {docno} with score {score}")

                # Load the raw text of the document
                with open(f"{input_path}/query_{qid}/doc_{i+1}.txt", 'r') as f:
                    text = f.read()
                    corpus[docno] = {"text": text}
    
    return corpus, results


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
    reranked_results = reranker.rerank(corpus, queries, results, top_k=top_k)

    # Evaluate the results
    logging.info("Retriever evaluation for k in: {}".format(
        retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, reranked_results, retriever.k_values)
    return ndcg, _map, recall, precision


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    config = configparser.ConfigParser()
    config.read("config.ini")

    dataset_name = config["META"]["DATASET_NAME"]
    # We take the part before the first hyphen in uppercase
    option = dataset_name.split("-")[0].upper()

    query_path = config[option]["QUERY_DESC_PATH"]
    qrels_path = config[option]["QRELS_PATH"]
    index_path = config[option]["INDEX_PATH"]
    rerank_model = config["RERANK"]["RERANKER"]    # Cross-encoder, monot5, etc.
    rerank_model_training = config["RERANK"]["TRAINING"]

    input_path = config["META"]["INPUT_PATH"]
    res_file = config["META"]["RES_FILE"]

    start = time.time()

    # Load the custom data
    corpus, queries, qrels = load_custom_data(
        query_path=query_path,
        qrels_path=qrels_path
    )

    # Load the pre-built corpus and BM25 results
    corpus, results = load_BM25_corpus(queries, input_path, res_file)

    # Evaluate using a reranker on top of the BM25 results
    ndcg, _map, recall, precision = rerank(queries, corpus, results, model=rerank_model, training=rerank_model_training)

    end = time.time()
    time_taken = end - start

    # Name of the file containing the queries
    query_file = query_path.split("/")[-1]
    # Name of the file containing the qrels
    qrels_file = qrels_path.split("/")[-1]

    print(f"Logging results for rerank_bm25+{rerank_model}_{rerank_model_training}")
    print(f"Time taken: {timedelta(seconds=time_taken)}")

    # Results: model, dataset, query type, qrels type, time taken (formatted in minutes and seconds), BEIR metrics
    row = [f"rerank_bm25+{rerank_model}_{rerank_model_training}", dataset_name, query_file, qrels_file, str(timedelta(seconds=time_taken)),
           _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"],
           precision["P@10"], precision["P@100"], precision["P@1000"],
           recall["Recall@10"], recall["Recall@100"], recall["Recall@1000"],
           ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    with open("../beir_stats_output.csv", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()
