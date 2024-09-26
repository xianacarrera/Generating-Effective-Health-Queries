from beir import LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer

import logging
import csv
import configparser
import time
from datetime import timedelta
import torch
import importlib.util
import os, pathlib  

if importlib.util.find_spec("tldr") is None:
    from tldr import TLDR


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
                # 5th column (results are already sorted by score)
                score = float(rows[4])
                results[qid][docno] = score

                # print(f"Query {qid}, doc {i+1}: {docno} with score {score}")

                # Load the raw text of the document
                with open(f"{input_path}/query_{qid}/doc_{i+1}.txt", 'r') as f:
                    text = f.read()
                    corpus[docno] = {"text": text}

    return corpus, results

def corpus_into_list(corpus: object):
    corpus_ids = sorted(corpus, key=lambda k: len(corpus[k]["text"]), reverse=True)
    corpus_list = [corpus[doc_id] for doc_id in corpus_ids]
    return corpus_list


def evaluate_dense(
    queries: object,
    corpus: object,
    results: object,
    model_name: str = "msmarco-distilbert-base-tas-b",
    score_function: str = "cosine"
):

    # Create the TLDR model instance providing the SBERT model path name
    tldr = models.TLDR(
        encoder_model=SentenceTransformer(model_name),
        n_components=128,
        n_neighbors=5,
        encoder="linear",
        projector="mlp-1-2048",
        verbose=2,
        knn_approximation=None,
        output_folder="data/"
    )

    # Starting to train the TLDR model with TAS-B model on the target dataset
    tldr.fit(corpus=corpus, batch_size=128, epochs=100, warmup_epochs=10, train_batch_size=1024, print_every=100)
    logging.info("TLDR model training completed\n")

    # Save the model
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "tldr", "inference_model.pt") 
    logging.info("TLDR model saved here: %s\n" % model_save_path)
    tldr.save(model_save_path)

    # Load back the trained model using the code below:
    tldr = TLDR()
    tldr.load(model_save_path, init=True)  # Loads both model parameters and weights

    retriever = EvaluateRetrieval(DRES(tldr), score_function=score_function)

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

    return ndcg, _map, recall, precision, mrr, recall_cap, hole


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

    model_name = "sentence-transformers/msmarco-distilbert-base-tas-b"
    score_function = "dot"
    abbrev = "TLDR_distilbert-base-tas-b_dot"

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

    # Get all the corpus documents as a list for tldr training
    corpus_list = corpus_into_list(corpus)

    # Evaluate using a dense model on top of the BM25 results
    ndcg, _map, recall, precision, mrr, recall_cap, hole = evaluate_dense(
        queries, corpus, results, model_name, score_function)

    end = time.time()
    time_taken = end - start

    # Name of the file containing the queries
    query_file = query_path.split("/")[-1]
    # Name of the file containing the qrels
    qrels_file = qrels_path.split("/")[-1]

    print(
        f"Logging results for dense_bm25+sentenceBERT-msmarco-distilbert-base-tas-b")
    print(f"Time taken: {timedelta(seconds=time_taken)}")

    # Results: model, abbrev, dataset, query type, qrels type, time taken (formatted in minutes and seconds), BEIR metrics
    row = [f"dense_bm25+sentenceBERT_{model_name}_{score_function}", abbrev,
           dataset_name, query_file, qrels_file, str(
               timedelta(seconds=time_taken)),
           _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"],
           precision["P@10"], precision["P@100"], precision["P@1000"],
           recall["Recall@10"], recall["Recall@100"], recall["Recall@1000"],
           ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    with open("../beir_dense_stats_output.csv", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()
