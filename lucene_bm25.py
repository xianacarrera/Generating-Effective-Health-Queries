from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

import logging
import requests
import csv
import json
import configparser

docker_beir_pyserini = "http://127.0.0.1:8000"

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


def evaluate_bm25(
    queries: object,
    qrels: object
):
    retriever = EvaluateRetrieval()
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]

    print("First 5 queries: ", query_texts[:5])
    print("First 5 qids: ", qids[:5])
    print("Number of queries: ", len(query_texts))
    print("Number of qids: ", len(qids))

    payload = {"queries": query_texts,
               "qids": qids, "k": max(retriever.k_values)}
    
    # Retrieve pyserini results (format of results is identical to qrels)
    results = json.loads(requests.post(
        docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]
    
    # print("Results: ", results)
    
    # Remove the query_id from the results if it is present
    for query_id in results:
        if query_id in results[query_id]:
            results[query_id].pop(query_id, None)

    # Evaluate the results
    logging.info("Retriever evaluation for k in: {}".format(
        retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    
    config = configparser.ConfigParser()
    config.read("config.ini")

    #query_path = config["C4"]["QUERY_QUESTION_PATH"]
    #qrels_path = config["C4"]["QRELS_PATH"]
    query_path = config["MISINFO"]["QUERY_DESC_PATH"]
    qrels_path = config["MISINFO"]["QRELS_PATH"]

    # Load the custom data
    corpus, queries, qrels = load_custom_data(            
        query_path=query_path,                
        qrels_path=qrels_path  
    )

    # Evaluate the BM25 model
    ndcg, _map, recall, precision = evaluate_bm25(queries, qrels)

    row = ["bm25", _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"], precision["P@10"], precision["P@100"], precision["P@1000"], recall["Recall@10"],
           recall["Recall@100"], recall["Recall@1000"], ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    with open("../bm25_output.csv", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()
