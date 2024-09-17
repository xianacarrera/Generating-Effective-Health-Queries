from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pyserini.search import SimpleSearcher
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

import logging
import csv
import configparser


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

def correct_format(
        results: object
):
    format_flag = -1
    # Remove the query_id from the results if it is present
    for query_id in results:
        # We check if any of the doc ids contain the string "uuid". If so, we adjust the format of all of them
        # If not, we always keep the format as it is
        if format_flag >=0: continue            # Check has already been done and no adjustment is needed

        if format_flag == 1 or any("uuid" in doc_id for doc_id in results[query_id]):
            # Adjust the format of the doc ids
            results[query_id] = {doc_id.split(":")[2].rstrip('>'): results[query_id][doc_id] 
                                for doc_id in results[query_id]}
            format_flag = 1
        else:   
            format_flag = 0     # Set the flag to 0 to indicate that no adjustment is needed 

    return results

def recover_docs(
        results: object,
        index_path: str,
        top_k: int = 100
):
    corpus = {}
    index_reader = SimpleSearcher(index_path)

    for query_id in results:
        if len(results[query_id]) > top_k:
            # We sort the results by score and keep the top k
            results[query_id] = {doc_id: results[query_id][doc_id] for doc_id in sorted(results[query_id], key=lambda x: results[query_id][x], reverse=True)[:top_k]}

        for doc_id in results[query_id]:
            doc = index_reader.doc(doc_id)
            corpus[doc_id] = {"title": doc.get("title"), "text": doc.get("text")}
        
    return corpus


def evaluate_bm25(
    queries: object,
    qrels: object,
    index_path: str,
    top_k : int = 100
):
    retriever = EvaluateRetrieval()
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]

    print("First 5 queries: ", query_texts[:5])
    print("First 5 qids: ", qids[:5])
    print("Number of queries: ", len(query_texts))
    print("Number of qids: ", len(qids))

    # Initialize Pyserini searcher
    searcher = SimpleSearcher(index_path)

    results = {}
    for qid, query in zip(qids, query_texts):
        hits = searcher.search(query, k=top_k)
        results[qid] = {hit.docid: hit.score for hit in hits}

    results = correct_format(results)

    corpus = recover_docs(results, index_path, top_k)

    return results, corpus, retriever

def rerank(
    queries: object,
    corpus: object,
    results: object,
    retriever: object,
    top_k : int = 100,
    ce_model : str = "cross-encoder/ms-marco-electra-base"
):
    # Initialize Cross-Encoder 
    cross_encoder = CrossEncoder(ce_model)
    reranker = Rerank(cross_encoder, batch_size=128)

    # Re-rank the top 100 results using the reranker
    reranked_results = reranker.rerank(corpus, queries, results, top_k=top_k)

    # Evaluate the results
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, reranked_results, retriever.k_values)
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
    ce_model = config["CROSS_ENCODER"]["CE_MODEL"]

    # Load the custom data
    corpus, queries, qrels = load_custom_data(            
        query_path=query_path,                
        qrels_path=qrels_path  
    )

    # Evaluate the BM25 model
    temp_results, corpus, retriever = evaluate_bm25(queries, qrels, index_path)
    ndcg, _map, recall, precision = rerank(queries, corpus, temp_results, retriever, ce_model=ce_model)

    row = ["rerank_bm25+cross_encoder", dataset_name, _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"], precision["P@10"], precision["P@100"], precision["P@1000"], recall["Recall@10"],
           recall["Recall@100"], recall["Recall@1000"], ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    with open("../beir_stats_output.csv", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()

