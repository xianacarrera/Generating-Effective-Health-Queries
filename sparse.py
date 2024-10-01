from beir import LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.sparse import SparseSearch
from bs4 import BeautifulSoup
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from huggingface_hub import login as hflogin
from beir.generation.models import QGenModel
from tqdm.autonotebook import trange
from pyserini.search import SimpleSearcher

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
                # 5th column (results are already sorted by score)
                score = float(rows[4])
                results[qid][docno] = score

                # print(f"Query {qid}, doc {i+1}: {docno} with score {score}")

                # Load the raw text of the document
                with open(f"{input_path}/query_{qid}/doc_{i+1}.txt", 'r') as f:
                    text = f.read()

                    # Parse the content of the document
                    soup = BeautifulSoup(text, "html.parser")
                    # Get the title of the document
                    title = soup.title.string if soup.title else " "

                    # Check if the type of title is NoneType
                    if type(title) == type(None):
                        title = " "

                    corpus[docno] = {"title": title, "text": text}

    return corpus, results


def doc_T5_query_generate_questions(corpus, queries):
    corpus_list = [corpus[doc_id] for doc_id in corpus]

    model_path = "castorini/docT5query-msmarco-passage"
    qgen_model = QGenModel(model_path, use_fast=False)

    gen_queries = {}
    num_return_sequences = 3   # Number of questions generated for each doc (try 3-5)
    batch_size = 80         # The bigger, the faster the generation

    for start_idx in trange(0, len(queries), batch_size, desc="question-generation"):
        size = len(corpus_list[start_idx: start_idx + batch_size])
        ques = qgen_model.generate(
            corpus=corpus_list[start_idx:start_idx + batch_size],
            ques_per_passage=num_return_sequences,
            max_length=64,
            top_p=0.95,
            top_k=10
        )

        assert len(ques) == size * num_return_sequences

        for idx in range(size):
            start_id = idx * num_return_sequences
            end_id = start_id + num_return_sequences
            gen_queries[queries[start_idx + idx]] = ques[start_id:end_id]

    return gen_queries

def search(qid, query, searcher):
    print(query)
    hits = searcher.search(query, 1000)

    results = []
    count = 1
    # The first thousand hits:
    for i in range(0, len(hits)):
        #json_doc = json.loads(hits[i].raw)
        docno = hits[i].docid
        if "uuid" in docno:
            docno = docno.split(":")[2].rstrip('>')

        dd = {"qid": qid, "Q0": "Q0", "docno": docno, "rank": count, "score": hits[i].score, "tag": "BM25"}
        results.append(dd)
        count +=1

    return results

def doc_T5_query_bm25_retrieval(corpus, gen_queries, index_path):
    searcher = SimpleSearcher(index_path)
    qids = list(gen_queries.keys())
    pass


def evaluate_sparse(
    queries: object,
    corpus: object,
    results: object,
    model_name: str = "SPARTA",
    access_token: str = ""
):
    if model_name == "SPARTA":
        model_path = "BeIR/sparta-msmarco-distilbert-base-v1"
        sparse_model = SparseSearch(models.SPARTA(model_path), batch_size=128)
        retriever = EvaluateRetrieval(sparse_model)
        results = retriever.retrieve(corpus, queries)
    elif model_name == "uniCOIL":
        model_path = "castorini/unicoil-d2q-msmarco-passage"
        sparse_model = SparseSearch(models.UniCOIL(model_path), batch_size=32)
        retriever = EvaluateRetrieval(sparse_model, score_function="dot")
        results = retriever.retrieve(corpus, queries, query_weights=True)
    elif model_name == "SPLADE":
        hflogin(access_token=access_token)

        model_path = "naver/splade_v2_distil"
        # Only works with agg="max"
        model = DRES(models.SPLADE(model_path), batch_size=128)
        retriever = EvaluateRetrieval(model, score_function="dot")
        results = retriever.retrieve(corpus, queries)
    elif model_name == "docT5query":
        gen_queries = doc_T5_query_generate_questions(corpus, queries)
        results = 

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

    dataset_name = config["META"]["DATASET_NAME"]
    # We take the part before the first hyphen in uppercase
    option = dataset_name.split("-")[0].upper()

    query_path = config[option]["QUERY_DESC_PATH"]
    qrels_path = config[option]["QRELS_PATH"]
    index_path = config[option]["INDEX_PATH"]

    model_name = config["SPARSE"]["MODEL_NAME"]
    abbrev = config["SPARSE"]["ABBREV"]

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

    # Evaluate using a dense model on top of the BM25 results
    ndcg, _map, recall, precision = evaluate_sparse(
        queries, corpus, results, model_name)

    end = time.time()
    time_taken = end - start

    # Name of the file containing the queries
    query_file = query_path.split("/")[-1]
    # Name of the file containing the qrels
    qrels_file = qrels_path.split("/")[-1]

    print(
        f"Logging results for dense_bm25+sparse_{model_name}")
    print(f"Time taken: {timedelta(seconds=time_taken)}")

    # Results: model, abbrev, dataset, query type, qrels type, time taken (formatted in minutes and seconds), BEIR metrics
    row = [f"dense_bm25+sparse_{model_name}", abbrev,
           dataset_name, query_file, qrels_file, str(
               timedelta(seconds=time_taken)),
           _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"],
           precision["P@10"], precision["P@100"], precision["P@1000"],
           recall["Recall@10"], recall["Recall@100"], recall["Recall@1000"],
           ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    with open("../beir_sparse_stats_output.csv", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()
