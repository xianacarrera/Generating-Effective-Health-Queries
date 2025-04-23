from beir import LoggingHandler
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.sparse import SparseSearch
from bs4 import BeautifulSoup
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.generation.models import QGenModel
from tqdm.autonotebook import trange

import logging
import csv
import time
import beir_helper as bh
from datetime import timedelta
import argparse


def load_sparse_BM25_corpus(
    queries: object,
    input_path: str,
    res_file: str = "res",
    use_title: bool = True
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
                    
                    if use_title:
                        # Parse the content of the document
                        soup = BeautifulSoup(text, "html.parser")
                        # Get the title of the document
                        title = soup.title.string if soup.title else " "

                        # Check if the type of title is NoneType
                        if type(title) == type(None):
                            title = " "
                    else:
                        title = " "

                    corpus[docno] = {"title": title, "text": text}

    print(f"Use title?: {use_title}")
    return corpus, results


def generate_queries_docT5query(corpus, queries):
    corpus_list = [corpus[doc_id] for doc_id in corpus]

    model_path = "castorini/docT5query-msmarco-passage"
    qgen_model = QGenModel(model_path, use_fast=False)

    gen_queries = {}
    # Number of questions generated for each doc (try 3-5)
    num_return_sequences = 3
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


def expand_corpus(corpus, gen_queries):
    expanded_corpus = {}
    for doc_id in corpus:
        expanded_queries = " ".join(gen_queries[doc_id])
        expanded_corpus[doc_id] = {
            "title": corpus[doc_id]["title"],
            "text": corpus[doc_id]["text"] + " " + expanded_queries
        }

    return expanded_corpus


def evaluate_sparse(
    queries: object,
    corpus: object,
    results: object,
    model_name: str = "SPARTA",
    splade_path: str = "naver/splade_v2_distil"
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
        model = DRES(models.SPLADE(splade_path), batch_size=128)
        retriever = EvaluateRetrieval(model, score_function="dot")
        results = retriever.retrieve(corpus, queries)


    logging.info("Retriever evaluation for k in: {}".format(
        retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values)

    return ndcg, _map, recall, precision, results


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=str, help="Index to load configuration from")
    args = parser.parse_args()

    conf = bh.load_config(args.index, "SPARSE")
    model_name = conf["model_name"]

    start = time.time()

    # Load the custom data
    corpus, queries, qrels = bh.load_custom_data(
        query_path=conf["query_path"],
        qrels_path=conf["qrels_path"],
    )

    print(f"Use_title before loading: {conf['use_title']}")
    # Load the pre-built corpus and BM25 results
    corpus, results = bh.load_BM25_corpus(queries, conf["input_path"], conf["res_file"])

    
    if conf["clean"]:
        bh.clean_html(corpus, conf['use_title'])

    # Evaluate using a dense model on top of the BM25 results
    ndcg, _map, recall, precision, results = evaluate_sparse(
        queries, corpus, results, model_name, conf["splade_training"])

    end = time.time()
    time_taken = end - start

    #full_name = f"sparse_bm25+{model_name}"
    full_name = f"sparse_bm25+{model_name}_title"
    
    if model_name == "SPLADE":
        full_name += f"_{conf['splade_training']}"


    print(f"Use_title after loading: {conf['use_title']}")
    if conf["use_title"] == "empty" or conf["use_title"] == "repeat":
        full_name += f"_{conf['use_title']}"
        conf["abbrev"] += f"-{conf['use_title']}"

    print(f"Logging results for {full_name}")
    print(f"Time taken: {timedelta(seconds=time_taken)}")

    # Results: full_model, abbrev, dataset, query type, qrels type, time taken 
    # (formatted in minutes and seconds), BEIR metrics
    bh.log_results(
        conf, full_name, time_taken, _map, precision, recall, ndcg, results
    )
