from operator import indexOf
import os
import json
from pathlib import Path
from bs4 import BeautifulSoup
from beir.generation.models import QGenModel
from tqdm.autonotebook import trange
import configparser
from beir.datasets.data_loader import GenericDataLoader

import csv
os.environ["JAVA_HOME"] = "/opt/citius/modules/software/Java/11.0.2"


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


def create_docT5query_corpus(corpus, gen_queries, write_dir):
    Path(f'{write_dir}').mkdir(parents=True, exist_ok=True)

    # Write the expanded corpus indexing it directly with Pyserini
    pyserini_jsonl = "docT5query.jsonl"
    with open(os.path.join(write_dir, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id, doc in corpus.items():
            title = doc["title"]
            text = doc["text"]
            query_text = " ".join(gen_queries[doc_id])
            dd = {"id": doc_id,
                    "title": title,
                    "contents": text,
                    "queries": query_text}
            json.dump(dd, fOut)
            fOut.write('\n')

            print(f'Wrote doc_{doc_id}')



def main(field: str = 'description'):

    config = configparser.ConfigParser()
    config.read("config.ini")

    method = "docT5query"
    out_dir = config["META"]["OUT_DIR"]
    res_file = config["META"]["RES_FILE"]

    dataset_name = config["META"]["DATASET_NAME"]
    # We take the part before the first hyphen in uppercase
    option = dataset_name.split("-")[0].upper()

    query_path = config[option]["QUERY_DESC_PATH"]
    qrels_path = config[option]["QRELS_PATH"]

    input_path = config["META"]["INPUT_PATH"]
    res_file = config["META"]["RES_FILE"]


    # Load the custom data
    corpus, queries, _ = load_custom_data(
        query_path=query_path,
        qrels_path=qrels_path
    )

    write_dir = f'{out_dir}/index_{dataset_name}/field_{field}/method_{method}'

    # Load the pre-built corpus and BM25 results
    corpus, _ = load_BM25_corpus(queries, input_path, res_file)
    print("Loaded BM25 corpus")

    # Expand the corpus with docT5query
    gen_queries = generate_queries_docT5query(corpus, queries)
    print("Generated queries")

    # Write the expanded corpus as a jsonl file
    create_docT5query_corpus(corpus, gen_queries, write_dir)
    print("Created docT5query corpus")


if __name__ == "__main__":
    main()
