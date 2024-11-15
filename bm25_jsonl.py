from operator import indexOf
import os
import pandas as pd
from pathlib import Path
import time
import json
import argparse

os.environ["JAVA_HOME"] = "/opt/citius/modules/software/Java/11.0.2"
import xml.etree.ElementTree as ET
from pyserini.search import SimpleSearcher
import beir_helper as bh

def create_jsonl(corpus, write_dir):
    Path(f'{write_dir}').mkdir(parents=True, exist_ok=True)

    # Write the expanded corpus indexing it directly with Pyserini
    pyserini_jsonl = f"bm25.jsonl"
    with open(os.path.join(write_dir, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id, doc in corpus.items():
            text = doc["text"]
            dd = {"id": doc_id,
                    "contents": text}
            json.dump(dd, fOut)
            fOut.write('\n')

            print(f'Wrote doc_{doc_id}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=str, help="Index to load configuration from")
    args = parser.parse_args()

    program = "sparse"
    conf = bh.load_config(args.index, program)

    # Load the custom data
    corpus, queries, _ = bh.load_custom_data(
        query_path=conf["query_path"],
        qrels_path=conf["qrels_path"],
    )

    # Load the pre-built corpus and BM25 results
    corpus, _ = bh.load_BM25_corpus(queries, conf["input_path"], conf["res_file"])
    print("Loaded BM25 corpus")

    create_jsonl(corpus, conf["input_path"] + "/all_docs")


if __name__ == "__main__":
    main()
