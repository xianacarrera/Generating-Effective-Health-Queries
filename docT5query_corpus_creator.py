from operator import indexOf
import os
import json
from pathlib import Path
from beir.generation.models import QGenModel
from tqdm.autonotebook import trange

import beir_helper as bh
import argparse
os.environ["JAVA_HOME"] = "/opt/citius/modules/software/Java/11.0.2"


def generate_queries_docT5query(
    corpus: object, 
    num_return_sequences: int = 3        # Number of questions generated for each doc (try 3-5)
):
    corpus_list = [corpus[doc_id] for doc_id in corpus]
    corpus_ids = list(corpus.keys())

    model_path = "castorini/doc2query-t5-base-msmarco"
    qgen_model = QGenModel(model_path, use_fast=False)

    gen_queries = {}
    batch_size = 80         # The bigger, the faster the generation

    for start_idx in trange(0, len(corpus_list), batch_size, desc="question-generation"):
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
            gen_queries[corpus_ids[start_idx + idx]] = ques[start_id:end_id]

    return gen_queries


def create_docT5query_corpus(corpus, gen_queries, write_dir, num_return_sequences):
    Path(f'{write_dir}').mkdir(parents=True, exist_ok=True)

    # Write the expanded corpus indexing it directly with Pyserini
    pyserini_jsonl = f"docT5query_{num_return_sequences}q.jsonl"
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=str, help="Index to load configuration from")
    args = parser.parse_args()

    program = "sparse"
    conf = bh.load_config(args.index, program)

    num_return_sequences = 3
    path_save_index = conf["output_path"] + "/docT5query_" + str(num_return_sequences) + "q"  

    # Load the custom data
    corpus, queries, _ = bh.load_custom_data(
        query_path=conf["query_path"],
        qrels_path=conf["qrels_path"],
    )

    # Load the pre-built corpus and BM25 results
    corpus, _ = bh.load_BM25_corpus(queries, conf["input_path"], conf["res_file"])
    print("Loaded BM25 corpus")

    if conf["clean"]:
        bh.clean_html(corpus, conf[program]["use_title"])
        path_save_index += "_clean"

    if conf[program]["use_title"] == "empty" or conf[program]["use_title"] == "repeat":
        path_save_index += f"_{conf[program]['use_title']}"

    # Expand the corpus with docT5query
    gen_queries = generate_queries_docT5query(corpus, num_return_sequences)
    print("Generated queries")

    # Write the expanded corpus as a jsonl file
    create_docT5query_corpus(corpus, gen_queries, path_save_index, num_return_sequences)
    print("Created docT5query corpus")


if __name__ == "__main__":
    main()
