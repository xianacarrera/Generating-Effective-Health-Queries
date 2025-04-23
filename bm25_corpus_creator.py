from datetime import timedelta
import time
import beir_helper as bh
from pyserini.search import SimpleSearcher
import xml.etree.ElementTree as ET
from operator import indexOf
import os
import pandas as pd
from pathlib import Path
import argparse

os.environ["JAVA_HOME"] = "/opt/citius/modules/software/Java/11.0.2"


def create_corpus(qid, query, searcher, write_dir, tag="BM25"):
    print(query)
    hits = searcher.search(query, 1000)

    results = []
    count = 1
    # The first thousand hits:
    for i in range(0, len(hits)):
        docno = hits[i].docid
        if "uuid" in docno:
            docno = docno.split(":")[2].rstrip('>')

        dd = {"qid": qid, "Q0": "Q0", "docno": docno,
              "rank": count, "score": hits[i].score, "tag": tag}
        results.append(dd)

        # Write the raw text of the document to a file
        with open(f'{write_dir}/query_{qid}/doc_{count}.txt', 'w') as f:
            f.write(hits[i].raw)

        print(f'Wrote doc_{count}.txt')
        count += 1

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=str,
                        help="Index to load configuration from")
    args = parser.parse_args()

    config = bh.load_config(args.index, "CORPUS_CREATOR")

    method = config["method"]     # e.g., "orig_RC1"

    output_path = config["output_path"]
    topics_path = config["topics_path"]
    # misinfo-2020, C4-2021, C4-2022, CLEF
    dataset_name = config["dataset_name"]
    llm_model = config["llm_model"]          # gpt, llama

    # We will read the variants from the xml file, which are stored in the fields "description", "question" or "title"
    if dataset_name == "C4-2022":
        field = "question"
    elif dataset_name == "CLEF":
        field = "title"
    else:
        field = "description"

    # To generate a top 1000 filter using BM25 and the original queries, use instead:
    # if dataset_name == "C4-2021" or dataset_name=="C4-2022":
    #     field = "query"
    # else:
    #     field = "originaltitle"

    start_time = time.time()

    searcher = SimpleSearcher(config["index_path"])
    print("Index loaded")
    print("=============")

    write_dir = f'{output_path}/index_{dataset_name}/method_{method}'

    all_results = []
    with open(topics_path) as f:
        root = ET.parse(topics_path).getroot()
        topic_tag = 'query' if dataset_name == "CLEF" else 'topic'
        for topic in root.findall(topic_tag):
            qid_tag = 'id' if dataset_name == "CLEF" else "number"
            qid = topic.find(qid_tag).text
            query = topic.find(field).text
            Path(f'{write_dir}/query_{qid}').mkdir(parents=True, exist_ok=True)

            results = create_corpus(qid, query, searcher, write_dir, method)
            all_results.extend(results)

            df = pd.DataFrame(results)
            df.set_index('qid', inplace=True)
            df.to_csv(
                f'{write_dir}/query_{qid}/{dataset_name}_{method}.csv', sep=' ', header=False)
            print(
                f'Query {qid}: finished. Wrote {dataset_name}_{method}.csv')

    df_all = pd.DataFrame(all_results)
    df_all.set_index('qid', inplace=True)

    if llm_model != "":
        df_all.to_csv(
            f'{write_dir}/{llm_model}_{dataset_name}_{method}.csv', sep=' ', header=False)
        print(f'Wrote combined results as {llm_model}_{dataset_name}_{method}.csv')
    else:
        df_all.to_csv(
            f'{write_dir}/{dataset_name}_{method}.csv', sep=' ', header=False)
        print(f'Wrote combined results as {dataset_name}_{method}.csv')

    end_time = time.time()
    time_taken = end_time - start_time

    print(f"Time taken: {timedelta(seconds=time_taken)}")


if __name__ == "__main__":
    main()
