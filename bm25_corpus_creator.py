from operator import indexOf
import os
import pandas as pd
from pathlib import Path
import argparse

os.environ["JAVA_HOME"] = "/opt/citius/modules/software/Java/11.0.2"
import xml.etree.ElementTree as ET
from pyserini.search import SimpleSearcher
import beir_helper as bh
import time
from datetime import timedelta

def create_corpus(qid, query, searcher, write_dir, use_rm3 = False, tag = "BM25"):
    # text = topic.find(field).text
    # number = topic.find("number").text
    # stance = topic.find("stance").text
    if use_rm3:
        searcher.set_rm3(10, 10, 0.5)

        # Check that rm3 is being used
        is_using_rm3 = searcher.is_using_rm3()
        print(f'Using RM3: {is_using_rm3}')
        if not is_using_rm3:
            # Cancel the program
            exit()

    print(query)
    hits = searcher.search(query, 1000)

    results = []
    count = 1
    # The first thousand hits:
    for i in range(0, len(hits)):
        docno = hits[i].docid
        if "uuid" in docno:
            docno = docno.split(":")[2].rstrip('>')

        dd = {"qid": qid, "Q0": "Q0", "docno": docno, "rank": count, "score": hits[i].score, "tag": tag}
        results.append(dd)

        # Write the raw text of the document to a file
        with open(f'{write_dir}/query_{qid}/doc_{count}.txt', 'w') as f:
            f.write(hits[i].raw)

        print(f'Wrote doc_{count}.txt')
        count +=1

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=str, help="Index to load configuration from")
    args = parser.parse_args()

    config = bh.load_config(args.index, "CORPUS_CREATOR")

    method = config["method"]
    use_rm3 = config["use_rm3"]

    output_path = config["output_path"]
    topics_path = config["topics_path"]
    dataset_name = config["dataset_name"]
    field = "question" if dataset_name == "C4-2022" else "description"

    start_time = time.time()

    searcher = SimpleSearcher(config["index_path"])
    print("Index loaded")
    print("=============")

    write_dir = f'{output_path}/index_{dataset_name}/field_{field}/method_{method}'

    all_results = []    
    with open(topics_path) as f:
        root = ET.parse(topics_path).getroot()
        for topic in root.findall('topic'):
            qid = topic.find("number").text
            query = topic.find(field).text
            Path(f'{write_dir}/query_{qid}').mkdir(parents=True, exist_ok=True)

            results = create_corpus(qid, query, searcher, write_dir, use_rm3, method)
            all_results.extend(results)

            df = pd.DataFrame(results)
            df.set_index('qid', inplace=True)
            df.to_csv(f'{write_dir}/query_{qid}/res_{dataset_name}_{method}_{field}.csv', sep=' ', header=False)
            print(f'Query {qid}: finished. Wrote res_{dataset_name}_{method}_{field}.csv')

    df_all = pd.DataFrame(all_results)
    df_all.set_index('qid', inplace=True)
    df_all.to_csv(f'{write_dir}/all_res_{dataset_name}_{method}_{field}.csv', sep=' ', header=False)
    print(f'Wrote combined results as all_res_{dataset_name}_{method}_{field}.csv')

    end_time = time.time()
    time_taken = end_time - start_time

    print(f"Time taken: {timedelta(seconds=time_taken)}")


if __name__ == "__main__":
    main()
