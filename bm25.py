import pandas as pd
import argparse 
import xml.etree.ElementTree as ET
import beir_helper as bh
from pyserini.search import SimpleSearcher
import time
from datetime import timedelta
from pathlib import Path

def search_query(qid, query, searcher, tag, use_rm3 = False):
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
        count +=1

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=str, help="Index to load configuration from")
    args = parser.parse_args()

    config = bh.load_config(args.index, "SPARSE")

    method = config["model_name"]
    use_rm3 = config["use_rm3"]
    output_path = config["output_path"]
    dataset_name = config["dataset_name"]
    topics_file = config["topics_path"]
    field = "question" if dataset_name == "C4-2022" else "description"

    start_time = time.time()
    
    searcher = SimpleSearcher(config["input_path"] + "/lucene_index")
    print("Index loaded")
    print("=============")

    with open(topics_file) as f:
        root = ET.parse(topics_file).getroot()
        results = []

        for topic in root.findall('topic'):
            qid = topic.find("number").text
            query = topic.find(field).text

            query_res = search_query(qid, query, searcher, method if not use_rm3 else f'{method}_rm3', use_rm3)
            results.extend(query_res)
            print(f'Finished query {qid}')

        df = pd.DataFrame(results)
        df.set_index('qid', inplace=True)

        # Take method and get everything that comes before the last underscore
        method_2 = method.split("_")[:-1]
        method_2 = "_".join(method_2)
        filepath = f'{output_path}/results/{method_2}{"_rm3" if use_rm3 else ""}'
        Path(filepath).mkdir(parents=True, exist_ok=True)
        df.to_csv(f'{filepath}/{method}{"_rm3" if use_rm3 else ""}.csv', sep=' ', header=False)
        print(f'Wrote res_{dataset_name}_{method}.csv')


    end_time = time.time()
    time_taken = end_time - start_time

    print(f"Time taken: {timedelta(seconds=time_taken)}")


if __name__ == "__main__":
    main()
