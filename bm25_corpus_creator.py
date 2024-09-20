from operator import indexOf
import os
import pandas as pd
from pathlib import Path

os.environ["JAVA_HOME"] = "/opt/citius/modules/software/Java/11.0.2"
import xml.etree.ElementTree as ET
from pyserini.search import SimpleSearcher

def create_corpus(qid, query, searcher, write_dir):
    # text = topic.find(field).text
    # number = topic.find("number").text
    # stance = topic.find("stance").text
    print(query)
    hits = searcher.search(query, 1000)

    results = []
    count = 1
    # The first thousand hits:
    for i in range(0, len(hits)):
        docno = hits[i].docid
        if "uuid" in docno:
            docno = docno.split(":")[2].rstrip('>')

        dd = {"qid": qid, "Q0": "Q0", "docno": docno, "rank": count, "score": hits[i].score, "tag": "BM25"}
        results.append(dd)

        # Write the raw text of the document to a file
        with open(f'{write_dir}/query_{qid}/doc_{count}.txt', 'w') as f:
            f.write(hits[i].raw)

        print(f'Wrote doc_{count}.txt')
        count +=1

    return results


def main(field: str = 'description',
         index: str = '/mnt/beegfs/groups/irgroup/indexes/misinfo-2020',
         topics_file: str = '/mnt/beegfs/home/xiana.carrera/BEIR/TREC_2020_BEIR/original-misinfo-resources-2020/topics/misinfo-2020-topics.xml',
         out_dir: str = '/mnt/beegfs/home/xiana.carrera/BEIR'):
    searcher = SimpleSearcher(index)
    print("Index loaded")
    print("=============")
    
    method = "bm25"
    index_name = index.split("/")[-1]
    write_dir = f'{out_dir}/index_{index_name}/field_{field}/method_{method}'

    with open(topics_file) as f:
        root = ET.parse(topics_file).getroot()
        for topic in root.findall('topic'):
            qid = topic.find("number").text
            query = topic.find(field).text
            Path(f'{write_dir}/query_{qid}').mkdir(parents=True, exist_ok=True)

            results = create_corpus(qid, query, searcher, write_dir)

            df = pd.DataFrame(results)
            df.set_index('qid', inplace=True)
            df.to_csv(f'{write_dir}/query_{qid}/res_{index_name}_{method}_{field}.csv', sep=' ', header=False)
            print(f'Wrote res_{index_name}_{method}_{field}.csv')


if __name__ == "__main__":
    main()
