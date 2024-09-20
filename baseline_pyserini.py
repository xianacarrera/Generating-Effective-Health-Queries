from operator import indexOf
import os
import pandas as pd

os.environ["JAVA_HOME"] = "/opt/citius/modules/software/Java/11.0.2"
import xml.etree.ElementTree as ET
from pyserini.search import SimpleSearcher

def search(qid, query, searcher):
    # text = topic.find(field).text
    # number = topic.find("number").text
    # stance = topic.find("stance").text
    print(query)
    hits = searcher.search(query, 1000)

    results = []
    count = 1
    #  the first thousand hits:
    for i in range(0, len(hits)):
        #json_doc = json.loads(hits[i].raw)
        docno = hits[i].docid
        if "uuid" in docno:
            docno = docno.split(":")[2].rstrip('>')

        dd = {"qid": qid, "Q0": "Q0", "docno": docno, "rank": count, "score": hits[i].score, "tag": "BM25"}
        results.append(dd)
        count +=1

    return results


def main(field: str = 'description',
         index: str = '/path-to-index/misinfo-2020',
         topics_file: str = '/path-to-topics/misinfo-2020-topics.xml',
         out_dir: str = '/path-to-out/baseline_out'):
    searcher = SimpleSearcher(index)
    print("Index loaded")
    print("=============")

    results = []
    with open(topics_file) as f:
        root = ET.parse(topics_file).getroot()
        for topic in root.findall('topic'):
            qid = topic.find("number").text
            query = topic.find(field).text
            results.extend(search(qid, query, searcher))

    df = pd.DataFrame(results)
    df.set_index("qid", inplace=True)
    index_name = index.split("/")[-1]
    method = "bm25"
    df.to_csv(f'{out_dir}/res_baseline_{index_name}_{method}_{field}.csv', sep=' ', header=False)



if __name__ == "__main__":
    main()
