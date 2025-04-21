from beir.datasets.data_loader import GenericDataLoader
from bs4 import BeautifulSoup

import csv
import configparser
from datetime import timedelta
from pathlib import Path


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

    print("Custom data loaded")
    return corpus, queries, qrels



def load_BM25_corpus(
    queries: object,
    input_path: str,
    res_file: str = "res"
):
    corpus = {}
    results = {}
    for qid in list(queries):
        results[qid] = {}
        with open(f"{input_path}/query_{qid}/{res_file}", mode='r') as infile:
            # Load csv as a dictionary
            reader = csv.reader(infile, delimiter=' ')

            for i, rows in enumerate(reader):
                docno = rows[2]              # 3rd column
                # 5th column (results are already sorted by score)
                score = float(rows[4])
                results[qid][docno] = score

                # Load the raw text of the document
                with open(f"{input_path}/query_{qid}/doc_{i+1}.txt", 'r') as f:
                    text = f.read()
                    corpus[docno] = {"text": text}

    print("Corpus loaded")
    return corpus, results



def clean_html_page(
    text: str,
):
    # Parse the content of the document
    soup = BeautifulSoup(text, "html.parser")
    # Get the title of the document
    title = soup.title.string if soup.title else " "

    # Check if the type of title is NoneType
    if type(title) == type(None):
        title = " "

    # Remove all script and style elements
    for script_or_style in soup(["script", "style", "meta",
                                    "head", "title", "header", 
                                    "footer", "nav", "noscript", "link"]):
        script_or_style.decompose()

    # Get the text of the document
    text = soup.get_text(separator="\n")

    # Concatenate the title and text
    text = str(title + " " + text)

    # Encode the text to utf-8 and ignore any errors
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = text.strip()

    return text, title
    


def clean_html(
    corpus: object,
    use_title: str = "no"
):
    for docno in corpus:
        text = corpus[docno]["text"]

        clean_text, title = clean_html_page(text)

        if use_title == "empty":
            corpus[docno] = {"title": " ", "text": clean_text}
        elif use_title == "repeat":
            corpus[docno] = {"title": title, "text": clean_text}
        else:
            corpus[docno] = {"text": clean_text}

    print("HTML cleaned")


def load_config(
    file_path: str = "config.ini",
    pr: str = "reranker"
):
    program = pr.upper()

    print(f"Loading config from {file_path}")

    parser = configparser.ConfigParser()
    parser.read(file_path)

    conf = {}

    if program == "RERANKER":
        conf["model_name"] = parser[program]["MODEL_NAME"]
        conf["abbrev"] = parser[program]["MODEL_NAME"]
        conf["model_training"] = parser["RERANKER"]["TRAINING"]
        conf["abbrev"] += f"_{parser['RERANKER']['TRAINING']}"
    elif program == "DENSE":
        conf["model_name"] = parser[program]["MODEL_NAME"]
        conf["abbrev"] = parser[program]["MODEL_NAME"]
        conf["score_function"] = parser["DENSE"]["SCORE_FUNCTION"]
    elif program == "SPARSE":
        conf["model_name"] = parser[program]["MODEL_NAME"]
        conf["abbrev"] = parser[program]["MODEL_NAME"]
        conf["use_title"] = parser["SPARSE"]["USE_TITLE"]
        conf["use_rm3"] = True if parser["SPARSE"]["USE_RM3"].upper() == "TRUE" else False
        if conf["use_title"] not in ["empty", "repeat", "none"]:
            print("Use_title not recognized")
            exit()
        conf["splade_training"] = parser["SPARSE"]["SPLADE_TRAINING"]
        conf["abbrev"] += f"_{parser['SPARSE']['SPLADE_TRAINING']}"
    elif program == "CORPUS_CREATOR":
        conf["method"] = parser["CORPUS_CREATOR"]["METHOD"]
        conf["index_path"] = parser["CORPUS_CREATOR"]["INDEX_PATH"]
        conf["use_rm3"] = True if parser["CORPUS_CREATOR"]["USE_RM3"].upper() == "TRUE" else False
    else:
        print("Type of program not recognized")
        exit()

    conf["dataset_name"] = parser["META"]["DATASET_NAME"]
    conf["output_path"] = parser["META"]["OUTPUT_PATH"]
    conf["clean"] = True if parser["META"]["CLEAN_HTML"].upper() == "TRUE" else False

    conf["query_path"] = parser["INDEX"]["QUERY_PATH"]
    conf["qrels_path"] = parser["INDEX"]["QRELS_PATH"]
    conf["input_path"] = parser["INDEX"]["INPUT_PATH"]
    conf["topics_path"] = parser["INDEX"]["TOPICS_PATH"]
    conf["res_file"] = parser["INDEX"]["RES_FILE"]

    print("Config loaded")
    return conf



def save_final_ranking(
    results: object,
    output_path: str,
    file_name: str,
    abbrev: str
):
    # Save with format
    # qid Q0 docno rank score model

    # Create path if it doesn't exist
    Path(f"{output_path}/results").mkdir(parents=True, exist_ok=True)

    with open(f"{output_path}/results/{file_name}.txt", 'w') as f:
        for qid in results:
            for rank, docno in enumerate(results[qid]):
                f.write(f"{qid} Q0 {docno} {rank+1} {results[qid][docno]} {abbrev}\n")
        f.close()

    print("Final ranking saved")


def log_results(
    conf: object,
    full_name: str,
    time_taken: float,
    _map: object,
    precision: object,
    recall: object,
    ndcg: object,
    results: object
):
    # Name of the file containing the queries
    query_file = conf["query_path"].split("/")[-1]
    # Name of the file containing the qrels
    qrels_file = conf["qrels_path"].split("/")[-1]

    row = [full_name, conf["abbrev"], conf["dataset_name"], query_file, qrels_file, 
           str(timedelta(seconds=time_taken)),
           _map["MAP@10"], _map["MAP@100"], _map["MAP@1000"],
           precision["P@10"], precision["P@100"], precision["P@1000"],
           recall["Recall@10"], recall["Recall@100"], recall["Recall@1000"],
           ndcg["NDCG@10"], ndcg["NDCG@100"], ndcg["NDCG@1000"]]

    with open("../beir_raw_stats_output.csv", 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()

    print("Stats logged")

    # Save the final ranking
    full_name = full_name.replace("/", "-")
    full_name = full_name.replace("+", "-")
    save_final_ranking(results, conf["output_path"], full_name, conf["abbrev"])


