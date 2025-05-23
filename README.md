# Generating Effective Health Queries

This is the public code release for the **ACM SIGIR 2025** short paper **"Generating Effective Health-Related Queries for Promoting Reliable Search Results"**, by **Xiana Carrera, Marcos Fernández-Pichel and David E. Losada**.

Our project introduces a novel method that leverages Large Language Models (LLMs) to generate alternative formulations of health-related search queries, with the aim of reducing the prevalence of misinformation in search results. Specifically, we generate synthetic narratives that guide the creation of these alternative queries, which are designed to retrieve more helpful and fewer hamrful documents compared to the original user queries.

This is done in a two-stage process shown in the following figure:

![](figs/architecture.jpg)

## Installation

First, clone this repository:

```bash
git clone https://github.com/xianacarrera/Generating-Effective-Health-Queries
```

Create a new environment from the `environment.yml` file. By default, its name will be `healquery`:

```bash
conda env create -f environment.yml
# To change the environment name:
# conda env create -f environment.yml -n new-env-name
```

Activate the environment:
```bash
conda activate healquery
# Alternatively:
# conda activate new-env-name
```

### LLMs

The `llm_connector.py` script can be run using one of two LLMs: GPT-4 (more precisely, GPT-4o) or LLaMA3 (llama3.1:8b-instruct-q8_0). 

To use GPT-4, a valid OpenAI API key must be provided in the configuration file.

To use LLaMA3, a local installation of Ollama is required. On Linux systems, Ollama may be installed by running the following command:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Installation instructions for other operating systems are available on the [official download page](https://ollama.com/download).

Once installed, the [llama3.1:8b-instruct-q8_0](https://ollama.com/library/llama3:8b-instruct-q8_0) model can be downloaded and tested from the command line using:

```bash
ollama pull llama3:8b-instruct-q8_0
ollama run llama3:8b-instruct-q8_0
```

## Use
Once the installation has been successfully completed, run the `llm_connector.py` script to try the narrative and query generation processes.

Additionally, we provide the following scripts, which rely on `beir_helper.py`:
* `baseline_bm25.py`, which computes the BM25 score for all queries in a given corpus.
* `bm25_corpus_creator.py`, which computes the BM25 score for all queries in a given corpus and creates a  new directory containing the top 1000 documents for each query.
* `docT5query_corpus_creator.py`, which is analogous to `bm25_corpus_creator.py`, but uses docT5query as the scoring method instead of BM25.
* `dense.py`, which is designed to retrieve the filtered corpus generated by `bm25_corpus_creator.py` and execute one of the following dense retrieval models on it, using the BEIR library: Tas-B, ANCE, DPR.
* `sparse.py`, which is designed to retrieve the filtered corpus generated by `bm25_corpus_creator.py` and execute one of the following models on it, using the BEIR library: SPARTA, uniCOIL, SPLADE.
* `reranker.py`, which is designed to retrieve the filtered corpus generated by `bm25_corpus_creator.py` and apply one of the reranking models available through the BEIR library, specifically those implemented via the `CrossEncoder` o `MonoT5` classes. In our experiments, we tried electra-base, MiniLM-L-4-v2, MiniLM-L-6-v2, MiniLM-L-12-v2, TinyBERT-L-2-v2, TinyBERT-L-4, TinyBERT-L-6 and MonoT5 (base, base-med and large).


## Generation resources

With the non-deterministic nature of LLMs in mind, for transparency and reproducibility we include all key resources used to produce the results presented in the paper:

* **Prompts**. The file `prompts.txt` contains the exact prompts used to generate the narratives and alternative queries with both GPT-4 and LLaMA3, along with a description of the different variations explored during experimentation. The prompts have already been programmatically integrated into the `llm_connector.py` module, which contains the code relevant to those steps of our pipeline where an LLM is used.

* **Generated narratives**. Narrative outputs generated by the LLMs are included in the `generated_narratives` directory. They are organized by dataset and LLM model (gpt-4 or LLaMA3). These narratives serve as contextual expansions of the user queries with the goal of influencing the generation of alternative queries to reduce misinformation with query-dependent context.

* **Alternative queries**. These are the generated variants of the user queries. They are included in the `alternative_queries` directory, and organized by dataset, model and configuration. The configuration has three parameters: 
    1. Role (R/no R) - presence/absence of system role
    2. Narrative (N/no N) - presence/absence of a narrative
    3. Chain of thought (C0/C1/C2) - level of reasoning requested, which can be none (0), basic (1) or complete (2).
    
    If a narrative is used, it may be either the original narrative provided with the dataset topics, which are available for all datasets but *CLEF*, or the generated narratives described above. By default, the generated narratives are used. When original narratives or no narratives are selected, the name is prefixed with `orig`.

* **Search rankings**. We also provide the ranked lists of search results obtained using BM25 or MiniLM-L-12-v2. Compatibility scores are then computed from these rankings by using the `compatibility.py` program, which is a slightly modified version of the original implementation from [https://github.com/claclark/compatibility](https://github.com/claclark/compatibility). Our modifications retain the scoring algorithm and only adapt the script to handle non-sequential query IDs in our datasets and to manage cases where queries lack positive/negative relevance judgments (qrels).



## Contact
For any questions or issues, feel free to reach out at [xiana.carrera@rai.usc.es](mailto:xiana.carrera@rai.usc.es).


## Acknowledgements
This project was funded by MICIU/AEI/10.13039/501100011033 (PID2022-137061OB-C22, supported by ERDF) and Xunta de Galicia-Consellería de Cultura, Educación, Formación Profesional e Universidades (ED431G 2023/04, ED431C 2022/19, supported by ERDF).