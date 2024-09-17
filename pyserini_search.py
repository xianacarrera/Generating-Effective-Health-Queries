from pyserini.search import SimpleSearcher

if __name__ == "__main__":
    searcher = SimpleSearcher('/mnt/hpc-irgroup/indexes/misinfo-2020')
    hits = searcher.search('covid')

    print(f'Total number of hits: {len(hits)}')



