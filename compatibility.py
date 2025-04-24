#!/usr/bin/env python3

"""Maximum similarity to an ideal ranking from files in TREC format

This code implements an evaluation metric called "compatibility", which
was developed and explored over three papers.  Start with the first
(i.e. most recent).

1) Charles L. A. Clarke, Alexandra Vtyurina, and Mark D. Smucker. 2020.
   Assessing top-k preferences
   Under review. See: https://arxiv.org/abs/2007.11682

2) Charles L. A. Clarke, Mark D. Smucker, and Alexandra Vtyurina. 2020.
   Offline evaluation by maximum similarity to an ideal ranking.
   29th ACM Conference on Information and Knowledge Management.

3) Charles L. A. Clarke, Alexandra Vtyurina, and Mark D. Smucker. 2020.
   Offline evaluation without gain.
   ACM SIGIR International Conference on the Theory of Information Retrieval.

"""

import argparse
import sys
import pandas as pd

# Default persistence of 0.95, which is roughly equivalent to NSCG@20.
# Can be changed on the command line.
P = 0.95

# An additional normalization step was introduced in paper #1 (above)
# to handle short, truncated ideal results.  I don't recommend changing
# it, so it's not an command line argument, but setting it to False
# is required to exactly reproduce the numbers in papers #2 and #3,
# as well as the un-normalized numbers in paper #1.
NORMALIZE = True

# Depth for RBO computation. There's probably no need to ever play with this.
DEPTH = 1000

TOPICS_2020 = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50)
TOPICS_2021 = (101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150)
TOPICS_2022 = (151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200)
TOPICS_CLEF = (101001,101002,101003,101004,101005,101006,102001,102002,102003,102004,102005,102006,103001,103002,103003,103004,103005,103006,104001,104002,104003,104004,104005,104006,105001,105002,105003,105004,105005,105006,106001,106002,106003,106004,106005,106006,107001,107002,107003,107004,107005,107006,108001,108002,108003,108004,108005,108006,109001,109002,109003,109004,109005,109006,110001,110002,110003,110004,110005,110006,111001,111002,111003,111004,111005,111006,112001,112002,112003,112004,112005,112006,113001,113002,113003,113004,113005,113006,114001,114002,114003,114004,114005,114006,115001,115002,115003,115004,115005,115006,116001,116002,116003,116004,116005,116006,117001,117002,117003,117004,117005,117006,118001,118002,118003,118004,118005,118006,119001,119002,119003,119004,119005,119006,120001,120002,120003,120004,120005,120006,121001,121002,121003,121004,121005,121006,122001,122002,122003,122004,122005,122006,123001,123002,123003,123004,123005,123006,124001,124002,124003,124004,124005,124006,125001,125002,125003,125004,125005,125006,126001,126002,126003,126004,126005,126006,127001,127002,127003,127004,127005,127006,128001,128002,128003,128004,128005,128006,129001,129002,129003,129004,129005,129006,130001,130002,130003,130004,130005,130006,131001,131002,131003,131004,131005,131006,132001,132002,132003,132004,132005,132006,133001,133002,133003,133004,133005,133006,134001,134002,134003,134004,134005,134006,135001,135002,135003,135004,135005,135006,136001,136002,136003,136004,136005,136006,137001,137002,137003,137004,137005,137006,138001,138002,138003,138004,138005,138006,139001,139002,139003,139004,139005,139006,140001,140002,140003,140004,140005,140006,141001,141002,141003,141004,141005,141006,142001,142002,142003,142004,142005,142006,143001,143002,143003,143004,143005,143006,144001,144002,144003,144004,144005,144006,145001,145002,145003,145004,145005,145006,146001,146002,146003,146004,146005,146006,147001,147002,147003,147004,147005,147006,148001,148002,148003,148004,148005,148006,149001,149002,149003,149004,149005,149006,150001,150002,150003,150004,150005,150006)

TOPICS_2020 = [str(i) for i in TOPICS_2020]
TOPICS_2021 = [str(i) for i in TOPICS_2021]
TOPICS_2022 = [str(i) for i in TOPICS_2022]
TOPICS_CLEF = [str(i) for i in TOPICS_CLEF]


def rbo(run, ideal, p):
    run_set = set()
    ideal_set = set()

    score = 0.0
    normalizer = 0.0
    weight = 1.0
    for i in range(DEPTH):
        if i < len(run):
            run_set.add(run[i])
        if i < len(ideal):
            ideal_set.add(ideal[i])
        score += weight*len(ideal_set.intersection(run_set))/(i + 1)
        normalizer += weight
        weight *= p
    return score/normalizer


def idealize(run, ideal, qrels):
    rank = {}
    for i in range(len(run)):
        rank[run[i]] = i
    ideal.sort(key=lambda docno: rank[docno] if docno in rank else len(run))
    ideal.sort(key=lambda docno: qrels[docno], reverse=True)
    return ideal


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p', type=float, default=P, help='persistence')
    parser.add_argument('qrels', type=str, help='TREC-style qrels')
    parser.add_argument('run', type=str, help='TREC-style run')
    parser.add_argument('-o', '--output', type=str, default='compatibility.csv',
                        help='output CSV file path (default: compatibility.csv)')
    args = parser.parse_args()

    if args.p < 0.01 or args.p > 0.99:
        print('Value of p = ' + str(args.p) + ' out of range [0.01,0.99]',
              file=sys.stderr)
        sys.exit(0)

    ideal = {}
    qrels = {}
    with open(args.qrels) as qrelsf:
        for line in qrelsf:
            (topic, q0, docno, qrel) = line.rstrip().split()
            qrel = float(qrel)
            if qrel > 0.0:
                if topic not in qrels:
                    ideal[topic] = []
                    qrels[topic] = {}
                if docno in qrels[topic]:
                    if qrel > qrels[topic][docno]:
                        qrels[topic][docno] = qrel
                else:
                    ideal[topic].append(docno)
                    qrels[topic][docno] = qrel

    runid = ""
    run = {}
    scores = {}
    with open(args.run) as runf:
        for line in runf:
            (topic, q0, docno, rank, score, runid) = line.rstrip().split()
            if topic not in run:
                run[topic] = []
                scores[topic] = {}
            run[topic].append(docno)
            scores[topic][docno] = float(score)

    for topic in run:
        run[topic].sort()
        run[topic].sort(key=lambda docno: scores[topic][docno], reverse=True)
        if topic in ideal:
            ideal[topic] = idealize(run[topic], ideal[topic], qrels[topic])

    count = 0
    total = 0.0
    run_name = args.run.split('/')[-1]
    qrels_name = args.qrels.split('/')[-1]
    results = {"run": run_name, "qrels": qrels_name, "p": args.p}
    for topic in ideal:
        if not (topic in run):
            count += 1
            score = 0.0
            print('compatibility', topic, "{:.4f}".format(score), sep='\t')
            results[topic] = "{:.4f}".format(score)
        else:
            score = rbo(run[topic], ideal[topic], args.p)
            if NORMALIZE:
                best = rbo(ideal[topic], ideal[topic], args.p)
                if best > 0.0:
                    score /= best
                else:
                    score = best
            count += 1
            total += score
            print('compatibility', topic, "{:.4f}".format(score), sep='\t')
            results[topic] = "{:.4f}".format(score)

    if count > 0:
        print('compatibility', 'all', "{:.4f}".format(total/count), sep='\t')
        results['all'] = "{:.4f}".format(total/count)
    else:
        print('compatibility', 'all', "{:.4f}".format(0.0), sep='\t')
        results['all'] = "{:.4f}".format(0.0)

    int_keys = [int(k) for k in ideal.keys()]
    smallest_topic = min(int_keys)
    largest_topic = max(int_keys)
    existing_topics = []
    for i in range(smallest_topic, largest_topic + 1):
        # The topic was not in the qrels
        if str(i) not in results:      
            # But the topic exists in the topics of the collections
            if str(i) in TOPICS_2020 or str(i) in TOPICS_2021 or str(i) in TOPICS_2022 or str(i) in TOPICS_CLEF:  
                results[str(i)] = "NA"      # So we mark it as NA
                existing_topics.append(str(i))
        else:
            existing_topics.append(str(i))

    column_order = ["run", "qrels", "p", "all"] + existing_topics

    df = pd.DataFrame(results, index=[0], columns=column_order)
    df.to_csv(args.output, index=False, mode='a', header=False)

if __name__ == "__main__":
    main()
