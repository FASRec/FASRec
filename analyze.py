import numpy as np
def analyze(path):
    record = np.load(path, allow_pickle=True).item()
    best_test_ndcg = {"@10":0}
    best_test_hit = {"@10":0}
    best_e = 0
    for e in record:
        if(e > 600): break
        if "test_ndcg" in record[e] and sum(record[e]["test_ndcg"].values()) > sum(best_test_ndcg.values()):
            best_test_ndcg = record[e]
            best_test_ndcg = record[e]["test_ndcg"]
            best_test_hit = record[e]["test_hit"]
            best_e = e
            # print(e, record[e]["loss"])
    print(best_e)
    print(best_test_ndcg)
    print(best_test_hit)

if __name__ == '__main__':
    analyze("/home/wtc/perl5/wpz/WX/EMA/+-/e300a.7/record.npy")
    # analyze("/home/wtc/perl5/wpz/WX/2.5/SRNS/Beauty/v-1s0t50/record.npy")
    # analyze("/home/wtc/perl5/wpz/WX/2.5/SRNS/Beauty/v5s1t50/record.npy")
    analyze("/home/wtc/perl5/wpz/WX/2.5/SRNS/Beauty/v-1s0t300/record.npy")
    
    # analyze("/home/wtc/perl5/wpz/WX/EMA/+-/e50a.5/record.npy")
    # analyze("/home/wtc/perl5/wpz/WX/baseline/base_all/record.npy")
    # analyze("/home/wtc/perl5/wpz/WX/baseline/base_Beauty/record.npy")

