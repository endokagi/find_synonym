import csv
import reader_writer_csv as rw
from progress.bar import IncrementalBar

neg = []
pos = []
pair = []

def main():
    file = rw.get_data(f'./max_range_3/tfidf_p95_t7.csv')
    max = len(file)
    bar = IncrementalBar(f'Progress', max=max,
                         suffix='%(percent)d%% %(elapsed_td)s')
    
    for row in file:
        word = row["word"]
        tfidf_neg = row["tf-idf-pos"]
        tfidf_pos = row["tf-idf-neg"]
        tfidf_val = row["tf-idf-val"]
        node_label = row["node-label"]
        try:       
            if tfidf_pos == "0" and tfidf_neg != "0":
                neg.append([word,tfidf_neg,node_label])
            elif tfidf_pos != "0" and tfidf_neg == "0": 
                pos.append([word,tfidf_pos,node_label])
            elif not(tfidf_pos is "0" and tfidf_neg is "0"):
                pair.append([word,tfidf_pos,tfidf_neg,tfidf_val,node_label])
        except:
            pass
        bar.next()
    bar.finish() 

    fieldnamesPOS = ['word','tf-idf-pos','node-label']
    fieldnamesNEG = ['word','tf-idf-neg','node-label']
    fieldnamesPAIR = ['word','tf-idf-pos','tf-idf-neg','tf-idf-val','node-label']

    with open(f'./tfidf_find/tfidfPOS.csv', mode='w', newline='', encoding='utf-8') as writefile:
        writer = csv.DictWriter(writefile, fieldnames=fieldnamesPOS)
        writer.writeheader()
        for row in pos :
            writer.writerow({'word':row[0],'tf-idf-pos':row[1],'node-label':row[2]})

    with open(f'./tfidf_find/tfidfNEG.csv', mode='w', newline='', encoding='utf-8') as writefile:
        writer = csv.DictWriter(writefile, fieldnames=fieldnamesNEG)
        writer.writeheader()
        for row in neg :
            writer.writerow({'word':row[0],'tf-idf-neg':row[1],'node-label':row[2]})

    with open(f'./tfidf_find/tfidfPAIR.csv', mode='w', newline='', encoding='utf-8') as writefile:
        writer = csv.DictWriter(writefile, fieldnames=fieldnamesPAIR)
        writer.writeheader()
        for row in pair :
            writer.writerow({'word':row[0],'tf-idf-pos':row[1],'tf-idf-neg':row[2],'tf-idf-val':row[3],'node-label':row[4]})

main()