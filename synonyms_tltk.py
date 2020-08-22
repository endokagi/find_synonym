import tltk
import csv
import reader_writer_csv as rw
from progress.bar import IncrementalBar

def find_synonyms(path_folder,namefile):
    word = rw.get_data(f'./{path_folder}/{namefile}.csv')
    max = len(word)
    bar = IncrementalBar(f'Progress', max=max,
                         suffix='%(percent)d%% %(elapsed_td)s')
    for row in word:
        trans_word = transform(row["word"])
        tltk.corpus.w2v_load()
        try:
            find_synonym = tltk.corpus.similar_words(trans_word, n=5, score="n")
            row.update({"synonyms":find_synonym})     
        except:
            pass
        bar.next()
    bar.finish()   
    fieldnames = ['word','synonyms','tf-idf-pos','tf-idf-neg','tf-idf-val','node-label']
    rw.write_data_by_columns(f"./synonyms/tltk/{path_folder}/{namefile}.csv", fieldnames, word)

def transform(word):
    word = word.replace('(*', '')
    word = word.replace('(', '')
    word = word.replace(' - VERB)', '')
    word = word.replace(' - ADV)', '')
    word = word.replace(' - ADJ)', '')
    return word

def main():
    find_synonyms('max_range_3','tfidf_p95_t7')
    
main()
