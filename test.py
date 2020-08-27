import tltk
import csv
import reader_writer_csv as rw
from progress.bar import IncrementalBar

def find_synonyms(path_folder,namefile):
    word = rw.get_data(f'./{path_folder}/{namefile}.csv')
    data = []
    max = len(word)
    bar = IncrementalBar(f'Progress', max=max,
                         suffix='%(percent)d%% %(elapsed_td)s')
    for row in word:
        trans_word = transform(row["word"])
        pos = row["node-label"]
        tltk.corpus.w2v_load()
        try:
            find_synonym = tltk.corpus.similar_words(trans_word, score="n")
            data.append({"word":trans_word,"pos":pos,"synonym":find_synonym})     
        except:
            pass
        bar.next()
    bar.finish()   
    fieldnames = ['word','pos','synonym']
    rw.write_data_by_columns("dataset.csv", fieldnames, data)

def transform(word):
    word = word.replace('(*', '')
    word = word.replace('(', '')
    word = word.replace(' - VERB)', '')
    word = word.replace(' - ADV)', '')
    word = word.replace(' - ADJ)', '')
    return word

# def transform_synonym(word):
#     word = word.replace('"[', '')
#     word = word.replace(']"', '')
#     word = word.replace(', ', '|')

def main():
    find_synonyms('max_range_3','tfidf_p95_t7_100')
    
main()
