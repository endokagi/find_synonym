from pythainlp.corpus import wordnet
import reader_writer_csv as rw
from progress.bar import IncrementalBar

def find_synonyms(path_folder,namefile):
    word = rw.get_data(f'./{path_folder}/{namefile}.csv')
    max = len(word)
    bar = IncrementalBar(f'Progress', max=max,
                         suffix='%(percent)d%% %(elapsed_td)s')
    for row in word:
        trans_word = transform(row["word"])
        try:
            find_synonyms = wordnet.synsets(trans_word)[0].lemma_names('tha')
            row.update({"synonyms":find_synonyms})
        except:
            pass
        bar.next()
    bar.finish()   
    fieldnames = ['word','synonyms','tf-idf-pos','tf-idf-neg','tf-idf-val','node-label']
    rw.write_data_by_columns(f"./synonyms/pythai/{path_folder}/{namefile}.csv", fieldnames, word)

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