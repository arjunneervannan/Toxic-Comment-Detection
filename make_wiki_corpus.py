from __future__ import print_function
import os.path
import six
import sys
from gensim.corpora.wikicorpus import *


def tokenize(content):
    # override original method in wikicorpus.py
    return [token.encode('utf8') for token in content.split()
            if len(token) <= 15 and not token.startswith('_')]


def process_article(args):
    # override original method in wikicorpus.py
    text, lemmatize, title, pageid = args
    text = filter_wiki(text)
    if lemmatize:
        result = utils.lemmatize(text)
    else:
        result = tokenize(text)
    return result, title, pageid


class MyWikiCorpus(WikiCorpus):
    def __init__(self, fname, processes=None, lemmatize=utils.has_pattern(), dictionary=None, filter_namespaces=('0',)):
        WikiCorpus.__init__(self, fname, processes, lemmatize, dictionary, filter_namespaces)

    def get_texts(self):
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0
        texts = ((text, self.lemmatize, title, pageid) for title, text, pageid in
                 extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces))
        pool = multiprocessing.Pool(self.processes)
        for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
            for tokens, title, pageid in pool.imap(process_article, group):  # chunksize=10):
                articles_all += 1
                positions_all += len(tokens)
            if len(tokens) < ARTICLE_MIN_WORDS or any(title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                continue
            articles += 1
            positions += len(tokens)
            if self.metadata:
                yield (tokens, (pageid, title))
            else:
                yield tokens
        pool.terminate()

        print(
            "finished iterating over Wikipedia corpus of %i documents with %i positions"
            " (total %i articles, %i positions before pruning articles shorter than %i words)",
            articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS)
        self.length = articles  # cache corpus length


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) != 3:
        print("Using: python process_wiki.py enwiki-20180601-pages-    articles.xml.bz2 wiki.en.text")
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    output = open(outp, 'w')
    wiki = MyWikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        if six.PY3:
            output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        else:
            output.write(space.join(text) + "\n")
        i = i + 1
        if i % 10000 == 0:
            print("Saved " + str(i) + " articles")

    output.close()
    print("Finished Saved " + str(i) + " articles")
