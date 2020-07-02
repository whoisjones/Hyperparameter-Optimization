import multiprocessing as mp
from functools import partial
from flair.embeddings import DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

def train(corpus, configuration):
    #TODO: START TRAINER HERE
    label_dict = corpus.make_label_dictionary()

def multiprocess(optimizer: object):
    number_of_processes = mp.cpu_count()
    corpus = optimizer.corpus
    with mp.Pool(number_of_processes) as pool:
        setup_train = partial(train, corpus)
        pool.map_async(setup_train, optimizer.population)
        pool.close()
        pool.join()