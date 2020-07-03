import multiprocessing as mp
from functools import partial
from flair.embeddings import DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

def train(corpus, labels, configuration):
    word_embeddings = configuration.embeddings
    document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=configuration.hidden_size)
    classifier = TextClassifier(document_embeddings, label_dictionary=labels)
    trainer = ModelTrainer(classifier, corpus)
    trainer.train('resources/taggers/trec/gen1',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=5,
                  max_epochs=150)

def multiprocess(optimizer: object):
    number_of_processes = mp.cpu_count()
    corpus = optimizer.corpus
    label_dict = corpus.make_label_dictionary()
    with mp.Pool(number_of_processes) as pool:
        setup_train = partial(train, corpus, label_dict)
        pool.map_async(setup_train, optimizer.population)
        pool.close()
        pool.join()