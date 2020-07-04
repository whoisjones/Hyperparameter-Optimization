import multiprocessing as mp
from functools import partial
from flair.embeddings import DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

def train(corpus, configuration):
    label_dict = corpus.make_label_dictionary()
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

def multiprocess(params, corpus):
    number_of_processes = mp.cpu_count()
    with mp.Pool(number_of_processes) as pool:
        setup_train = partial(train, corpus)
        pool.map_async(setup_train, params)
        pool.close()
        pool.join()