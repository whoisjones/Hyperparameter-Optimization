import multiprocessing as mp
from functools import partial
import contextlib
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import DocumentRNNEmbeddings, WordEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

from multiprocessing import Pool
def doubler(config):


    # 1. get the corpus
    corpus: Corpus = TREC_6()

    # 2. create the label dictionary
    label_dict = corpus.make_label_dictionary()

    # 3. make a list of word embeddings
    word_embeddings = [WordEmbeddings('glove')]

    # 4. initialize document embedding by passing list of word embeddings
    # Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
    document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=config['hidden_size'])

    # 5. create the text classifier
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

    # 6. initialize the text classifier trainer
    trainer = ModelTrainer(classifier, corpus)

    # 7. start the training
    result = trainer.train('resources/taggers/trec/{}'.format(config['hidden_size']),
                  learning_rate=0.1,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  patience=5,
                  max_epochs=10)


    curr_scores = result["dev_loss_history"][-3:]

    return curr_scores

if __name__ == '__main__':
    configs = [{'hidden_size': 128}, { 'hidden_size': 256},{'hidden_size': 512}]
    pool = Pool(processes=1)
    print(pool.map_async(doubler, configs).get())