import torch
import torchtext.vocab
import spacy
import constants as c
import re
from logger import Logger as Log


class NumericRepresentationService:

    batch_size = 64
    # To load english use 'python -m spacy download en'
    # init nlp - we only use the tokenizer
    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

    # Load pre trained word embedding
    # name = Name of the file containing the vector cache
    # dim = The dimensionality of the vectors
    glove = torchtext.vocab.GloVe(name='6B', dim=100)

    def __init__(self):
        self.Text = []
        self.Label = []

        Log.print_and_log("Regular", f"Glove data base size: {len(self.glove.itos)}")

    def get_numeric_representation_of_final_data(self):

        Log.print_and_log("Regular", f"Converting final data base to numeric representation...")

        # Defines a data type together with instructions for converting to Tensor.
        self.Text = torchtext.data.Field(tokenize=self.tokenizer)
        self.Label = torchtext.data.LabelField(dtype=torch.int)

        # Load data from csv
        data_fields = [(c.FINAL_DATABASE_PREDICTION_COLUMN, self.Label),
                       (c.FINAL_DATABASE_TEXT_COLUMN, self.Text)]
        train, test = torchtext.data.TabularDataset.splits(path=c.FINAL_DATABASE_FOLDER,
                                                           train=c.FINAL_DATABASE_TRAIN,
                                                           test=c.FINAL_DATABASE_TEST,
                                                           format='csv',
                                                           skip_header=True,
                                                           fields=data_fields)
        Log.print_and_log("Regular", f'Number of training examples: {len(train)}')
        Log.print_and_log("Regular", f'Number of testing examples: {len(test)}')
        Log.print_and_log("Regular", f'Train example: {train.examples[0].Tweet}')

        # Construct the Vocab object for this field from the data set
        # Words outside the 25000 words vector will be initialize
        # using torch.Tensor.normal (word mean and standard deviation)
        self.Text.build_vocab(train, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
        self.Label.build_vocab(train)
        Log.print_and_log("Regular", f"Pre trained embedding size{self.Text.vocab.vectors.shape}")
        Log.print_and_log("Regular", f'Most frequent word in vocabulary: {self.Text.vocab.freqs.most_common(10)}')

        # Defines an iterator that batches examples of similar lengths together.
        # Minimizes amount of padding needed while producing freshly shuffled batches for each new
        train_iterator, test_iterator = torchtext.data.BucketIterator.splits(
            (train, test),
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.Tweet),
            sort_within_batch=False)

        self.print_data_base(train_iterator, 1)
        return train_iterator, test_iterator

    # Clean text, tokenize with nlp and return with lower case characters
    @staticmethod
    def tokenizer(s):
        return [w.text.lower() for w in NumericRepresentationService.nlp(NumericRepresentationService.clean_tweets(s))]

    @staticmethod
    def clean_tweets(text):
        # Get rid of urls
        text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', '', text)

        # Get rid of all non letters or numbers
        text = re.sub(r'[^A-Za-z0-9.]+', ' ', text)

        return text.strip()

    def print_data_base(self, iterator, num):
        count = 0
        for batch in iterator:
            if count == num:
                return
            count += 1

            for tweet_index in range(len(batch.Tweet)):
                print(f'Prediction in tensor: {batch.Prediction[tweet_index]}')
                prediction_tensor = batch.Prediction[tweet_index]
                # prediction = self.Label.vocab.itos[prediction_tensor]
                # print(f'Translate Prediction: '
                #      f'{prediction}')
                tweet = self.get_tweet_from_batch(batch.Tweet, tweet_index)
                print(f'Tweet in tensor: {tweet}')
                print(f'Translate Tweet: '
                      f'{self.translate_vector_from_numeric(tweet, self.Text.vocab)}')

    @staticmethod
    def get_tweet_from_batch(batch, tweet_index):
        tweet = []
        for index in range(len(batch)):
            tweet.append(batch[index][tweet_index])
        return tweet

    @staticmethod
    def translate_vector_from_numeric(vector, vocabulary):
        return [vocabulary.itos[num] for num in vector]


