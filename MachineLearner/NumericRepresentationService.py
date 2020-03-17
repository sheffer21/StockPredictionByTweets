import torch
import torchtext.vocab
import spacy
import common.constants as const
import re


class NumericRepresentationService:
    batch_size = 64
    # To load english use 'python -m spacy download en'
    # init nlp - we only use the tokenizer
    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

    # Load pre trained word embedding
    # name = Name of the file containing the vector cache
    # dim = The dimensionality of the vectors
    glove = torchtext.vocab.GloVe(name='6B', dim=100)

    def __init__(self, logger):
        self.Text = []
        self.Label = []

        self.logger = logger

        self.logger.printAndLog(const.MessageType.Regular, f"Glove data base size: {len(self.glove.itos)}")

    def getNumericRepresentationOfFinalData(self):

        self.logger.printAndLog(const.MessageType.Regular, f"Converting final data base to numeric representation...")

        # Defines a data type together with instructions for converting to Tensor.
        self.Text = torchtext.data.Field(tokenize=self.tokenizer, dtype=torch.int64)
        self.Label = torchtext.data.LabelField(dtype=torch.int64)

        data_fields = [(const.PREDICTION_COLUMN, self.Label),
                       (const.TEXT_COLUMN, self.Text)]

        train, test = torchtext.data.TabularDataset.splits(path=const.finalDatabaseFolder,
                                                           train=const.trainFileDebug,
                                                           # train=const.trainFile,
                                                           test=const.testFileDebug,
                                                           # test=const.testFile,
                                                           format='csv',
                                                           skip_header=True,
                                                           fields=data_fields)

        self.logger.printAndLog(const.MessageType.Regular, f'Number of training examples: {len(train)}')
        self.logger.printAndLog(const.MessageType.Regular, f'Number of testing examples: {len(test)}')
        self.logger.printAndLog(const.MessageType.Regular, f'Train example: {train.examples[0].Tweet}')

        # Construct the Vocab object for this field from the data set
        # Words outside the 25000 words vector will be initialize
        # using torch.Tensor.normal (word mean and standard deviation)
        self.Text.build_vocab(train, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
        self.Label.build_vocab(train)
        vocab_size = len(self.Text.vocab.vectors)
        prediction_vocab_size = len(self.Label.vocab)
        self.logger.printAndLog(const.MessageType.Regular, f"Pre trained embedding size{vocab_size}")
        self.logger.printAndLog(const.MessageType.Regular,
                                f'Most frequent word in vocabulary: {self.Text.vocab.freqs.most_common(10)}')

        # Defines an iterator that batches examples of similar lengths together.
        # Minimizes amount of padding needed while producing freshly shuffled batches for each new
        train_iterator, test_iterator = torchtext.data.BucketIterator.splits(
            (train, test),
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.Tweet),
            sort_within_batch=False)

        self.printDataBase(train_iterator, 1)
        return train_iterator, test_iterator, vocab_size, prediction_vocab_size

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

    def printDataBase(self, iterator, num):
        count = 0
        for batch in iterator:
            if count == num:
                return
            count += 1

            for tweet_index in range(len(batch.Tweet)):
                self.logger.printAndLog(const.MessageType.Regular.value,
                                        f'Prediction in tensor: {batch.Prediction[tweet_index]}')
                prediction_tensor = batch.Prediction[tweet_index].item()
                prediction = self.Label.vocab.itos[prediction_tensor]
                self.logger.printAndLog(const.MessageType.Regular.value, f'Translate Prediction: '
                                                                         f'{prediction}')
                tweet = self.getTweetFromBatch(batch.Tweet, tweet_index)
                self.logger.printAndLog(const.MessageType.Regular.value, f'Tweet in tensor: {tweet}')
                self.logger.printAndLog(const.MessageType.Regular.value, f'Translate Tweet: '
                                                                         f'{self.translateVectorFromNumeric(tweet, self.Text.vocab)}')

    @staticmethod
    def getTweetFromBatch(batch, tweet_index):
        tweet = []
        for index in range(len(batch)):
            tweet.append(batch[index][tweet_index])
        return tweet

    @staticmethod
    def translateVectorFromNumeric(vector, vocabulary):
        return [vocabulary.itos[num] for num in vector]
