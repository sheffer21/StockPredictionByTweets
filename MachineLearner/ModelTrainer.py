import tensorflow as tf
import random
import torch
import MachineLearner.DataAnalyzer.StatisticsProvider as stat
import common.constants as const
import pandas as pd
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import os
from abc import ABC
import matplotlib.pyplot as plt


class ModelTrainer(ABC):

    def __init__(self, logger, num_labels, classify, runName, MAX_LEN, epochs, batch_size, resultAnalyzer, dataFilter,
                 loadFromPreTrained=False, preTrainedSourceDir=""):
        # Save parameter
        self.filter = dataFilter
        self.dataAnalyzer = resultAnalyzer
        self.batch_size = batch_size
        self.epochs = epochs
        self.MAX_LEN = MAX_LEN
        self.logger = logger
        self.classify = classify
        self.runName = f'{runName}_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
        self.num_labels = num_labels

        # Load model
        self.device = self.GetGPUDevice()

        if loadFromPreTrained:
            self.model, self.tokenizer = self.Load_From_PreTrained(preTrainedSourceDir)
        else:
            self.tokenizer = self.Load_Tokenizer()
            self.model = self.Load_BERT()

    def Train(self, trainDataSetPath):
        # Load train
        self.logger.printAndLog(const.MessageType.Regular, f'Training model {self.runName}')
        sentences, labels, _, _ = self.Load_DataSet(trainDataSetPath)
        input_ids = self.Tokenize_Sentences(sentences, self.tokenizer)
        input_ids = self.Pad_Sequences(input_ids, self.tokenizer)
        attention_mask = self.Get_Attention_Mask(input_ids)
        train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = \
            self.Split_DataSet(input_ids, labels, attention_mask)
        train_inputs, train_labels, train_masks, validation_inputs, validation_labels, validation_masks = \
            self.Convert_To_PyTorch(train_inputs, validation_inputs, train_labels, validation_labels, train_masks,
                                    validation_masks)
        train_dataLoader, validation_dataLoader = self.Create_Iterator(train_inputs, train_masks, train_labels,
                                                                       validation_inputs, validation_masks,
                                                                       validation_labels)
        optimizer, scheduler = self.Get_Optimizer(self.model, train_dataLoader)

        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # Store the average loss after each epoch so we can plot them.
        loss_values = []

        # For each epoch...
        for epoch_i in range(0, self.epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            self.logger.printAndLog(const.MessageType.Regular, "")
            self.logger.printAndLog(const.MessageType.Regular, f'======== Epoch {epoch_i + 1} / {self.epochs} ========')
            self.logger.printAndLog(const.MessageType.Regular, 'Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataLoader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    self.logger.printAndLog(const.MessageType.Regular,
                                            f'Batch {step}  of  {len(train_dataLoader)}.    Elapsed: {elapsed}.')

                # Unpack this training batch from our dataLoader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # This will return the loss (rather than the model output) because we
                # have provided the `labels`.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple.
                loss = outputs[0]

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataLoader)

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

            self.logger.printAndLog(const.MessageType.Regular, "")
            self.logger.printAndLog(const.MessageType.Regular, f"  Average training loss: {avg_train_loss:.2f}")
            self.logger.printAndLog(const.MessageType.Regular,
                                    f"  Training epcoh took: {self.format_time(time.time() - t0)}")

            self.PerformValidation(validation_dataLoader)

        self.Save_Model()
        self.logger.printAndLog(const.MessageType.Regular, "")
        self.logger.printAndLog(const.MessageType.Regular, "Training complete!")
        stat.Plot_Training_Loss(loss_values, self.runName)

    def PerformValidation(self, validation_dataLoader):
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        self.logger.printAndLog(const.MessageType.Regular, "")
        self.logger.printAndLog(const.MessageType.Regular, "Running Validation...")
        self.dataAnalyzer.StartValidation()

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()

        # Tracking variables
        eval_loss, coefficient, eval_meanSquare = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataLoader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataLoader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", whichyes
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            self.dataAnalyzer.PerformValidationStep(logits, label_ids)

        self.dataAnalyzer.FinishValidation()
        # Report the final accuracy for this validation run.
        self.logger.printAndLog(const.MessageType.Regular,
                                f"  Validation took: {self.format_time(time.time() - t0)}")

    def Test(self, testPath, batchTesting=False):
        self.logger.printAndLog(const.MessageType.Regular, '---------------------------------------------')
        self.logger.printAndLog(const.MessageType.Regular, f"Start testing result on test {self.runName}")

        # Load the dataset into a pandas dataframe.
        sentences, labels, companies, dates = self.Load_DataSet(testPath)

        # Report the number of sentences.
        self.logger.printAndLog(const.MessageType.Regular, f'Number of test sentences: {len(sentences)}\n')

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = self.Tokenize_Sentences(sentences, self.tokenizer)

        # Pad our input tokens
        input_ids = self.Pad_Sequences(input_ids, self.tokenizer)

        # Create attention masks
        # Create a mask of 1s for each token followed by 0s for padding
        attention_masks = self.Get_Attention_Mask(input_ids)

        # Convert to tensors.
        prediction_inputs = torch.tensor(input_ids)
        prediction_masks = torch.tensor(attention_masks)
        prediction_labels = torch.tensor(labels)

        # Set the batch size.

        # Create the DataLoader.
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataLoader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)

        # Prediction on test set

        self.logger.printAndLog(const.MessageType.Regular,
                                f'Predicting labels for {len(prediction_inputs)} test sentences...')

        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in prediction_dataLoader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        if batchTesting:
            # batch prediction only available for linear regression
            if self.num_labels != 1:
                raise Exception()
            true_labels, predictions = self.GetBatchPredictions(true_labels, predictions, companies, dates)

        self.dataAnalyzer.PrintTestResult(true_labels, predictions, self.runName)
        self.logger.printAndLog(const.MessageType.Regular, '    DONE.')

    def GetGPUDevice(self):
        # Get the GPU device name.
        device_name = tf.test.gpu_device_name()

        # The device name should look like the following:
        if device_name == '/device:GPU:0':
            self.logger.printAndLog(const.MessageType.Regular, f'Found GPU at: {device_name}')
        else:
            raise SystemError('GPU device not found')

        # If there's a GPU available...
        if torch.cuda.is_available():

            # Tell PyTorch to use the GPU.
            device = torch.device("cuda")

            self.logger.printAndLog(const.MessageType.Regular,
                                    f'There are {torch.cuda.device_count()} GPU(s) available.')

            self.logger.printAndLog(const.MessageType.Regular, f'We will use the GPU:{torch.cuda.get_device_name(0)}')

        # If not...
        else:
            self.logger.printAndLog(const.MessageType.Regular, 'No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        return device

    def Load_BERT(self):
        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classification layer on top.
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=self.num_labels,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        # Tell pytorch to run this model on the GPU.
        model.cuda()

        # Get all of the model's parameters as a list of tuples.
        params = list(model.named_parameters())

        self.logger.printAndLog(const.MessageType.Regular,
                                f'The BERT model has {len(params)} different named parameters.\n')

        self.logger.printAndLog(const.MessageType.Regular, '==== Embedding Layer ====\n')

        for p in params[0:5]:
            self.logger.printAndLog(const.MessageType.Regular, f"{p[0]} {p[1].size()}")

        self.logger.printAndLog(const.MessageType.Regular, '\n==== First Transformer ====\n')

        for p in params[5:21]:
            self.logger.printAndLog(const.MessageType.Regular, f"{p[0]} {str(tuple(p[1].size()))}")

        self.logger.printAndLog(const.MessageType.Regular, '\n==== Output Layer ====\n')

        for p in params[-4:]:
            self.logger.printAndLog(const.MessageType.Regular, f"{p[0]} {str(tuple(p[1].size()))}")

        return model

    def Load_From_PreTrained(self, source_dir):
        self.logger.printAndLog(const.MessageType.Regular, f'Loading model from: {source_dir}')
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(source_dir)
        tokenizer = BertTokenizer.from_pretrained(source_dir)

        # Copy the model to the GPU.
        model.to(self.device)
        return model, tokenizer

    def Load_Tokenizer(self):
        # Load the BERT tokenizer.
        self.logger.printAndLog(const.MessageType.Regular, 'Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Print the original sentence.
        # self.logger.printAndLog(const.MessageType.Regular, f' Original: {sentences[0]}')

        # Print the sentence split into tokens.
        # self.logger.printAndLog(const.MessageType.Regular, f'Tokenized: {tokenizer.tokenize(sentences[0])}')

        # Print the sentence mapped to token ids.
        # self.logger.printAndLog(const.MessageType.Regular,
        #                        f'Token IDs:{tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0]))}')

        return tokenizer

    def Load_DataSet(self, dataSetPath):
        self.logger.printAndLog(const.MessageType.Regular, '-------------------------------------------------')
        self.logger.printAndLog(const.MessageType.Regular, f'Loading DataSet...')

        # Load the dataset into a pandas dataframe.
        df = pd.read_csv(dataSetPath)
        # df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None,
        #                 names=['sentence_source', 'label', 'label_notes', 'sentence'])

        # Get the lists of sentences and their labels.
        # sentences, labels, followers = zip(*((d.Tweet, self.classify(d.Prediction), d[const.USER_FOLLOWERS_COLUMN])
        #                                     for index, d in df.iterrows()
        #                                     if type(d.Tweet) is str and self.filter(d)))

        df = pd.concat([d for index, d in df.iterrows()
                        if type(d.Tweet) is str and self.filter(d)])

        sentences = df.Tweet.values
        labels = [self.classify(prediction) for prediction in df.Prediction.values]
        followers = df[const.USER_FOLLOWERS_COLUMN].values
        companies = df[const.COMPANY_COLUMN].values
        dates = df[const.DATE_COLUMN].values

        # labels = [float(i) for i in df.Prediction.values]
        # labels = [self.classify(i) for i in df.Prediction.values]

        # sentences = df.sentence.values
        # labels = df.label.values
        # Analyze data
        # self.dataAnalyzer.AnalyzeDataSet(sentences, labels, followers, self.runName)

        # Report the number of sentences.
        self.logger.printAndLog(const.MessageType.Regular, 'Number of training sentences: {:,}'.format(len(sentences)))
        self.logger.printAndLog(const.MessageType.Regular, "Examples from the dataSet:")
        # Display 10 random rows from the data.
        for index in range(10):
            self.logger.printAndLog(const.MessageType.Regular, '---------------------------------------------')
            self.logger.printAndLog(const.MessageType.Regular, f'Row number:   {index}:')
            self.logger.printAndLog(const.MessageType.Regular, f'Prediction:   {labels[index]}')
            self.logger.printAndLog(const.MessageType.Regular, f'Tweet:        {sentences[index]}')
            self.logger.printAndLog(const.MessageType.Regular, f'Followers:    {followers[index]}')
            self.logger.printAndLog(const.MessageType.Regular, f'Company:      {companies[index]}')
        self.logger.printAndLog(const.MessageType.Regular, '---------------------------------------------')
        return sentences, labels, companies, dates

    def Tokenize_Sentences(self, sentences, tokenizer):
        # Tokenize all of the sentences and map the tokens to their word IDs.
        input_ids = []

        # For every sentence...
        for sent in sentences:
            if type(sent) is not str:
                continue
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            encoded_sent = tokenizer.encode(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'

                # This function also supports truncation and conversion
                # to pytorch tensors, but we need to do padding, so we
                # can't use these features :( .
                # max_length = 128,          # Truncate all sentences.
                # return_tensors = 'pt',     # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_sent)

        # Print sentence 0, now as a list of IDs.
        self.logger.printAndLog(const.MessageType.Regular, "Tokenized input example:")
        self.logger.printAndLog(const.MessageType.Regular, f'Original: {sentences[0]}')
        self.logger.printAndLog(const.MessageType.Regular, f'Token IDs: {input_ids[0]}')
        self.logger.printAndLog(const.MessageType.Regular,
                                f'Max sentence length: {max([len(sen) for sen in input_ids])}')

        return input_ids

    def Pad_Sequences(self, input_ids, tokenizer):
        self.logger.printAndLog(const.MessageType.Regular, "---------------------------------")
        self.logger.printAndLog(const.MessageType.Regular,
                                f'Padding/truncating all sentences to {self.MAX_LEN} values...')

        self.logger.printAndLog(const.MessageType.Regular,
                                f'Padding token: "{tokenizer.pad_token}", ID: {tokenizer.pad_token_id}')

        # Pad our input tokens with value 0.
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        input_ids = pad_sequences(input_ids, dtype='int64', maxlen=self.MAX_LEN,
                                  value=0, truncating="post", padding="post")

        self.logger.printAndLog(const.MessageType.Regular, 'Done.')

        return input_ids

    @staticmethod
    def Get_Attention_Mask(input_ids):
        # Create attention masks
        attention_masks = []

        # For each sentence...
        for sent in input_ids:
            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id > 0) for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks.append(att_mask)

        return attention_masks

    @staticmethod
    def Split_DataSet(input_ids, labels, attention_mask):
        # Use 90% for training and 10% for validation.
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                            random_state=2018,
                                                                                            test_size=0.1)
        # Do the same for the masks.
        train_masks, validation_masks, _, _ = train_test_split(attention_mask, labels,
                                                               random_state=2018, test_size=0.1)

        return train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks

    @staticmethod
    def Convert_To_PyTorch(train_inputs, validation_inputs, train_labels, validation_labels, train_masks,
                           validation_masks):
        # Convert all inputs and labels into torch tensors, the required datatype
        # for our model.
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        return train_inputs, train_labels, train_masks, validation_inputs, validation_labels, validation_masks

    def Create_Iterator(self, train_inputs, train_masks, train_labels, validation_inputs, validation_masks,
                        validation_labels):
        # The DataLoader needs to know our batch size for training, so we specify it
        # here.
        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataLoader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataLoader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)

        return train_dataLoader, validation_dataLoader

    def Get_Optimizer(self, model, train_dataLoader):
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataLoader) * self.epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        return optimizer, scheduler

    def Save_Model(self):
        output_path = f"{const.TrainedModelDirectory}{self.runName}"

        # Create output directory if needed
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.logger.printAndLog(const.MessageType.Regular, f"Saving model to {output_path}")

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.model.module if hasattr(self.model,
                                                     # Take care of distributed/parallel training
                                                     'module') else self.model

        model_to_save.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Good practice: save your training arguments together with the trained model
        args = f"Run {self.runName} with Epochs: {self.epochs}, MAX_LEN: {self.MAX_LEN}, Batch_size: {self.batch_size}"
        torch.save(args, os.path.join(output_path, 'training_args.bin'))

    @staticmethod
    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    @staticmethod
    def GetBatchPredictions(true_labels, predictions, companies, dates):
        batch_trueLabels, batch_predictions = [], []

        true_labels = [item for sublist in true_labels for item in sublist]
        predictions = [item for sublist in predictions for item in sublist]

        data = pd.concat([pd.DataFrame([
            [datetime.datetime.strptime(dates[index], const.databaseDateFormat).date(),
             companies[index],
             predictions[index][0],
             true_labels[index]]],
            columns=['date', 'company', 'prediction', 'true_label'])
            for index in range(len(predictions))])

        grouped_data_by_date_and_company = data.groupby(['date', 'company'])

        index = 0
        for group_name, df_group in grouped_data_by_date_and_company:
            index += 1
            fig = plt.figure()
            ax = fig.add_subplot(111)
            predictions = df_group['prediction']
            n, bins, patches = ax.hist(predictions, bins=10, normed=True, fc='k', alpha=0.3)
            prediction = bins[np.argmax(n)]

            for i, item in df_group.iterrows():
                batch_trueLabels.append(item['true_label'])
                batch_predictions.append(prediction)

            # plt.show()

        return batch_trueLabels, batch_predictions
