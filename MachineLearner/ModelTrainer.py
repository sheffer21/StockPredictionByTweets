import tensorflow as tf
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
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

MAX_LEN = 64
# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.
batch_size = 32


class ModelTrainer:

    def __init__(self, logger):
        self.logger = logger
        self.device = self.GetGPUDevice()
        self.dataSet, self.sentences, self.labels = self.Load_DataSet()
        self.tokenizer = self.Load_Tokenizer()
        self.input_ids = self.Split_to_tokens()
        self.Ped_Sequences()
        self.attention_mask = self.Get_AttentionMask()
        self.train_inputs, self.validation_inputs, self.train_labels, self.validation_labels, self.train_masks, self.validation_masks = self.Split_DataSet()
        self.Convert_To_PyTorch()
        self.train_dataloader, self.validation_dataloader = self.Create_Iterator()
        self.model = self.Load_BERT()

    def Get_Optimizer(self):
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(self.model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(self.train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

    def Split_DataSet(self):
        # Use 90% for training and 10% for validation.
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(self.input_ids, self.labels,
                                                                                            random_state=2018,
                                                                                            test_size=0.1)
        # Do the same for the masks.
        train_masks, validation_masks, _, _ = train_test_split(self.attention_mask, self.labels,
                                                               random_state=2018, test_size=0.1)

        return train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks

    def Convert_To_PyTorch(self):
        # Convert all inputs and labels into torch tensors, the required datatype
        # for our model.
        self.train_inputs = torch.tensor(self.train_inputs)
        self.validation_inputs = torch.tensor(self.validation_inputs)

        self.train_labels = torch.tensor(self.train_labels)
        self.validation_labels = torch.tensor(self.validation_labels)

        self.train_masks = torch.tensor(self.train_masks)
        self.validation_masks = torch.tensor(self.validation_masks)

    def Get_AttentionMask(self):
        # Create attention masks
        attention_masks = []

        # For each sentence...
        for sent in self.input_ids:
            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id > 0) for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks.append(att_mask)

        return attention_masks

    def Ped_Sequences(self):
        self.logger.printAndLog(const.MessageType.Regular, f'\nPadding/truncating all sentences to {MAX_LEN} values...')

        self.logger.printAndLog(const.MessageType.Regular,
                                f'\nPadding token: "{self.tokenizer.pad_token}", ID: {self.tokenizer.pad_token_id}')

        # Pad our input tokens with value 0.
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        self.input_ids = pad_sequences(self.input_ids, dtype='long', maxlen=MAX_LEN,
                                       value=np.long(0), truncating="post", padding="post")

        self.logger.printAndLog(const.MessageType.Regular, '\nDone.')

    def Split_to_tokens(self):
        # Tokenize all of the sentences and map the tokens to their word IDs.
        input_ids = []

        # For every sentence...
        for sent in self.sentences:
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            encoded_sent = self.tokenizer.encode(
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
        self.logger.printAndLog(const.MessageType.Regular, f'Original: {self.sentences[0]}')
        self.logger.printAndLog(const.MessageType.Regular, f'Token IDs: {input_ids[0]}')
        self.logger.printAndLog(const.MessageType.Regular,
                                f'Max sentence length: {max([len(sen) for sen in input_ids])}')

        return input_ids

    def Load_Tokenizer(self):
        # Load the BERT tokenizer.
        self.logger.printAndLog(const.MessageType.Regular, 'Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Print the original sentence.
        self.logger.printAndLog(const.MessageType.Regular, f' Original: {self.sentences[0]}')

        # Print the sentence split into tokens.
        self.logger.printAndLog(const.MessageType.Regular, f'Tokenized: {tokenizer.tokenize(self.sentences[0])}')

        # Print the sentence mapped to token ids.
        self.logger.printAndLog(const.MessageType.Regular,
                                f'Token IDs:{tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.sentences[0]))}')

        return tokenizer

    def Load_DataSet(self):
        self.logger.printAndLog(const.MessageType.Regular, f'Loading data set')

        # Load the dataset into a pandas dataframe.
        df = pd.read_csv(f'{const.finalDatabaseFolder}{const.testFileDebug}')

        # Get the lists of sentences and their labels.
        sentences = df.Tweet.values
        labels = df.Prediction.values

        # Report the number of sentences.
        self.logger.printAndLog(const.MessageType.Regular, 'Number of training sentences: {:,}\n'.format(df.shape[0]))

        # Display 10 random rows from the data.
        for sample in df.sample(10):
            self.logger.printAndLog(const.MessageType.Regular, f'{sample}')
        return df, sentences, labels

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
                                    f'There are %d GPU(s) available. {torch.cuda.device_count()}')

            self.logger.printAndLog(const.MessageType.Regular, f'We will use the GPU:{torch.cuda.get_device_name(0)}')

        # If not...
        else:
            self.logger.printAndLog(const.MessageType.Regular, 'No GPU available, using the CPU instead.')
            device = torch.device("cpu")

    def Create_Iterator(self):
        # The DataLoader needs to know our batch size for training, so we specify it
        # here.
        # Create the DataLoader for our training set.
        train_data = TensorDataset(self.train_inputs, self.train_masks, self.train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(self.validation_inputs, self.validation_masks, self.validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        return train_dataloader, validation_dataloader

    def Load_BERT(self):
        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classification layer on top.
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=1,  # The number of output labels--2 for binary classification.
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

    def Train(self):
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
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            self.logger.printAndLog(const.MessageType.Regular, "")
            self.logger.printAndLog(const.MessageType.Regular, f'======== Epoch {epoch_i + 1} / {epochs} ========')
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
            for step, batch in enumerate(self.train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    self.logger.printAndLog(const.MessageType.Regular,
                                            f'Batch {step > 5,}  of  {len(self.train_dataloader) > 5,}.    Elapsed: {elapsed}.')

                # Unpack this training batch from our dataloader.
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
                self.optimizer.step()

                # Update the learning rate.
                self.scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(self.train_dataloader)

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

            self.logger.printAndLog(const.MessageType.Regular, "")
            self.logger.printAndLog(const.MessageType.Regular, f"  Average training loss: {avg_train_loss:.2f}")
            self.logger.printAndLog(const.MessageType.Regular,
                                    f"  Training epcoh took: {self.format_time(time.time() - t0)}")

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            self.logger.printAndLog(const.MessageType.Regular, "")
            self.logger.printAndLog(const.MessageType.Regular, "Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.model.eval()

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in self.validation_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(self.device) for t in batch)

                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have
                    # not provided labels.
                    # token_type_ids is the same as the "segment ids", which
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

                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)

                # Accumulate the total accuracy.
                eval_accuracy += tmp_eval_accuracy

                # Track the number of batches
                nb_eval_steps += 1

            # Report the final accuracy for this validation run.
            self.logger.printAndLog(const.MessageType.Regular, f"  Accuracy: {eval_accuracy / nb_eval_steps:.2f}")
            self.logger.printAndLog(const.MessageType.Regular,
                                    f"  Validation took: {self.format_time(time.time() - t0)}")

        self.logger.printAndLog(const.MessageType.Regular, "")
        self.logger.printAndLog(const.MessageType.Regular, "Training complete!")
        self.Plot_Training_Loss(loss_values)

    def Plot_Training_Loss(self, loss_values):
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(loss_values, 'b-o')

        # Label the plot.
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.show()

    def Test(self):
        # Load the dataset into a pandas dataframe.
        df = pd.read_csv(f"{const.finalDatabaseFolder}{const.testFileDebug}")

        # Report the number of sentences.
        self.logger.printAndLog(const.MessageType.Regular, f'Number of test sentences: {df.shape[0]}\n')

        # Create sentence and label lists
        sentences = df.sentence.values
        labels = df.label.values

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []

        # For every sentence...
        for sent in sentences:
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            encoded_sent = self.tokenizer.encode(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            )

            input_ids.append(encoded_sent)

        # Pad our input tokens
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                                  dtype="long", truncating="post", padding="post")

        # Create attention masks
        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

            # Convert to tensors.
        prediction_inputs = torch.tensor(input_ids)
        prediction_masks = torch.tensor(attention_masks)
        prediction_labels = torch.tensor(labels)

        # Set the batch size.
        batch_size = 32

        # Create the DataLoader.
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        # Prediction on test set

        self.logger.printAndLog(const.MessageType.Regular,
                                f'Predicting labels for {len(prediction_inputs)} test sentences...')

        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in prediction_dataloader:
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

        self.logger.printAndLog(const.MessageType.Regular, '    DONE.')

        self.logger.printAndLog(const.MessageType.Regular,
                                f'Positive samples: {df.label.sum()} of {len(df.label)} ({(df.label.sum() / len(df.label) * 100.0)})')

    # Function to calculate the accuracy of our predictions vs labels
    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    @staticmethod
    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))
