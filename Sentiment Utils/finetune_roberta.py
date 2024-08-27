# from transformers import pipeline
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from config import Stopwords_list
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict, deque

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.utils.serialization as xser
except:
    pass

print('Loading files...')
nltk.download('punkt')
# nltk.download('stopwords')
nltk.download('wordnet')
print('Files loaded.')

############################### Dataset prepping class for torch ###############################
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, text_column_name, label_column_name, group_column_name=None):
        """
        A custom dataset class for sentiment analysis.

        Args:
        - data: The input DataFrame containing text and sentiment labels.
        - tokenizer: The tokenizer object for tokenizing the text.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        if group_column_name:
            self.group_column_name = group_column_name
        else:
            self.group_column_name = False
        self.sort_data_by_length()

    def sort_data_by_length(self):
        """
        Sorts the data by input length in descending order. also preserves the order of the dataframe.
        """
        self.data.reset_index(drop=True, inplace=True)
        # Tokenize the text without fixed padding to get lengths
        self.data['input_length'] = self.data[self.text_column_name].apply(
            lambda x: len(self.tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True))
        )
        # self.data = self.data.sort_values(by='input_length', ascending=False).reset_index(drop=True)
        self.data = self.data.sort_values(by='input_length', ascending=False)
        self.changed_order = self.data.index
        self.data = self.data.reset_index(drop=True)
        
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)   

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the given index.

        Args:
        - idx: The index of the item to retrieve.

        Returns:
        A dictionary containing the tokenized input, attention mask, and label tensors.
        """
        row = self.data.iloc[idx]
        label = row[self.label_column_name]
        text = row[self.text_column_name]

        # Tokenize the text
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            # padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        if self.group_column_name:
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long),
                'group': row[self.group_column_name]
                
            }
        
        else:
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        

################################# Utility functions ###############################
def dynamic_collate_fn(batch, tokenizer):
    """
    Collate function for dynamic padding.
    """

    #Dyanamic padding
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # Calculate max length for the current batch
    # max_length = max(len(ids) for ids in input_ids)

    # Pad sequences to the max length
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    if 'group' in batch[0]:
        groups = [item['group'] for item in batch]

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded,
            'label': labels,
            'groups': groups
        }

    else:
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded,
            'label': labels
        }
    
def preprocess_text(text):
        """
        Preprocess texts by removing links, mentions, hashtags, emojis, and stopwords, and lemmatizing words.

        """
        # Convert text to lowercase
        # print(text)
        text = text.lower()

        # Remove links
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)

        # Remove mentions and hashtags
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'#\S+', '', text)

        # Remove emojis
        text = re.sub(r'[\U0001f600-\U0001f650]', '', text)

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Tokenize the text into individual words
        words = nltk.word_tokenize(text)

        # Remove stopwords
        # stop_words = set(stopwords.words('english'))
        # words = [word for word in words if word not in Stopwords_list]

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # Join the words back into a single string
        processed_text = ' '.join(words)

        return processed_text

def split_dataset(df, label_column_name='Sentiment', split=0.8):
    """
    Splits the dataset into train and test sets based on the specified split ratio over each sentiment.
    the values are the values of the label_column_name upon which the split is performed.
    """
    values = df[label_column_name].unique()
    tr_list = []
    te_list = []
    # # Create separate DataFrames for each class
    # Sample an equal number of samples from each class for training and testing
    for value in values:
        class_df = df[df[label_column_name] == value]
        if len(class_df) > 1:
            train_class_df, test_class_df = train_test_split(class_df, train_size=split, random_state=42)
            tr_list.append(train_class_df)
            te_list.append(test_class_df)
        else:
            tr_list.append(class_df)
            te_list.append(pd.DataFrame())

    # Concatenate the sampled data for the final training and testing DataFrames
    train_df = pd.concat(tr_list)
    test_df = pd.concat(te_list)

    return train_df, test_df


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, tokenizer, **kwargs):
        super().__init__(dataset, batch_size=batch_size, collate_fn=None, **kwargs)

        self.tokenizer = tokenizer

    def __iter__(self): 
        # Organize samples by group
        grouped_samples = defaultdict(list)

        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            group_id = sample['group']
            grouped_samples[group_id].append(sample)

        # Sort each group's samples by descending length of 'input_ids'
        for group_id in grouped_samples:
            grouped_samples[group_id].sort(key=lambda x: len(x['input_ids']), reverse=True)

        group_ids = list(grouped_samples.keys())
        num_groups = len(group_ids)

        # Initialize a deque to keep track of available groups
        available_groups = deque(group_ids)

        while available_groups:
            current_batch = []
            current_groups = set()

            for _ in range(self.batch_size):
                if not available_groups:
                    break
                group_id = available_groups.popleft()
                if grouped_samples[group_id]:
                    current_batch.append(grouped_samples[group_id].pop(0))
                    current_groups.add(group_id)
                    if grouped_samples[group_id]:
                        available_groups.append(group_id)

            # Convert to tensors
            if current_batch:
                yield dynamic_collate_fn(current_batch, self.tokenizer)



############################### Main class ###############################

class finetune_roberta:
    """
    Fintune Roberta easy with your dataset
    """
    def __init__(self, model_path=None, tokenizer_path=None, device='gpu'):
        """
        Initialize the model and tokenizer.
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        

        if device.lower()=='gpu'or device.lower() == 'cuda' or device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device_type=device.lower()

        elif device.lower()=='tpu':            
            
            self.device = xm.xla_device()
            self.device_type=device.lower()

        else:
            self.device = device
            self.device_type='Custom'

        self.load_model()

    ###################SAVE AND LOAD CODE#####################
        
    def save_model( self, model_path, tokenizer_path):
        """
        Save the model and tokenizer to specified path.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.model.to(device)
        
        # Save the model and tokenizer
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
    

    def load_model(self):
        """
        Load the model and tokenizer from specified path or if passed directly and if not any of before loads from scratch.
        """

        if isinstance(self.model_path, nn.Module):
            self.model = self.model_path.to(self.device)
            self.tokenizer = self.tokenizer_path
            print('Model and tokenizer taken as input.')
        
        elif self.model_path!=None and self.tokenizer_path!=None:
            # Load the pre-trained model and tokenizer
            self.model = RobertaForSequenceClassification.from_pretrained(self.model_path).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_path, is_split_into_words=True)
            print('Model and tokenizer from specified path loaded.')
        else:
            default_model = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
            # Load the pre-trained model and tokenizer
            self.model = RobertaForSequenceClassification.from_pretrained(default_model).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(default_model, is_split_into_words=True)
            
            print('Default Model and tokenizer loaded.')

        
    ################## Utility #####################
    

    def supply_data(self, df, text_column_name='Review Text', label_column_name='Sentiment', group_column_name=None, batch_size=4, split=0.8):
        """
        Preprocess the data and split into training and testing datasets.
        Supply a dataframe with atleast 2 columns, one for text and one for sentiment labels (0 negative, 1 neutral, 2 positive)
        """
        
        #original data is stored in og_data variable
        self.og_data = df
        print('Preprocessing data...')
        df.loc[:,'Review Text'] = df['Review Text'].apply(preprocess_text)
        print('Data preprocessed.')

        # if data is grouped then split the data into train and test based on each group and it's sentiment ratio
        if group_column_name is not None:
            temp_tr_dfs=[]
            temp_test_dfs=[]
            for group in df[group_column_name].unique():
                # Create separate DataFrames for each group
                group_df = df[df[group_column_name] == group]
                train_df, test_df = split_dataset(group_df, label_column_name, split)

                temp_tr_dfs.append(train_df)
                temp_test_dfs.append(test_df)

            train_df = pd.concat(temp_tr_dfs)
            test_df = pd.concat(temp_test_dfs)

        else: 
            train_df, test_df = split_dataset(df, label_column_name, split)
            

        # # Shuffle the data to ensure randomness
        # train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        # test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # # Randomize the data and split into training and testing datasets
        # # Randomizing allows in distributing the labels evenly across the datasets
        # randomized_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        # train_split=int(len(randomized_df)*split)
        # train_df = randomized_df[:train_split]
        # test_df = randomized_df[train_split:]
      
        print('Creating dataloaders...')
        # Create the training dataset and dataloader

        tr_dataset = SentimentDataset(train_df, self.tokenizer, text_column_name, label_column_name, group_column_name)
        te_dataset = SentimentDataset(test_df, self.tokenizer, text_column_name, label_column_name, group_column_name)
        
       
        # if grouping is already done on data and a column named group is present on dataframe
        if group_column_name:
            tr_dataloader=CustomDataLoader(tr_dataset, batch_size, tokenizer=self.tokenizer)
            te_dataloader=CustomDataLoader(te_dataset, batch_size, tokenizer=self.tokenizer)

        else:
            tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: dynamic_collate_fn(batch, self.tokenizer))
            te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: dynamic_collate_fn(batch, self.tokenizer))
        

        self.te_dataset = te_dataset
        self.tr_dataset = tr_dataset

        print('Dataloaders created.')
        self.tr_dataloader = tr_dataloader
        self.te_dataloader = te_dataloader
    
    ###################TRAINING CODE#####################

    def ready_optimizer_loss_fn(self, lr=2e-5, opt=None, loss=None):
        """
        Ready the optimizer and loss function (IMPORTANT: Run this after ready_model_tokenizer and supply_data and before start_training)
        """
        if opt==None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            print('Using provided optimizer, you sure? (y/n)')
            if input() == 'y':
                self.optimizer = opt
            else:
                return None
        if loss==None:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            print('Using provided loss function, you sure? (y/n)')
            if input() == 'y':
                self.loss_fn = loss
            else:
                return None



    def start_training(self, num_epochs = 3):
        """
        Kickstart the training process
        """

        # Define the number of training steps
        total_steps = len(self.tr_dataloader)  * num_epochs

        # Create a learning rate scheduler with warm-up
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=5e-5,  # Adjust the maximum learning rate as needed
            total_steps=total_steps,
            pct_start=0.1,  # The fraction of steps used for increasing the learning rate
            anneal_strategy='linear',  # You can choose 'linear' or 'cosine' annealing
        )

        loss_list = []

        # Fine-tuning loop
        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
            if self.device_type=='tpu':
                xm.rendezvous('training')
                para_loader = pl.ParallelLoader(self.tr_dataloader, [self.device])
                dataloader = para_loader.per_device_loader(self.device)
            else:
                dataloader = self.tr_dataloader
            
            # Wrap the dataloader with tqdm for the progress bar
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

            for batch in progress_bar:
                # Move the batch tensors to the device for computation
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Compute the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                loss_list.append(loss.item())  # Append loss to the list

                # Backward pass and optimization
                loss.backward()
                if self.device_type=='tpu':
                    xm.optimizer_step(self.optimizer)
                    xm.mark_step()
                else:
                    self.optimizer.step()

                # Update the learning rate using the one-cycle scheduler
                scheduler.step()
                self.optimizer.zero_grad()

                # Update the progress bar with the current loss
                progress_bar.set_postfix({'Loss': loss.item()})

            # Print average loss for the epoch
            avg_loss = total_loss / len(self.tr_dataloader) 
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

        return loss_list


    ###################TESTING CODE#####################

    def get_metrics(self, dataset='Test'):
        """
        Get the accuracy, precision, recall, and F1-score for the model on the given dataset.
        """

        # Evaluation settings
        self.model.eval()
        if dataset=='Test':
            dataset = self.te_dataloader
        elif dataset=='Train':
            dataset = self.tr_dataloader
        else:
            if isinstance(dataset, DataLoader):
                dataset = dataset
            else:
                print('Dataset not recognized, please provide a dataloader or use Test or Train')
                return None
        if self.device_type=='tpu':
            para_loader = pl.ParallelLoader(dataset, [self.device])
            dataset = para_loader.per_device_loader(self.device)

        # Variables to keep track of correct predictions and total samples
        correct_predictions = 0
        total_samples = 0

        # Lists to store true labels and predicted labels
        true_labels = []
        predicted_labels = []

        # Iterate over the dataset
        for batch in dataset:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # Disable gradient calculations
            with torch.no_grad():
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # Predicted labels
            batch_predicted_labels = torch.argmax(logits, dim=1)

            # Update the counts
            correct_predictions += torch.sum(batch_predicted_labels == labels).item()
            total_samples += len(labels)

            # Append true labels and predicted labels
            true_labels.extend(labels.tolist())
            predicted_labels.extend(batch_predicted_labels.tolist())

        # Calculate accuracy
        accuracy = correct_predictions / total_samples
    
        precision = precision_score(true_labels, predicted_labels, average=None)
        recall = recall_score(true_labels, predicted_labels, average=None)
        f1 = f1_score(true_labels, predicted_labels, average=None)
        confusion = confusion_matrix(true_labels, predicted_labels)
        print(f"Accuracy: {accuracy:.4f}")
        for i in range(len(precision)):
            print(f"Precision for class {i-1}: {precision[i]:.4f}")
            print(f"Recall for class {i-1}: {recall[i]:.4f}")
            print(f"F1-score for class {i-1}: {f1[i]:.4f}")
            print()

        print(f"Confusion matrix: \n{confusion}")

        return accuracy, precision, recall, f1

    ###################PREDICTION CODE#####################

    def predict_sentiment(self, text):
        """
        Predict the sentiment of the given text in postive, negative or neutral.
        """
        senti_score = self.sentiment_score(text)

        # Get the predicted sentiment label
        predicted_label = np.argmax(senti_score, axis=1)

        # Convert the predicted label to sentiment score
        sentiment_score = predicted_label.item()

        if sentiment_score == 0:
            return 'Negative'
        elif sentiment_score == 1:
            return 'Neutral'
        else:
            return 'Positive'
        
    def sentiment_score(self, text):
        """
        Predict the sentiment score of the given text in 0, 1 or 2.
        """

        try:
            
            # Preprocess the text
            text = preprocess_text(text)

            # Tokenize the text
            encoded_input = self.tokenizer.encode_plus(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            # Forward pass through the model
            outputs = self.model(**encoded_input)
            sentiment_score = outputs.logits
            sentiment_score = torch.softmax(sentiment_score, dim=1).cpu().detach().numpy()

            return sentiment_score

        except Exception as e:
            # Handle any exceptions and provide meaningful error messages
            error_msg = f"Error occurred during sentiment scoring: {str(e)}"
            raise ValueError(error_msg)

