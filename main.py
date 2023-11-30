import argparse
from functools import partial

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def preprocessing(input_text, tokenizer):
    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )


class MyDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        dataset = list()
        index = 0
        for data in sentences:
            tokens = data.split(' ')
            labels_id = labels[index]
            index += 1
            dataset.append((tokens, labels_id))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self.sentences)


def my_collate(batch, tokenizer):
    tokens, label_ids = map(list, zip(*batch))

    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=320,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    return text_ids, torch.tensor(label_ids)


def load_dataset(train_batch_size, test_batch_size, workers):
    df = pd.read_csv('movie_data.csv')
    text = df.review.values
    labels = df.sentiment.values

    # split train_set and test_set, random state = 0
    train_text, test_text, train_label, test_label = train_test_split(text, labels, train_size=0.8, random_state=0)

    train_set = MyDataset(train_text, train_label)
    test_set = MyDataset(test_text, test_label)

    # DataLoader
    collate_fn = partial(my_collate, tokenizer=tokenizer)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                              collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
                             collate_fn=collate_fn, pin_memory=True)
    return train_loader, test_loader


# Try to use the softmax、relu、tanh and logistic
class Lstm_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.Lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=320,
                            num_layers=1,
                            batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        lstm_output, _ = self.Lstm(tokens)
        outputs = lstm_output[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
    base_model = AutoModel.from_pretrained('roberta-base')
    model = Lstm_Model(base_model, 2, 768)
    model.to(args.device)

    # get movie data
    train_dataloader, validation_dataloader = load_dataset(
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        workers=args.workers)

    _params = filter(lambda x: x.requires_grad, model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(_params, lr=args.lr, weight_decay=args.weight_decay)

    l_acc, l_trloss, l_teloss, l_epo = [], [], [], []
    best_loss, best_acc = 0, 0

    for epoch in range(args.num_epoch):

        # start train
        train_loss, n_correct, n_train = 0, 0, 0
        model.train()
        # create progress bar
        for inputs, targets in tqdm(train_dataloader):
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            targets = targets.to(args.device)
            predicts = model(inputs)
            loss = criterion(predicts, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)
        train_loss, train_acc = train_loss / n_train, n_correct / n_train

        # Val
        val_loss, n_correct, n_test = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(validation_dataloader):
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                targets = targets.to(args.device)
                predicts = model(inputs)
                loss = criterion(predicts, targets)

                val_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)
        val_loss, val_acc = val_loss / n_test, n_correct / n_test

        l_epo.append(epoch), l_acc.append(val_acc), l_trloss.append(train_loss), l_teloss.append(val_loss)
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss = val_acc, val_loss
        print('{}/{} - {:.2f}%'.format(epoch + 1, args.num_epoch, 100 * (epoch + 1) / args.num_epoch))
        print('[Train] loss: {:.4f}, accuracy: {:.2f}'.format(train_loss, train_acc * 100))
        print('[Validation] loss: {:.4f}, accuracy: {:.2f}'.format(val_loss, val_acc * 100))
    print('best loss: {:.4f}, best accuracy: {:.2f}'.format(best_loss, best_acc * 100))
