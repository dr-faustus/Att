import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import dataset, dataloader
from torch.backends import cudnn
import random
from tqdm import *


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


class topkCE(nn.Module):
    def __init__(self, k=0.7):
        super(topkCE, self).__init__()
        self.loss = nn.NLLLoss()
        self.k = k
        self.softmax = nn.LogSoftmax(dim=1)
        return

    def forward(self, input, target):
        softmax_result = self.softmax(input)

        loss = Variable(torch.Tensor(1).zero_())
        if torch.has_cudnn:
            loss = loss.cuda()
        for idx, row in enumerate(softmax_result):
            gt = target[idx]
            pred = torch.unsqueeze(row, 0)
            cost = self.loss(pred, gt.unsqueeze(0))
            loss = torch.cat((loss, cost.unsqueeze(0)), 0)
        loss = loss[1:]
        if self.k == 1.0:
            valid_loss = loss
        else:
            index = torch.topk(loss, int(self.k * loss.size()[0]))
            valid_loss = loss[index[1]]
        return torch.mean(valid_loss)


class TopicAttention(nn.Module):
    def __init__(self, num_of_topics=6, hidden_size=150, input_size=300, classification_size=5, topic_hidden_size=20,
                 drop_out_prob=0.6, sentence_length=65):
        super(TopicAttention, self).__init__()
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.has_cudnn:
            torch.cuda.manual_seed(0)
        cudnn.benchmark = True
        random.seed(0)
        self.num_of_topics = num_of_topics
        self.sentence_length = sentence_length
        self.classification_size = classification_size
        self.topic_hidden_size = topic_hidden_size
        self.drop_out = nn.Dropout(p=drop_out_prob)

        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.attn_context = nn.Parameter(torch.randn(num_of_topics, hidden_size * 2))
        self.topic_hidden = nn.ModuleList([
                nn.Linear(hidden_size * 2, topic_hidden_size) for _ in range(num_of_topics)])
        self.output_layer = nn.Linear(topic_hidden_size * num_of_topics, classification_size, bias=True)

        self.attn_aspect_context = nn.Parameter(torch.randn(classification_size, topic_hidden_size))
        self.linear_aspect = nn.ModuleList([nn.Linear(topic_hidden_size * num_of_topics, 2 * topic_hidden_size)
                                            for _ in range(self.classification_size)])

        self.reconstruction_layer = nn.Sequential(
            nn.Linear(topic_hidden_size * num_of_topics, 2 ** 10),
            nn.ReLU(),
            nn.Linear(2 ** 10, 2 ** 12),
            nn.ReLU(),
            nn.Linear(2 ** 12, sentence_length * input_size),
            nn.Sigmoid()
        )
        self.reconstruction_loss_function = nn.MSELoss()

    def forward(self, x, validate=False):
        init_input = x
        x = self.drop_out(x)
        x, hidden = self.rnn(x)
        x = self.drop_out(x)
        context_values = None
        x_ = None
        for i in range(self.num_of_topics):
            context = torch.stack([self.attn_context[i]] * x.size(0))
            if context_values is None:
                context_values = context.unsqueeze(1)
            else:
                context_values = torch.cat((context_values, context.unsqueeze(1)), dim=1)
            energy = torch.bmm(x, context.unsqueeze(-1))
            probs = F.softmax(energy, dim=1)
            out = torch.bmm(x.transpose(1, 2), probs).squeeze(-1)
            out = self.topic_hidden[i](out)
            out = squash(out)
            if x_ is None:
                x_ = out
            else:
                x_ = torch.cat((x_, out), dim=1)
        x = torch.zeros(x_.size(0), self.classification_size)
        for i in range(self.classification_size):
            temp = self.linear_aspect[i](x_)
            temp = squash(temp)
            x[:, i] = torch.norm(temp, dim=-1)

        return x, self.regularization_loss()

    def regularization_loss(self):
        normalized_context_values = F.normalize(self.attn_context, dim=1)
        reg_term = torch.mm(normalized_context_values, normalized_context_values.t())
        identity = Variable(torch.eye(self.num_of_topics, self.num_of_topics))
        if torch.has_cudnn:
            identity = identity.cuda()
        return torch.norm(reg_term - identity)

    def reconstruction_loss(self, hidden_topic, init_input):
        recon_input = self.reconstruction_layer(hidden_topic)
        init_input_flat = None
        for i in range(init_input.size(1)):
            if init_input_flat is None:
                init_input_flat = init_input[:, i]
            else:
                init_input_flat = torch.cat((init_input_flat, init_input[:, i]), dim=1)
        return self.reconstruction_loss_function(recon_input, init_input_flat)


class VanillaAttention(nn.Module):
    def __init__(self, hidden_size=150, input_size=300, classification_size=5, drop_out_prob=0.6):
        super(VanillaAttention, self).__init__()
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.has_cudnn:
            torch.cuda.manual_seed(0)
        cudnn.benchmark = True
        random.seed(0)
        self.drop_out = nn.Dropout(drop_out_prob)
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.attn_context = nn.Parameter(torch.randn(hidden_size * 2))
        self.hidden_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, classification_size)

    def forward(self, x, validate=False):
        x = self.drop_out(x)
        x, hidden = self.rnn(x)
        x = self.drop_out(x)

        energy = torch.bmm(x, torch.stack([self.attn_context] * x.size(0)).unsqueeze(-1))
        probs = F.softmax(energy, dim=1)
        x = torch.bmm(x.transpose(1, 2), probs).squeeze(-1)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        x = F.softmax(x, dim=1)

        return x


class Net:
    def __init__(self, epochs, train_loader, test_loader, validation_loader, learning_rate, model_type,
                 num_of_topics=6, hidden_size=150, input_size=300,
                 topic_hidden_size=20, drop_out_prob=0.6):
        if model_type == 'topic-attention':
            self.model = TopicAttention(num_of_topics=num_of_topics, hidden_size=hidden_size, input_size=input_size,
                                        classification_size=train_loader.get_num_of_classes(),
                                        topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob,
                                        sentence_length=train_loader.get_sentence_length())
        elif model_type == 'vanilla-attention':
            self.model = VanillaAttention(hidden_size=hidden_size, input_size=input_size,
                                          classification_size=train_loader.get_num_of_classes(), drop_out_prob=drop_out_prob)
        if torch.has_cudnn:
            self.model.cuda()
        self.model_type = model_type
        self.num_epochs = epochs
        # self.LossFunction = topkCE(k=k)
        # self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = nn.MSELoss()
        self.validate_loss = nn.MSELoss()
        # self.loss_function = nn.MultiMarginLoss()
        # self.validate_loss = nn.MultiMarginLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # lr_decay = lambda epoch: 1 / (1 + epoch * 0.2)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_decay)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader

    def to_var(self, x):
        # x = torch.from_numpy(x).float()
        try:
            x = x.float()
        except AttributeError:
            x = torch.from_numpy(x).float()
        if torch.has_cudnn:
            x = Variable(x).cuda()
        else:
            x = Variable(x)
        return x

    def train(self, batch_size, validate=False):
        best_result = [0.0, 0]
        validate_loss = []
        train_loss = []
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            total_loss = []
            reg_total_loss = []
            data_loader = dataloader.DataLoader(dataset=self.train_loader,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0)
            for i, (datas, labels) in enumerate(data_loader):
                reg_loss = 0.0
                datas = self.to_var(datas)

                labels = self.to_var(labels)
                self.optimizer.zero_grad()
                if self.model_type == 'topic-attention':
                    outputs, reg_loss = self.model(datas)
                elif self.model_type == 'vanilla-attention':
                    outputs = self.model(datas)

                label_loss = self.loss_function(outputs, labels.squeeze(1))
                if self.model_type == 'topic-attention':
                    loss = label_loss + reg_loss
                elif self.model_type == 'vanilla-attention':
                    loss = label_loss

                loss.backward()
                self.optimizer.step()
                total_loss.append(label_loss)
                reg_total_loss.append(reg_loss)

            # print('Epoch [%d/%d], Avr Loss: %.4f, Avr Reg Loss: %.4f'
            #       % (epoch + 1, self.num_epochs, sum(total_loss) / len(total_loss), sum(reg_total_loss) / len(reg_total_loss)))
            if validate is False:
                f1 = self.test()
                print(f1)
                if f1[0] > best_result[0]:
                    best_result[0] = f1[0]
                    best_result[1] = epoch
            else:
                valid_loss = self.validate()
                validate_loss.append(valid_loss)
                train_loss.append(float(sum(total_loss) / len(total_loss)))
                # print(valid_loss)
        if validate is True:
            return train_loss, validate_loss
        else:
            return self.test()

    def test(self):
        self.model.eval()
        pred_labels = []
        true_labels = []
        for data, label in self.test_loader:
            data = self.to_var(data)
            data = data.unsqueeze(0)
            label = label
            if self.model_type == 'topic-attention':
                output, _ = self.model(data)
            elif self.model_type == 'vanilla-attention':
                output = self.model(data)

            output = output.squeeze(0)
            pred_labels.append(list(output))
            true_labels.append(label)
        best_result = [0.0, 0]
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            TP = 0
            FP = 0
            FN = 0
            for idx in range(len(pred_labels)):
                if idx == 12:
                    continue
                output = pred_labels[idx]
                label = true_labels[idx]
                for i in range(len(list(output))):
                    if float(output[i]) >= threshold:
                        if i in label:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if i in label:
                            FN += 1
            try:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1 = 0.0
            if f1 > best_result[0]:
                best_result[0] = f1
                best_result[1] = threshold

        return best_result

    def validate(self):
        self.model.eval()
        valid_loss = 0
        data_loader = dataloader.DataLoader(dataset=self.validation_loader,
                                            batch_size=len(self.validation_loader),
                                            shuffle=False,
                                            num_workers=0)
        for datas, labels in data_loader:
            datas = self.to_var(datas)
            labels = self.to_var(labels)
            if self.model_type == 'topic-attention':
                outputs, _ = self.model(datas, validate=True)
            elif self.model_type == 'vanilla-attention':
                outputs = self.model(datas, validate=True)
            valid_loss += self.validate_loss(outputs, labels.squeeze(1))
        valid_loss = float(valid_loss.data[0])
        return valid_loss
