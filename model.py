import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import dataloader
from torch.backends import cudnn
import random
from tqdm import *
import decimal
from copy import deepcopy
from dataset import return_similar_word_to_vector
from transformer import EncoderLayer
from transformer import PositionalEncoder
from copy import deepcopy

# torch.has_cudnn = False


def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


def load_saved_model(num_of_topics=11, hidden_size=128, topic_hidden_size=32, drop_out_prob=0.6,
                     model_path='./topic-attention'):
    model = TopicAttention(num_of_topics=num_of_topics, hidden_size=hidden_size,
                           topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def examine_context_vectors(num_of_topics, model_path):
    model = load_saved_model(num_of_topics, model_path)
    for vector in model.attn_context:
        print(return_similar_word_to_vector(vector.detach().numpy()))


def get_sentence_weights(model_path, sentence):
    model = load_saved_model()
    sentence = Variable(torch.from_numpy(sentence)).float()
    if torch.has_cudnn:
        sentence.cuda()
    weights, probs = model.return_weights(sentence.unsqueeze(0))
    return weights, probs


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


class TopicAttentionLayer(nn.Module):
    def __init__(self, input_size, num_of_topics):
        super(TopicAttentionLayer, self).__init__()
        self.attention_context = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_of_topics)])
        self.attention_linear_trans = nn.ModuleList([nn.Linear(input_size, input_size)
                                                       for _ in range(num_of_topics)])
        self.num_of_topics = num_of_topics
        self.attention_drop_out = nn.Dropout(.1)

    def forward(self, x, mask=None):
        topic_output = []
        for i in range(self.num_of_topics):
            energy = self.attention_context[i](x)
            if mask is not None:
                energy = energy.masked_fill(mask, -np.inf)
            probs = F.softmax(energy, dim=1)
            probs = self.attention_drop_out(probs)
            out = torch.sum(x * probs, dim=1)

            out = self.attention_linear_trans[i](out)
            out = F.relu(out)

            topic_output.append(out.unsqueeze(1))
        return torch.cat(topic_output, dim=1)

    def regularization_loss(self):
        attn_contexts = torch.cat([self.attention_context[i].weight.view(-1).unsqueeze(0) for i in range(self.num_of_topics)], dim=0)
        normalized_context_values = F.normalize(attn_contexts, dim=1)
        reg_term = torch.mm(normalized_context_values, normalized_context_values.t())
        identity = Variable(torch.eye(self.num_of_topics, self.num_of_topics))
        if torch.has_cudnn:
            identity = identity.cuda()
        return torch.norm(reg_term - identity, p=2.0)


class MemNet(nn.Module):
    def __init__(self, input_size):
        super(MemNet, self).__init__()
        self.query = nn.Parameter(torch.randn(input_size), requires_grad=True)

        self.linear_transform = torch.nn.Conv1d(input_size, input_size, kernel_size=5, padding=2)
        self.bn = torch.nn.BatchNorm1d(input_size)
        self.drop_out = nn.Dropout(.6)

        self.layer_norm = nn.LayerNorm(input_size)

        self.halt_prob = nn.Linear(input_size, 2)
        self.max_hop = 10

    def forward(self, x, mask):
        query = torch.cat([self.query.unsqueeze(0) for _ in range(x.size(0))], dim=0).contiguous()
        halted = torch.zeros(x.size(0)).byte().contiguous()
        halted = Variable(halted)
        if torch.has_cudnn:
            halted = halted.cuda()

        halting_hop = [self.max_hop] * x.size(0)
        n_hop = 0
        while not (halted.any() or n_hop == self.max_hop):
            energies = torch.bmm(x, query.unsqueeze(-1)).squeeze(-1)
            energies = energies.masked_fill(mask.squeeze(-1), -np.inf)
            probs = F.softmax(energies, dim=-1)

            x = F.leaky_relu(self.bn(self.linear_transform(x.transpose(1, 2))).transpose(1, 2))
            x = x.masked_fill(mask, .0)

            o = torch.bmm(x.transpose(1, 2), probs.unsqueeze(-1)).squeeze(-1)
            halt_prob = F.softmax(self.halt_prob(o + query), dim=-1)
            query = o + query * (1 - halted).unsqueeze(1).float()

            for b in range(len(halted)):
                halted[b] = halted[b] or (torch.argmax(halt_prob[b], dim=-1) == 1)
                if halting_hop[b] == self.max_hop and halted[b]:
                    halting_hop[b] = n_hop
            n_hop += 1
        # if not self.training:
        #     print(max_prob)
        #     print(min_prob)
        #     print(sum(halting_hop) / len(halting_hop))
        return query


class TopicAttention(nn.Module):
    def __init__(self, num_of_topics=11, hidden_size=150, input_size=300, classification_size=12, topic_hidden_size=20,
                 drop_out_prob=0.6, sentence_length=65, embeddings=None):
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
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.vocab_size = len(embeddings)

        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=300, padding_idx=self.vocab_size - 1)
        self.embeddings.weight = nn.Parameter(torch.from_numpy(embeddings).float(), requires_grad=False)

        self.dim_reduction = nn.Linear(300, 128)
        self.pos_emb = PositionalEncoder(300, 66, 128, self.vocab_size - 1)

        self.layer_norm = torch.nn.LayerNorm(128)

        self.mem_net = nn.ModuleList([
            MemNet(128) for _ in range(num_of_topics)
        ])
        self.output_layer = torch.nn.Linear(num_of_topics * 128, classification_size)
        self.drop_out = nn.Dropout(p=drop_out_prob)

        self.topic_hidden = nn.ModuleList([
                nn.Linear(128, topic_hidden_size) for _ in range(num_of_topics)])
        self.attn_aspect_context = nn.Parameter(torch.randn(classification_size, topic_hidden_size))
        self.linear_aspect = nn.ModuleList([nn.Linear(topic_hidden_size * num_of_topics, 2 * topic_hidden_size)
                                            for _ in range(self.classification_size)])
        self.encoder = torch.nn.GRU(input_size=input_size, hidden_size=64, batch_first=True, bidirectional=True)

    def forward(self, x_indices, lengths, validate=False):
        mask = Variable(torch.ones(x_indices.size(0), x_indices.size(1), 1))
        if torch.has_cudnn:
            mask = mask.cuda()
        max_len = lengths[torch.argmax(lengths)].item()
        for i, l in enumerate(lengths):
            if l < max_len:
                mask[i, l:] = 0
        mask = mask.byte()
        # attention_mask = 1 - torch.bmm(mask, mask.transpose(1, 2))
        mask = 1 - mask

        x = self.embeddings(x_indices) + self.pos_emb(x_indices)
        x = self.dim_reduction(x)
        x = x.masked_fill(mask, .0)
        x = self.drop_out(x)

        # x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        # x, _ = self.encoder(x)
        # x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # x = self.layer_norm(x)

        x = torch.cat([self.mem_net[i](x, mask) for i in range(self.num_of_topics)], dim=-1)
        x = torch.sigmoid(self.output_layer(x))
        return x, self.regularization_loss()

    def regularization_loss(self):
        attn_contexts = torch.cat([self.mem_net[i].query.unsqueeze(0) for i in range(self.num_of_topics)], dim=0)
        normalized_context_values = F.normalize(attn_contexts, dim=1)
        reg_term = torch.mm(normalized_context_values, normalized_context_values.t())
        identity = Variable(torch.eye(self.num_of_topics, self.num_of_topics))
        if torch.has_cudnn:
            identity = identity.cuda()
        return torch.norm(reg_term - identity, p=2.0)

    def return_weights(self, x):
        weights = []
        existance = []
        x, hidden = self.rnn(x)
        for i in range(self.num_of_topics):
            context = torch.stack([self.attn_context[i]] * x.size(0))
            energy = torch.bmm(x, context.unsqueeze(-1))
            probs = F.softmax(energy, dim=1)
            weights.append(probs)
            out = torch.bmm(x.transpose(1, 2), probs).squeeze(-1)
            out = self.topic_hidden[i](out)
            out = squash(out)
            existance.append(torch.norm(out[0]))
        return weights, existance


class TopicAttentionWithoutSquash(nn.Module):
    def __init__(self, num_of_topics=6, hidden_size=150, input_size=300, classification_size=5, topic_hidden_size=20,
                 drop_out_prob=0.6, sentence_length=65):
        super(TopicAttentionWithoutSquash, self).__init__()
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
        x = self.drop_out(x)
        self.rnn.flatten_parameters()
        x, hidden = self.rnn(x)
        x = self.drop_out(x)
        x_ = None
        for i in range(self.num_of_topics):
            context = torch.stack([self.attn_context[i]] * x.size(0))
            energy = torch.bmm(x, context.unsqueeze(-1))
            probs = F.softmax(energy, dim=1)
            out = torch.bmm(x.transpose(1, 2), probs).squeeze(-1)
            out = self.topic_hidden[i](out)
            out = F.relu(out)
            if x_ is None:
                x_ = out
            else:
                x_ = torch.cat((x_, out), dim=1)
        x = self.output_layer(x_)
        x = torch.sigmoid(x)
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

    def return_weights(self, x):
        weights = []
        existance = []
        x, hidden = self.rnn(x)
        for i in range(self.num_of_topics):
            context = torch.stack([self.attn_context[i]] * x.size(0))
            energy = torch.bmm(x, context.unsqueeze(-1))
            probs = F.softmax(energy, dim=1)
            weights.append(probs)
            out = torch.bmm(x.transpose(1, 2), probs).squeeze(-1)
            out = self.topic_hidden[i](out)
            out = squash(out)
            existance.append(torch.norm(out[0]))
        return weights, existance


class VanillaAttention(nn.Module):
    def __init__(self, hidden_size=150, input_size=300, classification_size=5, drop_out_prob=0.6, embeddings=None):
        super(VanillaAttention, self).__init__()
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.has_cudnn:
            torch.cuda.manual_seed(0)
        cudnn.benchmark = True
        random.seed(0)

        self.vocab_size = len(embeddings)

        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=300, padding_idx=self.vocab_size - 1)
        self.embeddings = self.embeddings.from_pretrained(torch.from_numpy(embeddings).float(), freeze=True)

        self.drop_out = nn.Dropout(drop_out_prob)
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.attn_context = nn.Linear(2 * hidden_size, 1)
        self.hidden_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, classification_size)

    def forward(self, x, validate=False):
        x = self.embeddings(x)
        x = self.drop_out(x)
        x, _ = self.rnn(x)
        x = self.drop_out(x)

        energy = self.attn_context(x)
        probs = F.softmax(energy, dim=1)
        x = torch.bmm(x.transpose(1, 2), probs).squeeze(-1)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        if not self.training:
            x = torch.sigmoid(x)
        return x


class Net:
    def __init__(self, epochs, train_loader, test_loader, validation_loader, learning_rate, model_type,
                 mode, min_delta, patience, num_of_topics=6, hidden_size=150, input_size=300,
                 topic_hidden_size=20, drop_out_prob=0.6, embeddings=None):
        if model_type == 'topic-attention':
            self.model = TopicAttention(num_of_topics=num_of_topics, hidden_size=hidden_size, input_size=input_size,
                                        classification_size=train_loader.get_num_of_classes(),
                                        topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob,
                                        sentence_length=train_loader.get_sentence_length(),
                                        embeddings=embeddings)
        elif model_type == 'vanilla-attention':
            self.model = VanillaAttention(hidden_size=hidden_size, input_size=input_size,
                                          classification_size=train_loader.get_num_of_classes(),
                                          drop_out_prob=drop_out_prob, embeddings=embeddings)
        elif model_type == 'topic-attention-without-squash':
            self.model = TopicAttentionWithoutSquash(num_of_topics=num_of_topics, hidden_size=hidden_size, input_size=input_size,
                                                     classification_size=train_loader.get_num_of_classes(),
                                                     topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob,
                                                     sentence_length=train_loader.get_sentence_length(), embeddings=embeddings)
        if torch.has_cudnn:
            self.model.cuda()

        self.model_type = model_type
        self.num_epochs = epochs
        # self.LossFunction = topkCE(k=k)
        # self.loss_function = nn.CrossEntropyLoss()
        # self.loss_function = nn.MSELoss()
        # self.validate_loss = nn.MSELoss()
        self.loss_function = nn.BCELoss()
        self.validate_loss = nn.BCELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # self.optimizer = torch.optim.Adadelta(self.model.parameters(), weight_decay=1e-5)
        self.early_stopping = EarlyStopping(mode, min_delta, patience)
        # lr_decay = lambda epoch: 1 / (1 + epoch * 0.2)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_decay)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader

        self.validation_data_loader = dataloader.DataLoader(dataset=self.validation_loader,
                                                            batch_size=128,
                                                            shuffle=False,
                                                            num_workers=0,
                                                            collate_fn=collate_fn)
        self.train_data_loader = dataloader.DataLoader(dataset=self.train_loader,
                                                       batch_size=128,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       collate_fn=collate_fn)
        self.test_data_loader = dataloader.DataLoader(dataset=self.test_loader,
                                                      batch_size=128,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      collate_fn=collate_fn)

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
        validate_loss = []
        train_loss = []

        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            total_loss = []
            reg_total_loss = []
            for i, (data, labels, lengths) in enumerate(self.train_data_loader):
                reg_loss = 0.0
                data = self.to_var(data).long()
                labels = self.to_var(labels)
                self.optimizer.zero_grad()
                if self.model_type == 'topic-attention' or self.model_type == 'topic-attention-without-squash':
                    outputs, reg_loss = self.model(data, lengths)
                elif self.model_type == 'vanilla-attention':
                    outputs = self.model(data)
                # print(outputs)
                label_loss = self.loss_function(outputs, labels.squeeze(1))
                if self.model_type == 'topic-attention' or self.model_type == 'topic-attention-without-squash':
                    loss = label_loss + reg_loss
                elif self.model_type == 'vanilla-attention':
                    loss = label_loss

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss.append(float(label_loss))
                reg_total_loss.append(float(reg_loss))


            # print('Epoch [%d/%d], Avr Loss: %.4f, Avr Reg Loss: %.4f'
            #       % (epoch + 1, self.num_epochs, sum(total_loss) / len(total_loss), sum(reg_total_loss) / len(reg_total_loss)))

            if validate is True:
                valid_loss = self.validate()
                validate_loss.append(valid_loss)
                train_loss.append(float(sum(total_loss) / len(total_loss)))
            else:
                # valid_f1 = self.eval_validation_data()[0]
                valid_loss = self.validate()
                result = self.test()
                print('F1: %.4f, precision: %.4f, recall: %.4f' % (result[2], result[0], result[1]))
                print('train loss: ' + str(float(sum(total_loss) / len(total_loss))))
                print('validation loss: ' + str(valid_loss))
                # print(valid_loss)
                train_loss.append(float(sum(total_loss) / len(total_loss)))
                if self.early_stopping.step(valid_loss, deepcopy(self.model)) is True:
                    print('Early stopping at ' + str(epoch))
                    self.model = self.early_stopping.best_model
                    result = self.test()
                    print('F1: ' + str(result[0]))
                    print('P: ' + str(result[1]))
                    print('R: ' + str(result[2]))
                    # break
        if validate is True:
            return train_loss, validate_loss
        else:
            torch.save(self.model.state_dict(), './' + self.model_type)
            return result

    def test(self):
        self.model.eval()
        pred_labels = []
        true_labels = []
        for data, labels, lengths in self.test_data_loader:
            data = data.long()
            labels = labels.long()
            lengths = lengths.long()
            if torch.has_cudnn:
                data = data.cuda()
                labels = labels.cuda()
                lengths = lengths.cuda()
            # print(data)
            if self.model_type == 'topic-attention' or self.model_type == 'topic-attention-without-squash':
                outputs, _ = self.model(data, lengths)
            elif self.model_type == 'vanilla-attention':
                outputs = self.model(data)
            pred_labels += deepcopy(list(outputs.cpu().detach().numpy()))
            true_labels += deepcopy(list(labels.cpu().squeeze(1).detach().numpy()))
        pred_labels = list(pred_labels)
        # print(pred_labels)
        true_labels = list(true_labels)
        threshold = self.eval_validation_data()[1]
        print('Best threshold value: ' + str(threshold))
        return self.calculate_F1(pred_labels, true_labels, threshold)

    def eval_validation_data(self):
        self.model.eval()
        pred_labels = []
        true_labels = []
        best_result = [0.0, 0]
        for data, labels, lengths in self.validation_data_loader:
            data = data.long()
            labels = labels.long()
            lengths = lengths.long()
            if torch.has_cudnn:
                data = data.cuda()
                labels = labels.cuda()
                lengths = lengths.cuda()
            if self.model_type == 'topic-attention' or self.model_type == 'topic-attention-without-squash':
                outputs, _ = self.model(data, lengths)
            elif self.model_type == 'vanilla-attention':
                outputs = self.model(data)
            pred_labels += deepcopy(list(outputs.cpu().detach().numpy()))
            true_labels += deepcopy(list(labels.cpu().squeeze(1).detach().numpy()))
        for threshold in list(frange(0, 1, decimal.Decimal('0.01'))):
            precision, recall, f1 = self.calculate_F1(pred_labels, true_labels, threshold)
            if f1 > best_result[0]:
                best_result[0] = f1
                best_result[1] = threshold
        print('validation f1: ' + str(best_result[0]))
        return best_result

    def calculate_F1(self, pred_labels, true_labels, threshold):
        TP = 0
        FP = 0
        FN = 0
        for idx in range(len(pred_labels)):
            output = pred_labels[idx]
            label = true_labels[idx]
            for i in range(len(list(output))):
                if output[i] >= threshold:
                    if label[i] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if label[i] == 1:
                        FN += 1
        try:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            precision = .0
            recall = .0
            f1 = .0
        # if len(pred_labels) == 642:
        #     print(str(TP) + ' ' + str(FP) + ' ' + str(FN))
        return precision, recall, f1

    def validate(self):
        self.model.eval()
        valid_loss = 0
        count = 0
        for data, labels, lengths in self.validation_data_loader:
            data = data.long()
            labels = labels.float()
            lengths = lengths.long()
            if torch.has_cudnn:
                data = data.cuda()
                labels = labels.cuda()
                lengths = lengths.cuda()
            if self.model_type == 'topic-attention' or self.model_type == 'topic-attention-without-squash':
                outputs, _ = self.model(data, lengths, validate=True)
            elif self.model_type == 'vanilla-attention':
                outputs = self.model(data)
            valid_loss += self.validate_loss(outputs, labels.squeeze(1)).item()
            count += 1
        valid_loss = float(valid_loss) / count
        return valid_loss


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics, model):
        if self.best is None:
            self.best = metrics
            self.best_model = model
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_model = model
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

    def return_best_model(self):
        return self.best_model


def collate_fn(data):
    sentence_lengths = [data[i][2] + 1 for i in range(len(data))]
    sorted_indices = np.argsort(sentence_lengths)[::-1]

    max_sentence_length = max(sentence_lengths)

    sentence_rep = np.array([data[idx][0][:max_sentence_length] for idx in sorted_indices])
    # try:
    label = np.array([data[i][1] for i in sorted_indices])
    sentence_lengths = np.array([sentence_lengths[idx] for idx in sorted_indices])

    sentence_rep = np.stack(sentence_rep)
    label = np.stack(label)
    sentence_lengths = np.stack(np.array(sentence_lengths))

    return torch.from_numpy(sentence_rep), torch.from_numpy(label), torch.from_numpy(sentence_lengths)

