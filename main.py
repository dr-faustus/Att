# from dataset import *
from semeval2016_dataset import *
from model import Net
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')

plt.switch_backend('agg')


def prepare_data_set(validation_percentage, word_embedding):
    # dataset = SimpleDataset('../datas/ABSA14_Restaurants_Train.xml',
    #                         '../datas/Restaurants_Test_Data_phaseB.xml', validation_percentage=validation_percentage)
    dataset = SimpleDataset('../datas/ABSA16_Restaurants_Train_SB1_v2.xml',
                            '../datas/EN_REST_SB1_TEST.xml.gold', validation_percentage=validation_percentage)
    train_loader = DataLoader(data='train', word_embeddings=word_embedding, simple_dataset=dataset)
    validation_loader = DataLoader(data='valid', word_embeddings=word_embedding, simple_dataset=dataset)
    test_loader = DataLoader(data='test', word_embeddings=word_embedding, simple_dataset=dataset)
    return train_loader, validation_loader, test_loader


def batch_size_validation_plot(num_of_epochs, word_embedding, learning_rate, batch_list):
    fig, ax = plt.subplots()
    for batch_size in batch_list:
        train_loader, validation_loader, test_loader = prepare_data_set(0.1, word_embedding)
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate)
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        ax.plot(np.arange(1, num_of_epochs + 1, 1), train_loss, '--', label=str(batch_size) + ' train')
        ax.plot(np.arange(1, num_of_epochs + 1, 1), validation_loss, '-', label=str(batch_size) + ' validation')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large', prop=fontP)
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.savefig('batch_size')


def number_of_word_embedding_validation_plot(num_of_epochs, word_embedding_list, learning_rate, batch_size, rho, k):
    fig, ax = plt.subplots()
    for word_embedding in word_embedding_list:
        train_loader, validation_loader, test_loader = prepare_data_set(0.1, word_embedding)
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate)
        train_loss, validation_loss = net.train(batch_size=batch_size)
        ax.plot(np.arange(1, num_of_epochs + 1, 1), train_loss, '--', label=str(word_embedding)+' train')
        ax.plot(np.arange(1, num_of_epochs + 1, 1), validation_loss, '-', label=str(word_embedding)+' validation')

    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large', prop=fontP)
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.savefig('word_embeddings')


def rho_validation_plot(num_of_epochs, word_embedding, learning_rate, batch_size, rho_list, k):
    fig, ax = plt.subplots()
    for rho in rho_list:
        train_loader, validation_loader, test_loader = prepare_data_set(0.1, word_embedding)
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate)
        train_loss, validation_loss = net.train(batch_size=batch_size)
        ax.plot(np.arange(1, num_of_epochs + 1, 1), train_loss, '--', label=str(rho) + ' train')
        ax.plot(np.arange(1, num_of_epochs + 1, 1), validation_loss, '-', label=str(rho) + ' validation')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large', prop=fontP)
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.savefig('rho')


def k_validation_plot(num_of_epochs, word_embedding, learning_rate, batch_size):
    fig, ax = plt.subplots()
    for k in k_list:
        train_loader, validation_loader, test_loader = prepare_data_set(0.1, word_embedding)
        net = Net(num_of_epochs, train_loader, validation_loader, test_loader, learning_rate)
        train_loss, validation_loss = net.train(batch_size=batch_size)
        ax.plot(np.arange(1, num_of_epochs + 1, 1), train_loss, '--', label=str(k) + ' train')
        ax.plot(np.arange(1, num_of_epochs + 1, 1), validation_loss, '-', label=str(k) + ' validation')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large', prop=fontP)
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.savefig('k')


def num_of_topics_validation_plot(num_of_epochs, word_embedding, learning_rate, batch_size, num_of_topics_list):
    fig, ax = plt.subplots()
    for num_of_topics in num_of_topics_list:
        train_loader, validation_loader, test_loader = prepare_data_set(0.1, word_embedding)
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate, num_of_topics=num_of_topics)
        train_loss, validation_loss = net.train(batch_size=batch_size)
        ax.plot(np.arange(1, num_of_epochs + 1, 1), train_loss, '--', label=str(rho) + ' train')
        ax.plot(np.arange(1, num_of_epochs + 1, 1), validation_loss, '-', label=str(rho) + ' validation')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large', prop=fontP)
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.savefig('topics')


def learning_rate_validation_plot(num_of_epochs, word_embedding, learning_rate_list, batch_size, num_of_topics):
    fig, ax = plt.subplots()
    for learning_rate in learning_rate_list:
        train_loader, validation_loader, test_loader = prepare_data_set(0.1, word_embedding)
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate, num_of_topics=num_of_topics)
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        ax.plot(np.arange(1, num_of_epochs + 1, 1), train_loss, '--', label=str(batch_size) + ' train')
        ax.plot(np.arange(1, num_of_epochs + 1, 1), validation_loss, '-', label=str(batch_size) + ' validation')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large', prop=fontP)
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.savefig('learning_rate')


def hidden_size_validation_plot(num_of_epochs, word_embedding, learning_rate, batch_size, num_of_topics, hidden_size_list):
    fig, ax = plt.subplots()
    for hidden_size in hidden_size_list:
        train_loader, validation_loader, test_loader = prepare_data_set(0.1, word_embedding)
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate,
                  num_of_topics=num_of_topics, hidden_size=hidden_size)
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        ax.plot(np.arange(1, num_of_epochs + 1, 1), train_loss, '--', label=str(batch_size) + ' train')
        ax.plot(np.arange(1, num_of_epochs + 1, 1), validation_loss, '-', label=str(batch_size) + ' validation')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large', prop=fontP)
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.savefig('hidden_size')


def context_size_validation_plot(num_of_epochs, word_embedding, learning_rate, batch_size, num_of_topics, hidden_size, context_size_list):
    fig, ax = plt.subplots()
    for context_size in context_size_list:
        train_loader, validation_loader, test_loader = prepare_data_set(0.1, word_embedding)
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate,
                  num_of_topics=num_of_topics, hidden_size=hidden_size, context_size=context_size)
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        ax.plot(np.arange(1, num_of_epochs + 1, 1), train_loss, '--', label=str(batch_size) + ' train')
        ax.plot(np.arange(1, num_of_epochs + 1, 1), validation_loss, '-', label=str(batch_size) + ' validation')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large', prop=fontP)
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.savefig('context_size')


def topic_hidden_size_validation_plot(num_of_epochs, word_embedding, learning_rate, batch_size, num_of_topics, hidden_size, context_size, topic_hidden_size_list):
    fig, ax = plt.subplots()
    for topic_hidden_size in topic_hidden_size_list:
        train_loader, validation_loader, test_loader = prepare_data_set(0.1, word_embedding)
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate,
                  num_of_topics=num_of_topics, hidden_size=hidden_size, context_size=context_size,
                  topic_hidden_size=topic_hidden_size)
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        ax.plot(np.arange(1, num_of_epochs + 1, 1), train_loss, '--', label=str(batch_size) + ' train')
        ax.plot(np.arange(1, num_of_epochs + 1, 1), validation_loss, '-', label=str(batch_size) + ' validation')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large', prop=fontP)
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.savefig('topic_hidden_size')


def drop_out_validation_plot(num_of_epochs, word_embedding, learning_rate, batch_size, num_of_topics, hidden_size,
                             context_size, topic_hidden_size, drop_out_prob_list):
    fig, ax = plt.subplots()
    for drop_out_prob in drop_out_prob_list:
        train_loader, validation_loader, test_loader = prepare_data_set(0.1, word_embedding)
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate,
                  num_of_topics=num_of_topics, hidden_size=hidden_size, context_size=context_size,
                  topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob)
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        ax.plot(np.arange(1, num_of_epochs + 1, 1), train_loss, '--', label=str(batch_size) + ' train')
        ax.plot(np.arange(1, num_of_epochs + 1, 1), validation_loss, '-', label=str(batch_size) + ' validation')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large', prop=fontP)
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.savefig('drop_out')


def validate(word_embedding, k, rho, learning_rate, batch_size):
    train_loader, validation_loader, test_loader = prepare_data_set(0.1, word_embedding)
    net = Net(250, train_loader, test_loader, validation_loader, learning_rate)
    best = net.train(batch_size=batch_size, validate=True)


def test(word_embedding, k, rho, learning_rate, batch_size):
    train_loader, _, test_loader = prepare_data_set(0.0, word_embedding)
    net = Net(300, train_loader, test_loader, _, learning_rate, input_size=input_size,
              num_of_topics=num_of_topics, hidden_size=hidden_size, context_size=context_size,
              topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob)
    best = net.train(batch_size=batch_size, validate=False)
    print(best)


k_list = [0.2, 0.4, 0.6, 0.8, 1.0]

batch_size_list = [16, 32, 64, 128, 256]
num_of_topics_list = [6, 8, 10, 15, 20]
learning_rate_list = [0.001, 0.0007, 0.0005, 0.0001]
hidden_size_list = [50, 100, 150, 200, 300]
context_size_list = [100, 200, 300, 500]
topic_hidden_size_list = [5, 10, 20, 50, 100]
drop_out_prob_list = [0.2, 0.4, 0.5, 0.6, 0.7]
# word_embedding = ['sg', 'cbow', 'glove']
# word_embedding = ['sg', 'glove']
word_embedding = ['sg']
k = 1.0
rho = 0.0
learning_rate = 0.001
batch_size = 128

num_of_topics = 15
hidden_size = 150
input_size = 300
classification_size = 5
context_size = 300
topic_hidden_size = 20
drop_out_prob = 0.6

# validate(word_embedding, k, rho, learning_rate, batch_size)
# num_of_topics_validation_plot(300, word_embedding, learning_rate, batch_size, num_of_topics_list)
# hidden_size_validation_plot(300, word_embedding, learning_rate, batch_size, num_of_topics, hidden_size_list)
# context_size_validation_plot(300, word_embedding, learning_rate, batch_size, num_of_topics, hidden_size, context_size_list)
# topic_hidden_size_validation_plot(300, word_embedding, learning_rate, batch_size, num_of_topics, hidden_size, context_size, topic_hidden_size_list)
# drop_out_validation_plot(300, word_embedding, learning_rate, batch_size, num_of_topics, hidden_size,
#                          context_size, topic_hidden_size, drop_out_prob_list)
# learning_rate_validation_plot(300, word_embedding, learning_rate_list, batch_size, num_of_topics)
# batch_size_validation_plot(300, word_embedding, learning_rate, batch_size_list, rho, k)
test(word_embedding, k, rho, learning_rate, batch_size)
