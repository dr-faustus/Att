from model import Net
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')

plt.switch_backend('agg')


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