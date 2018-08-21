from plotting import *
from dataset import get_data_loaders
from dataset import SimpleDataset
from dataset import DataLoader
from model import get_sentence_weights
from model import examine_context_vectors

fontP = FontProperties()
fontP.set_size('small')

plt.switch_backend('agg')

validation_percentage = 0.05


def validate(learning_rate, batch_size, dataset_name, model_type):
    train_loader, validation_loader, test_loader = get_data_loaders(validation_percentage, dataset_name)
    net = Net(250, train_loader, test_loader, validation_loader, learning_rate, model_type=model_type)
    train_loss, test_loss = net.train(batch_size=batch_size, validate=True)


def test(learning_rate, batch_size, dataset_name, model_type, early_stopping_mode, early_stopping_min_delta, early_stopping_patience):
    train_loader, validation_loader, test_loader = get_data_loaders(validation_percentage, dataset_name)
    net = Net(300, train_loader, test_loader, validation_loader, learning_rate, model_type,
              early_stopping_mode, early_stopping_min_delta, early_stopping_patience,
              input_size=input_size, num_of_topics=num_of_topics, hidden_size=hidden_size,
              topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob)
    result = net.train(batch_size=batch_size, validate=False)
    print(result)


def sentence_weight_examine(idx):
    dataset = SimpleDataset(validation_percentage=validation_percentage, dataset_name=dataset_name)
    loader = DataLoader(data='train', simple_dataset=dataset, dataset_name='sem-2016', padding=False)
    item = loader[idx][0]
    print(loader[idx][1])
    weights, existance = get_sentence_weights('./topic-attention', item)
    print(dataset.train_original_sentence[idx])
    print(list(weights))
    print(list(existance))


batch_size_list = [16, 32, 64, 128, 256]
num_of_topics_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
learning_rate_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
hidden_size_list = [50, 100, 150, 200, 300]
topic_hidden_size_list = [5, 10, 20, 50, 100]
drop_out_prob_list = [0.2, 0.4, 0.5, 0.6, 0.7]

learning_rate = 0.001
batch_size = 128

num_of_topics = 11
hidden_size = 150
input_size = 300
topic_hidden_size = 20
drop_out_prob = 0.6

dataset_name = 'sem-2016'
model_type = 'topic-attention'

early_stopping_mode = 'min'
early_stopping_min_delta = 0
early_stopping_patience = 10

# num_of_topics_validation(300, dataset_name, learning_rate, batch_size, model_type, num_of_topics_list, hidden_size, topic_hidden_size, drop_out_prob)
# dataset_name = 'sem-2014'
# num_of_topics_validation(300, dataset_name, learning_rate, batch_size, model_type, num_of_topics_list, hidden_size, topic_hidden_size, drop_out_prob)

test(learning_rate, batch_size, dataset_name, model_type, early_stopping_mode, early_stopping_min_delta, early_stopping_patience)
# sentence_weight_examine(6)
# examine_context_vectors(15, './topic-attention')
