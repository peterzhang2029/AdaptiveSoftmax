from torch import nn
from torch.nn import functional as F
from adaptive_softmax import *

class LanguageModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_input,
                 n_hidden,
                 n_layer,
                 drop_rate=0.25,
                 adaptive_softmax=True,
                 cutoff=[2000, 10000]):
        super(LanguageModel, self).__init__()
        self.drop_rate = drop_rate
        self.embedding = nn.Embedding(vocab_size, n_input)

        self.rnn = nn.LSTM(n_input, n_hidden, n_layer, batch_first=True)

        if adaptive_softmax:
            cutoff.append(vocab_size)
            self.linear = AdaptiveSoftmax(n_hidden, cutoff)
        else:
            self.linear = nn.Linear(n_hidden, vocab_size)

        self.adaptive_softmax = adaptive_softmax

        self.init_weights()

        self.n_layer = n_layer
        self.n_hidden = n_hidden

    def forward(self, input, hidden, target=None, training=True):
        embed = self.embedding(input)
        embed = F.dropout(embed, self.drop_rate, training)
        output, hidden = self.rnn(embed, hidden)
        output = F.dropout(output, self.drop_rate, training)

        if self.adaptive_softmax:
            self.linear.set_target(target.data)

        linear = self.linear(output.contiguous() \
                .view(output.size(0) * output.size(1), output.size(2)))

        return linear, hidden

    def log_prob(self, input, hidden, target):
        embed = self.embedding(input)
        output, hidden = self.rnn(embed, hidden)
        linear = self.linear.log_prob(output.contiguous() \
                .view(output.size(0) * output.size(1), output.size(2)))

        return linear, hidden

    def init_weights(self):
        init = 0.1

        self.embedding.weight.data.uniform_(-init, init)

        if not self.adaptive_softmax:
            nn.init.xavier_normal(self.linear.weight)
            self.linear.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        return (Variable(weight.new(self.n_layer,
                    batch_size, self.n_hidden).zero_()),
                Variable(weight.new(self.n_layer,
                    batch_size, self.n_hidden).zero_()))
