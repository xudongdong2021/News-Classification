import torch
import torch.nn as nn
import torch.nn.functional as F
class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.num_filters_total = args.num_filters * len(args.filter_sizes)
        self.W = nn.Embedding(args.vocab_size, args.embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, args.num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([args.num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, args.num_filters, (size, args.embedding_size)) for size in args.filter_sizes])
        self.args = args

    def forward(self, X):
        embedded_chars = self.W(X)
        embedded_chars = embedded_chars.unsqueeze(1)

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(embedded_chars))
            mp = nn.MaxPool2d((self.args.sequence_length - self.args.filter_sizes[i] + 1, 1))
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(self.args.filter_sizes))
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])
        model = self.Weight(h_pool_flat) + self.Bias
        return model


# class Bert(nn.Module):
