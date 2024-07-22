import torch
import testtool

def test_AdaptiveLogSoftmaxWithLoss():
    class TestAdaptiveLogSoftmaxWithLoss(torch.nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, cutoffs):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
            self.rnn = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.adaptive_softmax = torch.nn.AdaptiveLogSoftmaxWithLoss(in_features=hidden_dim, n_classes=vocab_size, cutoffs=cutoffs)

        def forward(self, x):
            embeds = self.embedding(x)
            output, _ = self.rnn(embeds)
            output = output.contiguous().view(-1, output.size(2)) 
            loss = self.adaptive_softmax(output,  torch.randint(0, vocab_size, (batch_size, seq_length)).view(-1)).loss
            return loss
    
    vocab_size = 100
    embed_dim = 64
    hidden_dim = 32
    cutoffs = [20, 40, 60]

    batch_size = 2
    seq_length = 3
    inputs = torch.randint(0, vocab_size, (batch_size, seq_length))

    testtool.testKQI(TestAdaptiveLogSoftmaxWithLoss(vocab_size, embed_dim, hidden_dim, cutoffs), inputs)

if __name__ == '__main__':
    test_AdaptiveLogSoftmaxWithLoss()
