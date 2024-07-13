import torch
import testtool

def test_PairwiseDistance():
    class TestPairwiseDistance(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(8 * 8, 25)
            self.fc2 = torch.nn.Linear(25, 10)
            self.pairwise_distance = torch.nn.PairwiseDistance(p = 2)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            distance = self.pairwise_distance(x, torch.randn(1, 10))
            return distance
    
    testtool.testKQI(TestPairwiseDistance(), torch.randn(1, 8 * 8))


def test_CosineSimilarity():
    class TestCosineSimilarity(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(8 * 8, 25)
            self.fc2 = torch.nn.Linear(25, 10)
            self.cos_similarity = torch.nn.CosineSimilarity(dim = 1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            cos_sim = self.cos_similarity(x, torch.randn(1, 10))
            return cos_sim
    
    testtool.testKQI(TestCosineSimilarity(), torch.randn(1, 8 * 8))


if __name__ == '__main__':
    test_PairwiseDistance()
    test_CosineSimilarity()
