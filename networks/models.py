import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, input_ids, labels, negative_samples):
        # 문맥 단어들의 임베딩을 조회
        context_embeds = self.embeddings(input_ids)
        # 타겟 단어의 임베딩 조회
        target_embeds = self.out_embeddings(labels)
        # 네거티브 샘플의 임베딩 조회
        negative_embeds = self.out_embeddings(negative_samples)

        # 3개의 단어를 평균함 == 평균은 동등 == 즉 모든 feature에 동등한 기회를 주겠다
        context_embeds_mean = torch.mean(context_embeds, dim=1)
        # 타겟 단어에 대한 유사도 관계 수치화
        positive_score = torch.bmm(target_embeds, context_embeds_mean.unsqueeze(2)).squeeze()
        # 네거티브 샘플에 대한 유사도 관계 수치화
        negative_score = torch.bmm(negative_embeds, context_embeds_mean.unsqueeze(2)).squeeze()
        return positive_score, negative_score

    def predict(self, input_ids):
        context_embeds = self.embeddings(input_ids)
        return context_embeds
