import torch
import torch.nn as nn

class HierarchicalQueryFiltering(nn.Module):
    def __init__(self, top_k=100):
        """
        Args:
            top_k (int): 保留最显著的 top_k 个查询
        """
        super(HierarchicalQueryFiltering, self).__init__()
        self.top_k = top_k  # 保留的查询数量

    def forward(self, queries, query_scores):
        """
        执行分层查询过滤，保留最显著的查询。
        
        Args:
            queries (Tensor): 输入的查询 (batch_size, num_queries, hidden_dim)
            query_scores (Tensor): 查询的得分 (batch_size, num_queries)

        Returns:
            filtered_queries (Tensor): 过滤后的查询 (batch_size, top_k, hidden_dim)
        """
        # 获取每个 batch 中的 top_k 查询索引
        top_k_indices = query_scores.topk(self.top_k, dim=1).indices
        
        # 通过索引提取对应的查询
        batch_size, num_queries, hidden_dim = queries.shape
        filtered_queries = torch.gather(queries, 1, top_k_indices.unsqueeze(-1).expand(batch_size, self.top_k, hidden_dim))

        return filtered_queries
