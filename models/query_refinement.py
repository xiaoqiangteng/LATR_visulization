import torch
import torch.nn as nn

class QueryRefinement(nn.Module):
    def __init__(self):
        super(QueryRefinement, self).__init__()
        self.fusion_layer = nn.Linear(256, 256)  # 假设 hidden_dim = 256
    
    def forward(self, encoded_queries, additional_features=None):
        """
        对 Transformer 编码后的查询进行精化。
        
        Args:
            encoded_queries (Tensor): Transformer 编码后的查询 (batch_size, num_queries, hidden_dim)
            additional_features (Tensor, optional): 其他层次的特征 (batch_size, num_queries, hidden_dim)
        
        Returns:
            refined_queries (Tensor): 精化后的查询
        """
        # 跨层特征融合：如果有附加特征，则将其融合
        if additional_features is not None:
            # 将编码后的查询和附加特征进行融合
            refined_queries = encoded_queries + additional_features
            refined_queries = self.fusion_layer(refined_queries)  # 融合后的查询通过线性层处理
        else:
            refined_queries = encoded_queries  # 如果没有附加特征，仅返回原始查询

        return refined_queries
