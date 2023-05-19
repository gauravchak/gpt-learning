class TwoTowerModel(nn.Module):
    def __init__(self, D_u, D_i, D_s, D_d, N, num_user_towers, num_item_towers):
        super(TwoTowerModel, self).__init__()

        self.num_user_towers = num_user_towers
        self.num_item_towers = num_item_towers

        # Define user MLP
        self.MLP_user = nn.Sequential(
            nn.Linear(D_u + D_i + D_s, D_i*(num_user_towers - 2)),
        )

        # Define user history MLP with attention
        self.MLP_user_history = nn.MultiheadAttention(embed_dim=D_i, num_heads=1)

        # Define a user agnostic parameter.
        self.user_agnostic_param = nn.Parameter(torch.randn(D_i))

        # Define item MLP
        self.MLP_item = nn.Sequential(
            nn.Linear(D_i + D_d, D_i*num_item_towers)
        )

        # Define W_u
        self.W_u = nn.Linear(D_u, D_i*num_user_towers)

    def forward(self, user_id, user_history, user_static, item_id, item_dense):
        B = user_id.shape[0]

        # Cross attention on user_history with item_id as query
        item_id_expanded = item_id.unsqueeze(1)
        out_user_esuhm, _ = self.MLP_user_history(query=item_id_expanded, key=user_history, value=user_history)
        out_user_esuhm = out_user_esuhm.squeeze(1)

        # Pass user features through MLP
        user_features = torch.cat([user_id, out_user_esuhm, user_static], dim=-1)
        # Output from user features
        out1 = self.MLP_user(user_features).view(B, self.num_user_towers - 2, D_i)
        # Expand user_agnostic_param from (D_i,) to (B, 1, D_i) and concatenate with user_embeddings
        expanded_param = self.user_agnostic_param.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)

        # Concatenate out1, out_user_esuhm, and expanded_param to create out_user
        out_user = torch.cat([out1, out_user_esuhm.unsqueeze(1), expanded_param], dim=1)

        # Create mult_user_embedding [B, num_user_towers * D_i]
        mult_user_embedding = out_user.view(B, -1)

        # Pass item features through MLP
        item_features = torch.cat([item_id, item_dense], dim=-1)
        # Create mult_item_embedding [B, num_item_towers * D_i]
        mult_item_embedding = self.MLP_item(item_features).view(B, -1)

        return mult_user_embedding, mult_item_embedding
