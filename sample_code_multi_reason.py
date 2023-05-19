class TwoTowerModel(nn.Module):
    """
    Depends on:
      - UserHistoryEncoder : examples of this are sum pooling, weighted sum pooling where wt is a function of 
        position and timegap.
    """
    def __init__(self, D_u, D_i, D_s, D_d, N_very_recent, N_medium_term, N_long_term, num_user_towers, num_item_towers):
        """
        Args:
            D_u: dimension of sparse viewer id feature
            D_i: dimension of sparse item id feature and also the output dimension
            D_s: dimension of the output of user sparse embedding arch
            D_d: dimension of item dense features
            N_very_recent: number of items in user history limited to most recent DVV event
            N_medium_term: number of items in user history limited to most recent DVV event
            N_long_term: number of items in user history (up to 1024)
            num_user_towers: final number of user towers >= 6 : 3 for user history, 1 for popularity, 1 for Pass 0 and 1 for TTSN
            num_item_towers: final number of item towers
        """
        super(TwoTowerModel, self).__init__()

        self.num_user_towers = num_user_towers
        self.num_item_towers = num_item_towers

        # Define user MLP for all but last 2 user towers.
        self.MLP_user = nn.Sequential(
            nn.Linear(D_u + D_s, D_i*(num_user_towers - 2)),
        )

        # Define user history encoder. Either Sum pooling or weighted sum pooling where wt
        # is function of timegap, watch time, DVV of that watch.
        self.MLP_user_history_s = UserHistoryEncoder(num_items=N_very_recent, embedding_dim=D_i)
        self.MLP_user_history_m = UserHistoryEncoder(num_items=N_medium_term, embedding_dim=D_i)
        self.MLP_user_history_l = UserHistoryEncoder(num_items=N_long_term, embedding_dim=D_i)

        # Define a user agnostic parameter.
        self.user_agnostic_param = nn.Parameter(torch.randn(D_i))

        # Define item MLP
        self.MLP_item = nn.Sequential(
            nn.Linear(D_i + D_d, D_i*num_item_towers)
        )

        # Define W_u
        self.W_u = nn.Linear(D_u, D_i*num_user_towers)
        # Define loss computing MLP with attention
        self.MLP_user_with_item = nn.MultiheadAttention(embed_dim=D_i, num_heads=1)


    def forward(self, user_id, user_history_s, user_history_m, user_history_l, user_static, item_id, item_dense):
        B = user_id.shape[0]

        # Cross attention on user_history with item_id as query
        item_id_expanded = item_id.unsqueeze(1)
        enc_user_history_s = self.MLP_user_history_s(user_history_s)  # [B, N_very_recent, D_i] --> [B, D_i] 
        enc_user_history_m = self.MLP_user_history_s(user_history_m)  # [B, N_medium_term, D_i] --> [B, D_i] 
        enc_user_history_l = self.MLP_user_history_s(user_history_l)  # [B, N_long_term, D_i] --> [B, D_i] 

        # Pass user features through MLP
        user_features = torch.cat([user_id, user_static], dim=-1)
        # Output from user features
        out1 = self.MLP_user(user_features).view(B, self.num_user_towers - 4, D_i)
        # Expand user_agnostic_param from (D_i,) to (B, 1, D_i) and concatenate with user_embeddings
        user_agnostic_emb = self.user_agnostic_param.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)

        # Concatenate out1, out_user_esuhm, and expanded_param to create out_user
        out_user = torch.cat(
            [
                out1, 
                enc_user_history_s.unsqueeze(1), 
                enc_user_history_m.unsqueeze(1), 
                enc_user_history_l.unsqueeze(1), 
                user_agnostic_emb
            ], dim=1
        )

        # Create mult_user_embedding [B, num_user_towers, D_i]
        mult_user_embedding = out_user.view(B, self.num_user_towers, -1)

        # Pass item features through MLP
        item_features = torch.cat([item_id, item_dense], dim=-1)
        # Create mult_item_embedding [B, num_item_towers, D_i]
        mult_item_embedding = self.MLP_item(item_features).view(B, self.num_item_towers, -1)

        # Compute losses
        # 1, -4, Softmax ... V2V short term
        # 1, -3, Softmax ... V2V medium term
        # 1, -2, Softmax ... V2V long term
        # 1, -2, Softmax ... V2V long term
        total_loss = self._compute_losses(mult_user_embedding, mult_item_embedding, loss_configs)
        return mult_user_embedding, mult_item_embedding, total_loss

    def _compute_losses(self, mult_user_embedding, mult_item_embedding, loss_configs):
        total_loss = 0
        item_embedding = torch.sum(mult_item_embedding, dim=1)  # [B, D_i]
        for config in loss_configs:
            # config.u_idx which user tower to use
            # config.loss_func how to compute loss of that logit.
            # config.wt input weight of that loss. P1: Use learnable loss weights on top of this.
            user_embedding = mult_user_embedding[:, config.u_idx, :]  # [B, D_i]
            loss = config.loss_func(user_embedding, item_embedding)  # [B]
            total_loss += config.wt * loss
        # Reshape tensors for batch matrix multiplication
        item_e_reshaped = item_embedding.unsqueeze(2)  # shape: [B, D, 1]
        mult_user_e_reshaped = mult_user_embedding.permute(0, 2, 1)  # shape: [B, D, T]

        # Compute dot products
        dots = torch.bmm(mult_user_e_reshaped, item_e_reshaped).squeeze(2)  # shape: [B, T]

        # Compute probabilities using softmax
        probs = torch.softmax(dots, dim=1)  # shape: [B, T]

        # Compute logit
        logit = (probs * dots).sum(dim=1)  # shape: [B]
        total_loss += wt_TTSN * TTSN_loss(logit)
        return total_loss
