import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from layers import GCN, DenseModel
from utils import sample_n


class SeeGera_Independent(nn.Module):
    def __init__(self,
                 num_nodes=2708,
                 input_dim=1433,
                 num_hidden=128,
                 out_dim=512,
                 noise_dim=5,
                 K=1,
                 J=1,
                 dropout=0.,
                 device=None,
                 encoder_type='gcn',
                 encoder_layers=2,
                 decoder_type='gcn',
                 decoder_layers=2):
        super(SeeGera_Independent, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        if self.encoder_type == 'gcn':
            self.node_mu_nn = GCN(in_dim=input_dim + noise_dim,
                                  hid_dim=num_hidden,
                                  out_dim=out_dim,
                                  n_layers=encoder_layers,
                                  dropout=dropout)
            self.node_var_nn = GCN(in_dim=input_dim,
                                   hid_dim=num_hidden,
                                   out_dim=out_dim,
                                   n_layers=encoder_layers,
                                   dropout=dropout)
        else:
            assert self.encoder_type == 'mlp'
            self.node_mu_nn = DenseModel(in_dim=input_dim + noise_dim,
                                         num_hidden=num_hidden,
                                         out_dim=out_dim,
                                         num_layers=encoder_layers,
                                         dropout=dropout)
            self.node_var_nn = DenseModel(in_dim=input_dim,
                                          num_hidden=num_hidden,
                                          out_dim=out_dim,
                                          num_layers=encoder_layers,
                                          dropout=dropout)

        self.attr_mu_nn = DenseModel(in_dim=num_nodes + noise_dim,
                                     num_hidden=num_hidden,
                                     out_dim=out_dim,
                                     num_layers=2,
                                     dropout=dropout)
        self.attr_var_nn = DenseModel(in_dim=num_nodes,
                                      num_hidden=num_hidden,
                                      out_dim=out_dim,
                                      num_layers=2,
                                      dropout=dropout)

        if self.decoder_type == 'gcn':
            self.decoder = GCN(in_dim=2 * out_dim,
                               hid_dim=num_hidden,
                               out_dim=input_dim,
                               n_layers=decoder_layers,
                               dropout=dropout)
        else:
            assert self.decoder_type == 'mlp'
            self.decoder = DenseModel(in_dim=2 * out_dim,
                                      num_hidden=num_hidden,
                                      out_dim=input_dim,
                                      num_layers=decoder_layers,
                                      dropout=dropout)
        self.noise_dim = noise_dim
        self.noise_dist = dist.Bernoulli(torch.tensor([.5], device=device))

        self.K = K
        self.J = J

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'attr_mu_nn' in name or 'attr_var_nn' in name:
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    assert 'bias' in name
                    torch.nn.init.zeros_(param)

    def encode(self, adj, x):
        attr_dim = x.shape[1]
        num_nodes = x.shape[0]
        node_mu_input = x.unsqueeze(1).expand(-1, self.K + self.J, -1)
        node_noise_e = self.noise_dist.sample(torch.Size([num_nodes, self.K + self.J, self.noise_dim]))
        node_noise_e = torch.squeeze(node_noise_e)
        noised_node_mu_input = torch.cat((node_noise_e, node_mu_input), 2)
        if self.encoder_type == 'gcn':
            node_mu = self.node_mu_nn(adj, noised_node_mu_input)
        else:
            assert self.encoder_type == 'mlp'
            node_mu = self.node_mu_nn(noised_node_mu_input)
        node_mu_iw = node_mu[:, :self.K, :]
        node_mu_star = node_mu[:, self.K:, :]
        node_mu_iw_vec = torch.mean(node_mu_iw, 1)

        if self.encoder_type == 'gcn':
            node_logv = self.node_var_nn(adj, x)
        else:
            assert self.encoder_type == 'mlp'
            node_logv = self.node_var_nn(x)
        node_logv_iw = node_logv.unsqueeze(1).expand(-1, self.K, -1)
        node_sigma_iw1 = torch.exp(0.5 * node_logv_iw)
        merged_node_sigma = node_sigma_iw1.unsqueeze(2).expand(-1, -1, self.J + 1, -1)

        node_z_samples_iw = sample_n(mu=node_mu_iw, sigma=node_sigma_iw1)
        merged_node_z_samples = node_z_samples_iw.unsqueeze(2).expand(-1, -1, self.J + 1, -1)
        node_mu_star1 = node_mu_star.unsqueeze(1).expand(-1, self.K, -1, -1)
        merged_node_mu = torch.cat((node_mu_star1, node_mu_iw.unsqueeze(2)), 2)

        attr_mu_input = x.transpose(0, 1).unsqueeze(1).expand(-1, self.K + self.J, -1)
        attr_noise_e = self.noise_dist.sample(torch.Size([attr_dim, self.K + self.J, self.noise_dim]))
        attr_noise_e = torch.squeeze(attr_noise_e)
        noised_attr_mu_input = torch.cat((attr_noise_e, attr_mu_input), 2)
        attr_mu = self.attr_mu_nn(noised_attr_mu_input)
        attr_mu_iw = attr_mu[:, :self.K, :]
        attr_mu_star = attr_mu[:, self.K:, :]
        attr_mu_iw_vec = torch.mean(attr_mu_iw, 1)

        attr_logv = self.attr_var_nn(x=x.transpose(0, 1))
        attr_logv_iw = attr_logv.unsqueeze(1).expand(-1, self.K, -1)
        attr_sigma_iw1 = torch.exp(0.5 * attr_logv_iw)
        merged_attr_sigma = attr_sigma_iw1.unsqueeze(2).expand(-1, -1, self.J + 1, -1)

        attr_z_samples_iw = sample_n(mu=attr_mu_iw, sigma=attr_sigma_iw1)
        merged_attr_z_samples = attr_z_samples_iw.unsqueeze(2).expand(-1, -1, self.J + 1, -1)
        attr_mu_star1 = attr_mu_star.unsqueeze(1).expand(-1, self.K, -1, -1)
        merged_attr_mu = torch.cat((attr_mu_star1, attr_mu_iw.unsqueeze(2)), 2)

        return merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
               merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
               node_mu_iw_vec, attr_mu_iw_vec

    def decode(self, node_z, attr_z, adj, x):
        outputs_u, outputs_a = None, None
        for i in range(self.K):
            input_u = node_z[:, i, :].squeeze()
            input_a = attr_z[:, i, :].squeeze()

            z_u = F.dropout(input_u, self.dropout, self.training)
            z_u_t = z_u.transpose(0, 1)
            logits_node = torch.matmul(z_u, z_u_t)

            if self.decoder_type == 'inner_product':
                z_a = F.dropout(input_a, self.dropout, self.training)
                z_a_t = z_a.transpose(0, 1)
                logits_attr = torch.matmul(z_u, z_a_t)
            else:
                assert self.decoder_type in ['gcn', 'mlp']
                z_a = F.dropout(input_a, self.dropout, self.training)
                weights = F.normalize(x, p=1, dim=1)
                fine_grained_features = torch.matmul(weights, z_a)
                concat_features = torch.cat((z_u, fine_grained_features), dim=1)
                if self.decoder_type == 'gcn':
                    logits_attr = self.decoder(adj, concat_features)
                else:
                    assert self.decoder_type == 'mlp'
                    logits_attr = self.decoder(concat_features)
            if i == 0:
                outputs_u = logits_node.unsqueeze(2)
                outputs_a = logits_attr.unsqueeze(2)
            else:
                outputs_u = torch.cat((outputs_u, logits_node.unsqueeze(2)), 2)
                outputs_a = torch.cat((outputs_a, logits_attr.unsqueeze(2)), 2)
        return outputs_u, outputs_a

    def forward(self, adj, x):
        merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
        merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
        node_mu_iw_vec, attr_mu_iw_vec = self.encode(adj, x)
        reconstruct_node_logits, reconstruct_attr_logits = self.decode(adj=adj,
                                                                       x=x,
                                                                       node_z=node_z_samples_iw,
                                                                       attr_z=attr_z_samples_iw)

        return merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
               merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
               reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec


class SeeGera_Correlated(SeeGera_Independent):
    def __init__(self,
                 num_nodes=2708,
                 input_dim=1433,
                 num_hidden=128,
                 out_dim=512,
                 noise_dim=5,
                 K=1,
                 J=1,
                 dropout=0.,
                 device=None,
                 encoder_type='gcn',
                 encoder_layers=2,
                 decoder_type='gcn',
                 decoder_layers=2):
        super(SeeGera_Correlated, self).__init__(num_nodes=num_nodes,
                                                 input_dim=input_dim,
                                                 num_hidden=num_hidden,
                                                 out_dim=out_dim,
                                                 noise_dim=noise_dim,
                                                 K=K,
                                                 J=J,
                                                 dropout=dropout,
                                                 device=device,
                                                 encoder_type=encoder_type,
                                                 encoder_layers=encoder_layers,
                                                 decoder_type=decoder_type,
                                                 decoder_layers=decoder_layers)
        self.attr_mu_nn = DenseModel(in_dim=out_dim, num_hidden=num_hidden, out_dim=out_dim, num_layers=2,
                                     dropout=dropout)
        self.attr_var_nn = DenseModel(in_dim=out_dim, num_hidden=num_hidden, out_dim=out_dim, num_layers=2,
                                      dropout=dropout)

    def encode(self, adj, x):
        attr_dim = x.shape[1]
        num_nodes = x.shape[0]
        node_mu_input = x.unsqueeze(1).expand(-1, self.K + self.J, -1)
        node_noise_e = self.noise_dist.sample(torch.Size([num_nodes, self.K + self.J, self.noise_dim]))
        node_noise_e = torch.squeeze(node_noise_e)
        noised_node_mu_input = torch.cat((node_noise_e, node_mu_input), 2)
        if self.encoder_type == 'gcn':
            node_mu = self.node_mu_nn(adj, noised_node_mu_input)
        else:
            assert self.encoder_type == 'mlp'
            node_mu = self.node_mu_nn(noised_node_mu_input)
        node_mu_iw = node_mu[:, :self.K, :]
        node_mu_star = node_mu[:, self.K:, :]
        node_mu_iw_vec = torch.mean(node_mu_iw, 1)

        if self.encoder_type == 'gcn':
            node_logv = self.node_var_nn(adj, x)
        else:
            assert self.encoder_type == 'mlp'
            node_logv = self.node_var_nn(x)
        node_logv_iw = node_logv.unsqueeze(1).expand(-1, self.K, -1)
        node_sigma_iw1 = torch.exp(0.5 * node_logv_iw)
        merged_node_sigma = node_sigma_iw1.unsqueeze(2).expand(-1, -1, self.J + 1, -1)

        node_z_samples_iw = sample_n(mu=node_mu_iw, sigma=node_sigma_iw1)
        merged_node_z_samples = node_z_samples_iw.unsqueeze(2).expand(-1, -1, self.J + 1, -1)
        node_mu_star1 = node_mu_star.unsqueeze(1).expand(-1, self.K, -1, -1)
        merged_node_mu = torch.cat((node_mu_star1, node_mu_iw.unsqueeze(2)), 2)

        weights = F.normalize(x.transpose(0, 1), p=1, dim=1).unsqueeze(1).expand(-1, self.K + self.J, -1)
        attr_mu_input = torch.matmul(weights.transpose(0, 1), node_mu.transpose(0, 1)).transpose(0, 1)

        attr_mu = self.attr_mu_nn(attr_mu_input)
        attr_mu_iw = attr_mu[:, :self.K, :]
        attr_mu_star = attr_mu[:, self.K:, :]
        attr_mu_iw_vec = torch.mean(attr_mu_iw, 1)

        weights = F.normalize(x.transpose(0, 1), p=1, dim=1)
        attr_logv_input = torch.matmul(weights, node_logv)
        attr_logv = self.attr_var_nn(x=attr_logv_input)

        attr_logv_iw = attr_logv.unsqueeze(1).expand(-1, self.K, -1)
        attr_sigma_iw1 = torch.exp(0.5 * attr_logv_iw)
        merged_attr_sigma = attr_sigma_iw1.unsqueeze(2).expand(-1, -1, self.J + 1, -1)

        attr_z_samples_iw = sample_n(mu=attr_mu_iw, sigma=attr_sigma_iw1)
        merged_attr_z_samples = attr_z_samples_iw.unsqueeze(2).expand(-1, -1, self.J + 1, -1)
        attr_mu_star1 = attr_mu_star.unsqueeze(1).expand(-1, self.K, -1, -1)
        merged_attr_mu = torch.cat((attr_mu_star1, attr_mu_iw.unsqueeze(2)), 2)

        return merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
               merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
               node_mu_iw_vec, attr_mu_iw_vec
