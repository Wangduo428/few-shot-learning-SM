import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FewShotLearning.methods import res12, resnet
from tqdm import tqdm

class ProtoNet_Sinkhorn(nn.Module):
    def __init__(self, args):
        super(ProtoNet_Sinkhorn, self).__init__()
        self.args = args

        if args.model == 'ResNet12':
            self.base = res12.ResNet(avg_pool=False, scale=args.feature_scale, drop_rate=args.drop_rate, drop_block=args.drop_block, dropblock_size=args.dropblock_size)
        elif args.model == 'ResNet10':
            self.base = resnet.resnet10(avg_pool=False)
        else:
            raise ValueError('Unknown model type.')

        if args.num_gpus > 1:
            self.base = nn.DataParallel(self.base)

        self.base = self.base.cuda()

        self.method = 'ProtoNet_Sinkhorn'

        print('Running model ProtoNet_Sinkhorn')

    def set_forward_score(self, xtrain, ytrain, xtest, ytest):
        # xtrain: [batch_size, n_way, n_shot, 3, 84, 84]
        # ytrain: [batch_size, n_way, n_shot]
        # xtest: [batch_size, n_way, n_query, 3, 84, 84]
        # ytest: [batch_size, n_way, n_query]
        # train: True means in training mode, false in testing mode

        batch_size, n_way, n_shot = xtrain.size()[0:3]
        n_query = xtest.size()[2]

        xtrain = xtrain.cuda()
        xtest = xtest.cuda()

        xtrain = xtrain.view(-1, *xtrain.size()[3:])
        xtest = xtest.view(-1, *xtest.size()[3:])
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)  # [N, 512, 6, 6]

        ftrain = f[:batch_size * n_way * n_shot]
        ftrain = ftrain.view(batch_size, n_way, n_shot, *f.size()[1:])
        ftrain = ftrain.mean(2)
        ftrain = ftrain.view(*ftrain.size()[0:3], -1)  # [bs, n_way, 640, 25]

        ftest = f[batch_size * n_way * n_shot:]
        ftest = ftest.view(batch_size, n_way, n_query, *f.size()[1:])
        ftest = ftest.view(batch_size, n_way * n_query, *f.size()[1:])
        ftest = ftest.view(*ftest.size()[0:3], -1)  # [bs, n_way*n_q, 640, 25]

        P, _, sinkhorn_score = self.sinkhorn(ftrain, ftest)  # P: [bs, n2, n1, 25, 25], dist: [bs, n2, n1]

        return sinkhorn_score

    def sinkhorn(self, f1, f2):
        # f1, f2: [bs, n1/n2, 640, 25]
        thresh = self.args.thresh
        maxiters = self.args.maxiters
        lam = self.args.lam
        eps = 1.0/lam   #0.01

        def SUM(Mat, u, v):
            return (-Mat + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps

        bs, n1 = f1.size()[0], f1.size()[1]
        n2 = f2.size()[1]

        # first calculate cost matrix by pairwise distance, may try different choices
        local_cos = self.local_cosine(f2, f1)  # [bs, n2, n1, 25, 25]
        M = 1-local_cos

        # then cal the distribution of each sample
        d1 = self.local_cross_match_att(f2, f1)  # [bs, n2, n1, 25]
        d2 = self.local_cross_match_att(f1, f2)  # [bs, n1, n2, 25]

        d2 = d2.transpose(1, 2)  # [bs, n2, n1, 25]

        # calculate sinkhorn distance
        M = M.view(-1, *M.size()[3:])
        d1 = d1.contiguous().view(-1, d1.size()[3])
        d2 = d2.contiguous().view(-1, d2.size()[3])

        u = torch.zeros_like(d1).cuda()
        v = torch.zeros_like(d2).cuda()

        for i in range(maxiters):
            u_tmp = u
            u = eps * (torch.log(d1+1e-8) - torch.logsumexp(SUM(M, u, v), dim=-1)) + u
            v = eps * (torch.log(d2+1e-8) - torch.logsumexp(SUM(M, u, v).transpose(-2, -1), dim=-1)) + v

            err = torch.abs(u-u_tmp).sum(-1).mean()

            if err.item() < thresh:
                break

        P = torch.exp(SUM(M, u, v))

        sinkhorn_dist = P*M
        sinkhorn_dist = sinkhorn_dist.sum(-1).sum(-1)
        sinkhorn_score = P*(1-M)
        sinkhorn_score = sinkhorn_score.sum(-1).sum(-1)

        P = P.view(bs, n2, n1, *P.size()[1:])
        sinkhorn_dist = sinkhorn_dist.view(bs, n2, n1)
        sinkhorn_score = sinkhorn_score.view(bs, n2, n1)

        # P: [bs, n2, n1, 25, 25], dist: [bs, n2, n1]
        return P, sinkhorn_dist, sinkhorn_score

    def local_cosine(self, f1, f2):
        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)
        f1_norm = f1_norm.unsqueeze(2)  # [bs, n1, 1, 640, 25]
        f2_norm = f2_norm.unsqueeze(1)  # [bs, 1, n2, 640, 25]
        M = torch.matmul(f1_norm.transpose(3, 4), f2_norm)   # [bs, n1, n2, 25, 25]
        return M

    def local_cross_match_att(self, f1, f2):
        # f1: [batch_size, n1, 640, 25]
        f1 = f1.unsqueeze(2)
        f2 = f2.unsqueeze(1)

        f2_fuse = f2.transpose(3, 4)  # [batch_size, 1, n2, 25, 640]

        f1_norm = F.normalize(f1, p=2, dim=3, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=3, eps=1e-12)

        f1_norm = f1_norm.transpose(3, 4)

        dist = torch.matmul(f1_norm, f2_norm)  # [batch_size, n1, n2, 25, 25]

        weight = torch.softmax(dist, dim=-1)  # indicate the similarity between local features

        f2_fuse = torch.matmul(weight, f2_fuse)  # [batch_size, n1, n2, 25, 640]
        # f2_fuse = F.normalize(f2_fuse, p=2, dim=4, eps=1e-12)

        cro = torch.sum(f1.transpose(3, 4) * f2_fuse, dim=-1)  # # [batch_size, n1, n2, 25]

        cro = nn.ReLU()(cro)  # to make sure all the values are positive, since its probability
        cro = cro + 1e-5  # To avoid nan when calculate P
        cro = cro / cro.mean(-1, True)  # normalize

        return cro

    def test_loop(self, test_loader):
        self.base.eval()
        test_accuracies = np.zeros((0))
        with torch.no_grad():
            t = tqdm(test_loader, desc='testing', ncols=80)
            for (xtrain, ytrain, _, xtest, ytest, _) in t:
                # xtrain: [batch_size, n_way, n_shot, 3, 84, 84]
                # ytrain: [batch_size, n_way, n_shot]
                # xtest: [batch_size, n_way, n_query, 3, 84, 84]
                # ytest: [batch_size, n_way, n_query]
                scores = self.set_forward_score(xtrain, ytrain, xtest, ytest)  # [batch_size, n_way*n_query, n_way]
                batch_size, n_test = scores.size()[0], scores.size()[1]

                scores = scores.view(batch_size * n_test, -1)  # [batch_size*n_way*n_query, n_way]
                ytest = ytest.view(-1)  # [batch_size*n_way*n_query]

                if torch.isnan(scores).any():
                    print('batch {} contains nan.'.format(i))

                _, preds = torch.max(scores.data.cpu(), 1)

                gt = (preds == ytest.data.cpu()).float()
                gt = gt.view(batch_size, n_test).numpy()  # [batch_size, n_way*n_query]
                acc = np.sum(gt, 1) / n_test
                acc = np.reshape(acc, (batch_size))
                test_accuracies = np.append(test_accuracies, acc)
                t.set_postfix(curr_acc=np.mean(test_accuracies, 0))

        accuracy_mean = np.mean(test_accuracies, 0)
        stds = np.std(test_accuracies, 0)
        ci95 = 1.96 * stds / np.sqrt(self.args.test_epoch_size)
        print('###########Test accuracy {:.2%}, std {:.2%}#############'.format(accuracy_mean, ci95))
        return accuracy_mean, ci95

    def load_model(self, model_path):
        state = torch.load(model_path)
        base_state = state['base_state']

        if self.args.num_gpus > 1:
            self.base.module.load_state_dict(base_state)
        else:
            self.base.load_state_dict(base_state)
