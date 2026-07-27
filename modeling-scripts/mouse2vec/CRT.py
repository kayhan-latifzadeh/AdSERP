import pdb, copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from base_models import MLP, resnet1d18, SSLDataSet
from sklearn.metrics import f1_score

class cnn_extractor(nn.Module):
    def __init__(self, dim, input_plane):
        super(cnn_extractor, self).__init__()
        self.cnn = resnet1d18(input_channels=dim, inplanes=input_plane)

    def forward(self, x):        
        x = self.cnn(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TFR(nn.Module):
    def __init__(self, seq_len, patch_len, dim, depth, heads, mlp_dim, channels=12,
                 dim_head=64, dropout=0., emb_dropout=0.):
        '''
        The encoder of CRT
        '''
        super().__init__()
        self.patch_len = patch_len
        self.seq_len = seq_len

        assert seq_len % (4 * patch_len) == 0, \
            'The seq_len should be 4 * n * patch_len, or there must be patch with both magnitude and phase data.'
        num_patches = seq_len // patch_len
        patch_dim = channels * patch_len

        self.to_patch = nn.Sequential(Rearrange('b c (n p1) -> b n c p1', p1=patch_len),
                            Rearrange('b n c p1 -> (b n) c p1'))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 3, dim))
        self.modal_embedding = nn.Parameter(torch.randn(3, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 3, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.cnn1 = cnn_extractor(dim=channels, input_plane=dim // 8) # For temporal data
        self.cnn2 = cnn_extractor(dim=channels, input_plane=dim // 8) # For magnitude data
        self.cnn3 = cnn_extractor(dim=channels, input_plane=dim // 8) # For phase data

    def forward(self, x, opt):
        batch, _, time_steps = x.shape
        t, m, p = x[:, :, :time_steps // 2], x[:, :, time_steps // 2: time_steps * 3 // 4], x[:, :, -time_steps // 4:]
        assert t.shape[-1] % opt.patch_len == 0 == m.shape[-1] % opt.patch_len == p.shape[-1] % opt.patch_len

        t, m, p = self.to_patch(t), self.to_patch(m), self.to_patch(p)
        patch2seq = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  Rearrange('(b n) c 1 -> b n c', b=batch))

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch)
        x = torch.cat((cls_tokens[:, 0:1, :], patch2seq(self.cnn1(t)),
                       cls_tokens[:, 1:2, :], patch2seq(self.cnn2(m)),
                       cls_tokens[:, 2:3, :], patch2seq(self.cnn3(p))), dim=1)

        b, t, c = x.shape
        time_steps = t - 3
        t_token_idx, m_token_idx, p_token_idx = 0, time_steps // 2 + 1, time_steps * 3 // 4 + 2
        x[:m_token_idx] += self.modal_embedding[:1]
        x[m_token_idx: p_token_idx] += self.modal_embedding[1:2]
        x[p_token_idx: ] += self.modal_embedding[2:]
        x += self.pos_embedding[:, : t]
        x = self.dropout(x)
        x = self.transformer(x)
        t_token, m_token, p_token = x[:, t_token_idx], x[:, m_token_idx], x[:, p_token_idx]
        avg = (t_token + m_token + p_token) / 3
        return avg


def TFR_Encoder(seq_len, patch_len, dim, in_dim, depth):
    vit = TFR(seq_len=seq_len,
              patch_len=patch_len,
              #num_classes=num_class,
              dim=dim,
              depth=depth,
              heads=8,
              mlp_dim=dim,
              dropout=0.2,
              emb_dropout=0.1,
              channels=in_dim)
    return vit

class CRT(nn.Module):
    def __init__(
            self,
            encoder,
            decoder_dim,
            decoder_depth=2,
            decoder_heads=8,
            decoder_dim_head=64,
            patch_len = 20,
            in_dim=12
    ):
        super().__init__()
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch = encoder.to_patch
        pixel_values_per_patch = in_dim * patch_len

        # decoder parameters
        self.modal_embedding = self.encoder.modal_embedding
        self.mask_token = nn.Parameter(torch.randn(3, decoder_dim))
        self.decoder = Transformer(dim=decoder_dim,
                                   depth=decoder_depth,
                                   heads=decoder_heads,
                                   dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.ModuleList([nn.Linear(decoder_dim, pixel_values_per_patch) for i in range(3)])
        self.projs = nn.ModuleList([nn.Linear(decoder_dim, decoder_dim) for i in range(2)])

        self.to_clicks = nn.Linear(decoder_dim, 2 * patch_len)

    def IDC_loss(self, tokens, encoded_tokens):
        B, T, D = tokens.shape
        tokens, encoded_tokens = F.normalize(tokens, dim=-1), F.normalize(encoded_tokens, dim=-1)
        encoded_tokens = encoded_tokens.transpose(2, 1)
        cross_mul = torch.exp(torch.matmul(tokens, encoded_tokens))
        mask = (1 - torch.eye(T)).unsqueeze(0).to(tokens.device)
        cross_mul = cross_mul * mask
        return torch.log(cross_mul.sum(-1).sum(-1)).mean(-1)

    def forward(self, x, clicks, mask_ratio=0.75, beta = 1e-4, beta2 = 1e-4):
        device = x.device
        patches = self.to_patch[0](x)
        batch, num_patches, c, length = patches.shape
        clickpatches = self.to_patch[0](clicks.unsqueeze(1).float())

        num_masked = int(mask_ratio * num_patches)
        rand_indices1 = torch.randperm(num_patches // 2, device=device)
        masked_indices1 = rand_indices1[: num_masked // 2].sort()[0]
        unmasked_indices1 = rand_indices1[num_masked // 2:].sort()[0]
        rand_indices2 = torch.randperm(num_patches // 4, device=device)
        masked_indices2, unmasked_indices2 = rand_indices2[: num_masked // 4].sort()[0], rand_indices2[num_masked // 4:].sort()[0]
        rand_indices = torch.cat((masked_indices1, unmasked_indices1, 
                         masked_indices2 + num_patches // 2, unmasked_indices2 + num_patches // 2,
                         masked_indices2 + num_patches // 4 * 3, unmasked_indices2 + num_patches // 4 * 3))

        masked_num_t, masked_num_f = masked_indices1.shape[0], 2 * masked_indices2.shape[0]
        unmasked_num_t, unmasked_num_f = unmasked_indices1.shape[0], 2 * unmasked_indices2.shape[0]

        tpatches = patches[:, : num_patches // 2, :, :]
        mpatches, ppatches = patches[:, num_patches // 2: num_patches * 3 // 4, :, :], patches[:, -num_patches // 4:, :, :]

        unmasked_tpatches = tpatches[:, unmasked_indices1, :, :]
        unmasked_mpatches, unmasked_ppatches = mpatches[:, unmasked_indices2, :, :], ppatches[:, unmasked_indices2, :, :]
        t_tokens, m_tokens, p_tokens = self.to_patch[1](unmasked_tpatches), self.to_patch[1](unmasked_mpatches), self.to_patch[1](unmasked_ppatches)
        t_tokens, m_tokens, p_tokens = self.encoder.cnn1(t_tokens), self.encoder.cnn2(m_tokens), self.encoder.cnn3(p_tokens)
        Flat = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                      Rearrange('(b n) c 1 -> b n c', b=batch))
        t_tokens, m_tokens, p_tokens = Flat(t_tokens), Flat(m_tokens), Flat(p_tokens)
        ori_tokens = torch.cat((t_tokens, m_tokens, p_tokens), 1).clone()
        
        cls_tokens = repeat(self.encoder.cls_token, '() n d -> b n d', b=batch)
        tokens = torch.cat((cls_tokens[:, 0:1, :], t_tokens,
                            cls_tokens[:, 1:2, :], m_tokens,
                            cls_tokens[:, 2:3, :], p_tokens), dim=1)

        t_idx, m_idx, p_idx = num_patches // 2 - 1, num_patches * 3 // 4 - 1, num_patches - 1
        pos_embedding = torch.cat((self.encoder.pos_embedding[:, 0:1, :], self.encoder.pos_embedding[:, unmasked_indices1 + 1, :],
             self.encoder.pos_embedding[:, t_idx + 2: t_idx + 3],
             self.encoder.pos_embedding[:, unmasked_indices2 + t_idx + 3, :],
             self.encoder.pos_embedding[:, m_idx + 3: m_idx + 4],
             self.encoder.pos_embedding[:, unmasked_indices2 + m_idx + 4, :]), dim=1)

        modal_embedding = torch.cat((repeat(self.modal_embedding[0], '1 d -> 1 n d', n=unmasked_num_t + 1),
                                     repeat(self.modal_embedding[1], '1 d -> 1 n d', n=unmasked_num_f // 2 + 1),
                                     repeat(self.modal_embedding[2], '1 d -> 1 n d', n=unmasked_num_f // 2 + 1)), dim=1)
        
        tokens = tokens + pos_embedding + modal_embedding

        encoded_tokens = self.encoder.transformer(tokens) # tokens: unmasked tokens + CLS tokens

        t_idx, m_idx, p_idx = unmasked_num_t, unmasked_num_f // 2 + unmasked_num_t + 1, -1

        idc_loss = self.IDC_loss(self.projs[0](ori_tokens), self.projs[1](torch.cat(([encoded_tokens[:, 1: t_idx+1], encoded_tokens[:, t_idx+2: m_idx+1], encoded_tokens[:, m_idx+2: ]]), dim=1)))

        decoder_tokens = encoded_tokens

        mask_tokens1 = repeat(self.mask_token[0], 'd -> b n d', b=batch, n=masked_num_t)
        mask_tokens2 = repeat(self.mask_token[1], 'd -> b n d', b=batch, n=masked_num_f // 2)
        mask_tokens3 = repeat(self.mask_token[2], 'd -> b n d', b=batch, n=masked_num_f // 2)
        mask_tokens = torch.cat((mask_tokens1, mask_tokens2, mask_tokens3), dim=1)

        decoder_pos_emb = self.decoder_pos_emb(torch.cat(
            (masked_indices1, masked_indices2 + num_patches // 2, masked_indices2 + num_patches * 3 // 4)))

        mask_tokens = mask_tokens + decoder_pos_emb
        decoder_tokens = torch.cat((decoder_tokens, mask_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        mask_tokens = decoded_tokens[:, -mask_tokens.shape[1]:]

        pred_pixel_values_t = self.to_pixels[0](torch.cat((decoder_tokens[:, 1: t_idx + 1], mask_tokens[:, : masked_num_t]), 1))
        pred_pixel_values_m = self.to_pixels[1](torch.cat((decoder_tokens[:, t_idx+2: m_idx+1], mask_tokens[:, masked_num_t: masked_num_f // 2 + masked_num_t]), 1))
        pred_pixel_values_p = self.to_pixels[2](torch.cat((decoder_tokens[:, m_idx+2: -mask_tokens.shape[1]], mask_tokens[:, -masked_num_f // 2:]), 1))
        pred_pixel_values = torch.cat((pred_pixel_values_t, pred_pixel_values_m, pred_pixel_values_p), dim=1)
        
        recon_loss = F.mse_loss(pred_pixel_values, rearrange(patches[:,rand_indices], 'b n c p -> b n (c p)'))

        rmvcls_embedding = torch.cat((decoder_tokens[:, 1: t_idx + 1], mask_tokens[:, : masked_num_t]), 1)
        click_pred = self.to_clicks(rmvcls_embedding)
        click_pred = rearrange(click_pred, 'b n (c p) -> b n c p', p=2)
        click_pred = rearrange(click_pred, 'b n c p -> (b n c) p')
        clicksGT = clickpatches[:, rand_indices1].squeeze()
        clicksGT = rearrange(clicksGT, 'b n c -> (b n c)')
        clicksOH = F.one_hot(clicksGT.to(torch.int64), num_classes=2)
        pos_weight = (clicks==0).sum()/(clicks==1).sum()
        click_pred_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(click_pred, clicksOH.float())
        
        return recon_loss + beta * idc_loss + beta2 * click_pred_loss

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.encoder = TFR_Encoder(seq_len=opt.seq_len,
                            patch_len=opt.patch_len,
                            dim=opt.dim,
                            in_dim=opt.in_dim,
                            depth=opt.depth)
        self.crt = CRT(encoder=self.encoder,
                       decoder_dim=opt.dim,
                       in_dim=opt.in_dim,
                       patch_len=opt.patch_len)

    def forward(self, x, clicks, ratio = 0.5):
        return self.crt(x, clicks, mask_ratio=ratio, beta2=self.opt.beta2)

def self_supervised_learning(model, X, clicks, opt, modelfile, min_ratio=0.3, max_ratio=0.8):
    optimizer = optim.Adam(model.parameters(), opt.AElearningRate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

    model.to(opt.device)
    model.train()

    dataset = SSLDataSet(X, clicks)
    dataloader = DataLoader(dataset, batch_size=opt.AEbatchSize, shuffle=True)

    losses = []
    for _ in range(opt.AEepochs):
        for idx, batch in enumerate(dataloader):
            x, clicks = tuple(t.to(opt.device) for t in batch)
            loss = model.to(opt.device)(x, clicks, ratio=max(min_ratio, min(max_ratio, _ / opt.AEepochs)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
            if idx % 20 == 0:
                print(idx,'/',len(dataloader), sum(losses) / len(losses))
        scheduler.step(_)
        torch.save(model.to('cpu'), modelfile)

class encoder_classifier(nn.Module):
    def __init__(self, AE, opt):
        super().__init__()
        self.opt = opt
        hidden_channels = 64
        self.inputdim = opt.dim
        self.encoder = copy.deepcopy(AE.encoder)
        self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p = opt.Headdropout),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p = opt.Headdropout),
                nn.Linear(hidden_channels, opt.num_class)
                )
        # initialize the classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        features = self.encoder(x, self.opt)
        out = self.classifier(features)
        return out  

def weight_init_layer(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
def weight_init_whole(m):
    children = list(m.children())
    if len(children)==0: # No more children layers
        m.apply(weight_init_layer)
    else: # Get children layers
        for c in children:
            weight_init_whole(c)
        
def finetuning(AE, train_set, valid_set, opt, modelfile, multi_label=True):
    stage_info = {0: 'Train the classifier only', 1: 'Finetune the whole model'}
    print('Stage %d: '%(opt.stage), stage_info[opt.stage])

    train_loader = DataLoader(train_set, batch_size=opt.HeadbatchSize, shuffle=True)
    model = encoder_classifier(AE, opt)
    model.train()
    
    best_res = 0
    step = 0
    if opt.stage == 0:
        optimizer = optim.Adam(model.classifier.parameters(), lr=opt.HeadlearningRate)
    else:
        lr = opt.HeadlearningRate/2
        optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)
    for epoch in range(opt.Headepochs):
        for batch_idx, batch in enumerate(train_loader):
            step += 1
            x, y, clicks = tuple(t.to(opt.device) for t in batch)            
            pred = model.to(opt.device)(x)
            if not multi_label:
                pos_weight = (y==0.).sum()/y.sum()
                y = F.one_hot(y, num_classes=2)
                #loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred, y.squeeze().float())
                loss = nn.BCEWithLogitsLoss()(pred, y.squeeze().float())
            else:
                if pred.shape[0]==1:
                    loss = nn.CrossEntropyLoss()(pred, y.long())
                else:
                    loss = nn.CrossEntropyLoss()(pred, y.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                [trainacc, trainf1] = test(model, train_set, opt.HeadbatchSize, multi_label)
                if trainf1 > best_res:
                    best_res = trainf1
                scheduler.step(best_res)

def test(model, dataset, batch_size, multi_label):
    def topAcc(GTlabel, pred_proba, top):
        count = 0
        for i in range(GTlabel.shape[0]):
            if (GTlabel[i]) in np.argsort(-pred_proba[i])[:top]:
                count+=1.0
        return count/GTlabel.shape[0]

    model.eval()
    model.to(model.opt.device)
    testloader = DataLoader(dataset, batch_size=batch_size)
    pred_prob = None
    with torch.no_grad():
        for batch in testloader:
            x, y, clicks = tuple(t.to(model.opt.device) for t in batch)            
            pred = model(x)
            if pred_prob is None:
                pred_prob = pred.cpu().detach().numpy()
            else:
                pred_prob = np.concatenate([pred_prob, pred.cpu().detach().numpy()], axis=0)
    acc = topAcc(dataset.label.squeeze(), pred_prob, 1)
    pred = np.argmax(pred_prob, axis=1)
    f1 = f1_score(dataset.label.squeeze(), pred, average='macro')

    model.train()
    return [acc, f1]