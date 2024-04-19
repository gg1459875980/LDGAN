import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MRIDataset
from model import Generator, Discriminator


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--channels", type=int, default=1, help="number of image channels (MRI is usually grayscale)")
opt = parser.parse_args()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
dataset = MRIDataset("data/", "dataset/clinic_score.csv", transform=transforms_)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)

# 初始化模型
G_AB = Generator().to(device)
G_BA = Generator().to(device)
D_A = Discriminator(opt.channels).to(device)
D_B = Discriminator(opt.channels).to(device)

# 初始化权重
G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# 损失函数
criterion_GAN = nn.MSELoss().to(device)
criterion_cycle = nn.L1Loss().to(device)

# 优化器
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 学习率调整
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, decay_start_epoch=opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, decay_start_epoch=opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, decay_start_epoch=opt.decay_epoch).step)


import time
import numpy as np
from torchvision.utils import save_image

# 训练循环
start_time = time.time()
for epoch in range(opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # 准备数据
        real_A = batch['image'].to(device)
        real_B = torch.roll(real_A, shifts=-1, dims=0)  # 将batch内的图像向前滚动以创建pairs
        score_A = batch['score'].to(device)
        score_B = torch.roll(score_A, shifts=-1, dims=0)

        # 真实数据标签和虚假数据标签
        valid = torch.ones((real_A.size(0), *D_A.output_shape), device=device)
        fake = torch.zeros((real_A.size(0), *D_A.output_shape), device=device)

        # ------------------
        #  训练生成器 G_AB 和 G_BA
        # ------------------
        G_AB.train()
        G_BA.train()
        optimizer_G.zero_grad()

        # 身份损失
        same_B = G_AB(real_B)
        loss_identity_B = criterion_cycle(same_B, real_B)
        same_A = G_BA(real_A)
        loss_identity_A = criterion_cycle(same_A, real_A)

        # GAN 损失
        fake_B = G_AB(real_A)
        pred_fake = D_B(fake_B)
        loss_GAN_AB = criterion_GAN(pred_fake, valid)

        fake_A = G_BA(real_B)
        pred_fake = D_A(fake_A)
        loss_GAN_BA = criterion_GAN(pred_fake, valid)

        # 循环一致性损失
        recovered_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, real_A)
        recovered_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recovered_B, real_B)

        # 总的生成器损失
        loss_G = (loss_GAN_AB + loss_GAN_BA) + 10.0 * (loss_cycle_A + loss_cycle_B) + 5.0 * (loss_identity_A + loss_identity_B)
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  训练鉴别器 D_A
        # -----------------------
        optimizer_D_A.zero_grad()

        # 真实数据损失
        loss_real = criterion_GAN(D_A(real_A), valid)
        # 生成的虚假数据损失
        loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
        # 总的鉴别器损失
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  训练鉴别器 D_B
        # -----------------------
        optimizer_D_B.zero_grad()

        # 真实数据损失
        loss_real = criterion_GAN(D_B(real_B), valid)
        # 生成的虚假数据损失
        loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
        # 总的鉴别器损失
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()

        # -----------------------
        #  打印训练状态
        # -----------------------
        elapsed_time = time.time() - start_time
        print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {loss_D_A.item() + loss_D_B.item()}] [G loss: {loss_G.item()}] elapsed_time: {elapsed_time}")

        # 如果需要，保存图像
        if i % 100 == 0:
            save_image(fake_A.data, f"images/fake_A_{epoch}_{i}.png", normalize=True)
            save_image(fake_B.data, f"images/fake_B_{epoch}_{i}.png", normalize=True)

    # 更新学习率
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # 如果需要，保存模型
    if epoch % 20 == 0:
        torch.save(G_AB.state_dict(), f"saved_models/G_AB_{epoch}.pth")
        torch.save(G_BA.state_dict(), f"saved_models/G_BA_{epoch}.pth")
        torch.save(D_A.state_dict(), f"saved_models/D_A_{epoch}.pth")
        torch.save(D_B.state_dict(), f"saved_models/D_B_{epoch}.pth")



