import torch
import torch.nn as nn


class Generator_eeg(nn.Module):
    def __init__(self,sequence_len,out_features,hidden_dim=50,latent_dim=200):
        super().__init__()
        self.sequence_len = sequence_len
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim,(sequence_len//64) * hidden_dim,bias=False),
            nn.LeakyReLU(0.2)
        )

        self.block1 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block2 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block3 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block4 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block5 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block6 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )


        self.last = nn.Sequential(nn.Conv1d(hidden_dim,out_features,kernel_size=5,padding="same"),nn.Tanh())


    def make_conv1d_block(self,in_channel,out_channel,kernel=3,upsample=True):
        block = []

        if upsample:
            block.append(nn.Upsample(scale_factor=2))
        
        block.append(nn.Conv1d(in_channel,out_channel,kernel,padding="same"))
        block.append(nn.BatchNorm1d(out_channel))
        block.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*block)

    def forward(self,noise):
        out = self.fc1(noise)
        out = torch.reshape(out,(-1,self.hidden_dim,self.sequence_len//64))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.last(out)
        out = torch.reshape(out,(-1,self.out_features,self.sequence_len))
        
        return out
    
class Discriminator_eeg(nn.Module):
    def __init__(self,sequence_len,in_features,hidden_dim):
        super().__init__()
        self.first = nn.Conv1d(in_features,hidden_dim,3,padding="same")

        self.block1 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block2 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block3 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block4 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block5 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block6 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )

        self.last = nn.Linear(hidden_dim*sequence_len//64,1)

    def make_conv1d_block(self,in_channel,out_channel,kernel=3,downsample=False):
        block = []

        block.append(nn.Conv1d(in_channel,out_channel,kernel,padding="same"))
        block.append(nn.LeakyReLU(0.2))
        if downsample:
            block.append(nn.AvgPool1d(kernel_size=3,padding=1,stride=2))

        return nn.Sequential(*block)
    
    def forward(self,x):
        _x = self.first(x)
        _x = self.block1(_x)
        _x = self.block2(_x)
        _x = self.block3(_x)
        _x = self.block4(_x)
        _x = self.block5(_x)
        _x = self.block6(_x)
        _x = torch.flatten(_x,start_dim=1)
        _x = self.last(_x)
        _x = self.sigmoid(_x)
        return _x
    



class GAN:
    def __init__(self,seq_len,features=3,n_critic=3,lr=5e-4,
                 g_hidden=50,d_hidden=50,max_iters=1000,latent_dim=200,
                 w_loss = False,
                 saveDir=None,ckptPath=None,prefix="T01"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Train on {}".format(self.device))

        self.w_loss = w_loss

    
        self.G = Generator_eeg(seq_len,features,g_hidden,latent_dim=latent_dim).to(self.device)
        self.D = Discriminator_eeg(seq_len,features,d_hidden).to(self.device)

        self.load_ckpt(ckptPath)

        self.lr = lr
        self.n_critic = n_critic

        self.g_optimizer = torch.optim.Adam(self.G.parameters(),lr=self.lr)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(),lr=self.lr)

        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim

        self.max_iters = max_iters
        self.saveDir = saveDir
        self.g_hidden = g_hidden
        self.d_hidden = d_hidden
        self.prefix = prefix

    def train(self,dataloader):
        data = self.get_infinite_batch(dataloader)

        for g_iter in range(self.max_iters):
            for p in self.D.parameters():
                p.requires_grad = True
            

            self.G.train()

            for d_iter in range(self.n_critic):
                self.D.zero_grad()
                self.G.zero_grad()
                
                sequence , _ = data.__next__()
                real= torch.autograd.Variable(sequence).float().to(self.device)

                batch_size = real.size(0)
                z = torch.randn(batch_size,1,self.latent_dim).to(self.device)

                fake = self.G(z)  

                d_loss = self._get_critic_loss(self.D(fake),self.D(real))
                d_loss.backward()

                self.d_optimizer.step()
                print(f'Discriminator iteration: {d_iter}/{self.n_critic}, Discriminator Loss: {d_loss}')

            self.G.zero_grad()
            self.D.zero_grad()

            z = torch.randn(batch_size,1,self.latent_dim).to(self.device)
            fake = self.G(z,fake)
            g_loss = self._get_generator_loss(self.D(fake))
            g_loss.backward()

            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.max_iters}, g_loss: {g_loss}')

            if g_iter % 50 == 0:
                self.save_model()


            torch.cuda.empty_cache()

        self.save_model()
        print("Finished Training!!")
    

    def _get_critic_loss(self,fake_prediction,real_prediction):
        batch_size = fake_prediction.size(0)

        if self.w_loss == False: 
            real_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(1), requires_grad=False).to(self.device)
            fake_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(0), requires_grad=False).to(self.device)

            loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_prediction,fake_label)
            loss_real = nn.functional.binary_cross_entropy_with_logits(real_prediction,real_label)
            loss_d = loss_fake + loss_real
        else:
            loss_d = fake_prediction.mean() - real_prediction.mean()
        
        return loss_d

    def _get_generator_loss(self,prediction):
        batch_size = prediction.size(0)

        if self.w_loss == False:
            real_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(1), requires_grad=False).to(self.device)
            g_loss = nn.functional.binary_cross_entropy_with_logits(prediction,real_label)
        else:
            g_loss = prediction.mean()
        
        return g_loss


    def load_ckpt(self,ckptPath):
        if ckptPath:
            print("Load Checkpoint....")
            ckpt = torch.load(ckptPath,map_location=self.device)
            self.G.load_state_dict(ckpt['G_param'])
            self.D.load_state_dict(ckpt['D_param'])


    def generate_samples(self,sample_size):
        z = torch.randn(sample_size,1,self.latent_dim).to(self.device)
        label = torch.randint(low=0,high=self.label_dim,size=(sample_size,),device=self.device)
        fakes = self.G(z,label).detach().cpu().numpy()
        
        return fakes,label.cpu().numpy()
    

    def save_model(self):
        torch.save({"G_param":self.G.state_dict(),"D_param":self.D.state_dict()},
                f"{self.saveDir}/{self.prefix}_net_G{self.g_hidden}_D{self.d_hidden}_ckpt.pth")
    

    def get_infinite_batch(self,dataloader):
        while True:
            for data in dataloader:
                yield data