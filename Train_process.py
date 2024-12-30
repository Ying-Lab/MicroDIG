import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as stats
from Network import VAE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiStepLR
# from Network import weights_init

from torch.utils.tensorboard import SummaryWriter   


def AE3_train(model,device,train_iter,val_iter,save_dir,loss_log_name):
    print("Using device:\n")
    print(device)
    model = model.to(device)
    def loss_fun(rec_output, x, corr_output,label):
        rec_loss = nn.MSELoss()
        loss_1 = rec_loss(rec_output,x)
        corr_loss = nn.MSELoss()
        loss_2 = corr_loss(corr_output,label)
        return  loss_1 , loss_2
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer,milestones=[100,300,350,400,450,500,550,600],gamma=0.8)
    writer = SummaryWriter('{0}/{1}'.format(save_dir,loss_log_name))
    for epoch in range(650):
        model.train()
        train_loss_all=0
        train_num_sample=0
        for _,(feature,label) in enumerate(train_iter):
            feature,label = feature.to(device),label.to(device)
            rec_output,corr_output = model(feature)
            loss_1 , loss_2 = loss_fun(rec_output,feature,corr_output,label)
            loss  = 0.4*loss_1 + 0.6*loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_all+=loss.item()
            train_num_sample+=feature.shape[0]
        writer.add_scalars('loss_all',{'train_loss':train_loss_all/train_num_sample},epoch)
        print("---------------epoch:{0},train_loss:{1}----------------".format(epoch,train_loss_all/train_num_sample))
        scheduler.step()
        if epoch%10==0:
            model.eval()
            val_loss_all = 0
            val_num_sample=0
            for _,(feature,label) in enumerate(val_iter):
                feature,label = feature.to(device),label.to(device)
                rec_output,corr_output = model(feature)
                loss_1,loss_2 = loss_fun(rec_output,feature,corr_output,label)
                loss  = 0.4*loss_1 + 0.6*loss_2
                val_loss_all+=loss.item()
                val_num_sample+=feature.shape[0]
            writer.add_scalars('loss_all',{'val_loss':val_loss_all/val_num_sample},epoch)
            print("---------------test_loss:{0}----------------".format(val_loss_all/val_num_sample))
    torch.save(model,save_dir+"/model.pth")

#-----------------------------------------------------------------------------------------------------------------------------

def AE_train(model,train_iter,test_iter,lr_rate,num_epoch,save_name):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    def loss_fun(rec_output, x, corr_output,label):
        rec_loss = nn.MSELoss()
        loss_1 = rec_loss(rec_output,x)
        corr_loss = nn.MSELoss()
        loss_2 = corr_loss(corr_output,label)
        return  loss_1 , loss_2
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',0.5)
    writer = SummaryWriter('/main/result/{0}_log'.format(save_name))
    for epoch in range(num_epoch):
        model.train()
        train_loss_all=0
        train_loss_1=0
        train_loss_2=0
        train_num_sample=0
        for _,(feature,label,_) in enumerate(train_iter):
            feature,label = feature.to(device),label.to(device)
            rec_output,corr_output = model(feature)
            loss_1 , loss_2 = loss_fun(rec_output,feature,corr_output.squeeze(),label)
            loss  = 0.4*loss_1 + 0.6*loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_all+=loss.item()
            train_loss_1+=loss_1.item()
            train_loss_2+=loss_2.item()
            train_num_sample+=feature.shape[0]
        writer.add_scalars('loss_all',{'train_loss':train_loss_all/train_num_sample},epoch)
        writer.add_scalars('loss_1',{'train_loss':train_loss_1/train_num_sample},epoch)
        writer.add_scalars('loss_2',{'train_loss':train_loss_2/train_num_sample},epoch)
        print("---------------epoch:{0},loss:{1}----------------".format(epoch,train_loss_all/train_num_sample))
        if epoch%10==0:
            model.eval()
            test_loss_all = 0
            test_loss_1=0
            test_loss_2=0
            test_num_sample=0
            for _,(feature,label, _) in enumerate(test_iter):
                feature,label = feature.to(device),label.to(device)
                rec_output,corr_output = model(feature)
                loss_1,loss_2 = loss_fun(rec_output,feature,corr_output.squeeze(),label)
                loss  = 0.4*loss_1 + 0.6*loss_2
                scheduler.step(loss)
                test_loss_all+=loss.item()
                test_loss_1+=loss_1.item()
                test_loss_2+=loss_2.item()
                test_num_sample+=feature.shape[0]
            writer.add_scalars('loss_all',{'test_loss':test_loss_all/test_num_sample},epoch)
            writer.add_scalars('loss_1',{'test_loss':test_loss_1/test_num_sample},epoch)
            writer.add_scalars('loss_2',{'test_loss':test_loss_2/test_num_sample},epoch)
            print("---------------test_loss:{0}----------------".format(test_loss_all/test_num_sample))
    # torch.save({'{0}_model'.format(save_name): model.state_dict()}, '{0}_model.pth'.format(save_name))



def MLP_train(model,train_iter,test_iter,lr_rate,num_epoch,save_name):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    # model.apply(weights_init)
    loss_fun = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',0.5)
    writer = SummaryWriter('/main/result/{0}_log'.format(save_name))
    for epoch in range(num_epoch):
        model.train()
        train_loss=0
        train_num_sample=0
        for _,(feature,label,_) in enumerate(train_iter):
            feature,label = feature.to(device),label.to(device)
            predict = model(feature)
            loss = loss_fun(predict.squeeze(),label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            train_num_sample+=feature.shape[0]
        writer.add_scalars('loss',{'train_loss':train_loss/train_num_sample},epoch)
        print("---------------epoch:{0},loss:{1}----------------".format(epoch,train_loss/train_num_sample))
        if epoch%10==0:
            model.eval()
            test_loss = 0
            test_num_sample=0
            for _,(feature,label,_) in enumerate(test_iter):
                feature,label =feature.to(device),label.to(device)
                pridict = model(feature)
                loss = loss_fun(pridict.squeeze(),label)
                scheduler.step(loss)
                test_loss+=loss.item()
                test_num_sample+=feature.shape[0]
            writer.add_scalars('loss',{'test_loss':test_loss/test_num_sample},epoch)
            print("---------------test_loss:{0}----------------".format(test_loss/test_num_sample))
    # torch.save({'{0}_model'.format(save_name): model.state_dict()}, '{0}_model.pth'.format(save_name))

def CNN_train(model,train_iter,test_iter,lr_rate,num_epoch,save_name):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    # model.apply(weights_init)
    loss_fun = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',0.5)
    writer = SummaryWriter('/main/result/{0}_log'.format(save_name))
    for epoch in range(num_epoch):
        model.train()
        train_loss=0
        train_num_sample=0
        for _,(feature,label,_) in enumerate(train_iter):
            feature,label = feature.to(device),label.to(device)
            predict = model(feature)
            loss = loss_fun(predict.squeeze(),label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            train_num_sample+=feature.shape[0]
        writer.add_scalars('loss',{'train_loss':train_loss/train_num_sample},epoch)
        print("---------------epoch:{0},loss:{1}----------------".format(epoch,train_loss/train_num_sample))
        if epoch%10==0:
            model.eval()
            test_loss = 0
            test_num_sample=0
            for _,(feature,label,_) in enumerate(test_iter):
                feature,label =feature.to(device),label.to(device)
                pridict = model(feature)
                loss = loss_fun(pridict.squeeze(),label)
                scheduler.step(loss)
                test_loss+=loss.item()
                test_num_sample+=feature.shape[0]
            writer.add_scalars('loss',{'test_loss':test_loss/test_num_sample},epoch)
            print("---------------test_loss:{0}----------------".format(test_loss/test_num_sample))
    # torch.save({'{0}_model'.format(save_name): model.state_dict()}, '{0}_model.pth'.format(save_name))

def VAE_train(train_iter,test_iter,input_dimension,lr_rate,num_epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dimension,only_decoder=False).to(device)
    # model.apply(weights_init)
    def loss_fun(recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss()
        BCE = recon_loss(recon_x,x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    writer = SummaryWriter('/YYY/log')
    for epoch in range(num_epoch):
        model.train()
        train_loss=0
        train_num_sample=0
        for _,data in enumerate(train_iter):
            data =data.to(device)
            output,z,mu,logvar = model(data)
            loss = loss_fun(output,data,mu,logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            train_num_sample+=data.shape[0]
        writer.add_scalars('loss',{'train_loss':train_loss/train_num_sample},epoch)
        print("---------------epoch:{0},loss:{1}----------------".format(epoch,train_loss/train_num_sample))
        if epoch%2!=0:
            model.eval()
            test_loss = 0
            test_num_sample=0
            for _,data in enumerate(test_iter):
                data = data.to(device)
                output,z,mu,logvar = model(data)
                loss = loss_fun(output,data,mu,logvar)
                test_loss+=loss.item()
                test_num_sample+=data.shape[0]
            writer.add_scalars('loss',{'test_loss':test_loss/test_num_sample},epoch)
            print("---------------test_loss:{0}----------------".format(test_loss/test_num_sample))
    
def GNN_train(model,train_iter,test_iter,lr_rate,num_epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    # model.apply(weights_init)
    loss_fun = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    writer = SummaryWriter('/main/log')
    for epoch in range(num_epoch):
        model.train()
        train_loss=0
        train_num_sample=0
        for _,data in enumerate(train_iter):
            feature,edge_index,batch,label = data.x,data.edge_index,data.batch,data.y
            feature,edge_index,batch,label =feature.to(device),edge_index.to(device),batch.to(device),label.to(device)
            pridict = model(feature,edge_index,batch)
            pridict = pridict.squeeze(-1)
            loss = loss_fun(pridict,label.to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            train_num_sample+=feature.shape[0]/2
        writer.add_scalars('loss',{'train_loss':train_loss/train_num_sample},epoch)
        print("---------------epoch:{0},loss:{1}----------------".format(epoch,train_loss/train_num_sample))
        if epoch%10==0:
            model.eval()
            test_loss = 0
            test_num_sample=0
            for _,data in enumerate(test_iter):
                feature,edge_index,batch,label = data.x,data.edge_index,data.batch,data.y
                feature,edge_index,batch,label =feature.to(device),edge_index.to(device),batch.to(device),label.to(device)
                pridict = model(feature,edge_index,batch)
                pridict = pridict.squeeze(-1)
                loss = loss_fun(pridict,label.to(torch.float32))
                test_loss+=loss.item()
                test_num_sample+=feature.shape[0]/2
            writer.add_scalars('loss',{'test_loss':test_loss/test_num_sample},epoch)
            print("---------------test_loss:{0}----------------".format(test_loss/test_num_sample))
    torch.save({'GNN_model': model.state_dict()}, 'GNN_model.pth')           

