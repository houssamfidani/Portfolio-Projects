import torch
import torch.nn as nn
import puissance4

sizeInput=2*puissance4.nbLignes*puissance4.nbColonnes
sizeHiddenLayer=64
sizeOutputPolicy=puissance4.nbColonnes
sizeOutputValue=1
init_bound=0.1

class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet,self).__init__()

        self.l1=nn.Linear(sizeInput,sizeHiddenLayer,bias=False)
        self.l2=nn.Linear(sizeHiddenLayer,sizeHiddenLayer,bias=False)
        self.policy=nn.Linear(sizeHiddenLayer,sizeOutputPolicy,bias=False)
        self.value=nn.Linear(sizeHiddenLayer,sizeOutputValue,bias=False)

    def forward(self,x):
        x=x.view(-1,sizeInput)
        x=torch.tanh(self.l1(x))
        x=torch.tanh(self.l2(x))
        p=self.policy(x)
        v=torch.tanh(self.value(x))
        return p,v

def model_init(net):
    for p in net.parameters():
        torch.nn.init.uniform_(p,-init_bound,init_bound)

def state_tensor(state,joueur):
    encodage=puissance4.encode_state(state,joueur)
    return torch.FloatTensor(encodage.reshape(sizeInput))

def policy_value(net,state,joueur):
    with torch.no_grad():
        p,v=net(state_tensor(state,joueur).unsqueeze(0))
        p=torch.softmax(p,dim=1)[0]
    return p.numpy(),float(v[0,0])

if __name__=="__main__":
    net=AlphaZeroNet()
    model_init(net)
    state=puissance4.initial_state()
    p,v=policy_value(net,state,puissance4.joueur1)
    print(p)
    print(v)
