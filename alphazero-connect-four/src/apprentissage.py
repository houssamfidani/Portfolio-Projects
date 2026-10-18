import random
import numpy
import torch
import matplotlib.pyplot as plt
import puissance4
import model
import mcts

nbParties=100
nbSimulations=50
nbEpochs=10
batchSize=64
alpha=0.001
nomModele="model.pth"
nomPlot="loss.png"

def choix_aleatoire_politique(pi):
    u=random.uniform(0,1)
    somme=0
    for action in range(0,puissance4.nbColonnes):
        somme=somme+pi[action]
        if u<somme:
            return action
    return int(numpy.argmax(pi))

def jouer_partie_auto(net,nb_simulations=nbSimulations):
    state=puissance4.initial_state()
    joueur=puissance4.joueur1
    etats=[]
    joueurs=[]
    politiques=[]

    while not puissance4.is_finished(state):
        pi,root=mcts.run_mcts(state,joueur,net,nb_simulations)
        etats.append(numpy.copy(state))
        joueurs.append(joueur)
        politiques.append(numpy.copy(pi))
        action=choix_aleatoire_politique(pi)
        state=puissance4.play(state,action,joueur)
        joueur=puissance4.autre_joueur(joueur)

    exemples=[]
    for i in range(0,len(etats)):
        z=puissance4.resultat(state,joueurs[i])
        exemples.append([etats[i],joueurs[i],politiques[i],z])

    return exemples,state

def generer_parties(net,nb_parties=nbParties,nb_simulations=nbSimulations):
    exemples=[]
    for i in range(0,nb_parties):
        if i%1==0:
            print(i)
        exemplesPartie,state=jouer_partie_auto(net,nb_simulations)
        exemples=exemples+exemplesPartie
    return exemples

def batch_tensors(exemples,indices):
    states=[]
    politiques=[]
    resultats=[]

    for indice in indices:
        states.append(model.state_tensor(exemples[indice][0],exemples[indice][1]))
        politiques.append(torch.FloatTensor(exemples[indice][2]))
        resultats.append(exemples[indice][3])

    states=torch.stack(states)
    politiques=torch.stack(politiques)
    resultats=torch.FloatTensor(resultats).reshape((len(indices),1))
    return states,politiques,resultats

def loss_alpha_zero(net,states,politiques,resultats):
    p,v=net(states)
    log_p=torch.log_softmax(p,dim=1)
    policy_loss=-torch.mean(torch.sum(politiques*log_p,dim=1))
    value_loss=torch.mean(torch.square(resultats-v))
    return value_loss+policy_loss

def entrainer_model(net,exemples,nb_epochs=nbEpochs,batch_size=batchSize):
    optimizer=torch.optim.Adam(net.parameters(),lr=alpha)
    pertes=[]

    for epoch in range(0,nb_epochs):
        print("epoch : %d" % epoch)
        random.shuffle(exemples)
        meanLoss=0
        nbBatchs=0

        for i in range(0,len(exemples),batch_size):
            indices=list(range(i,min(i+batch_size,len(exemples))))
            states,politiques,resultats=batch_tensors(exemples,indices)
            loss=loss_alpha_zero(net,states,politiques,resultats)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meanLoss=meanLoss+float(loss.detach())
            nbBatchs=nbBatchs+1

        perte=meanLoss/nbBatchs
        pertes.append(perte)
        print("mean loss : %f" % perte)

    return pertes

def plot_pertes(pertes,nom_fichier=nomPlot):
    plt.figure(figsize=(6,4))
    plt.plot(range(0,len(pertes)),pertes,marker="o",linewidth=2)
    plt.xlabel("Epoque")
    plt.ylabel("Perte moyenne")
    plt.title("Evolution de la perte")
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(nom_fichier,dpi=160)
    plt.close()

def entrainement(nb_parties=nbParties,nb_simulations=nbSimulations,nb_epochs=nbEpochs,nom_modele=nomModele,nom_plot=nomPlot):
    net=model.AlphaZeroNet()
    model.model_init(net)

    exemples=generer_parties(net,nb_parties,nb_simulations)
    print(len(exemples))
    pertes=entrainer_model(net,exemples,nb_epochs,batchSize)
    plot_pertes(pertes,nom_plot)
    torch.save(net.state_dict(),nom_modele)
    print(nom_modele)
    return pertes

if __name__=="__main__":
    entrainement(nbParties,nbSimulations,nbEpochs,nomModele,nomPlot)
