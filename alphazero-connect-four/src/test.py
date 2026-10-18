import random
import torch
import puissance4
import model
import mcts

nbEpisodes=50
nbSimulations=50
nomModele="model.pth"

def action_aleatoire(state):
    actions=puissance4.legal_actions(state)
    return random.choice(actions)

def action_gagnante(state,joueur):
    actions=puissance4.legal_actions(state)
    for action in actions:
        next_state=puissance4.play(state,action,joueur)
        if puissance4.winner(next_state)==joueur:
            return action
    return None

def action_amelioree(state,joueur):
    action=action_gagnante(state,joueur)
    if action is not None:
        return action
    action=action_gagnante(state,puissance4.autre_joueur(joueur))
    if action is not None:
        return action
    return action_aleatoire(state)

def action_ia(state,joueur,net,nb_simulations=nbSimulations):
    action,pi,root=mcts.choose_action_with_mcts(state,joueur,net,nb_simulations)
    return action

def action_adversaire(state,joueur,type_adversaire):
    if type_adversaire=="ameliore":
        return action_amelioree(state,joueur)
    return action_aleatoire(state)

def jouer_partie(net,joueurIA,nb_simulations=nbSimulations,type_adversaire="aleatoire"):
    state=puissance4.initial_state()
    joueur=puissance4.joueur1

    while not puissance4.is_finished(state):
        if joueur==joueurIA:
            action=action_ia(state,joueur,net,nb_simulations)
        else:
            action=action_adversaire(state,joueur,type_adversaire)

        state=puissance4.play(state,action,joueur)
        joueur=puissance4.autre_joueur(joueur)

    return puissance4.resultat(state,joueurIA)

def tester_joueur(net,joueurIA,nb_episodes=nbEpisodes,nb_simulations=nbSimulations,type_adversaire="aleatoire",afficher=True):
    averageNumberSuccesses=0
    averageNumberDraws=0

    for i in range(0,nb_episodes):
        if afficher and i%10==0:
            print(i)
        resultat=jouer_partie(net,joueurIA,nb_simulations,type_adversaire)
        if resultat==1:
            averageNumberSuccesses=averageNumberSuccesses+1
        if resultat==0:
            averageNumberDraws=averageNumberDraws+1

    averageNumberSuccesses=averageNumberSuccesses/nb_episodes
    averageNumberDraws=averageNumberDraws/nb_episodes
    return averageNumberSuccesses,averageNumberDraws

def charger_model(nom_modele=nomModele):
    net=model.AlphaZeroNet()
    net.load_state_dict(torch.load(nom_modele))
    net.eval()
    return net

def tester_model(nom_modele=nomModele,nb_episodes=nbEpisodes,nb_simulations=nbSimulations,type_adversaire="aleatoire",afficher=True):
    net=charger_model(nom_modele)
    j1=tester_joueur(net,puissance4.joueur1,nb_episodes,nb_simulations,type_adversaire,afficher)
    if afficher:
        print(j1[0])
        print(j1[1])
    j2=tester_joueur(net,puissance4.joueur2,nb_episodes,nb_simulations,type_adversaire,afficher)
    if afficher:
        print(j2[0])
        print(j2[1])
    return j1,j2

if __name__=="__main__":
    tester_model(nomModele,nbEpisodes,nbSimulations)
