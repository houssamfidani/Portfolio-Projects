import numpy
import puissance4
import model

cpuct=1.0
nbSimulations=50

class Node:
    def __init__(self,state,joueur):
        self.state=state
        self.joueur=joueur
        self.children={}
        self.N=numpy.zeros(puissance4.nbColonnes)
        self.W=numpy.zeros(puissance4.nbColonnes)
        self.Q=numpy.zeros(puissance4.nbColonnes)
        self.P=numpy.zeros(puissance4.nbColonnes)
        self.developpe=False

def developper(node,net):
    p,v=model.policy_value(net,node.state,node.joueur)
    actions=puissance4.legal_actions(node.state)
    somme=0
    for action in actions:
        node.P[action]=p[action]
        somme=somme+node.P[action]

    if somme==0:
        for action in actions:
            node.P[action]=1.0/len(actions)
    else:
        for action in actions:
            node.P[action]=node.P[action]/somme

    node.developpe=True
    return v

def selection(node):
    meilleureAction=None
    meilleurScore=-1000000000
    actions=puissance4.legal_actions(node.state)

    for action in actions:
        score=node.Q[action]+cpuct*node.P[action]/(1+node.N[action])
        if score>meilleurScore:
            meilleurScore=score
            meilleureAction=action

    return meilleureAction

def simulation(node,net):
    if puissance4.is_finished(node.state):
        return puissance4.resultat(node.state,node.joueur)

    if not node.developpe:
        return developper(node,net)

    action=selection(node)

    if action not in node.children:
        next_state=puissance4.play(node.state,action,node.joueur)
        node.children[action]=Node(next_state,puissance4.autre_joueur(node.joueur))

    value=-simulation(node.children[action],net)

    node.N[action]=node.N[action]+1
    node.W[action]=node.W[action]+value
    node.Q[action]=node.W[action]/node.N[action]

    return value

def run_mcts(state,joueur,net,nb_simulations=nbSimulations):
    root=Node(state,joueur)

    for i in range(0,nb_simulations):
        simulation(root,net)

    pi=numpy.zeros(puissance4.nbColonnes)
    somme=numpy.sum(root.N)
    actions=puissance4.legal_actions(state)

    if somme==0:
        for action in actions:
            pi[action]=1.0/len(actions)
    else:
        for action in actions:
            pi[action]=root.N[action]/somme

    return pi,root

def choose_action_with_mcts(state,joueur,net,nb_simulations=nbSimulations):
    pi,root=run_mcts(state,joueur,net,nb_simulations)
    return int(numpy.argmax(pi)),pi,root

if __name__=="__main__":
    net=model.AlphaZeroNet()
    model.model_init(net)

    state=puissance4.initial_state()
    state=puissance4.play(state,0,puissance4.joueur1)
    state=puissance4.play(state,0,puissance4.joueur2)
    state=puissance4.play(state,1,puissance4.joueur1)
    state=puissance4.play(state,1,puissance4.joueur2)
    state=puissance4.play(state,2,puissance4.joueur1)
    state=puissance4.play(state,2,puissance4.joueur2)

    action,pi,root=choose_action_with_mcts(state,puissance4.joueur1,net,50)
    puissance4.display(state)
    print(pi)
    print(action)
