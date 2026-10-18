import numpy

nbLignes=6
nbColonnes=7
nbAlignes=4

joueur1=1
joueur2=-1
caseVide=0

def initial_state():
    return numpy.zeros([nbLignes,nbColonnes],dtype=int)

def autre_joueur(joueur):
    return -joueur

def legal_actions(state):
    actions=[]
    for action in range(0,nbColonnes):
        if state[0,action]==caseVide:
            actions.append(action)
    return actions

def play(state,action,joueur):
    if action not in legal_actions(state):
        raise ValueError("action illegale")

    next_state=numpy.copy(state)
    for ligne in range(nbLignes-1,-1,-1):
        if next_state[ligne,action]==caseVide:
            next_state[ligne,action]=joueur
            return next_state

    return next_state

def is_full(state):
    return len(legal_actions(state))==0

def winner(state):
    directions=[
        [0,1],
        [1,0],
        [1,1],
        [1,-1]
    ]

    for ligne in range(0,nbLignes):
        for colonne in range(0,nbColonnes):
            joueur=state[ligne,colonne]
            if joueur!=caseVide:
                for direction in directions:
                    compteur=1
                    for k in range(1,nbAlignes):
                        ligneTest=ligne+k*direction[0]
                        colonneTest=colonne+k*direction[1]
                        if ligneTest<0 or ligneTest>=nbLignes or colonneTest<0 or colonneTest>=nbColonnes:
                            break
                        if state[ligneTest,colonneTest]!=joueur:
                            break
                        compteur=compteur+1
                    if compteur==nbAlignes:
                        return joueur
    return caseVide

def is_finished(state):
    return winner(state)!=caseVide or is_full(state)

def resultat(state,joueur):
    gagnant=winner(state)
    if gagnant==joueur:
        return 1
    if gagnant==autre_joueur(joueur):
        return -1
    return 0

def encode_state(state,joueur):
    encodage=numpy.zeros([2,nbLignes,nbColonnes],dtype=float)
    for ligne in range(0,nbLignes):
        for colonne in range(0,nbColonnes):
            if state[ligne,colonne]==joueur:
                encodage[0,ligne,colonne]=1.0
            elif state[ligne,colonne]==autre_joueur(joueur):
                encodage[1,ligne,colonne]=1.0
    return encodage

def display(state):
    print("0 1 2 3 4 5 6")
    for ligne in range(0,nbLignes):
        ligneTexte=""
        for colonne in range(0,nbColonnes):
            if state[ligne,colonne]==joueur1:
                symbole="X"
            elif state[ligne,colonne]==joueur2:
                symbole="O"
            else:
                symbole="."
            ligneTexte=ligneTexte+symbole
            if colonne<nbColonnes-1:
                ligneTexte=ligneTexte+" "
        print(ligneTexte)
