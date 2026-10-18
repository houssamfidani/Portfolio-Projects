import os
import pygame
import torch
import puissance4
import model
import mcts

tailleCase=90
hauteurTexte=80
largeur=puissance4.nbColonnes*tailleCase
hauteur=puissance4.nbLignes*tailleCase+hauteurTexte
rayon=35
nbSimulations=50

couleurFond=(245,245,245)
couleurPlateau=(40,90,180)
couleurVide=(255,255,255)
couleurJoueur1=(220,50,50)
couleurJoueur2=(240,210,40)
couleurTexte=(20,20,20)

joueurHumain=puissance4.joueur1
joueurIA=puissance4.joueur2

def charger_model():
    net=model.AlphaZeroNet()
    chemin=os.path.join(os.path.dirname(__file__),"model_3.pth")
    net.load_state_dict(torch.load(chemin,map_location="cpu"))
    net.eval()
    return net

def texte_resultat(state):
    gagnant=puissance4.winner(state)
    if gagnant==joueurHumain:
        return "Humain gagne"
    if gagnant==joueurIA:
        return "IA gagne"
    if puissance4.is_full(state):
        return "Match nul"
    return "A vous de jouer"

def draw(screen,font,state):
    screen.fill(couleurFond)
    texte=font.render(texte_resultat(state),True,couleurTexte)
    screen.blit(texte,(20,25))

    pygame.draw.rect(screen,couleurPlateau,(0,hauteurTexte,largeur,hauteur-hauteurTexte))

    for ligne in range(0,puissance4.nbLignes):
        for colonne in range(0,puissance4.nbColonnes):
            x=colonne*tailleCase+tailleCase//2
            y=hauteurTexte+ligne*tailleCase+tailleCase//2
            if state[ligne,colonne]==puissance4.joueur1:
                couleur=couleurJoueur1
            elif state[ligne,colonne]==puissance4.joueur2:
                couleur=couleurJoueur2
            else:
                couleur=couleurVide
            pygame.draw.circle(screen,couleur,(x,y),rayon)

    pygame.display.flip()

def jouer_ia(state,net):
    if puissance4.is_finished(state):
        return state
    action,pi,root=mcts.choose_action_with_mcts(state,joueurIA,net,nbSimulations)
    return puissance4.play(state,action,joueurIA)

def nouvelle_partie():
    return puissance4.initial_state()

pygame.init()
screen=pygame.display.set_mode((largeur,hauteur))
pygame.display.set_caption("Puissance 4 AlphaZero")
font=pygame.font.SysFont("Arial",28)
clock=pygame.time.Clock()

net=charger_model()
state=nouvelle_partie()
running=True

while running:
    draw(screen,font,state)

    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False

        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_r:
                state=nouvelle_partie()

        if event.type==pygame.MOUSEBUTTONDOWN:
            if not puissance4.is_finished(state):
                colonne=event.pos[0]//tailleCase
                if colonne in puissance4.legal_actions(state):
                    state=puissance4.play(state,colonne,joueurHumain)
                    draw(screen,font,state)
                    state=jouer_ia(state,net)

    clock.tick(30)

pygame.quit()
