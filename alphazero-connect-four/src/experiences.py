import apprentissage
import test

configurations=[
    ["config_1",100,50,10,"model_1.pth","loss_1.png"],
    ["config_2",200,100,20,"model_2.pth","loss_2.png"],
    ["config_3",300,150,50,"model_3.pth","loss_3.png"]
]

nomResultats="resultats_experiences.txt"

with open(nomResultats,"w") as fichier:
    for configuration in configurations:
        nom=configuration[0]
        nbParties=configuration[1]
        nbSimulations=configuration[2]
        nbEpochs=configuration[3]
        nomModele=configuration[4]
        nomPlot=configuration[5]

        print(nom)
        pertes=apprentissage.entrainement(nbParties,nbSimulations,nbEpochs,nomModele,nomPlot)
        aleatoire_j1,aleatoire_j2=test.tester_model(nomModele,test.nbEpisodes,test.nbSimulations,"aleatoire",False)
        ameliore_j1,ameliore_j2=test.tester_model(nomModele,test.nbEpisodes,test.nbSimulations,"ameliore",False)

        fichier.write("%s\n" % nom)
        fichier.write("parties : %d\n" % nbParties)
        fichier.write("simulations : %d\n" % nbSimulations)
        fichier.write("epochs : %d\n" % nbEpochs)
        fichier.write("perte finale : %f\n" % pertes[-1])
        fichier.write("test aleatoire joueur 1 : %f %f\n" % (aleatoire_j1[0],aleatoire_j1[1]))
        fichier.write("test aleatoire joueur 2 : %f %f\n" % (aleatoire_j2[0],aleatoire_j2[1]))
        fichier.write("test amélioré joueur 1 : %f %f\n" % (ameliore_j1[0],ameliore_j1[1]))
        fichier.write("test amélioré joueur 2 : %f %f\n\n" % (ameliore_j2[0],ameliore_j2[1]))
        fichier.flush()

print(nomResultats)
