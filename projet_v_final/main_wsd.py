#enconding : utf8
import argparse
import numpy as np
from time import time

from Lexeme4wsd import Lexeme
from Clustering4wsd import run_clusters



lexemes = ['abattre', 'aborder', 'affecter', 'comprendre', 'compter']

#On recupère les arguments donnés par l'utilisateur
parser = argparse.ArgumentParser()
parser.add_argument('lexeme', default=0, help='Quel lexeme utiliser ?\n\tabattre = 0\n\taborder = 1\n\taffecter = 2\n\tcomprendre = 3\n\tcompter = 4', type=int)
parser.add_argument('nb_cluster', default=3, help='Nombre de clusters à chercher', type=int)
parser.add_argument('t_exemple', default='lineaire', help='on construit les exemple de façon "lineaire" ou "dependance" ?', type=str)
parser.add_argument('t_feature', default=1, help='Quelles caractéristiques prendre en compte ?\n\tTextuel = 1\n\tSyntaxique (POS tagging normal) = 3\n\tSyntaxique (Deep POS tagging) = 4\n\tRelation syntaxique=7', type=int)
parser.add_argument('t_modele', default='tf-idf', help='Une représentation vectorielle par "tf-idf" ou par "embeds"?', type=str)
parser.add_argument('n_grams', default=2, help='Quels nombres de n-grams pour la représentation vectorielle ?', type=int)
parser.add_argument('--test', action="store_true", help='Activer pour faire tourner un test')

args = parser.parse_args()


if args.t_feature != 1 and args.t_modele == "embeds":
    print("Veuillez selectionner la feature textuelle (feat=1) pour lancer une vectorisation par embeddings.")
    exit()

#Chargement des données d'apprentissage et instasation des vecteurs
lexeme = Lexeme(lexemes[args.lexeme],  args.t_exemple, args.t_feature, args.t_modele, args.n_grams, args.test)

#On vectorise et on recupère les données issues de la vectorization et les classes gold correpondantes
X = lexeme.exemples
Y = lexeme.targets

#Partie clustering
run_clusters(args.nb_cluster, X, Y)
#On fait un plot du dendrogram pour le hierarchical cluster
#plot_cluster(Clusters.algorithms[1])
