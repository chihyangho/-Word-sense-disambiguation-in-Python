###########################################################
          PROJET DE DESAMBIGUATION LEXICAL
                    README
###########################################################
           HO Chih-Yang, KOMADINA Santiago

*Sur certaines initialisations le programme plante, faut relancer en variant les paramettres, merci*

fichier principal : main_wsd.py
  arguments: lemme(int), nb_cluster(int), t_exemple(str), t_feature(int), t_model(str), n_grmas(int), --test
   

Ce programme vise a trouver la meilleur clusterisation pour un ensemble de phrases où est présent un verbe à désambiguer parmi "abattre", "aborder", "compter", "comprendre" et "affecter".
Conceptuellement le programme se divise en trois parties:
  1) création d'exemples
  2) vectorisation d'exemples
  3) clusterisation des exemples (avec KMeans et Hierarchical) & récupération du meilleur modèle

Nous proposons deux types d'exemples :
  a) l'exemple linéaire : on prend les features selectionnées dans contexte linéaire de taille n (n mots avant et après notre verbe dans la phrase)
  b) l'exemple de dépendance : on prend le noued père et enfants(éventuel(s)) du noued de notre verbe dans l'abre de dépendance syntaxique associé à la phrase en question.

  features :
    Nous proposons de selectionner 5 possibles features présentes dans les données conll mis à disposition:
      - texte (conll[1]) - lemme(conll[2]) -POS (normal) (conll[3]) - POS (dep) (conll[4]) - relation de dépendance syntaxique avec le père (conll[7], attention, il faut parser ces données)

Pour la vectorisation deux possibilités existent :
  1) Soit la vectorisation par tf-idf qui fonctionne avec n'importe quelle feature pour les exemples linéaires
  2) Soit en chargeant des embeddings de mots pré entrainés (actuellemnt marche avec les vecteurs mis à disposition par fastText à https://fasttext.cc/docs/en/crawl-vectors.html)
    et en réalisant la somme des embeddings pour chaque mot dans un exemple.
    Marche pour tout type d'exemple MAIS JUSTE AVEC LA FEATURE DE TEXTE (conll[1])

Pour la partie clustering on utilise deux algorithmes KMeans et HierarchicalClustering(que l'on va tester avec différents paramètres de distance et de linkage)
Qu'on lance sur le même corpus puis on affiche les performances.
