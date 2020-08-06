#enconding : utf8
import numpy as np

types_feat = {
    1 : 'texte',
    2 : 'lemme',
    3 : 'POS (normal)',
    4 : 'POS (dep)',
    7 : 'relation de dépendance syn'
    }

def load_conll_file(path):
    """
    Lit un fichier .conll et renvoit une liste de listes de listes modélisant les exemples
    """
    examples = []
    with open(path, 'r', encoding="utf8") as txt:
        cas = []
        for s in txt.readlines():
            if s != "\n":
                cas += [s.split()]
            else :
                examples.append(cas)
                cas = []
    return examples

def read_one_val_per_line(path):
    """
    Charge un fichier contenant une valeure numérique par ligne et renvoie une liste contenant les valeurs indicées par leur numéro de ligne
    """
    val = []
    with open(path, 'r', encoding="utf8") as doc:
        val = [int(val) for val in doc.readlines()]
    return val

def extract_data(data, field):
    """
    - data : list[list[list]] : l'information chargé en format .conllu
    - field : int : indice du champ à extraire
    retourne une liste de string où chaque élément est un exemple du corpus d'apprentissage
    """
    data_extracted = []

    for exemple in data:
        token = []
        for e in exemple:
            token.append(e[field])
        data_extracted.append(" ".join(token))

    return data_extracted

def save_embeds(i2w, embeds, filename):
    """
    Enregistre les embeddings de mot dans un fichier
    """
    with open("embeds/"+filename, 'w', encoding="utf8") as doc:
        for i, word in enumerate(i2w):
            doc.write("\n"+word+" "+" ".join(str(emb) for emb in embeds[i]))



def save_text_data_for_training(lexemes, nom_fichier):
    """
    Enregistre les champts de data un exemple par ligne
    """
    data = []
    tokens = []
    targets = []
    for lexeme in lexemes:
        raw_data = load_conll_file("data/"+lexeme+"/"+lexeme+".conll")
        tokens += read_one_val_per_line("data/"+lexeme+"/"+lexeme+".tok_ids")
        targets += read_one_val_per_line("data/"+lexeme+"/"+lexeme+".gold")

        data += extract_data(raw_data, 1).tolist()
    """
    for i, tok_id, gold in zip(range(len(data)), tokens, targets):
        data[i].split()[tok_id] = str(data[i].split()[tok_id])+str(gold)
    """
    with open(nom_fichier, "w", encoding="utf8") as f:
        for row in data:
            f.write("\n"+str(row))
    print("Fichier de données extraites enregistré.")

def load_text_data(path):
    """
    Lit un fichier de phrases ligne par ligne
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        data = [str(ligne[:len(ligne)-1]).split() for ligne in f.readlines()]
    return data


def create_lin_exemples(data, token_ids, n, feature):
    """
    Crée des examples d'apprentissage en prenant n mot comme contexte au tour du verbe à analyser
    On pred soit le contexte linéaire textuel (feature=1)
    Soit le contexte linéaire syntaxique (POS tagging /POS dep feature = 3, 4)
    On ne prend pas le mot cible en considération
    """
    lin_data = extract_data(data, feature)
    #On gère pour les caractères de début et fin de phrase, en espérant que les embeddings les prenne pour UKN
    char_deb = ["*d"+str(i)+"*" for i in range(n)]
    char_fin = ["*f"+str(i)+"*" for i in range(n)]
    norm_data = [char_deb + exemple.split() + char_fin for exemple in lin_data]
    lin_exemples = [norm_data[te][ti-1:ti+n-1]+norm_data[te][ti+n:ti+2*n] for te, ti in enumerate(token_ids)]

    return lin_exemples



#Fonctions pour traiter les arbres de dépendance syntaxique
def get_father(sample, tok_id, feature):
    """
    Retourne le père éventuel d'un noeud token à l'intérieur de l'arbre de dépendance syntaxique d'une phrase
    """
    father = []
    id = str(sample[tok_id-1][6])
    if '|' in id:
        #print(id)
        i = id.index('|')
        id = int(id[:i])
    else :
        id = int(id)
    if id > 0:
        father = [sample[id][feature]]
    return father

def get_children(sample, tok_id, feature):
    """
    Retourne les noeuds enfants dans l'arbre de dépendance syntaxique associés à un token
    """
    children = []
    for token in sample:
        if "|" not in str(token[6]):
            if int(token[6])==tok_id:
                children.append(token[feature])
        else:
            i = token[6].index('|')
            id = int(token[6][:i])
            if id == tok_id:
                children.append(token[feature])
    return children

def make_deep_tree(sample, tok_id, feature):
    """
    Crée l'arbre de dépendance syntaxique à une branche de notre verbe (un père et les enfants directs si existance)
    """
    word = [sample[tok_id-1][feature]]
    father = get_father(sample, tok_id, feature)
    children = get_children(sample, tok_id, feature)
    """
    while i < depth:
        if id_f > 0:
            id_f, t_father = get_father(sample, id_f, feature)
            father = father+t_father
    """
    return father+word+children

def make_exemples(data, tok_id, t_exemple, feature, n):
    """
    Crée les exemples prêts à la vectorisation
    t_exemple = {'lin', 'dep'} lin pour prendre le contexte (len = n) linéaire de la feature
    feature = information à extraire :
        - feature = 1 : texte
        - feature = 2 : lemme
        - feature = 3 : POS (normal)
        - feature = 4 : POS (dep)
        - feature = 7 : relation de dépendance syntaxique avec le père dans l'arbre
    """

    if t_exemple == 'lineaire':
        exemples = create_lin_exemples(data, tok_id, n, feature)
    else:
        #t_exemple == 'dependance':
        exemples = [make_deep_tree(sample, tok_id[i], feature) for i, sample in enumerate(data)]
    print("Exemples de type %s avec %s features créés" % (t_exemple, types_feat[feature]))
    return exemples
