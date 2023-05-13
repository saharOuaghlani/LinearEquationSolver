import sys
import time
import numpy as np #bibliotheque pour manipuler les matrices
import random
import numpy.linalg as alg
from numpy import array, zeros, fabs, linalg

#Le but est la résolution de l'equation linéaire 𝐴𝑋 = 𝑏 avec la méthode de Gauss et la décomposition LU.

print("                 --------------------------------------------------------")
print("                |                      ~ Mini Projet ~                   |")
print("                |           PROGRAMME DE RESOLUTION DE AX = b            |")
print("                 --------------------------------------------------------")


#Calcul de conditionnement de A

def Cond(A,nA,s): #A est la matrice, nA est la taille de la matrice
                  #s est le choix de la norme de conditionnement
    valAbsA=[] #liste contenant les valeurs absolues qui sont nécessaires 
               #pour le calcul de la norme de A
    valAbsInvA=[] #liste contenant les valeurs absolues qui sont nécessaires 
                  #pour le calcul de la norme de la matrice inverse de A

    if (s==1):  #Cond1(A)
                #sommes des éléments des colonnes
        #Calcul de norme de A
        for i in range(nA):
            ni=0
            for j in range(nA):
                ni+=abs(A[j,i]) 
            valAbsA.append(ni)
        normeA=max(valAbsA) #la valeur max entre les sommes
        #Calcul de norme de l'inverse de A
        for i in range(nA):
            ni=0
            for j in range(nA):
                ni+=abs(round(InvA[j,i],2)) 
            valAbsInvA.append(ni)
        normeInvA=max(valAbsInvA) #la valeur max entre les sommes
        CondA=normeA*normeInvA

    elif (s==2): #Cond2(A)
        CondA=alg.cond(A, 2)

    else: #CondInfini(A)
          #sommes des éléments des lignes
        #Calcul de norme de A
        for i in range(nA):
            ni=0
            for j in range(nA):
                ni+=abs(A[i,j]) 
            valAbsA.append(ni)
        normeA=max(valAbsA) #la valeur max entre les sommes
        #Calcul de norme de l'inverse de A
        for i in range(nA):
            ni=0
            for j in range(nA):
                ni+=abs(round(InvA[i,j],2)) 
            valAbsInvA.append(ni)
        normeInvA=max(valAbsInvA) #la valeur max entre les sommes
        CondA=normeA*normeInvA
    return CondA


#Methode de Gauss

def Gauss(a,b, n): 
    x = zeros(n, float) #Initialiser les éléments de la  matrice solution à 0

    for k in range(n-1): #Préciser la ligne de pivot
        if fabs(a[k,k]) < 1.0e-12: #traiter le cas où le pivot est null
            for i in range(k+1, n): #chercher dans les lignes au dessous de la ligne pivot
                                    # un autre pivot différent de 0
                if fabs(a[i,k]) > fabs(a[k,k]):  
                    a[[k,i]] = a[[i,k]] #permuter les lignes de la matrice A 
                    b[[k,i]] = b[[i,k]] #permuter les lignes de la matrice b
                    break
    
    #appliquer l'élimination sur les lignes au dessous de la ligne de pivot (pour les deux matrices)
        for i in range(k+1,n): 
            if a[i,k] == 0:continue  #on traite uniquement les éléments au dessous de pivot qui sont différents de 0
                                     #car quand a[i,j]=0, il n y aura pas de modification sur la ligne
            facteur = a[i,k]/a[k,k]
            for j in range(k,n): #Changement des éléments au dessous du pivot
                a[i,j] = a[i,j] - a[k,j]*facteur 
                
            b[i] = b[i] - b[k]*facteur #changement des éléments de vecteurs b

    #calculer la solution x
    x[n-1] = b[n-1] / a[n-1, n-1] #les valeurs de x doivent être calculées de la dernière équation à la première équation.
    for i in range(n-2, -1, -1): 
        som_ax = 0
  
        for j in range(i+1, n): #la sommation devrait commencer à partir de i+1 
                                #puisque seules les valeurs de x nous sont connues des calculs précédents. 
            som_ax += a[i,j] * x[j]
        
        x[i] = (b[i] - som_ax) / a[i,i] #Le résultat est ensuite divisé par l'élément correspondant dans la diagonale principale

    return x


#Decomposition LU

def luDecomposition(A,b, n):
    #Créer les matrices lower et upper de taille nxn et les remplir par des 0
    lower = [[0 for x in range(n)]
             for y in range(n)]
    upper = [[0 for x in range(n)] 
             for y in range(n)]
 
    # Decomposition de la matrice A en lower et upper
    for i in range(n):
        #on commence par le remplissage de matrice upper puisqu'elle est triangulaire supérieure 
        #et il existe des éléments de lower*upper qui dependent de upper seulement 
        for k in range(i, n): # matrice upper
            # Sommation de lower(i, j) * upper(j, k)
            sum = 0
            for j in range(i): 
                sum += (lower[i][j] * upper[j][k])

            upper[i][k] = A[i][k] - sum # Remplir upper(i, k)
 
        for k in range(i, n): # matrice lower
            if (i == k):
                lower[i][i] = 1  # Mettre les diagonales en 1 
            else:
                # Sommation de lower(k, j) * upper(j, i)
                sum = 0
                for j in range(i):
                    sum += (lower[k][j] * upper[j][i])
                
                lower[k][i] = round(float((A[k][i] - sum) /upper[i][i]),2) # Remplir lower(k, i)
    #La méthode solve permet de résoudre une equation linéaire ax=b 
    Y=alg.solve(lower,b) #Résolution de l'equation LY=b (l'inconnu est Y)
    X=alg.solve(upper,Y) #Résolution de l'equation Ux=Y (l'inconnu est x)
    return upper, lower, X



b=[]
print("\n \t ** Choisir une option ****\n")
print("\n\n\t\t\t+-----------------------------------+\n")
print("\t\t\t|  1. Saisir votre propre matrice   |\n")
print("\t\t\t+-----------------------------------+\n")
print("\t\t\t|  2. Utiliser la matrice par défaut|\n")
print("\t\t\t+-----------------------------------+\n")
s=0
while (s not in [1,2]):   
    s=int(input("\t\tVotre choix:"))


#remplir les matrices A et b
if(s==1): #l'utilisateur va donner ses propres matrices A et b 
    n= int(input("\n\t\tDonner la taille de matrice: ")) #n=taille
    A= np.zeros((n,n), float)
    #remplissage de la matrice A
    print("\n\t\t** Remplissage de la matrice A **")
    for i in range(n): #parcourir les lignes
        for j in range(n): #parcourir les colonnes
            A[i][j]=float(input("\t\tA["+str(i)+"]["+str(j)+"]="))
    print("\n\n")
    #remplissage de la matrice b
    print("\n\t\t** Remplissage de la matrice B **")
    for i in range(n):#parcourir les lignes
        b.append(float(input("\t\tb["+str(i)+"]=")))
else: #l'utilisateur ne donne pas les matrices, donc elles sont choisies aléatoirement où la taille est fixée 5
    A = np.zeros((7,7), float) 
    #remplissage de la matrice A aléatoirement
    for i in range(7): #parcourir les lignes
        for j in range(7): #parcourir les colonnes
            A[i][j]=round(random.uniform(-1000.00, 1000.00), 2) #choisir des valeurs aléatoires entre -1000.00 et 1000.00
    #remplissage de la matrice b aléatoirement
    for i in range(7): #parcourir les lignes
        b.append(round(random.uniform(-1000.00, 1000.00), 2)) #choisir des valeurs aléatoires entre -1000.00 et 1000.00



#Calcul de Cond(A)
print("\n \t ** Choisir un conditionnement ****\n")
print("\n\n\t\t\t+-----------------------------+\n")
print("\t\t\t|     1. Norme 1              |\n")
print("\t\t\t+-----------------------------+\n")
print("\t\t\t|     2. Norme 2              |\n")
print("\t\t\t+-----------------------------+\n")
print("\t\t\t|     3. Norme infinie        |\n")
print("\t\t\t+-----------------------------+\n")
s=0
while(s not in [1, 2, 3] ):
    s=int(input("\t\tVotre Choix :"))

nA=len(A) #taille de la matrice
if(alg.det(A)==0.0): #vérifier si le determinant de la matrice A est different de 0
    #Cond(A) est le produit de la norme de A et la norme de l'inverse de A, donc c'est obligatoire de vérifier que la matrice A soit inversible
     print("Le determinant de la matrice A est égal à 0. Impossible de calculer le conditionnement et d'appliquer les deux méthodes")
else:
    InvA = alg.inv(A) #la matrice inverse de A
    print("\nMatrice A:",A)
    print("\nMatrice b:",b)
    print("\n\tLe conditionnement de A: ", Cond(A,nA,s))
    print("\n \t ** Choisir La méthode de résolution: ****\n")
    print("\n\n\t\t\t+-----------------------------------+\n")
    print("\t\t\t|     1. Méthode de Gauss           |\n")
    print("\t\t\t+-----------------------------------+\n")
    print("\t\t\t|     2. Décomposition LU           |\n")
    print("\t\t\t+-----------------------------------+\n")
    choix=int(input("\t\tVotre Choix : "))
    while (choix not in [1,2]):
        choix=int(input("\t\tVotre Choix :"))
    if(choix==1):
        t1 = time.time() #pour calculer le temps de calcul du programme
        x=Gauss(A,b,nA)
        t2 = time.time()
        print("\n\t\t** Méthode de Gauss **")
        print("\n\tLa solution est le vecteur :", x)
        print("\n\tLe temps de calcul:",t2-t1)
    else:
        #vérifier si tous les sous déterminants sont différents de 0
        for i in range(1,5):
            mat_slice = A[:i,:i] #decouper la matrice en sous matrice
            det=alg.det(mat_slice) #le determinant de sous matrice
            if(det==0.0):
                print("Le déterminant de sous matrice numéro ",str(i)," est egale a 0. Impossible d'appliquer LU")
            else:
                t1 = time.time() #pour calculer le temps de calcul 
                U,L,x=luDecomposition(A,b,nA)
                t2 = time.time()
                print("\n\t\t** Décomposition LU **")
                print("\nMatrice de composition L:",L)
                print("\nMatrice de décomposition U:",U)
                print("\n\tLa solution est le vecteur:",x)
                print("\n\tLe temps de calcul:",t2-t1)
    




