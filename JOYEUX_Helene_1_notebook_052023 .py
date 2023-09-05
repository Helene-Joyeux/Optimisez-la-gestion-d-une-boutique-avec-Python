#!/usr/bin/env python
# coding: utf-8

# # Chargement des libraires utiles dans cette analyse

# In[180]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # 1) Analyse exploratoire des fichiers fournis

# ## 1.1 Fichier erp

# In[181]:


# Importation du document et visualisation
erp = pd.read_excel('erp.xlsx')
display(erp.head())

# Informations
erp.info()

#Recherche valeurs nulles
print(erp.isna().sum())

# Modification du type de données de product_id car ce n'est pas un entier mais un identifiant
erp['product_id'] = erp['product_id'].astype(str)

# Description de mes données
display(erp.describe(include='all'))

# Recherche doublons
doublons = erp.duplicated(keep=False)
print(erp[doublons])
if erp[doublons].empty :
    print("ERP ne contient pas de doublon.")
else :
    print("ERP contient des doublons.")

# Operation pour trouver la clé primaire (nombre de valeurs uniques = nombre de lignes)
print(erp.nunique() == erp.shape[0])


# #### Product_id est la clé primaire du fichier erp

# In[182]:


# Tri des valeurs product_id afin de détecter d'éventuelles erreurs dans la variable product_id
print(erp['product_id'].sort_values())
# pas d'erreur

# Calcul nombre de produits vendus en ligne
compte = erp['onsale_web'].value_counts()
print(compte)
total_produit_vendu_web = (compte[1])
total_produit_vendu_hors_lignes = (compte[0])
print('{} produits sont vendus en lignes'.format(total_produit_vendu_web))
print('{} produits sont vendus hors lignes'.format(total_produit_vendu_hors_lignes))

# Vérification des stocks
# Nombre de produits sans stock
produit_stock_vide = erp.loc[erp['stock_status'] == 'outofstock']
nb_produit_stock_vide = produit_stock_vide.shape[0]
print("{} produits n'ont plus de stock".format(nb_produit_stock_vide))

# Vérification que le minimum/maximum/moyenne de quantité des produits sans stock soit bien zéro
display(produit_stock_vide.describe())

# Nombre de produits en stock
produit_en_stock = erp.loc[erp['stock_status'] == 'instock']
nb_produit_en_stock = produit_en_stock.shape[0]
print("{} produits en stock".format(nb_produit_en_stock))

# Vérification que le minimum de quantité des produits avec stock soit bien supérieur à zéro
display(produit_en_stock.describe())

# Identification du produit qui a un stock de zéro alors qu'indiqué en stock
print(produit_en_stock.loc[produit_en_stock['stock_quantity'] == 0])

# Il faut vérifier la variable prix mais je le fais après de manière plus approfondi (voir suite)


# #### A première vu il n'y a rien d'aberrant dans le minimum/maximum/moyenne de la variable prix. A vérifier de manière plus approdondi.

# #### Le product_id 4954 est indiqué comme en stock alors qu'il à 0 en stock quantité

# ## 1.2 Fichier web

# In[183]:


# Importation du document et visualisation
web = pd.read_excel('web.xlsx')
display(web.head())

# Informations
web.info()

# Description de mes données
display(web.describe(include='all'))

# Création d'une copie du fichier web
web_corrige = web.copy()

# Suppression des colonnes vides et celles remplis par 0 par défault c'est à dire avec une moyenne de 0
del web_corrige['virtual']
del web_corrige['downloadable']
del web_corrige['tax_class']
del web_corrige['rating_count']
del web_corrige['average_rating']
del web_corrige['menu_order']
del web_corrige['comment_count']
del web_corrige['post_content']
del web_corrige['post_password']
del web_corrige['post_parent']
del web_corrige['post_content_filtered']
del web_corrige['post_mime_type']

# Recherche valeurs nulles
print(web_corrige.isna().sum())

# Recherche des sku nulles
mask_sku_null = web_corrige['sku'].isnull()
display(web_corrige[mask_sku_null])

# Identification des lignes sku NaN mais avec des produits associés
mask_postname_notnull = web_corrige['post_name'].notnull()
web_sans_sku = web_corrige. loc[(mask_sku_null) & (mask_postname_notnull)]
display(web_sans_sku.head())

# Conservation des lignes ayant un sku car c'est le point commun avec les autres fichiers
mask_sku_notnull = web_corrige['sku'].notnull()
web_notnull = web_corrige.loc[mask_sku_notnull]
display(web_notnull.head())

# Modification du type de données de total sales
web_notnull['total_sales'] = web_notnull['total_sales'].astype(int)


# In[184]:


# Recherche de doublons dans tous le dataframe web_notnull
doublons = web_notnull.duplicated(keep=False)
print(web_notnull[doublons])
if web_notnull[doublons].empty :
    print("Web ne contient pas de doublon.")
else :
    print("Web contient des doublons.")

# Valeurs uniques
print(web_notnull.nunique())

# Rappel nombre de valeurs dans sku
print(web_notnull['sku'].notnull().sum())

# Identification des doublons de sku
display(web_notnull.loc[web_notnull['sku'].duplicated(keep=False),:])

# Valeurs des deux colonnes ayant deux valeurs
print(web_notnull['post_author'].value_counts())
print(web_notnull['post_type'].value_counts())

#Sélection des lignes produits pour supprimer les doublons (supression des lignes avec les pièce-jointes)
web_product = web_notnull.loc[web_notnull['post_type'] == 'product']
display(web_product.head())

# Vérification qu'il n'y a plus de doublons de sku dans le dataframe web_product
display(web_product.loc[web_product[['sku']].duplicated(keep=False),:])

# Remise à jour des index
web_product.reset_index(inplace=True, drop = True)


# In[185]:


# Tri des valeurs sku afin de détecter d'éventuelles erreurs dans la variable sku
# print(web_product['sku'].sort_values())
# Impossible car il y a des str

# web product est déjà une sélection des sku non null

# Modification du type de données de sku non nulls en int, errors_coerce 
# pour que les valeurs qui ont un mauvais format se tranforment en NA et qu'elles soient identifiables
sku_nonnull_int = pd.to_numeric(web_product['sku'], errors = 'coerce')
display(sku_nonnull_int.head())

# Identification des sku qui sont devenus des NA
display(sku_nonnull_int[sku_nonnull_int.isna()])
display(web_product.iloc[[0]])
display(web_product.iloc[[712]])


# #### '13127-1' correspond à une sous-catégorie de vin
# 
# #### Le produit bon cadeau à un sku en chaîne de caractères
# 

# In[186]:


# Operation pour trouver la clé primaire (nombre de valeurs uniques = nombre de lignes)
print(web_product.nunique() == web_product.shape[0])

# Visualisation des valeurs uniques par variable
print(web_product.nunique())

# Vérification doublons post_title
# Visualisation de la colonne entièrement
pd.set_option('max_colwidth',None)
mask = web_product[['post_title']].duplicated(keep=False)
display(web_product.loc[mask, ['post_title', 'post_name']])


# #### En ce qui concerne les doublons de titres : Domaine Hauvette IGP Alpilles Jaspe 2017 et Clos du Mont-Olivet Châteauneuf-du-Pape 2007, il doit s'agir d'un vin blanc et d'un vin rouge.
# 
# #### En ce qui concerne Marc Colin Et Fils Chassagne-Montrachet Blanc Les Vide-Bourses 1er Cru 2016 est-ce un réel doublon ?

# #### La clé primaire du fichier web sera sku

# ## 1.3 Fichier liaison

# In[187]:


# Importation du document et visualisation
liaison = pd.read_excel('liaison.xlsx')
display(liaison.head())

# Renommage de la colonne id_web
liaison.rename(columns = {'id_web' : 'sku'}, inplace=True)

# Informations
liaison.info()

# Modification du type de données de product_id
liaison['product_id'] = liaison['product_id'].astype(str)

# Description de mes données
display(liaison.describe(include='all'))

# Recherche valeurs nulles
print(liaison.isna().sum())

# Calcul nombre de produits associés à une référence sku
nb_sku_associes_product_id = liaison['sku'].value_counts().sum()
print("{} product_id sont associés à une référence sku".format(nb_sku_associes_product_id))

# Recherche doublons
doublons = liaison.duplicated(keep=False)
print(liaison[doublons])
if liaison[doublons].empty :
    print("Liaison ne contient pas de doublon.")
else :
    print("Liaison contient des doublons.")

# Operation pour trouver la cle primaire (nombre de valeurs uniques = nombre de lignes)
print(liaison.nunique() == liaison.shape[0])


# #### Product_id est la clé primaire du fichier liaison

# In[188]:


# Tri des valeurs product_id afin de détecter d'éventuelles erreurs dans la variable product_id
print(liaison['product_id'].sort_values())
# pas d'erreur

# Tri des valeurs sku afin de détecter d'éventuelles erreurs dans la variable product_id
# print(liaison['sku'])
# Impossible car il y a des str

# Sélection des sku non nulls
mask = liaison['sku'].notnull()
sku_nonnull = liaison.loc[mask]
display(sku_nonnull.head())

# Modification du type de données de sku non nulls en int, errors_coerce 
# pour que les valeurs qui ont un mauvais format se tranforment en NA et qu'elles soient identifiables
sku_nonnull_int = pd.to_numeric(sku_nonnull['sku'], errors = 'coerce')

# Remise à jour des index
liaison.reset_index(inplace=True, drop = True)

# Identification des sku qui sont devenus des NA
mask = sku_nonnull_int.isna()
print(sku_nonnull_int.loc[mask])
display(liaison.iloc[[443]])
display(liaison.iloc[[822]])
display(liaison.iloc[[823]])

liaison.info()


# #### 13127-1 et 14680-1 correspondent à une sous-catégorie
# 
# #### Le produit bon cadeau à un sku en chaîne de caractères

# ## 2) Rapprochement des fichiers

# ### 2.1) Jointure entre le fichier erp et le fichier liaison 

# In[189]:


# Jointure à gauche entre le dataframe erp et le dataframe liaison pour garder l'ensemble des product_id présent dans erp
erp_liaison = erp.merge(liaison, on ='product_id', how ='left', indicator = True)
display(erp_liaison.head())

#Vérification que tous les produits d'erp sont bien présents
print(erp_liaison['_merge'].value_counts())

# Info sur le dataframe erp_liaison
erp_liaison.info()

#Suppression de la colonne merge
del erp_liaison['_merge']


# ### 2.2) Jointure entre le fichier erp_liaison et le fichier web

# In[190]:


# Jointure externe entre le dataframe erp_liaison et le dataframe web_product 
# pour dans un premier temps conserver les produits des deux dataframes.
erp_liaison_web_outer = erp_liaison.merge(
    web_product, on = 'sku', how ='outer', indicator = True)
display(erp_liaison_web_outer.head())

#Vérification de la jointure externe
print(erp_liaison_web_outer['_merge'].value_counts())

# Identification des produits qui étaient dans erp_liaison et pas dans web_product
mask = erp_liaison_web_outer['_merge'] == 'left_only'
produits_presents_queERP = erp_liaison_web_outer[mask]
display(produits_presents_queERP)
print("{} produits sont absents du fichier web".format(produits_presents_queERP.shape[0]))

# Vérification de si ses produits sont vendus dans la boutique en ligne
nb_produits_absents_web = produits_absents_web['onsale_web'].value_counts()
display(nb_produits_absents_web)
display('{} produits sont censés être vendus en ligne et mais sont sans sku, donc il y a pas de correspondance'.format(
    nb_produits_absents_web[1]))

# Identification des 3 produits qui sont censés être vendus en ligne
# mais qui n'ont pas de correspondance dans web car pas de sku
display(erp_liaison_web_outer.loc[(erp_liaison_web_outer['onsale_web'] == 1) & (
    erp_liaison_web_outer['sku'].isnull())])

# Identification des produits vendus hors ligne donc qui ne devraient pas trouver de correspondance dans web 
# mais qui ont un sku
produits_vendus_hors_ligne = erp_liaison_web_outer.loc[(erp_liaison_web_outer['onsale_web'] == 0) & (
    erp_liaison_web_outer['sku'].notnull())]
display(produits_vendus_hors_ligne)
print('{} produits ont un sku dans erp mais ne sont pas vendus en ligne, et sont donc absent du fichier web.'.format(
    produits_vendus_hors_ligne.shape[0]))

# Nous conservons donc uniquement les produits qui ont une correspondance dans les deux tables
# (ce qui correspond à une jointure interne.
erp_liaison_web = erp_liaison_web_outer.loc[erp_liaison_web_outer['_merge'] == 'both']

# Identification du produit qui avait un problème de stock dans le fichier erp
mask = erp_liaison_web['product_id'] == '4954'
display(erp_liaison_web.loc[mask])


# ## 3) Calcul du chiffre d'affaires par produit

# In[191]:


# Vérification que tous les produits sont bien vendus en ligne
print(erp_liaison_web['onsale_web'].value_counts())

# Calcul et création de la colonne correspondante au CA
erp_liaison_web['CA'] = erp_liaison_web['price']*erp_liaison_web['total_sales']

# Tri de la colonne CA et visualtion de certaines colonnes
display(erp_liaison_web[['product_id','post_name','CA','total_sales']].sort_values('CA', ascending=False))

# Sélection des 10 produits rapportant le plus de CA
top_10_produit_CA = erp_liaison_web[[
    'product_id','post_name','CA','total_sales']].sort_values('CA', ascending=False).head(10)
display(top_10_produit_CA.head(10))


# ##  4) Calcul du chiffre d'affaires en ligne

# In[192]:


# tous les produits présents dans la jointure sont vendus en ligne
Total_CA_enligne = erp_liaison_web['CA'].sum()

print('Le CA total de vente en ligne s élève à {} euros'.format(Total_CA_enligne))


# # 5) Analyse univariée variable prix

# In[193]:


#Description de la variable prix : min, max, écart-type, moyenne, quantiles...
display(erp_liaison_web['price'].describe())

# Calcul de l'étendue
prix_max = erp_liaison_web['price'].max()
prix_min = erp_liaison_web['price'].min()

etendue_prix = prix_max - prix_min
print("L'étendue de la plage de prix est de {} euros.".format(etendue_prix))

# Calcul du milieu de l'étendu
milieu_etendu_prix = (prix_max + prix_min)/2
print("Le milieu de l'étendue de la plage de prix est de {} euros.".format(milieu_etendu_prix))

# Calcul de la variance empirique corrigée
variance_corrigee = round(erp_liaison_web['price'].var(ddof=0),2)
print("La variance corrigée est de {}". format(variance_corrigee))

# calcul du coefficient de variation
coef_variation_prix = round(erp_liaison_web['price'].std()/erp_liaison_web['price'].mean()*100,2)
print('Le coefficient de variation est de {} % les données sont donc très dispersées.'.format(coef_variation_prix))


# #### La valeur max paraît très élévée comparé à la moyenne.
# 
# #### L’écart-type mesure à quel point les valeurs de la série statistique diffèrent de la moyenne de la série.
# #### Plus une distribution est dispersée, plus son écart-type est élevé, les valeurs de la variable prix sont donc très dispersées.
# 
# #### La valeur 50% correspond à la médiane,ce qui veut dire qu'il y a autant de vins vendus à plus 23,55 euros qu'à moins de 23,55 euros.

# #### Plus l'étendue est grande plus les données sont dispersées, est ici l'étendu est très importante.

# #### Variance très élevée donc la dispersion est très importante

# In[194]:


# Historgramme pour visualiser la répartition des données
fig, ax = plt.subplots(1,1)

liste_prix = erp_liaison_web['price'].tolist()

bins = 5

n, bins, patches = ax.hist((liste_prix))

ax.set_ylabel('Nombre de vins')
ax.set_xlabel('Prix')
plt.title("Répartition des prix de vin", fontsize = 15)


# #### On observe visuellement qu'une grande partie des vins ont un prix inférieur à 25 euros et puis une partie moins importante à moins de 50 euros,etc.

# In[195]:


# Calcul de l'écart interquartile
q3, q1 = np.percentile (erp_liaison_web['price'],[75,25])
iqr = q3 - q1
print("75% des vins sont vendus à un prix inférieur à {} euros ".format(round(q3,2))) 

# Calcul de la skewness 
skewness = round(erp_liaison_web['price'].skew(),2)
display("Le skweness est {}".format(skewness))
if skewness > 0.5 :
    print('La skewness étant positive cela signifie que la majorité de nos données se situent à droite et que notre dataset est asymétrique.')
elif skewness < -0.5  :
    print('La skewness étant négative cela signifie que la majorité de nos données se situent à gauche et que notre dataset est asymétrique.')
else :
    print('La skewness est proche de 0 cela signifie que la majorité de nos données se situent au centre et que notre dataset est symétrique.')
    
    # Calcul mode
mode_prix = erp_liaison_web['price'].mode()
print("Le prix le plus donné, est {} euros.".format(mode_prix[0]))


# In[196]:


# Représentation des données de la variable pris en boîte à moustaches,
erp_liaison_web.boxplot(column="price", vert=False)
plt.show()


# #### Cette visualisation nous permet d'identifier des outliers, les ronds au dessus de la moustaches supérieures soit 80 euros sont des outliers.

# In[197]:


# Calcul du kurtosis
kurtosis = round(erp_liaison_web['price'].kurtosis(),2)
print('Le kurtosis est de {}.'.format(kurtosis))
if kurtosis > 0 :
    print("Etant donné que le kurtosis est supérieur à 0, cela indique qu'il y a un regroupement d'outliers.")
else :
    print("Etant donné que le kurtosis est inférieur à 0, cela indique qu'il n'y a pas de regroupement d'outliers.")

# Calcul moyenne et écart-type
moyenne_prix = erp_liaison_web['price'].mean()
ecart_type = erp_liaison_web['price'].std()

# Identification des outliers grâce au Zscore
erp_liaison_web['zcore_prix'] = ((erp_liaison_web['price'] - moyenne_prix) / ecart_type)

# Aperçu du Z-score
display(erp_liaison_web['zcore_prix'].describe())

# Recherche outliers au dessus du prix habituel
outliers_zcore = erp_liaison_web.loc[abs(erp_liaison_web['zcore_prix']) > 2]
nb_valeurs_inhabituelles = outliers_zcore['product_id'].value_counts().sum()
print("Il y a {} valeurs inhabituelles dans la variable prix".format(nb_valeurs_inhabituelles))

# Top 10 des outliers
display(outliers_zcore.sort_values('zcore_prix', ascending = False).head(10))


# In[198]:


# Courbe de lorenz sur la variable ventes
# Plus la courbe de Lorenz est proche de la première bissectrice, plus la répartition est égalitaire.
ventes = erp_liaison_web['total_sales'].values

n = len(ventes)

lorenz = np.cumsum(np.sort(ventes))/ventes.sum()
lorenz = np.append([0], lorenz)

xaxis = np.linspace(0-1 / n,1+1 / n,len(lorenz))

bins = np.linspace(0, 1)

plt.plot(xaxis,lorenz,drawstyle ='steps-post',label = 'courbe de concentration')
plt.plot(bins,bins,'--', label = 'première bissectrice')
plt.title('Courbe de lorenz de la variable ventes des vins')
plt.xlabel('Pourcentage total des ventes')
plt.ylabel('Ventes cumulées de bouteille de vins')
plt.grid(axis = 'x')
plt.grid(axis = 'y')
plt.yticks([0,0.25,0.5,0.75,1],['0','25%','50%','75%','100%'])
plt.xticks([0,0.25,0.5,0.75,1],['0','25%','50%','75%','100%'])
plt.legend()
plt.show()

# Calcul indice de gini
# AUC = Area under the curve
# Calcul air sous la courbe de lorenz
AUC = (lorenz.sum()-lorenz[-1]/2 - lorenz[0]/2)/n

# Calcul aire entre la courbe de lorenz est la première bissectrices
S = 0.5 - AUC

# Calcul indice de gini
gini= 2*S
print("L'indice de gini est de {}".format(round(gini,2)))


# #### 50% des vins realisent 100% du total des ventes de vins.

# #### Plus l'indice de Gini est proche de 0 plus la répartition des données est égalitaire. Plus l'indice de Gini est proche de 1 moins la répartition est égalitaire. La répartition des ventes de vins est donc très inégalitaire.
