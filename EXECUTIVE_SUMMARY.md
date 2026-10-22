# Retail Intelligence : prévision de la demande et optimisation des stocks

**Auteur :** Abderrahmane Azzou · **Dépôt :** [github.com/Abderrahmane-mega/retail-intelligence](https://github.com/Abderrahmane-mega/retail-intelligence)

## Problème

Une chaîne de magasins doit décider chaque jour combien commander pour chaque produit. Commander trop peu provoque des ruptures et des ventes perdues, commander trop immobilise du capital. Le projet transforme l'historique de ventes en une prévision à 14 jours, puis en une quantité à commander.

## Approche

- **Données :** Corporación Favorita (Kaggle), 54 magasins × 33 familles de produits, soit 1 782 séries et environ 3 millions de lignes (2013 à août 2017). Pipeline reproductible : ingestion, validation, puis SQL en trois couches (raw, clean, features) sous DuckDB.
- **Modèle :** LightGBM global, avec toutes les variables décalées d'au moins 14 jours (aucune fuite de données). Validation chronologique : entraînement jusqu'au 20 juin 2017, puis 28 jours de validation et 28 jours de test.
- **Décision :** politique de réapprovisionnement à niveau cible (prévisions × facteur de sécurité calibré par famille sur la validation). Le niveau de service (0,80) est choisi en minimisant le coût simulé sur la validation, puis évalué une seule fois sur le test.
- **Livrable :** dashboard Streamlit avec prévisions, simulateur de stock et recommandation de commande.

## Résultats sur le test (19 juillet au 15 août 2017)

| | LightGBM | Naïve (moy. 3 lags) | Naïve (lag 14) |
|---|---|---|---|
| **WAPE** | **14,2 %** | 16,1 % | 18,6 % |

À procédure de stock de sécurité identique, ne changer que la prévision donne :

- **coût simulé : -8,5 %** ;
- **stock moyen : -20,9 %** ;
- taux de service quasi inchangé (95,8 % contre 96,0 %).

Face à une règle empirique (marge fixe de 15 %), le coût baisse de 12,3 % et le stock de 40,5 %, mais le service est plus bas (95,8 % contre 97,1 %, avec 45 % d'unités manquées en plus). Le résultat est robuste : LightGBM bat la baseline naïve aux 8 niveaux de service testés, avec une réduction de coût de 2,8 % à 12,2 %.

## Hypothèses et limites

- Le jeu de données ne contient ni prix ni marge. Les coûts sont des hypothèses (rupture = 1 par unité, stockage = 0,1 par unité et par jour). L'impact est **estimé par simulation**, pas mesuré sur une entreprise réelle.
- Une seule fenêtre de test de 28 jours, hors période de fêtes. Les ventes observées sont assimilées à la demande.
- Le gain est concentré sur les familles à gros volume. Neuf familles sur 33, de petite taille, sont moins bien servies que par la règle empirique.
- Près de 90 % de l'importance du modèle vient de l'historique récent (`lag_14` : 74 %, moyenne glissante sur 7 jours : 15 %). Les promotions, le trafic et le pétrole apportent peu à cet horizon.

## Prochaines étapes

Validation walk-forward sur plusieurs fenêtres (dont décembre), intégration de prix et de marges réels, intervalles de prédiction par régression quantile, déploiement.
