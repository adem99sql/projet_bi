Projet de Business Intelligence : Analyse des réservations hôtelières
1. Description du projet
Ce projet se concentre sur l’analyse des données de réservation hôtelière afin d’en extraire des informations exploitables pour le secteur de l’hôtellerie. Il implique la création d’une solution complète de Business Intelligence, de l’extraction et de la transformation des données (ETL) à l’entreposage des données, en passant par la création de tableaux de bord et la modélisation prédictive. Le jeu de données principal utilisé est celui de la « Demande de réservations hôtelières », qui contient des informations détaillées sur les réservations d’un hôtel urbain et d’un complexe hôtelier entre juillet 2015 et août 2017.

2. Objectifs du projet
Les principaux objectifs de ce projet sont les suivants :

Conception de l’entrepôt de données : Concevoir et mettre en œuvre un entrepôt de données performant pour stocker et organiser les données de réservation hôtelière à l’aide d’un modèle dimensionnel.
Implémentation du processus ETL : Développer un pipeline ETL robuste pour extraire les données de la source, les transformer dans un format adapté et les charger dans l’entrepôt de données.
Création de tableaux de bord : Concevoir et développer des tableaux de bord interactifs permettant de visualiser les indicateurs clés de performance (KPI) et de présenter des analyses.
Application d'apprentissage automatique : Appliquer des techniques d'apprentissage automatique à l'entrepôt de données afin de développer des modèles prédictifs pour résoudre des problèmes métier spécifiques (par exemple, prévoir les annulations de réservations).
Documentation et présentation : Documenter minutieusement l'ensemble du processus et présenter les résultats de manière professionnelle.
3. Ensemble de données
Nom : Demande de réservations hôtelières
Source : (Originaire d'un article de Nuno Antonio, Ana Almeida et Luis Nunes, mais souvent trouvée sur Kaggle et d'autres référentiels de données. Vous pouvez préciser votre source exacte si vous la connaissez.)
Description : Contient environ 119 390 enregistrements de réservation pour un hôtel urbain et un complexe hôtelier, avec 32 variables couvrant les détails de la réservation, les informations client, les détails du séjour et le statut d'annulation.
Période : Réservations effectuées entre juillet 2015 et août 2017.
4. Technologies utilisées
Visualisation des données et tableaux de bord : Microsoft Power BI
ETL (Extraction, Transformation, Chargement) : Talend Open Studio (ou version Talend spécifiée)
Data Warehouse : MySQL
Machine Learning et analyse statistique : R
Version Control : Git et GitHub
5. Structure du projet
Une structure de projet bien organisée est essentielle à la maintenabilité et à la collaboration. Voici une suggestion de structure.

6. Phases du projet (résumé)
Analyse et conception (1 semaine) :
Étude du jeu de données et compréhension des variables.
Conception d’un modèle dimensionnel (schéma en étoile, en flocon ou en constellation).
Définition des dimensions et des faits.
Mise en œuvre de l’ETL et de l’entrepôt de données (2 semaines) :
Configuration de l’environnement technique (MySQL).
Développement de scripts ETL (Talend).
Mise en œuvre du processus de chargement des données dans l’entrepôt de données.
Validation de l’intégrité des données.
Analyse et visualisation (1 semaine) :
Développement de tableaux de bord interactifs (Power BI).
Création de visualisations pour les indicateurs clés de performance (KPI).
Apprentissage automatique (2 semaines) :
Formulation d’une problématique métier (par exemple, prédiction des annulations).
Préparation des données pour l’apprentissage automatique.
Sélection et test des algorithmes.
Évaluation des performances du modèle et sélection du meilleur modèle.
7. Analyse des indicateurs clés de performance (KPI)
Ce projet vise à analyser divers KPI, notamment :

Performance des réservations :
Taux d’occupation (par type d’hôtel, mois, saison)
Revenu moyen par chambre disponible (RevPAR)
Durée moyenne de séjour (par segment de marché)
Répartition des réservations par canal
Comportement client :
Taux d’annulation global (et par segment)
Délai moyen entre la réservation et l’arrivée
Taux de fidélisation des clients
Origine géographique des clients (principaux pays)
Efficacité opérationnelle :
Écart entre le type de chambre demandé et le type de chambre attribué
Nombre moyen de demandes spéciales par réservation
Variations saisonnières de la demande
Prévisions :
Prévision du taux d’occupation à court terme
8. Utilisation et exécution du projet
Prérequis :
Serveur MySQL
Talend Open Studio (ou la version utilisée)
R et RStudio (ou votre environnement R préféré)
Power BI Desktop
Configuration :
Cloner le dépôt : git Clone https://github.com/adem99sql/projet_bi/Hotel-Booking-Analysis
Base de données :
Créez la base de données MySQL.
Exécutez les scripts de création de schéma situés dans data/.
ETL :
Importez les jobs Talend depuis etl/ dans votre Talend Open Studio.
Configurez les connexions aux bases de données dans les jobs Talend.
Exécutez le job ETL principal pour alimenter l'entrepôt de données à partir des données brutes (assurez-vous que data/raw/ contient le jeu de données).
Machine Learning :
Ouvrez les scripts R depuis Machine learning/ dans RStudio.
Assurez-vous que tous les packages R requis sont installés (vous pouvez inclure un fichier requirements.R ou lister les packages ici).
Définissez le répertoire de travail et les informations de connexion à la base de données dans les scripts.
Exécutez les scripts pour l'analyse et l'entraînement du modèle.
Tableaux de bord :
Ouvrez les fichiers .pbix depuis dashboards/ dans Power BI Desktop.
Actualisez les sources de données, en vous assurant que Power BI peut se connecter à votre entrepôt de données MySQL ou aux extraits de données fournis.
Adem khadhraoui,Mariem dridi
