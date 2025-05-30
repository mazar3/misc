# Chapitre 1 : Introduction à l’Intelligence Artificielle

## A1. Introduction : Qu'est-ce que l'IA et où la trouve-t-on ?
- **Questions de définition et d'applications** : L'IA cherche à créer des machines capables de simuler l'intelligence humaine pour effectuer des tâches variées. Ses applications sont vastes et en pleine expansion.
- **Exemples d'applications concrètes** :
    - **Voiture autonome** : Prise de décision, perception de l'environnement.
    - **Assistant Alexa** : Compréhension du langage naturel, réponse aux requêtes.
    - **Robocup 2018** : Coordination d'agents autonomes (robots footballeurs).
    - **Robot Boston Dynamics** : Mobilité avancée, équilibre, interaction physique.

## A2. Prérequis de l’IA : Composants Essentiels et Fondations (incluant Vision de Turing)
- **Concepts fondamentaux** : Nécessité de bases en logique, probabilités, statistiques, théorie de l'information.
- **Algorithmes** : Séquences d'instructions précises pour résoudre un problème ou effectuer une tâche. Au cœur de l'IA.
- **Big Data** : Vastes ensembles de données nécessaires pour entraîner et améliorer les modèles d'IA, notamment en apprentissage machine.
- **Cloud computing** : Fournit la puissance de calcul et le stockage nécessaires pour traiter le Big Data et exécuter des algorithmes d'IA complexes.
- **Principales étapes du parcours de l’IA (vision d'Alan Turing)** : Référence aux idées pionnières de Turing sur la possibilité pour les machines de "penser" et les capacités qu'elles devraient démontrer.

## A3. Définitions de l’IA : Quatre Approches et Leurs Caractéristiques
Il existe quatre perspectives principales pour définir l'IA :

- **1. Systèmes qui pensent comme les humains (Approche Cognitive)**
    - **Besoins** : Comprendre le fonctionnement interne du cerveau humain (via Sciences cognitives, Neurosciences cognitives).
    - **But** : Modéliser et simuler le processus de pensée humain pour la résolution de problèmes.
- **2. Systèmes qui agissent comme des humains (Approche Comportementale Humaine)**
    - **But** : Imiter le comportement humain de manière indiscernable.
    - **Test de Turing (Alan Turing, 1950)** : Un évaluateur humain dialogue (par écrit) avec une entité. Si l'évaluateur ne peut distinguer l'IA d'un autre humain, l'IA "réussit" le test.
        - **Capacités requises pour l'ordinateur (selon Turing)** :
            - Traitement du langage naturel
            - Représentation des connaissances
            - Raisonnement automatisé
            - Apprentissage (Machine Learning)
            - Vision artificielle
            - Robotique (pour le test de Turing "total")
- **3. Systèmes rationnels : Les «lois de la pensée» (Approche Logique)**
    - **Fondement** : Pensée logique, suivant des règles formelles (syllogismes d'Aristote, ex: "Socrate est un homme, tous les hommes sont mortels, donc Socrate est mortel").
    - **Logique formelle** : Utilisation de systèmes logiques pour représenter les connaissances et raisonner.
    - **Problèmes des systèmes rationnels** :
        - Difficulté à formaliser toutes les connaissances du monde réel.
        - Complexité computationnelle (explosion combinatoire).
        - Incertitude difficile à gérer.
- **4. Systèmes qui agissent rationnellement (Approche de l'Agent Rationnel)**
    - **Agent** : Entité qui perçoit son environnement via des **capteurs** (sensors) et agit sur cet environnement via des **effecteurs** (actuators).
    - **Agent rationnel** : Agent qui agit de manière à maximiser une **mesure de performance** attendue, compte tenu des informations disponibles et de ses connaissances. Il fait "la bonne chose".
    - **Exemples et avantages** : Plus général que la pensée rationnelle (l'action optimale n'est pas toujours une déduction logique), plus testable et scientifiquement rigoureux que les approches basées sur le comportement/pensée humaine.
    - **Prise de décisions rationnelles** : Choisir l'action qui mène au meilleur résultat attendu.

## A4. Histoire de l’IA : Des Origines aux Développements Récents
- **Gestation de l’IA (1943 - 1955)**
    - **McCulloch et Pitts (1943)** : Premier modèle mathématique de neurone artificiel.
    - **Minsky et Edmonds (1951)** : SNARC, premier ordinateur à réseau de neurones.
    - **Alan Turing (1950)** : Article "Computing Machinery and Intelligence" (Test de Turing).
- **Naissance de l’IA (1956)**
    - **Séminaire de Dartmouth (John McCarthy)** : Terme "Intelligence Artificielle" inventé.
    - **Logic Theorist (Newell et Simon)** : Premier programme d'IA, capable de prouver des théorèmes mathématiques.
- **Enthousiasme des débuts : grandes espérances (1952 - 1969)**
    - **General Problem Solver (GPS)** (Newell et Simon) : Tentative de créer un solveur de problèmes universel.
    - **Geometry Theorem Prover (Herbert Gelernter)** : Prouve des théorèmes de géométrie.
    - **Monde des blocs (SHRDLU - Terry Winograd)** : Compréhension du langage naturel dans un micro-monde.
- **Épreuve de la réalité (1966 - 1973) - "Premier hiver de l'IA"**
    - **Excès de confiance** : Promesses non tenues.
    - **Suspension du financement** : Notamment pour la traduction automatique (rapport ALPAC).
    - **Difficultés de passage à l'échelle (scalability)** : Problèmes combinatoires.
- **Systèmes fondés sur les connaissances (1969 - 1979)**
    - **DENDRAL** : Premier système expert (analyse de structures moléculaires).
    - **Systèmes experts** : Capturent la connaissance d'experts humains sous forme de règles (ex: MYCIN).
    - **Langages de représentation de connaissances** : RDF (Resource Description Framework), W3C, Semantic Web (vision d'un web de données structurées).
- **L’IA devient une industrie (1980 - présent)**
    - **R1 (XCON)** : Système expert commercialement réussi chez Digital Equipment Corporation (DEC).
    - **Projet 5ème génération (Japon, 1982)** : Vaste projet national visant à développer des ordinateurs massivement parallèles pour l'IA.
- **Retour des réseaux de neurones (1986 - présent)**
    - **Apprentissage par rétro-propagation (Rumelhart, Hinton, Williams)** : Algorithme clé pour entraîner les réseaux de neurones multicouches.
- **L’IA devient une science (1987 - présent)**
    - Accent sur la **reproductibilité**, l'évaluation rigoureuse, et l'**amélioration de l'existant** plutôt que des inventions totalement nouvelles.
- **Agents intelligents (1995 - présent)**
    - Développement d'**agents rationnels**, intégration dans des systèmes plus larges (ex: moteurs de recherche).
- **Disponibilités de vastes ensembles de données (2001 - présent) - "Printemps de l'IA"**
    - Le Big Data et l'augmentation de la puissance de calcul propulsent le Machine Learning et le Deep Learning.
- **Dates clés de l’IA moderne**
    - **Assistants vocaux (Siri, 2010)**
    - **GAN (Generative Adversarial Networks, Ian Goodfellow, 2014)**
    - **AlphaGo (DeepMind, 2016)** : Bat le champion du monde de Go.
    - **Transformers (Google, 2017)** : Architecture de réseau de neurones révolutionnant le NLP.
    - **ChatGPT (OpenAI, 2018 pour le modèle initial GPT, 2022 pour la version grand public)**
    - **AI Act (Union Européenne, proposition 2021, accord 2023)** : Régulation de l'IA.
- **Ligne de temps et évolution de l’IA** : Une figure visuelle récapitule ces grandes étapes et les tendances.

## A5. Approches de l’IA : Méthodologies (Symbolique, Apprentissage) et Formes (Générative, Augmentée)
- **Programmation symbolique vs. Machine Learning (connexionniste)**
    - **Détail : Programmation symbolique (IA classique / GOFAI - Good Old-Fashioned AI)**
        - **Principe** : Représentation explicite des connaissances sous forme de symboles et de règles logiques (ex: `si <condition> alors <action>`).
        - **Limites** : Difficile de créer et maintenir pour des problèmes complexes, peu adaptable aux données nouvelles ou bruitées, difficulté à acquérir les connaissances.
    - **Détail : Machine Learning (Apprentissage Machine)**
        - **Principe** : Les systèmes apprennent à partir de données sans être explicitement programmés pour chaque tâche. L'algorithme identifie des patrons (patterns) dans les données.
    - **Détail : Deep Learning (Apprentissage Profond)**
        - **Relation avec le Machine Learning** : Sous-domaine du Machine Learning utilisant des réseaux de neurones artificiels avec de nombreuses couches ("profonds").
        - **Extraction de features (caractéristiques)** : Le Deep Learning excelle dans l'apprentissage automatique des caractéristiques pertinentes (features) directement à partir des données brutes (ex: pixels d'une image), contrairement au ML classique où l'ingénierie des features est souvent manuelle.
- **Formes d’Intelligence Artificielle (selon le niveau d'intelligence/autonomie)**
    - **Intelligence Humaine** : Référence.
    - **Intelligence Artificielle (Étroite / Faible)** : Spécialisée dans une ou quelques tâches spécifiques (ex: reconnaissance d'image, traduction). C'est l'état actuel de l'IA.
    - **Intelligence Artificielle Générale (Forte)** : Capacité hypothétique d'une IA à comprendre, apprendre et appliquer des connaissances dans n'importe quel domaine, au niveau humain (non atteint).
    - **Intelligence Augmentée** : Utilisation de l'IA comme un outil pour amplifier les capacités et l'intelligence humaines, plutôt que de les remplacer.
- **Intelligence Artificielle Générative (GenAI)**
    - **Processus** : Crée de nouveaux contenus (texte, image, audio, code, etc.) originaux, en apprenant les motifs et la structure de données d'entraînement.
    - **Formes (modèles et techniques)** :
        - **GAN (Generative Adversarial Networks)** : Deux réseaux (générateur, discriminateur) en compétition pour créer des données réalistes.
        - **Modèles de Diffusion** : Apprennent à inverser un processus de "bruitage" pour générer des données à partir de bruit.
        - **LLMs (Large Language Models)** : Modèles de langage massifs entraînés sur d'immenses corpus de texte (ex: GPT-3, BERT).
        - **Audio/Video Gen** : Modèles spécialisés dans la génération de sons, musique, ou séquences vidéo.
        - **LVM/VLM (Large Vision Models / Visual Language Models)** : Modèles combinant la compréhension d'images et de texte (ex: DALL-E, CLIP).

## A6. Applications de l’IA : Domaines d'Impact, Exemples Concrets et Paradoxe de Moravec
- **Généralités et impact** : L'IA transforme de nombreux secteurs (santé, finance, transport, divertissement, etc.) et a un impact croissant sur la société.
- **Principales applications (domaines)** :
    - **Vision par ordinateur** :
        - **Tâches** : Détection d’objets, Reconnaissance de formes/visages, Segmentation d'images (délimiter des zones), Classification d'images.
        - **Exemple** : **Tesla automatic car driving** (analyse de l'environnement routier).
    - **Reconnaissance et synthèse vocale** :
        - **Exemples** : **Alexa, SIRI, Google assistant** (comprendre la parole et générer une réponse vocale).
    - **Traitement du langage naturel (NLP)** :
        - **Tâches** : Compréhension de texte, traduction, génération de texte, analyse de sentiment.
        - **Exemple** : **ChatGPT** (interaction conversationnelle avancée).
    - **Raisonnement (Logique et Planification)** :
        - **Exemples** : Démonstration automatique de théorèmes mathématiques, Vérification de preuves logiques, planification d'actions pour des robots.
    - **Robotique** :
        - **Tâches** : Navigation, manipulation d'objets, interaction avec l'environnement.
        - **Exemple** : **Robocup (robots Nao)** (coordination d'agents physiques autonomes).
    - **Jeux vidéos et Jeux de stratégie** :
        - **Exemples** : **Deep Blue (IBM)** battant Kasparov aux échecs ; **AlphaGo (DeepMind)** battant Lee Sedol au Go.
- **Paradoxe de Moravec** :
    - **Constat** : Les tâches qui sont faciles pour les humains (ex: perception sensorielle, motricité, reconnaissance de visages) sont souvent très difficiles à réaliser pour l'IA. Inversement, des tâches considérées comme complexes pour les humains (ex: calculs, résolution de jeux formels comme les échecs ou le Go) peuvent être relativement "faciles" pour l'IA (avec suffisamment de puissance de calcul et les bons algorithmes).
    - **Explication (partielle)** : Les capacités humaines "faciles" sont le fruit de millions d'années d'évolution, tandis que le raisonnement abstrait est plus récent et moins optimisé biologiquement.

# Chapitre 2 : Agents Intelligents et Systèmes Multi-Agents

## B1. Agents Intelligents au Cœur de l'IA et Rôle des Agents Rationnels
- **Définition intuitive d'un agent intelligent** : Entité logicielle ou physique capable de percevoir son environnement, de raisonner et d'agir de manière autonome pour atteindre des objectifs.
    - **Exemples** : Médecin virtuel (diagnostic), Netflix (recommandation), Siri (assistant vocal), robots (tâches physiques).
- **Place centrale des agents rationnels** : L'objectif est de construire des agents qui agissent de la **meilleure façon possible** (rationnellement) pour atteindre leurs buts.

## B2. Agents et Environnements : Définitions Fondamentales (Agent, Capteurs, Effecteurs, Séquence, Fonction vs. Programme)
- **Définition d'un agent** : Tout ce qui peut être considéré comme **percevant** son environnement à travers des **capteurs** (sensors) et **agissant** sur cet environnement à travers des **effecteurs** (actuators).
- **Agent humain vs. agent robotique/logiciel** :
    - **Humain** : Yeux, oreilles (capteurs) ; mains, jambes, voix (effecteurs).
    - **Robotique/Logiciel** : Caméras, microphones (capteurs) ; moteurs, écran, haut-parleurs (effecteurs).
- **Séquence de perception** : L'historique complet de tout ce que l'agent a perçu jusqu'à présent. Le choix d'action de l'agent à un instant `t` peut dépendre de toute cette séquence.
- **Fonction d’agent (externe)** : Spécification abstraite/mathématique qui associe une séquence de perceptions à une action. `f: P* -> A` (où P* est l'ensemble des séquences de perceptions possibles et A l'ensemble des actions). Décrit le *quoi*.
- **Programme d’agent (interne)** : Implémentation concrète de la fonction d'agent. C'est le code ou l'algorithme qui s'exécute sur l'architecture de l'agent. Décrit le *comment*.
- **Exemple : Le monde de l’aspirateur**
    - **Environnement** : Deux cases (A, B), pouvant être sales ou propres.
    - **Capteurs** : Localisation (A ou B), état de la case (Sale ou Propre).
    - **Effecteurs** : Aller à gauche, Aller à droite, Aspirer, Ne rien faire.
    - **Fonction d'agent (table)** : Peut être représentée par une table qui, pour chaque séquence de perceptions, indique l'action à effectuer.
        - Ex: `[A, Sale] -> Aspirer` ; `[A, Propre] -> Aller à Droite`
    - **Programme d'agent (code)** : Ensemble de règles ou un algorithme qui implémente cette fonction.
        - Ex: `si case_actuelle = Sale alors aspirer sinon si case_actuelle = A alors aller_à_droite sinon aller_à_gauche`
- **Fonction d’agent vs Programme d’agent** : La fonction est la spécification du comportement désiré (le "quoi"), le programme est l'implémentation qui réalise ce comportement (le "comment"). Plusieurs programmes peuvent implémenter la même fonction.

## B3. Concept de Rationalité : Être Rationnel, Mesures et Facteurs Clés (PEAS pour la Rationalité, Omniscience, Apprentissage, Autonomie)
- **Agent rationnel** : Pour chaque séquence de perceptions possible, un agent rationnel doit sélectionner une action qui est censée **maximiser sa mesure de performance**, compte tenu des informations fournies par la séquence de perceptions et de toute connaissance intégrée qu'il possède.
- **Mesure de performance** : Critère qui évalue le succès du comportement de l'agent. Elle est définie par le concepteur et doit être objective.
    - **Exemple du robot aspirateur (choix de mesure)** :
        - Nombre de cases propres sur une période donnée.
        - Consommation d'énergie.
        - Temps de fonctionnement.
        - *Une bonne mesure serait par ex. : +1 point par case nettoyée, -0.1 point par déplacement, -1 point par unité d'énergie consommée sur une durée de N pas de temps.*
- **Facteurs de rationalité (ce qui détermine la rationalité d'un choix)** :
    1.  **Mesure de performance** (qui définit le succès).
    2.  **Connaissance préalable de l’environnement** par l'agent.
    3.  **Actions possibles** pour l'agent.
    4.  **Séquence de perception** de l'agent à ce jour.
- **Rationalité de l'agent aspirateur (exemple)** : Si la mesure de performance récompense la propreté, un agent rationnel aspirera si la case est sale et se déplacera si elle est propre (en fonction de sa connaissance et de sa stratégie pour explorer).
- **Omniscience, apprentissage et autonomie** :
    - **Agent omniscient vs. rationalité** :
        - Un agent **omniscient** connaît l'issue réelle de ses actions et agit en conséquence. C'est impossible dans la réalité.
        - La **rationalité** concerne l'action attendue comme optimale, compte tenu de ce que l'agent *sait* et *perçoit*. Un agent rationnel peut faire une erreur si ses informations sont incomplètes, mais il aura fait le meilleur choix *possible*.
    - **Récolte d’informations et apprentissage** : Un agent rationnel devrait explorer son environnement pour apprendre (si l'environnement est inconnu) et améliorer ses performances futures. L'apprentissage est essentiel pour la rationalité dans des environnements complexes ou changeants.
    - **Autonomie** : Un agent est autonome si son comportement est déterminé par sa propre expérience (apprise) plutôt que uniquement par ses connaissances initiales (programmées par le concepteur). Un agent rationnel devrait être autonome.

## B4. Environnements : Caractérisation (PEAS) et Typologie (Observable, Agents, Déterminisme, Épisodicité, Dynamisme, Nature, Connaissance) et Structure/Types d'Agents
- **Définition de l'environnement de travail (PEAS)** : Cadre pour concevoir un agent rationnel.
    - **P**erformance (Measure) : Comment le succès est mesuré.
    - **E**nvironnement : Où l'agent opère.
    - **A**ctionneurs (Actuators) : Comment l'agent agit.
    - **S**enseurs (Sensors) : Comment l'agent perçoit.
- **Exemple : Robot chauffeur de taxi (description PEAS)**
    - **Performance** : Sécurité, rapidité, légalité, confort du passager, profit.
    - **Environnement** : Routes, autres véhicules, piétons, clients, météo.
    - **Actionneurs** : Volant, accélérateur, frein, clignotant, klaxon, affichage.
    - **Senseurs** : Caméras, sonar, GPS, compteur de vitesse, odomètre, clavier (pour destination).
- **Types d’environnements** (propriétés qui influencent la conception de l'agent) :
    - **Entièrement observable vs Partiellement observable (vs Non observable)** :
        - *Entièrement* : Les capteurs donnent accès à l'état complet de l'environnement à chaque instant (ex: jeu d'échecs avec plateau visible).
        - *Partiellement* : L'agent ne connaît qu'une partie de l'état (ex: poker, robot avec capteurs limités).
        - *Non observable* : L'agent n'a aucun capteur.
    - **Mono-agent vs Multi-agent** :
        - *Mono-agent* : L'agent opère seul (ex: mots croisés).
        - *Multi-agent* : Plusieurs agents dans l'environnement. Peut être :
            - **Compétitifs** (ex: échecs, jeu à somme nulle).
            - **Coopératifs** (ex: robots dans un entrepôt coordonnant leurs actions, équipe de Robocup).
    - **Déterministe vs Stochastique (vs Incertain)** :
        - *Déterministe* : L'état suivant de l'environnement est complètement déterminé par l'état actuel et l'action de l'agent (ex: jeu d'échecs sans dés).
        - *Stochastique* : L'état suivant est incertain, des probabilités peuvent être associées aux issues (ex: jeu avec dés, robot avec effecteurs imprécis).
        - *Incertain (ou non-déterministe stratégique)* : L'environnement est non-déterministe du point de vue de l'agent car les actions d'autres agents (non modélisés par des probabilités) affectent l'issue.
    - **Épisodique vs Séquentiel** :
        - *Épisodique* : L'expérience de l'agent est divisée en épisodes atomiques. Le choix de l'action dans un épisode ne dépend que de l'épisode lui-même (pas des épisodes passés). (ex: classification d'images une par une).
        - *Séquentiel* : La décision actuelle peut affecter toutes les décisions futures. L'historique est important (ex: échecs, conduite).
    - **Statique vs Dynamique (vs Semi-dynamique)** :
        - *Statique* : L'environnement ne change pas pendant que l'agent délibère (ex: mots croisés).
        - *Dynamique* : L'environnement peut changer pendant que l'agent réfléchit (ex: conduite, bourse).
        - *Semi-dynamique* : L'environnement ne change pas avec le temps, mais la performance de l'agent oui (ex: un jeu avec un temps limité par coup).
    - **Discret vs Continu** :
        - *Discret* : Nombre fini (ou dénombrable) d'états, de perceptions et d'actions distincts (ex: échecs, monde de l'aspirateur).
        - *Continu* : États, perceptions, actions ou temps évoluent sur des plages continues (ex: conduite, contrôle d'un bras robotique).
    - **Connu vs Inconnu** :
        - *Connu* : Les "lois de la physique" de l'environnement (effets des actions) sont connues de l'agent.
        - *Inconnu* : L'agent doit apprendre comment fonctionne l'environnement (nécessite exploration).
    - **Tableau récapitulatif des types d'environnements pour divers tâches** : Utile pour classer rapidement des problèmes d'IA (ex: Échecs = Entièrement obs., Multi-agent compétitif, Déterministe, Séquentiel, Statique, Discret, Connu).
- **Structure et types d’agents** :
    - **Agent = Architecture + Programme**
        - *Architecture* : Dispositif de calcul sur lequel le programme s'exécute (matériel : capteurs, effecteurs, CPU).
        - *Programme* : Implémentation de la fonction d'agent.
    - **Programme d’agent (méthodes d'implémentation)** :
        - **Tables de consultation** (lookup tables).
        - **Systèmes de règles** (si-alors).
        - **Algorithmes de recherche** (pour trouver la meilleure séquence d'actions).
        - **Algorithmes d'apprentissage** (pour s'adapter et améliorer les performances).
    - **Agents pilotés par table** :
        - **Problèmes** : La table devient immense pour la plupart des environnements réels (explosion combinatoire du nombre d'états/perceptions). Non adaptable.
        - **Exemple voiture autonome** : Impossible de lister toutes les séquences de perceptions possibles (images de caméras, etc.) et l'action correspondante.
    - **Types de programmes d’agents (en complexité croissante)** :
        1.  **Agents réactifs simples (Simple reflex agents)** : Agissent uniquement sur la base de la perception actuelle (règles condition-action). Ignorent l'historique. (Ex: `si voiture_devant_freine alors freiner`).
        2.  **Agents réactifs à base de modèle (Model-based reflex agents)** : Maintiennent un état interne pour suivre les aspects du monde non visibles actuellement. Nécessitent un modèle de l'environnement (comment le monde évolue, comment les actions affectent le monde). (Ex: savoir où sont les autres voitures même si temporairement masquées).
        3.  **Agents à base de buts (Goal-based agents)** : Agissent pour atteindre des buts explicites. Utilisent la recherche et la planification pour trouver des séquences d'actions menant au but. (Ex: un robot aspirateur avec le but "nettoyer toutes les pièces").
        4.  **Agents à base d’utilité (Utility-based agents)** : Choisissent les actions qui maximisent une fonction d'utilité (mesure du "bonheur" ou de la désirabilité d'un état). Utiles quand il y a des buts conflictuels ou incertains. (Ex: choisir l'itinéraire le plus rapide mais aussi le plus sûr).
        5.  **Agents capables d’apprentissage (Learning agents)** : Peuvent améliorer leurs performances avec l'expérience. Composés d'un "élément de performance" (choix des actions), d'un "élément d'apprentissage", d'une "critique" (évaluation des actions) et d'un "générateur de problèmes" (suggestion d'actions exploratoires).

## B5. Systèmes Multi-Agents (SMA) : Définition, Composants (AEIO), Types d'Agents, Interactions et Intelligence Collective
- **Définition et utilité des SMA** : Un SMA est un système composé de plusieurs agents autonomes qui interagissent dans un environnement commun pour résoudre des problèmes qui sont difficiles ou impossibles à résoudre par un agent monolithique.
    - **Utilité** : Résolution de problèmes distribués, robustesse, parallélisme, modularité, systèmes ouverts.
- **SMA = Agents + Environnement + Interactions + Organisations (AEIO)** (selon [Ferber])
    - **Agents** : Entités autonomes (voir ci-dessous).
    - **Environnement** : Espace partagé où les agents existent et interagissent.
    - **Interactions** : Mécanismes par lesquels les agents s'influencent (directes ou indirectes).
    - **Organisations** : Structures et règles qui gouvernent les relations et interactions entre agents (rôles, hiérarchies).
- **Agents (définition de Jacques Ferber pour les SMA)** : Une entité physique ou abstraite capable d'agir sur elle-même et son environnement, qui dispose d'une représentation partielle de cet environnement et qui peut communiquer avec d'autres agents.
    - **Propriétés d’un agent (dans un SMA)** :
        - **Autonome** : Agit sans intervention humaine directe, contrôle sur ses actions et son état interne.
        - **Proactif (ou orienté but)** : Prend des initiatives pour atteindre ses objectifs.
        - **Flexible (ou réactif)** : Perçoit son environnement et répond aux changements.
        - **Social** : Peut interagir et communiquer avec d'autres agents.
        - **Situé** : Est immergé dans un environnement qu'il perçoit et sur lequel il agit.
- **Typologie des agents (dans les SMA)** :
    - **Agent réactif** : Comportement simple stimulus-réponse. Pas de représentation symbolique du monde ni de raisonnement complexe.
    - **Agent cognitif (ou délibératif)** : Possède une représentation symbolique de l'environnement, des buts, et des capacités de raisonnement/planification. Modèle BDI (Beliefs, Desires, Intentions) est un exemple.
    - **Agent hybride** : Combine des aspects réactifs (pour la vitesse) et cognitifs (pour les décisions complexes).
    - **Couplage à l’environnement** :
        - **Fort (Tight coupling)** : L'agent est fortement influencé par l'environnement, souvent réactif.
        - **Faible (Loose coupling)** : L'agent a plus d'autonomie par rapport à l'environnement, souvent cognitif.
        - **Réactif/Cognitif-BDI** : Ces architectures influencent le type de couplage.
- **Interactions entre agents** :
    - **Communication indirecte (stigmergie)** : Les agents interagissent en modifiant l'environnement (ex: phéromones des fourmis).
    - **Communication directe** : Les agents échangent des messages via un langage de communication (ex: KQML, FIPA-ACL). Peut être : coordination, coopération, négociation.
- **Intelligence collective (ou émergente)** :
    - **Penseur isolé vs communauté de penseurs** : La résolution de problèmes complexes peut émerger des interactions d'agents simples plutôt que d'un seul agent "génial". "Le tout est plus que la somme des parties."
    - **Exemples naturels** : Colonies de fourmis (construction de nids, recherche de nourriture), bancs de poissons, volées d'oiseaux (comportements coordonnés sans leader central).

## B6. Applications & Plateformes de Développement des SMA
- **Application exemple : évacuation du public d’un stade (vidéo)** : Simulation de foules où chaque individu est un agent avec des comportements simples, dont les interactions mènent à des phénomènes d'évacuation complexes.
- **Plateformes de développement des SMA (exemples)** : Frameworks et bibliothèques facilitant la création de SMA.
    - **JADE (Java Agent DEvelopment Framework)** : Populaire, basé sur Java, conforme FIPA.
    - **Madkit** : Modulaire, basé sur des rôles organisationnels.
    - **AgentBuilder** : Outil graphique pour construire des agents.
    - **JADEX** : Intègre le raisonnement BDI avec JADE.
    - **PADE (Python Agent DEvelopment Framework)** : Framework pour agents en Python.
    - **SPADE (Smart Python Agent Development Environment)** : Autre framework Python populaire.

# Chapitre 3 : Apprentissage Automatique (Machine Learning)

## C1. Fondations et Types d'Apprentissage Machine
- **Prérequis de l’IA et Approches de l’IA (récapitulatif)** : Rappel que le ML est une approche clé de l'IA, complémentaire à la programmation symbolique, et qu'elle s'appuie sur les données et les algorithmes.
- **Programmation symbolique (limites)** : Difficulté à gérer la complexité, l'incertitude, et l'adaptation à de nouvelles situations sans reprogrammation explicite.
- **Machine Learning (concept général, types)** :
    - **Concept** : Permettre aux ordinateurs d'apprendre à partir de données sans être explicitement programmés pour chaque tâche spécifique.
    - **Types principaux** :
        - **Apprentissage Supervisé** : Apprend à partir de données étiquetées.
        - **Apprentissage Non Supervisé** : Apprend à partir de données non étiquetées, découvre des structures.
        - **Apprentissage par Renforcement** : Apprend par essais et erreurs en interagissant avec un environnement pour maximiser une récompense.
        - **Deep Learning (Apprentissage Profond)** : Sous-domaine du ML utilisant des réseaux de neurones profonds, applicable aux trois types ci-dessus.

## C2. Définition du Machine Learning : Processus et Objectif de Généralisation
- **Machine Learning : méthodes et processus d'apprentissage** : Ensemble de méthodes permettant aux ordinateurs d'améliorer leurs performances sur une tâche donnée grâce à l'expérience (les données). Le processus implique la sélection d'un modèle, son entraînement sur des données, et son évaluation.
- **Informatique traditionnelle vs. Apprentissage machine** :
    - **Traditionnelle** : `Données + Programme -> Résultats` (on code les règles).
    - **Machine Learning** : `Données + Résultats (exemples) -> Programme (modèle)` (la machine "découvre" les règles).
- **But principal : Généralisation** : Capacité du modèle appris à bien performer sur des données **nouvelles et invisibles** (non vues pendant l'entraînement), et pas seulement sur les données d'entraînement.

## C3. Types d’Apprentissage : Supervisé, Non Supervisé, Semi-Supervisé, Auto-Supervisé, Renforcement et Formulation
- **Apprentissage supervisé (Supervised Learning)** :
    - **Entrée** : **Données étiquetées** (Labeled Data), c'est-à-dire des paires (entrée, sortie désirée).
    - **Processus** : Le **modèle ML** (ML Model) apprend une fonction de mappage des entrées vers les sorties.
    - **Sortie** : **Prédictions** (Predictions) sur de nouvelles données.
    - **Tâches typiques** : Classification (prédire une catégorie), Régression (prédire une valeur continue).
- **Apprentissage non supervisé (Unsupervised Learning)** :
    - **Entrée** : **Données non étiquetées** (Unlabelled Data).
    - **Processus** : La **machine** (Machine/algorithme) essaie de trouver des structures, des motifs, ou des regroupements intrinsèques dans les données.
    - **Sortie** : **Résultats** (Results) comme des clusters, des dimensions réduites, des règles d'association.
    - **Tâches typiques** : Clustering (regroupement), Réduction de dimension, Détection d'anomalies.
- **Apprentissage semi-supervisé (Semi-supervised learning)** :
    - **Principe** : Utilise un mélange de **données étiquetées (peu nombreuses)** et de **données non étiquetées (nombreuses)** pour l'entraînement.
    - **Cas d'usage (use-case)** : Utile quand l'étiquetage des données est coûteux ou long.
- **Apprentissage auto-supervisé (Self-supervised learning - SSL)** :
    - **Principe** : Forme d'apprentissage non supervisé où les **étiquettes sont générées automatiquement à partir des données d'entrée elles-mêmes** en définissant une **tâche prétexte**.
    - **Tâche prétexte (pretext task)** : Une tâche pour laquelle les étiquettes peuvent être créées à partir des données brutes (ex: prédire une partie masquée d'une image à partir du reste).
    - **Objectif** : Apprendre des représentations utiles des données qui peuvent ensuite être transférées à des tâches en aval (avec peu ou pas de données étiquetées pour ces tâches).
    - **Exemples de tâches prétexte** :
        - **Position relative** (prédire la position relative de deux patchs d'une image).
        - **Prédiction de rotation** (prédire l'angle de rotation appliqué à une image).
        - **Inpainting** (reconstruire une partie masquée d'une image).
        - **Colorisation** (prédire les couleurs d'une image en niveaux de gris).
        - **Résolution de puzzle (Jigsaw puzzle)** (réassembler les patchs mélangés d'une image).
- **Apprentissage par renforcement (Reinforcement Learning - RL)** :
    - **Composants clés** :
        - **Agent** : L'apprenant ou le preneur de décision.
        - **Environnement** : Le monde avec lequel l'agent interagit.
        - **État (State `s`)** : La situation actuelle perçue par l'agent.
        - **Action (Action `a`)** : Ce que l'agent peut faire.
        - **Récompense (Reward `r`)** : Feedback de l'environnement indiquant si l'action était bonne ou mauvaise dans un état donné. L'agent cherche à maximiser la récompense cumulée.
    - **Online Interactions vs. Offline Reinforcement Learning** :
        - **Online (classique)** : L'agent apprend en interagissant directement et continuellement avec l'environnement.
        - **Offline (Batch RL)** : L'agent apprend à partir d'un ensemble fixe de données d'interactions collectées précédemment (pas d'interaction directe pendant l'apprentissage).

## C4. Terminologie du Machine Learning : Exemples, Modèles, Phases, Régression vs. Classification
- **Termes de base** :
    - **Exemple (ou instance, observation, feature vector) `x`** : Une entrée unique pour le modèle (ex: les caractéristiques d'une maison).
    - **Exemple étiqueté `(x, y)`** : Un exemple `x` avec sa sortie ou cible correcte `y` (utilisé en apprentissage supervisé).
    - **Entraîner le modèle (Train the model)** : Processus d'ajustement des paramètres du modèle en utilisant des données d'entraînement.
    - **Exemple sans étiquette `(x, ?)`** : Un exemple `x` dont la sortie est inconnue (utilisé en apprentissage non supervisé ou pour la prédiction).
    - **Modèle (Model)** : Le résultat de l'apprentissage ; une fonction ou structure qui peut faire des prédictions `y'` pour de nouvelles entrées.
- **Exemple tabulaire (housingMedianAge)** : Les données sont souvent organisées en tableaux où les lignes sont des exemples et les colonnes sont des **caractéristiques (features)** (ex: `housingMedianAge`, `totalRooms`) et potentiellement une **étiquette (label)** (ex: `medianHouseValue`).
- **Modèles d’apprentissage (relation caractéristiques/étiquettes)** : Le modèle cherche à découvrir la relation mathématique ou logique entre les caractéristiques d'entrée et l'étiquette de sortie.
    - **Phase d'Apprentissage (entraînement)** : Le modèle est exposé aux données d'entraînement pour ajuster ses paramètres internes.
    - **Phase d'Inférence (prédiction)** : Le modèle entraîné est utilisé pour faire des prédictions sur de nouvelles données non vues.
- **Régression vs Classification (types de tâches supervisées)** :
    - **Modèle de régression** : Prédit des **valeurs continues** (numériques) (ex: prédire le prix d'une maison, la température).
    - **Modèle de classification** : Prédit des **valeurs discrètes** (catégories ou classes) (ex: prédire si un email est un spam ou non, si une image contient un chat ou un chien).

## C5. Réduction de la Perte et Descente du Gradient : Optimisation des Modèles
- **Exemple de régression (graphique prix maison vs surface)** :
    - Modèle linéaire simple : `Y = WX + b` (ou `y = mx + c`)
        - `Y` : Étiquette (prix prédit).
        - `X` : Caractéristique (surface).
        - `W` (ou `w1`, `m`) : Poids (weight) ou pente de la caractéristique.
        - `b` (ou `w0`, `c`) : Biais (bias) ou ordonnée à l'origine.
- **Perte (Loss)** : Mesure à quel point la prédiction du modèle est "mauvaise" par rapport à la valeur réelle. Une valeur unique.
- **Perte nulle** : Signifie que les prédictions du modèle sont parfaites pour les exemples considérés.
- **Erreur (Error)** : Différence entre la valeur réelle (Actual) et la valeur prédite (Predicted) pour un seul exemple. La perte est souvent une agrégation des erreurs.
- **Perte L2 (L2 Loss) ou Perte Quadratique (Squared Loss)** :
    - **Formule** : `L2 Loss = Σ (y - y_pred)²` (somme des carrés des erreurs).
    - **MSE (Mean Squared Error - Erreur Quadratique Moyenne)** : `MSE = (1/N) * Σ (y - y_pred)²` (moyenne des carrés des erreurs). C'est une fonction de perte courante pour la régression.
- **Question sur MSE de deux ensembles** : Comparer la MSE sur deux ensembles de prédictions pour voir quel modèle/ensemble de prédictions est meilleur (MSE plus faible = meilleur).
- **Réduction de la perte : approche itérative (diagramme)** :
    - L'entraînement d'un modèle consiste à trouver les valeurs des paramètres (ex: `W`, `b`) qui minimisent la fonction de perte.
    - C'est un processus itératif : on ajuste les paramètres petit à petit pour réduire la perte.
- **Réduction de la perte : descente du gradient (Gradient Descent)** (algorithme d'optimisation courant) :
    - **Principe (graphique coût vs pondération)** : Imaginez une surface de perte. La descente du gradient essaie de "descendre la pente" pour atteindre le point le plus bas (minimum de perte).
    - **Point de départ** : Les paramètres sont initialisés (souvent aléatoirement).
    - **Gradient** : Vecteur qui indique :
        - **Direction** : La direction de la plus forte augmentation de la perte. On se déplace dans la direction *opposée* au gradient.
        - **Magnitude** : La "force" de la pente.
- **Réduction de la perte : taux d’apprentissage (learning rate, `α` ou `η`)** :
    - Hyperparamètre qui contrôle la taille des pas effectués dans la direction opposée au gradient.
    - **Taux faible** : Convergence lente, risque de rester coincé dans des minima locaux peu profonds.
    - **Taux élevé** : Peut dépasser le minimum (overshooting), voire diverger.
    - **Taux efficace (juste)** : Atteint le minimum de manière efficiente.
    - **Hyperparamètres** : Paramètres de l'algorithme d'apprentissage qui ne sont pas appris à partir des données mais fixés avant l'entraînement (ex: taux d'apprentissage, nombre de couches dans un réseau).
- **Réduction de la perte : point de départ et convexité** :
    - **Problèmes convexes** : La fonction de perte a un seul minimum global. Le point de départ n'influence pas la solution finale trouvée (on atteindra toujours le même minimum).
    - **Problèmes non convexes** (exemple : réseaux de neurones profonds) : La fonction de perte a de multiples minima locaux. Le point de départ *peut* influencer le minimum local atteint.
- **Descente de gradient pour DNN (Deep Neural Networks)** : La même idée s'applique, mais le calcul du gradient (via la rétropropagation) et la mise à jour des poids concernent tous les poids et biais du réseau.
- **Descente de gradient classique (Batch) vs Stochastique (SGD) vs par Mini-lot (Mini-Batch)** :
    - **Batch Gradient Descent (Classique)** :
        - Calcule le gradient sur l'**ensemble des données d'entraînement** avant chaque mise à jour des poids.
        - Précis mais lent et gourmand en mémoire pour de grands datasets.
        - **Formule** : Implique une somme sur tous les exemples.
    - **Stochastic Gradient Descent (SGD)** :
        - Calcule le gradient et met à jour les poids pour **chaque exemple individuellement**.
        - Rapide, moins de mémoire, mais estimations du gradient "bruyantes" (trajectoire erratique). Peut aider à échapper aux minima locaux.
    - **Mini-Batch Gradient Descent** :
        - Compromis : Calcule le gradient et met à jour les poids sur de **petits sous-ensembles (mini-lots)** des données d'entraînement.
        - Efficace, bonnes estimations du gradient, tire parti de la vectorisation matérielle. Le plus utilisé en pratique.
        - **Formule** : Somme sur les exemples du mini-lot.
    - **Époques (Epochs)** : Une époque correspond à un passage complet de l'algorithme d'apprentissage sur l'ensemble des données d'entraînement. (Ex: si 1000 exemples et mini-lot de 100, une époque = 10 itérations de mise à jour).
    - **Comparaison visuelle (Batch, Stochastic, Mini-Batch Gradient Descent)** : Graphiques montrant la trajectoire de convergence (Batch = lisse, SGD = bruité, Mini-Batch = entre les deux).

## C6. Généralisation et Représentation des Données : Éviter le Surapprentissage et Structurer les Données
- **Généralisation** : La capacité d'un modèle à faire des prédictions précises sur des données **nouvelles, non vues précédemment**, après avoir été entraîné sur un ensemble de données spécifique. C'est l'objectif principal du ML.
    - **Surapprentissage (Overfitting)** : Le modèle apprend "trop bien" les données d'entraînement, y compris le bruit et les spécificités. Il performe très bien sur les données d'entraînement mais mal sur les nouvelles données (mauvaise généralisation).
        - *Symptômes* : Perte d'entraînement faible, perte de validation/test élevée.
- **Modèle simple vs complexe** :
    - **Modèle trop simple (Underfitting)** : Ne capture pas la structure sous-jacente des données. Perte élevée sur l'entraînement et le test.
    - **Modèle trop complexe (Overfitting)** : S'adapte trop aux données d'entraînement. Perte faible sur l'entraînement, élevée sur le test.
    - **Bon modèle** : Compromis, capture la tendance générale sans mémoriser le bruit.
- **Ensemble d’apprentissage (Training set) et Ensemble d’évaluation/test (Validation/Test set)** :
    - **Ensemble d'apprentissage** : Utilisé pour entraîner le modèle (ajuster les paramètres).
    - **Ensemble de validation** : Utilisé pour ajuster les hyperparamètres du modèle et pour une évaluation intermédiaire afin de détecter le surapprentissage.
    - **Ensemble de test** : Utilisé pour une évaluation finale et impartiale de la performance du modèle final sur des données complètement nouvelles.
    - **Importance des ensembles de données** :
        - **Volume suffisant** : Nécessaire pour un bon apprentissage et une évaluation fiable.
        - **Divergence (non-chevauchement)** : Les ensembles d'entraînement, de validation et de test doivent être distincts et idéalement provenir de la même distribution de données que celle que le modèle rencontrera en production.
        - **Pas de doublons (ou fuite de données)** : S'assurer qu'aucun exemple de l'ensemble de test/validation n'est présent dans l'ensemble d'entraînement.
- **Processus : entraînement / validation / test (diagramme)** :
    1.  **Entraînement** : Modèle apprend sur le `training set`.
    2.  **Validation** : Modèle évalué sur le `validation set` pour régler les hyperparamètres (ex: taux d'apprentissage, complexité du modèle) et choisir le meilleur modèle.
    3.  **Test** : Le modèle final sélectionné est évalué une dernière fois sur le `test set` (qui n'a jamais été utilisé auparavant) pour estimer sa performance de généralisation.

# Chapitre 4 : L’Apprentissage Profond (Deep Learning)

## D1. Émergence, Définition et Fonctionnement de Base
- **Artificial Intelligence / Machine Learning / Deep Learning (chronologie et emboîtement)** :
    - AI (concept général) -> ML (sous-domaine de l'AI, apprend à partir des données) -> DL (sous-domaine du ML, utilise des réseaux de neurones avec de nombreuses couches, dits "profonds").
- **Facteurs d'émergence du Deep Learning** :
    1.  **Big Data** : Disponibilité de très grandes quantités de données pour entraîner les modèles.
    2.  **HPC/GPU (High Performance Computing / Graphics Processing Units)** : Puissance de calcul massive, notamment grâce aux GPU, pour effectuer les calculs matriciels intensifs des réseaux profonds.
    3.  **Modèles flexibles et améliorations algorithmiques** : Développement de nouvelles architectures de réseaux, fonctions d'activation, techniques de régularisation et optimiseurs.
- **Réseau de neurones non profond vs. profond (diagramme)** :
    - **Non profond (Shallow)** : Peu de couches cachées (souvent 0 ou 1).
    - **Profond (Deep)** : Multiples couches cachées, permettant d'apprendre des hiérarchies de caractéristiques de plus en plus abstraites.
- **Exemple de fonctionnement (cycle d'un réseau)** :
    1.  **Forward Pass (Propagation avant)** : Les données d'entrée (`x`) traversent le réseau, couche par couche, jusqu'à produire une sortie (`Y=f(x)`). Chaque neurone calcule une sortie basée sur ses entrées et ses poids.
    2.  **Calcul de l'erreur (Loss)** : La sortie prédite est comparée à la sortie réelle (cible) pour calculer une perte.
    3.  **Backward Pass (Rétropropagation)** : L'erreur est propagée à rebours à travers le réseau pour calculer le gradient de la perte par rapport à chaque poids.
    4.  **Mise à jour des poids** : Les poids sont ajustés en utilisant le gradient (ex: descente de gradient) pour réduire l'erreur. Ce cycle est répété.

## D2. Le Perceptron : Unité de Base et Algorithme d'Apprentissage
- **Définition du Perceptron (neurone formel)** :
    - **Pré-activation (`Z`)** : Somme pondérée des entrées plus un biais. `Z = Σ(wi * xi) + b`.
    - **Activation (`A(z)`)** : Application d'une fonction d'activation (souvent non-linéaire) à la pré-activation. Ex: `A(z) = sigmoid(z)`.
- **Structure (éléments d'un neurone)** :
    - **Inputs (`xi`)** : Valeurs d'entrée.
    - **Weights (`wi`)** : Poids associés à chaque entrée, modulent l'importance de l'entrée.
    - **Net input function (Sommation)** : Calcule la somme pondérée (`Z`).
    - **Activation function (`A`)** : Introduit la non-linéarité, détermine si le neurone "s'active".
    - **Output (`y`)** : Sortie du neurone (`A(Z)`).
    - **Bias (`b`)** : Terme additif, permet de décaler la fonction d'activation.
- **Perceptron Learning Algorithm (diagramme)** : Algorithme pour entraîner un perceptron simple (mono-couche) pour la classification binaire sur des données linéairement séparables. Ajuste les poids si la prédiction est incorrecte.
- **Fonctions d’activation et non-linéarité** :
    - **Rôle** : Permettent aux réseaux de neurones (surtout multicouches) d'apprendre des relations complexes et non linéaires dans les données. Sans non-linéarité, un réseau multicouche se comporterait comme un réseau monocouche.
- **Exemples de fonctions d'activation courantes** :
    - **Sigmoid (Logistique)** : `σ(z) = 1 / (1 + e^(-z))`. Sortie entre 0 et 1. Souffre du "vanishing gradient".
    - **Tanh (Tangente Hyperbolique)** : `tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`. Sortie entre -1 et 1. Moins sujette au vanishing gradient que la sigmoïde mais toujours présente.
    - **ReLU (Rectified Linear Unit)** : `ReLU(z) = max(0, z)`. Sortie `z` si `z > 0`, sinon `0`. Simple, efficace, aide à lutter contre le vanishing gradient. Peut souffrir du problème des "neurones morts" (dead neurons).
    - **Leaky ReLU** : `LeakyReLU(z) = z` si `z > 0`, sinon `αz` (où `α` est petit, ex: 0.01). Variante de ReLU pour éviter les neurones morts.
- **Perceptron (fonction Softmax)** : Souvent utilisée dans la **couche de sortie** pour les tâches de **classification multi-classes**. Transforme un vecteur de scores bruts (logits) en un vecteur de probabilités (chaque sortie entre 0 et 1, et la somme des sorties est 1).
    - `Softmax(zi) = e^zi / Σ(e^zj)` pour chaque score `zi`.
- **Fonctions d’activation : problèmes rencontrés** :
    - **Vanishing Gradient (Gradient qui s'évanouit)** : Dans les réseaux profonds, les gradients deviennent très petits lors de la rétropropagation, rendant l'apprentissage des premières couches très lent ou inexistant. Surtout avec Sigmoid/Tanh.
    - **Exploding Gradient (Gradient qui explose)** : Les gradients deviennent très grands, entraînant des mises à jour de poids instables et une divergence de l'apprentissage.
    - **Dead neurons (Neurones morts)** : Avec ReLU, si un neurone reçoit des entrées qui le font toujours sortir 0, son gradient sera toujours 0, et il ne pourra plus jamais s'activer ou apprendre.
- **Fonctions d’activation : recommandation de choix (tableau comparatif)** :
    - **ReLU** est un bon point de départ pour les couches cachées.
    - **Leaky ReLU** ou variantes si problème de "dead neurons".
    - **Sigmoid** pour la classification binaire en sortie (avec perte Cross-Entropy binaire).
    - **Softmax** pour la classification multi-classes en sortie (avec perte Cross-Entropy catégorielle).
    - **Tanh** parfois utilisée, mais moins que ReLU.
- **Comment ça marche ? (Cycle d'entraînement d'un réseau de neurones)**
    1.  **Initialization** : Initialiser les poids (souvent aléatoirement, mais avec soin).
    2.  **Forward Pass** : Propager les données d'entrée à travers le réseau pour obtenir une prédiction.
    3.  **Error calculation (Loss)** : Calculer l'erreur entre la prédiction et la valeur réelle.
    4.  **Backpropagation** : Calculer les gradients de la perte par rapport à chaque poids.
    5.  **Weight Update & Iterate** : Mettre à jour les poids en utilisant les gradients (ex: descente de gradient) et répéter à partir de l'étape 2.
- **Processus de Backpropagation (animation)** : Illustration visuelle de la propagation de l'erreur et du calcul des gradients de la couche de sortie vers les couches d'entrée.
- **Entraînement d'un réseau de neurones : résumé (animation)** : Vue d'ensemble animée du cycle complet forward-loss-backward-update.

## D3. Types de Réseaux de Neurones Profonds (ANN, MLP, CNN, RNN, etc.)
- **Vue d'ensemble illustrée (architectures classiques)** :
    - **ANN (Artificial Neural Network)** : Terme générique.
    - **MLP (Multilayer Perceptron)** : Réseau de neurones "feedforward" (connexions uniquement vers l'avant) avec une ou plusieurs couches cachées. Chaque neurone d'une couche est connecté à tous les neurones de la couche suivante (couches denses ou "fully connected").
    - **CNN (Convolutional Neural Network)** : Spécialisé pour les données de type grille (ex: images). Utilise des couches de convolution pour extraire des caractéristiques locales.
    - **RNN (Recurrent Neural Network)** : Spécialisé pour les données séquentielles (ex: texte, séries temporelles). Possède des connexions récurrentes (boucles) permettant de maintenir un "état" ou une "mémoire" du passé.
- **Autres architectures importantes** :
    - **GAN (Generative Adversarial Network)** : Deux réseaux (générateur, discriminateur) qui s'entraînent en compétition pour générer de nouvelles données réalistes.
    - **Autoencoders (AE)** : Apprennent à compresser (encoder) les données en une représentation de faible dimension, puis à les décompresser (décoder) pour les reconstruire. Utilisés pour la réduction de dimension, la génération, la détection d'anomalies.
    - **Transformers** : Architecture basée sur des mécanismes d'attention, initialement pour le NLP, mais maintenant étendue à d'autres domaines (vision). Très performants.
    - **Vision Transformers (ViT)** : Appliquent l'architecture Transformer directement à des patchs d'images.
    - **LLMs (Large Language Models)** : Transformers massifs entraînés sur d'énormes corpus de texte (ex: GPT-3, BERT).
    - **VLMs (Visual Language Models)** : Modèles combinant la compréhension du langage et de la vision (ex: CLIP, DALL-E).
- **Multilayer Perceptron (MLP)** :
    - **Structure** :
        - **Couche d'entrée (Input Layer)** : Reçoit les données brutes.
        - **Couches cachées (Hidden Layers)** : Une ou plusieurs couches entre l'entrée et la sortie, où se fait l'essentiel de l'apprentissage des représentations.
        - **Couche de sortie (Output Layer)** : Produit la prédiction finale (ex: avec Softmax pour la classification).
    - **Traitement d'une image pour un MLP** : Une image 2D (ou 3D avec couleurs) doit être **aplatie (flatten)** en un vecteur 1D pour être donnée en entrée à un MLP classique (couches denses). Cela perd l'information spatiale.
- **Exemple : Classification d'images MNIST avec MLP** (chiffres manuscrits de 0 à 9)
    - **Modèle simple : classification avec SoftMax (régression logistique multiclasse)**
        - **Logits (pré-activation de la couche de sortie)** : `L = X.W + b` (où `X` est l'entrée, `W` les poids, `b` le biais).
        - **Prédictions (probabilités)** : `Y = softmax(X.W + b)`.
    - **Dimensions des tenseurs (exemples)** :
        - `predictions` : [batch_size, nb_classes] (ex: [100, 10] pour MNIST).
        - `images` (entrée aplatie) : [batch_size, hauteur * largeur] (ex: [100, 784] pour MNIST 28x28).
        - `weights` (`W`) : [nb_features_entree, nb_neurones_sortie] (ex: [784, 10]).
        - `biases` (`b`) : [nb_neurones_sortie] (ex: [10]).
    - **Visualisation des poids pour une sortie** : Pour un neurone de sortie (ex: celui pour le chiffre '3'), les poids `W` peuvent être remodelés en une image 28x28. Cette image montre quelles régions de l'image d'entrée "activent" le plus ce neurone (ce que le neurone a appris à "reconnaître").
    - **Exemple de calcul d’erreur (Cost, "What's the cost of this difference?")** : Si la cible est '3' (ex: `[0,0,0,1,0,0,0,0,0,0]`) et la prédiction Softmax donne une faible probabilité pour la classe '3', la fonction de coût (ex: Cross-Entropy) sera élevée.
    - **Cost function (général)** : Prend en entrée les prédictions du modèle (`Input`), les vraies étiquettes (`Output`), et dépend des `Parameters` du modèle. Son objectif est de quantifier l'erreur du modèle.

## D4. Paramètres d’Entraînement, Syntaxe (PyTorch Lightning) et Évaluation de Modèles (MNIST en pratique)
- **Descente de gradient et mise à jour des poids (récapitulatif formules)** :
    - `Loss = L(y_true, y_pred)`
    - `Gradients = ∇_W L`
    - `W_new = W_old - learning_rate * Gradients`
- **Calcul d'erreur (visualisation réseau MNIST)** : Visualisation du forward pass, de la sortie Softmax, et de la comparaison avec la vraie étiquette pour obtenir la perte.
- **Encodage et calcul d’erreur** :
    - **One-hot encoding** : Représentation des étiquettes catégorielles où chaque catégorie est un vecteur binaire avec un '1' à l'index de la catégorie et des '0' ailleurs (ex: '3' -> `[0,0,0,1,0,0,0,0,0,0]` pour 10 classes).
    - **L2 Loss (MSE)** : `Σ (y_true - y_pred)²`. Plutôt pour la régression.
    - **Cross-Entropy Loss** : Couramment utilisée pour les problèmes de classification. Mesure la différence entre deux distributions de probabilités (la vraie distribution one-hot et la distribution prédite par Softmax).
- **La descente du gradient (récapitulatif graphique)** : Rappel de la descente vers le minimum de la fonction de coût.
- **Premier résultat MNIST avec MLP simple (Accuracy 92%)** : Performance de base avec un modèle linéaire (régression logistique Softmax).
- **Analyse et améliorations (série d'itérations sur le modèle MNIST)** :
    - **Amélioration N°01 : ajout de couches cachées + Sigmoid** : Introduction de non-linéarité pour améliorer la capacité du modèle.
    - **Amélioration N°02 : remplacement Sigmoid par ReLU (Accuracy ~96%)** : ReLU est souvent plus performante et aide à un apprentissage plus rapide.
    - **Problème d'overfitting ("Accuracy" bruitée, cross entropy loss divergente)** :
        - L'accuracy sur l'ensemble d'entraînement continue de s'améliorer, mais celle sur l'ensemble de validation stagne ou se dégrade.
        - La perte sur l'ensemble de validation commence à augmenter alors que celle sur l'entraînement diminue.
    - **Amélioration N°03 : rate decay (learning rate decay/scheduling) (Accuracy ~98%)** : Réduire progressivement le taux d'apprentissage pendant l'entraînement peut aider à converger plus finement vers un bon minimum.
    - **Analyse : Overfitting (graphique et illustration)** : Le modèle apprend le bruit des données d'entraînement au lieu de la structure générale.
    - **Dropout (technique de régularisation)** : Pendant l'entraînement, désactive aléatoirement (met à zéro) une fraction des neurones à chaque itération. Cela empêche les neurones de co-adapter leurs poids de manière trop spécifique et force le réseau à apprendre des caractéristiques plus robustes.
    - **Amélioration N°04 : 5 couches + ReLU + rate decay + Dropout (75%) (Accuracy ~98.2%)** : Combinaison de techniques pour obtenir une meilleure performance et réduire l'overfitting.
- **Paramètres d’entrainement (liés au processus itératif)** :
    - **Epochs** : Nombre de fois que l'ensemble des données d'entraînement est parcouru par l'algorithme.
    - **Batch_size** : Nombre d'exemples d'entraînement utilisés dans une seule itération (un forward/backward pass) pour estimer le gradient.
    - **Itérations (ou Steps)** : Nombre de mises à jour des poids. `Itérations = (Nombre total d'exemples / Batch_size) * Epochs`.
- **Syntaxe et outils : PyTorch Lightning** (framework de haut niveau pour PyTorch)
    - **Définition du modèle (`LightningModule`)** : Classe encapsulant le modèle PyTorch (`nn.Module`), la fonction de perte, l'optimiseur, et les boucles d'entraînement/validation/test.
        - `__init__` : Définir les couches.
        - `forward` : Définir le passage avant.
        - `training_step` : Calculer et retourner la perte pour un batch d'entraînement.
        - `validation_step`, `test_step` : Idem pour validation et test.
        - `configure_optimizers` : Définir l'optimiseur.
    - **Préparation/chargement des données (`LightningDataModule`)** : Classe pour organiser le chargement, la transformation et la division des données (DataLoaders pour train, val, test).
        - `setup` : Télécharger/préparer les données.
        - `train_dataloader`, `val_dataloader`, `test_dataloader` : Retourner les DataLoaders respectifs.
    - **`trainer.fit(model, datamodule)`** : Lance le processus d'entraînement et de validation.
- **Préparation et division des données (avec PyTorch Lightning)** :
    - **Training Set** : Utilisé pour l'entraînement. Visualisation de la `model loss` (perte sur l'ensemble d'entraînement).
    - **Validation Set** : `val_loader` ajouté. Visualisation des pertes `train_loss` vs `val_loss` pour détecter l'overfitting.
    - **Avec Dropout** : Visualisation de l'effet du dropout sur les courbes de perte d'entraînement et de validation (souvent, rapproche les deux courbes).
    - **Test Set** :
        - `train_test_split` (scikit-learn) : Fonction pour diviser les données.
        - `trainer.test(datamodule=...)` : Évalue le modèle final sur l'ensemble de test.
        - **Matrice de confusion** : Tableau qui résume les performances d'un modèle de classification (vrais positifs, vrais négatifs, faux positifs, faux négatifs pour chaque classe).

## D5. Réseaux de Neurones Convolutionnels (CNNs / ConvNets) : Traitement d'Images
- **Motivation : aller plus loin avec les images** : Les MLP ne sont pas idéaux pour les images car ils aplatissent l'entrée (perte de structure spatiale 2D/3D) et ont beaucoup de paramètres (chaque pixel connecté à chaque neurone de la première couche cachée). Les CNNs exploitent la structure locale des images.
- **Petit rappel : une image et une région de l'image (pixels)** : Une image est une grille de pixels. Les CNNs analysent des petites régions (patchs) de l'image.
- **Extraction de caractéristiques et filtrage d’image : convolutions** :
    - **Opération de Convolution** : Un **noyau (kernel)** ou **filtre** (petite matrice de poids) glisse sur l'image d'entrée. À chaque position, un produit scalaire est calculé entre le noyau et la partie de l'image qu'il recouvre. Le résultat est un pixel dans une **carte d'activation (feature map)**.
    - **Kernel** : Définit le type de caractéristique que la convolution va détecter (ex: bords, textures).
    - **Exemple de calcul de convolution (blur)** : Un noyau de moyenne (ex: [[1,1,1],[1,1,1],[1,1,1]]/9) floute l'image.
    - **Filtres de convolution (plusieurs filtres dans une couche)** : Une couche de convolution apprend typiquement plusieurs filtres en parallèle, chacun spécialisé dans la détection d'une caractéristique différente.
    - **Exemples de filtres (manuels)** : Filtres de Sobel pour les contours (verticaux, horizontaux), filtres de gradient. En DL, ces filtres sont *appris* par le réseau.
    - **Exemples de convolution (différents kernels et effets)** : Sharpen, edge detection, blur, etc.
- **Premier classifieur d’images (MNIST) avec CNN** : Les CNNs surpassent largement les MLP sur MNIST.
- **Architecture des CNNs (composants clés)** :
    - **Image 2D (ou 3D pour RGB)** : Entrée du CNN.
    - **Couche de Convolution (Convolution Layer)** :
        - Applique un ensemble de filtres appris à l'entrée pour produire des cartes d'activation.
        - **Stride** : Pas de déplacement du filtre sur l'image. Stride de 1 (déplacement d'un pixel à la fois) ou 2 (déplacement de deux pixels, réduisant la taille de la sortie).
        - **Padding** : Ajout de pixels (souvent des zéros) autour des bords de l'image d'entrée. Permet de contrôler la taille de la sortie et d'appliquer le filtre aux pixels du bord.
            - `Padding='same'` : La sortie a la même taille spatiale que l'entrée (si stride=1).
            - `Padding='valid'` : Pas de padding, la taille de sortie est réduite.
        - **Matrices de poids (W)** : Les filtres eux-mêmes sont les poids appris. Dimension : `[hauteur_filtre, largeur_filtre, canaux_entrée, canaux_sortie (nb_filtres)]`.
        - Création des "activation maps" (cartes de caractéristiques).
    - **Couche de Pooling (Pooling Layer)** :
        - Réduit la dimension spatiale (sous-échantillonnage) des cartes d'activation, rendant la représentation plus compacte et plus robuste aux petites translations.
        - **Max Pooling** : Prend la valeur maximale dans une fenêtre de la carte d'activation. Le plus courant.
        - **Average Pooling** : Prend la moyenne des valeurs dans la fenêtre.
        - **Inconvénients** : Perte d'information spatiale fine.
    - **Couches Entièrement Connectées (Fully Connected Layers)** : Souvent placées à la fin du CNN (après plusieurs couches de convolution et de pooling) pour effectuer la classification finale basée sur les caractéristiques de haut niveau extraites.
    - **Architecture typique LeNet-5 pour MNIST (diagramme détaillé)** : Un des premiers CNNs réussis. Séquence type : `INPUT -> CONV -> POOL -> CONV -> POOL -> FC -> FC -> OUTPUT(Softmax)`.
        - `CONV1`: ex: 6 filtres 5x5
        - `POOL1`: ex: Max pooling 2x2, stride 2
        - `CONV2`: ex: 16 filtres 5x5
        - `POOL2`: ex: Max pooling 2x2, stride 2
        - `FC1`: ex: 120 neurones
        - `FC2`: ex: 84 neurones
        - `OUTPUT`: 10 neurones (Softmax)
    - **Résultat MNIST avec CNN (99.3%+)** : Les CNNs atteignent des performances très élevées sur la reconnaissance d'images.

# Chapitre 5 : Apprentissage Profond pour la Vision par Ordinateur

## E1. Rappel CNN et Tâche de Classification
- **Exemple de classification (Chat/Chien)** : Tâche fondamentale en vision : attribuer une étiquette (ex: "chat" ou "chien") à une image entière.

## E2. Architectures (Deep) de Classification d’Images : D'AlexNet aux Transformers
- **Le Challenge ImageNet (ILSVRC - ImageNet Large Scale Visual Recognition Challenge)** : Compétition annuelle (maintenant terminée pour la classification) qui a grandement stimulé la recherche en architectures de CNN. Consiste à classifier des images parmi 1000 catégories.
- **Liste des architectures notables (chronologique et par performance)** :
    - **LeNet (1998)** : Pionnier, pour la reconnaissance de chiffres manuscrits.
    - **AlexNet (2012)** : Vainqueur d'ImageNet, a popularisé les CNN profonds et l'utilisation des GPU.
    - **VGG (ex: VGG16, VGG19) (2014)** : Architecture simple et uniforme, utilisant de petits filtres (3x3), mais très profonde.
    - **GoogLeNet (Inception) (2015)** : Introduit le "module Inception" qui effectue des convolutions de différentes tailles en parallèle, plus efficace en calcul.
    - **ResNet (Residual Networks) (2015)** : Introduit les "connexions résiduelles" (skip connections) pour faciliter l'entraînement de réseaux très profonds (plus de 100 couches) en luttant contre le vanishing gradient.
    - **DenseNet (2017)** : Chaque couche est connectée à toutes les couches suivantes dans un bloc dense, améliorant le flux d'informations et de gradients.
    - **EfficientNet (2019)** : Utilise une méthode de "mise à l'échelle composée" (scaling) pour optimiser la profondeur, la largeur et la résolution du réseau de manière équilibrée.
    - **Vision Transformers (ViT) (2020/2021)** : Applique l'architecture Transformer (basée sur l'attention) directement à des patchs d'images, obtenant des résultats SOTA (State-Of-The-Art).
    - **ConvNeXt (2022)** : Modernise les architectures CNN classiques en s'inspirant des succès des Transformers, obtenant des performances comparables ou supérieures aux ViT avec une complexité parfois moindre.
- **Comparaison des taux d’erreurs sur ImageNet (graphiques)** :
    - **2010-2017** : Montre la chute drastique du taux d'erreur (top-5) avec l'arrivée d'AlexNet, puis les améliorations successives (VGG, GoogLeNet, ResNet).
    - **2016-2023** : Montre la continuation des améliorations, l'arrivée des Transformers (ViT) et des architectures hybrides ou optimisées comme ConvNeXt.
- **Détails des architectures** :
    - **LeNet (Yann LeCun, 1998, 6 couches effectives)** : CONV-POOL-CONV-POOL-FC-FC-OUTPUT.
    - **AlexNet (Krizhevsky et al., 2012, 8 couches)** : CONV-POOL-CONV-POOL-CONV-CONV-CONV-POOL-FC-FC-FC-OUTPUT. Utilisation de ReLU, Dropout, Data Augmentation.
    - **VGG16/VGG19 (Simonyan & Zisserman, 2014, 16/19 couches)** : Utilisation exclusive de filtres 3x3 pour les convolutions, empilés pour augmenter la profondeur. Simple mais lourd en paramètres.
    - **GoogLeNet (Szegedy et al., 2015, 22 couches, module Inception)** : Le module Inception combine des convolutions 1x1, 3x3, 5x5 et du max pooling, dont les sorties sont concaténées. Réduction de dimension avec des convolutions 1x1 pour limiter les coûts.
    - **Inception V1, V2, V3, V4** : Évolutions du module Inception avec des améliorations (factorisation des convolutions, normalisation par batch, etc.).
    - **ResNet (He et al., 2015)** : Les **skip connections** (raccourcis) permettent au gradient de se propager plus facilement. `H(x) = F(x) + x`. `F(x)` est ce que le bloc apprend (le résidu).
    - **DenseNet (Huang et al., 2017)** : Chaque couche reçoit en entrée les feature maps de toutes les couches précédentes (au sein d'un bloc dense) et ses propres feature maps sont passées à toutes les couches suivantes. Favorise la réutilisation des caractéristiques.
    - **Vision Transformers (ViT) (Dosovitskiy et al., 2020/2021)** : L'image est divisée en patchs, aplatis, puis traités comme des "mots" par un encodeur Transformer standard. Nécessite souvent de grands ensembles de données pré-entraînement.
- **Comment les utiliser ? (Exemple VGG16 en Keras/TensorFlow)** :
    - `from tensorflow.keras.applications import VGG16`
    - `model = VGG16(weights='imagenet', include_top=True)` (`include_top=True` pour la classification ImageNet, `False` pour le transfert).

## E3. Transfert Learning et Fine Tuning : Réutiliser la Connaissance
- **Entraînement d’un CNN (coûteux)** : Entraîner un CNN profond from scratch (à partir de zéro) sur un grand dataset comme ImageNet demande beaucoup de données, de temps et de ressources de calcul (GPU).
- **Transfer Learning (Apprentissage par transfert) : définition et avantages** :
    - **Définition** : Technique où un modèle pré-entraîné sur une tâche source volumineuse (ex: ImageNet) est réutilisé comme point de départ pour une tâche cible, souvent avec moins de données.
    - **Avantages** :
        - Nécessite moins de données pour la tâche cible.
        - Entraînement plus rapide.
        - Peut atteindre de meilleures performances (surtout si les données cibles sont limitées).
    - **Diagrammes de comparaison** :
        - **Training from scratch vs. Transfer Learning** : From scratch part de poids aléatoires ; Transfer Learning part de poids pré-entraînés pertinents.
        - **Traditional ML vs. Transfer Learning** : Traditional ML entraîne des modèles isolés ; Transfer Learning transfère la connaissance.
        - **Source model/data vs. Target model/data** : Le modèle apprend sur `Source data` (grand) pour créer `Source model`, puis adapté à `Target data` (petit) pour `Target model`.
- **Fine tuning (Ajustement fin) : deux possibilités principales** :
    1.  **Feature Extractor (Geler les couches initiales)** :
        - On retire la couche de classification d'origine du modèle pré-entraîné.
        - On **gèle les poids** des couches de convolution initiales (elles agissent comme extracteur de caractéristiques génériques).
        - On ajoute une nouvelle couche de classification adaptée à la tâche cible et on n'entraîne que cette nouvelle couche (et éventuellement quelques couches finales du modèle pré-entraîné).
    2.  **Fine-tuning (Geler peu ou pas de couches)** :
        - On remplace la couche de classification.
        - On entraîne l'ensemble du réseau (ou une grande partie) sur les nouvelles données, mais avec un **taux d'apprentissage plus faible** pour ne pas "détruire" les poids pré-entraînés trop rapidement. Les couches initiales apprennent des caractéristiques plus génériques, les couches finales des caractéristiques plus spécifiques.
    - **Remplacement de la couche de classification (exemple AlexNet)** : La dernière couche FC de 1000 classes (ImageNet) est remplacée par une couche FC avec le nombre de classes de la nouvelle tâche.
- **Transfer Learning vs. Fine tuning (Quand geler ou ajuster ?)** :
    - **Petites données cibles, similaires à la source** : Utiliser comme extracteur de features (geler la plupart des couches).
    - **Grandes données cibles, similaires à la source** : Fine-tuner plus de couches, voire toutes, avec un petit learning rate.
    - **Petites données cibles, différentes de la source** : Difficile. Essayer de geler beaucoup, ou voir si les couches initiales sont utiles.
    - **Grandes données cibles, différentes de la source** : Fine-tuner une grande partie du réseau, voire entraîner from scratch si les données sont très différentes et volumineuses.
- **Pratiquement : exemple de code PyTorch Lightning (FireDetectionModel avec VGG16 pré-entraîné)** :
    - Charger VGG16 avec `weights='IMAGENET1K_V1'`.
    - Geler les paramètres de `features` (couches de convolution).
    - Remplacer `classifier` par une nouvelle séquence de couches adaptées à la détection d'incendie (ex: 2 classes).
    - Entraîner uniquement le nouveau classifieur.
- **Paramètres d’entraînement & Callbacks utiles (PyTorch Lightning)** :
    - **ModelCheckpoint** : Sauvegarde le meilleur modèle (ou les poids) pendant l'entraînement en fonction d'une métrique surveillée (ex: `val_loss`).
    - **EarlyStopping** : Arrête l'entraînement si une métrique surveillée (ex: `val_loss`) ne s'améliore plus pendant un certain nombre d'époques (`patience`), pour éviter l'overfitting et gagner du temps.

## E4. Architectures (Deep) de Localisation et Détection d’Objets
- **Comparaison des performances (mAP vs. time)** : Graphique montrant le compromis entre la précision (mAP - mean Average Precision) et la vitesse (FPS - Frames Per Second) pour différents détecteurs d'objets.
- **Tâches de vision (complexité croissante)** :
    1.  **Classification** : Quelle classe d'objet est présente dans l'image ? (1 étiquette par image)
    2.  **Classification + Localisation** : Quelle classe et où se trouve-t-elle (avec une seule boîte englobante - bounding box) ?
    3.  **Object Detection** : Quelles classes sont présentes et où sont leurs boîtes englobantes (multiples objets par image) ?
    4.  **Instance Segmentation** : Quelles classes, où sont leurs boîtes, ET quels pixels appartiennent à chaque instance d'objet (masque de segmentation par objet) ?
- **Évolution des performances sur PASCAL VOC (graphique)** : Montre l'amélioration du mAP sur le dataset PASCAL VOC au fil des ans avec l'arrivée des CNN (R-CNN, Fast R-CNN, Faster R-CNN, YOLO, SSD).
- **Métrique : IoU (Intersection Over Union)** :
    - Mesure la superposition entre une boîte englobante prédite et la boîte englobante réelle (ground truth).
    - `IoU = Aire de l'Intersection / Aire de l'Union`.
    - Utilisée pour déterminer si une détection est un Vrai Positif (TP) (ex: si IoU > 0.5).
- **Bases de données courantes pour la détection/segmentation** :
    - **PASCAL VOC (Visual Object Classes)** : ~20 classes, ~11k images.
    - **COCO (Common Objects in Context)** : 80 classes, >200k images, plus d'objets par image, segmentation.
    - **ImageNet** : Utilisé aussi pour la détection (sous-ensemble avec boîtes).
    - **YouTube-BoundingBoxes / YouTube-8M / Objects365** : Très grands datasets.
- **Détection d’objets comme une classification : Fenêtre glissante (Sliding Window)** :
    - Faire glisser une fenêtre de taille fixe sur l'image.
    - À chaque position, utiliser un classifieur CNN pour dire si un objet est présent.
    - **Problème** : Très coûteux en calcul (beaucoup de fenêtres à évaluer, différentes tailles et ratios d'aspect).
- **Approches basées sur les Propositions de Régions (Region Proposals)** :
    - **Idée** : D'abord générer un petit nombre de régions candidates (RoI - Regions of Interest) susceptibles de contenir des objets, puis appliquer un CNN à ces régions.
    - **Selective Search** : Algorithme classique pour générer des propositions de régions basé sur la similarité de couleur, texture, taille, etc.
    - **1. R-CNN (Regions with CNN features) (Girshick et al., 2014)** :
        1.  Génère ~2k propositions de régions (Selective Search).
        2.  Déforme (warp) chaque région à une taille fixe.
        3.  Passe chaque région déformée dans un CNN pré-entraîné (ex: AlexNet) pour extraire les caractéristiques.
        4.  Classifie chaque région avec des SVMs linéaires (un par classe).
        5.  Ajuste les boîtes englobantes avec une régression linéaire (Bbox regression).
        - **Inconvénients** : Lent (CNN sur chaque région), entraînement multi-étapes.
    - **2. Fast R-CNN (Girshick, 2015)** :
        1.  Passe l'**image entière** une seule fois dans un CNN pour obtenir une carte de caractéristiques globale.
        2.  Projette les propositions de régions (toujours générées par ex. Selective Search) sur cette carte de caractéristiques.
        3.  Utilise une couche **RoI Pooling** pour extraire un vecteur de caractéristiques de taille fixe pour chaque RoI à partir de la carte de caractéristiques.
        4.  Passe ces vecteurs dans des couches FC pour la classification (Softmax) et la régression de la boîte.
        - **Avantages** : Beaucoup plus rapide que R-CNN (partage des calculs de convolution).
    - **3. Faster R-CNN (Ren et al., 2015)** :
        1.  Introduit le **RPN (Region Proposal Network)** : un petit CNN qui apprend à générer des propositions de régions directement à partir des cartes de caractéristiques du CNN principal. Remplace Selective Search.
        2.  Le RPN et le réseau de détection (Fast R-CNN) partagent les couches de convolution.
        - **Avantages** : Encore plus rapide, approche de bout en bout (end-to-end) pour la génération de propositions et la détection.
        - **Comparaison des vitesses** : R-CNN (secondes/image) -> Fast R-CNN (millisecondes/image) -> Faster R-CNN (encore plus rapide, temps réel possible).
- **Détection sans Propositions (Single-Shot Detectors) : YOLO / SSD** :
    - **Principe général** : Traitent la détection comme un problème de régression. Prédisent les boîtes et les classes directement à partir de l'image entière en un seul passage.
    - **Grille** : L'image est divisée en une grille de cellules.
    - **Boîtes d'ancrage (Anchor boxes) / Boîtes de base (Default boxes)** : Chaque cellule de la grille est responsable de prédire un certain nombre de boîtes englobantes avec des formes et tailles prédéfinies (ancres).
    - **Prédictions par boîte** :
        - Coordonnées de la boîte (`dx, dy, dw, dh` - décalages par rapport à l'ancre).
        - Score de confiance de la présence d'un objet (`confidence` ou `P(Object)`).
        - Probabilités des classes (`C` classes, `P(Classe_i | Objet)`).
    - **1. YOLO (You Only Look Once) (Redmon et al., 2016)** :
        - **Division en grille** : Ex: grille 7x7. Chaque cellule prédit `B` boîtes (ex: B=2) et `C` probabilités de classes.
        - **Prédiction des scores P(Objet)** : Indique si une boîte contient un objet.
        - **Confiance du cadre de sélection (Confidence Score)** : `P(Objet) * IoU(prédite, réelle)`.
        - **Prédiction des probabilités de classes P(Classe|Objet)** : Probabilité conditionnelle de chaque classe, étant donné qu'il y a un objet.
        - **Combinaison des prédictions** : Score final par classe pour chaque boîte = `P(Objet) * IoU * P(Classe_i | Objet)`.
        - **NMS (Non-Maximum Suppression)** : Élimine les boîtes redondantes qui détectent le même objet. Garde la boîte avec le plus haut score de confiance et supprime celles qui ont une IoU élevée avec elle.
        - **Architecture unique** : Un seul CNN qui prédit tout en même temps.
        - **Performances** : Très rapide (temps réel), mAP initialement un peu plus faible que Faster R-CNN, mais s'est amélioré avec les versions.
        - **Historique d’évolution de YOLO** :
            - **YOLOv1 à YOLOv11 (et plus)** : Améliorations successives (YOLO9000/v2, YOLOv3, v4, v5, v7, v8, v9, etc.) avec de meilleures backbones (ex: Darknet, CSPNet), techniques d'entraînement, fonctions de perte, tailles d'ancres.
            - **YOLO-NAS (Neural Architecture Search)**, **YOLO-World** (Open-Vocabulary).
        - **Démo YOLOv3, YOLOv12 (probablement une typo, peut-être YOLOvX ou une version future conceptuelle)** : Visualisation des détections en temps réel.
    - **2. SSD (Single Shot MultiBox Detector) (Liu et al., 2016)** :
        - Similaire à YOLO, mais utilise des cartes de caractéristiques de différentes échelles (multiples couches de convolution) pour prédire des boîtes. Permet de mieux détecter des objets de différentes tailles.

# Chapitre 6 : Nouvelles Architectures Neuronales Profondes & Multimodales (Transformers, ViT, LLM & VLM)

## F1. Évolution Récente de l'IA et des Architectures
- **Ligne de temps et évolution de l’IA (répétition)** : Rappel de la progression rapide des modèles, notamment avec l'avènement des Transformers et des modèles à grande échelle.

## F2. Rappel : Réseaux de Neurones Convolutionnels (CNN)
- **CNN Explainer (outil interactif)** : Référence à un outil permettant de visualiser et comprendre interactivement le fonctionnement interne des CNNs (couches de convolution, pooling, etc.). *Utile pour se remémorer les concepts de base des CNN avant de comparer avec les Transformers.*

## F3. Réseaux de Neurones Récurrents (RNN) : Traitement de Séquences et Limites
- **Exemple : réseau de neurones pour la prédiction du mot suivant (Next Word Prediction)** : Tâche classique pour les RNN où le contexte des mots précédents est utilisé pour prédire le mot suivant dans une phrase.
- **Vanilla Recurrent Neural Network (RNN simple)** :
    - **Structure déroulée** : Un RNN peut être vu comme une série de mêmes blocs neuronaux, chacun traitant un élément de la séquence et passant une information d'état (hidden state) au bloc suivant. La sortie à un instant `t` dépend de l'entrée à `t` et de l'état caché à `t-1`.
    - `h_t = f_W(h_{t-1}, x_t)`
- **LSTM (Long Short-Term Memory)** :
    - **Problème des RNN simples** : Difficulté à capturer les dépendances à long terme dans les séquences à cause du **vanishing/exploding gradient**.
    - **Structure interne d'une cellule LSTM** : Plus complexe qu'un neurone RNN simple. Comprend :
        - **Cell State (état de la cellule)** : "Autoroute" pour l'information, permettant de conserver des informations sur de longues périodes.
        - **Forget Gate (Porte d'oubli)** : Décide quelles informations de l'état de la cellule doivent être oubliées.
        - **Input Gate (Porte d'entrée)** : Décide quelles nouvelles informations doivent être stockées dans l'état de la cellule.
        - **Output Gate (Porte de sortie)** : Décide quelle partie de l'état de la cellule doit être sortie (et passée comme état caché au pas de temps suivant).
- **Problèmes des RNN (même avec LSTM/GRU)** :
    - **Vanishing/Exploding Gradient** (moins sévère avec LSTM/GRU mais toujours possible pour des séquences très longues).
    - **Difficulté de parallélisation** : Le calcul est intrinsèquement séquentiel (l'état `t` dépend de `t-1`), ce qui limite l'exploitation des GPU.
    - **Rôle des Transformers** : Les Transformers ont été proposés en partie pour surmonter ces limitations, notamment grâce à la parallélisation et au mécanisme d'attention pour capturer les dépendances à longue portée.

## F4. Transformers : Révolutionner le Traitement Séquentiel avec l'Attention
- **Définition (Google Brain, "Attention Is All You Need", 2017)** :
    - Architecture de réseau de neurones initialement proposée pour la traduction automatique.
    - Repose principalement sur le **mécanisme d'auto-attention (self-attention mechanism)** pour pondérer l'importance des différents éléments d'une séquence d'entrée par rapport aux autres éléments de la même séquence.
    - Permet une **parallélisation** significative du traitement des séquences (contrairement aux RNN).
    - Est devenu la **base de nombreux modèles génératifs** de pointe (LLM, modèles de diffusion).
- **Étapes principales (flux de données simplifié dans un Transformer)** :
    1.  **Tokenisation** : Le texte d'entrée est décomposé en unités plus petites (tokens : mots, sous-mots).
    2.  **Embedding Generation (Génération des plongements)** : Chaque token est converti en un vecteur dense (embedding). Un embedding de position est ajouté pour indiquer l'ordre des tokens.
    3.  **Attention Mechanism (Mécanisme d'Attention)** : (Coeur du Transformer)
        - **Self-Attention** : Calcule comment chaque token est lié aux autres tokens de la même séquence.
        - **Multi-Head Attention** : Exécute plusieurs mécanismes d'attention en parallèle (différentes "têtes" apprennent différentes relations) et concatène leurs résultats.
    4.  **Transformer Layers (Blocs Encodeur/Décodeur)** :
        - **Encodeur** : Traite la séquence d'entrée (composé de couches d'auto-attention et de réseaux feed-forward).
        - **Décodeur** : Génère la séquence de sortie, en utilisant également l'attention sur la sortie de l'encodeur (cross-attention) et l'auto-attention sur les tokens déjà générés.
- **Transformers : explained (outil interactif)** : Référence à des visualisations (ex: The Illustrated Transformer) qui expliquent en détail les composants (Query, Key, Value, Scaled Dot-Product Attention, etc.).

## F5. Vision Transformers (ViT) : Appliquer l'Attention à la Vision
- **Définition (Google, 2020/2021)** : Architecture qui adapte le modèle Transformer pour les tâches de **vision par ordinateur**, se positionnant comme une **alternative aux CNNs**.
- **Comparaison CNN vs ViT** :
    - **CNN** :
        - **Biais inductif fort** : Localité (les convolutions traitent des voisinages locaux), invariance par translation (partage de poids).
        - Efficace sur des datasets de taille modérée.
        - Extrait des hiérarchies de caractéristiques locales puis plus globales.
    - **ViT** :
        - **Moins de biais inductif spécifique à la vision** : Traite les patchs d'image comme une séquence.
        - Le mécanisme d'attention peut capturer des **dépendances globales** entre les patchs dès les premières couches.
        - Nécessite souvent de **très grands ensembles de données de pré-entraînement** (ex: ImageNet-21k, JFT-300M) pour bien généraliser, ou des techniques de régularisation fortes.
- **Architecture ViT (générale pour la classification)** :
    1.  **Découpage de l'image en patchs** (ex: 16x16 pixels).
    2.  **Aplatissement et projection linéaire de chaque patch** pour obtenir des "patch embeddings".
    3.  Ajout d'**embeddings de position** pour conserver l'information spatiale.
    4.  Un **token [CLS]** (classification) spécial est souvent ajouté à la séquence de patch embeddings.
    5.  La séquence résultante est passée à travers un **Encodeur Transformer** standard.
    6.  La représentation du token [CLS] en sortie de l'encodeur est utilisée pour la classification (via une tête MLP).
- **Étapes principales de ViT (détaillées)** :
    1.  **Découpage et préparation de l’image** :
        - L'image est divisée en une grille de **patchs** non chevauchants (ou peu).
        - Chaque patch est **aplati** en un vecteur.
        - Ce vecteur est projeté linéairement pour créer un **patch embedding** de dimension fixe.
        - Un **embedding de position** (appris ou fixe) est ajouté à chaque patch embedding.
    2.  **Self-Attention (au sein d'une tête d'attention)** :
        - Pour chaque patch embedding, on dérive trois vecteurs : **Query (Q)**, **Key (K)**, **Value (V)** via des projections linéaires.
        - **Score d'attention** : Calculé pour chaque Query avec toutes les Keys (souvent par produit scalaire, suivi d'une mise à l'échelle et d'un Softmax). `Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V`.
        - **Pondération** : Les scores d'attention (après Softmax) sont utilisés pour pondérer les vecteurs Value. Le résultat est une somme pondérée des Values, représentant comment chaque patch "prête attention" aux autres.
    3.  **Multi-Head Attention** :
        - Le processus de Self-Attention (Q, K, V) est exécuté `h` fois en parallèle (les "têtes"), avec des projections linéaires différentes pour Q, K, V à chaque tête.
        - Les sorties des `h` têtes sont concaténées et reprojetées linéairement pour obtenir la sortie finale du bloc Multi-Head Attention. Permet au modèle d'apprendre différentes relations contextuelles.
    4.  **Bloc Transformer (Encoder Block)** : Une couche d'encodeur Transformer typique contient :
        - Un module Multi-Head Self-Attention.
        - Une **connexion résiduelle (skip connection)** autour de ce module.
        - Une couche de **Normalisation de couche (Layer Normalization)**.
        - Un réseau **Feed-Forward (MLP - Multi-Layer Perceptron)** simple (appliqué indépendamment à chaque position).
        - Une connexion résiduelle autour du MLP.
        - Une autre couche de Normalisation de couche.
        - Ces blocs sont empilés plusieurs fois.
    5.  **Token classification (pour la classification d'image)** :
        - La représentation finale du token **[CLS]** (introduit au début de la séquence) en sortie du dernier bloc Transformer est passée à une tête de classification (souvent un MLP simple avec une couche Softmax) pour prédire la classe de l'image.

## F6. LLM et VLM : Modèles Géants de Langage et Modèles Vision-Langage
- **LLMs (Large Language Models - Grands Modèles de Langage)** :
    - **Cœur** : Reposent fondamentalement sur l'architecture **Transformer**, en particulier le **mécanisme d'attention**.
    - **Modèles emblématiques** :
        - **GPT (Generative Pre-trained Transformer) - famille (OpenAI)** : GPT-3, GPT-3.5, **GPT-4**. Principalement des décodeurs Transformer.
        - **LLaMA (Meta)** : Modèles open-source performants.
        - **Mistral (Mistral AI)** : Modèles européens performants, souvent avec des architectures optimisées (ex: Mixture of Experts).
        - (BERT, T5 sont d'autres familles importantes, BERT étant un encodeur).
    - **Étapes de fonctionnement (génération de texte type GPT)** :
        1.  **Tokenisation** : Texte d'entrée (prompt) décomposé en tokens.
        2.  **Embeddings** : Tokens convertis en vecteurs + embeddings de position.
        3.  **Transformer (Décodeur)** : Les embeddings traversent les couches du décodeur Transformer (auto-attention).
        4.  **Prédiction (Prochain token)** : La sortie du Transformer est utilisée pour prédire une distribution de probabilité sur le prochain token du vocabulaire. Un token est échantillonné.
        5.  **Boucle (Loop)** : Le token prédit est ajouté à la séquence d'entrée, et le processus recommence pour générer le token suivant, jusqu'à une condition d'arrêt.
        6.  **Entraînement (Training)** : Phase de pré-entraînement massive sur d'énormes corpus de texte (apprentissage auto-supervisé, ex: prédire le mot suivant).
        7.  **Fine-tuning (Ajustement fin)** :
            - **Supervised Fine-Tuning (SFT)** : Sur des paires instruction-réponse de haute qualité.
            - **RLHF (Reinforcement Learning from Human Feedback)** : Apprentissage par renforcement où les récompenses sont basées sur les préférences humaines pour améliorer l'alignement, la serviabilité et réduire les biais.
- **VLMs (Vision Language Models - Modèles Vision-Langage)** :
    - **Capacité** : Peuvent traiter et comprendre des informations provenant de **modalités visuelles (images, vidéos) et textuelles simultanément**.
    - **Support des modalités en entrée** : Prennent typiquement une image et/ou du texte en entrée.
    - **Exemples de tâches** :
        - **Classification d'images basée sur du texte (zero-shot/few-shot)** : Ex: CLIP (associe images et descriptions textuelles).
        - **Génération de texte à partir d'images (Image Captioning)** : Décrire une image avec du texte.
        - **Visual Question Answering (VQA)** : Répondre à des questions textuelles sur une image.
        - **Génération d'images à partir de texte (Text-to-Image)** : Ex: DALL-E, Stable Diffusion (souvent basés sur des modèles de diffusion guidés par des embeddings textuels de type CLIP).
        - **Génération multimodale** : Ex: générer une histoire avec des images illustratives.
    - **Architectures typiques** : Souvent combinent un encodeur d'image (CNN ou ViT) et un encodeur/décodeur de texte (Transformer), avec des mécanismes pour fusionner ou aligner les représentations des deux modalités (ex: cross-attention).

# Chapitre 7 : Pistes de Développement et Challenges en Apprentissage Profond

## G1. Diversité des Applications du Deep Learning
- **Exemples d'applications DL (diversifiées)** :
    - **Classification d'images** (ex: reconnaître des objets).
    - **Image Retrieval (Recherche d'images par le contenu)**.
    - **Segmentation d'images** (délimiter les pixels appartenant à chaque objet).
    - **CAD (Computer-Aided Diagnosis) médical** (aide au diagnostic à partir d'images médicales).
    - **Social Distancing Monitoring** (détection de la distance entre personnes).
    - *Illustre la polyvalence et l'impact du DL.*

## G2. Deep Learning : Création de Modèles & Analyse de Performance (Biais-Variance)
- **Courbes de performance (Loss/Accuracy pour training/validation)** : Outils essentiels pour diagnostiquer le comportement du modèle pendant l'entraînement.
    - `train_loss` vs `val_loss` ; `train_accuracy` vs `val_accuracy`.
    - Des écarts importants entre les courbes d'entraînement et de validation indiquent souvent un surapprentissage.
- **Dilemme Biais & Variance (Compromis Biais-Variance)** :
    - **Underfitting (Sous-apprentissage - Biais élevé)** : Le modèle est trop simple pour capturer la structure des données. Mauvaises performances sur l'ensemble d'entraînement ET de validation/test.
    - **Optimum** : Bon équilibre, le modèle généralise bien.
    - **Overfitting (Surapprentissage - Variance élevée)** : Le modèle apprend "par cœur" les données d'entraînement, y compris le bruit. Très bonnes performances sur l'entraînement, mais mauvaises sur la validation/test.
- **Définition du Biais (Bias)** :
    - Erreur due à des hypothèses erronées dans l'algorithme d'apprentissage. Un biais élevé signifie que le modèle ne parvient pas à capturer la relation réelle entre les caractéristiques et la sortie (modèle trop simple).
    - Mesure à quel point les prédictions moyennes du modèle sont éloignées de la vérité.
- **Définition de la Variance** :
    - Erreur due à la sensibilité du modèle aux fluctuations des données d'entraînement. Une variance élevée signifie que le modèle change considérablement si les données d'entraînement changent légèrement (modèle trop complexe, s'adapte au bruit).
    - Mesure la dispersion des prédictions du modèle pour un point de données donné.
- **Exemples de classification (chat) pour illustrer Biais/Variance** :
    - **High Bias** : Un classifieur linéaire simple essayant de séparer des classes non linéairement séparables.
    - **High Variance** : Un classifieur très complexe (ex: arbre de décision très profond) qui s'adapte parfaitement aux points d'entraînement mais crée une frontière de décision très irrégulière.
    - **Just Right** : Une frontière de décision qui capture la tendance générale sans sur-ajuster.
- **Processus d'amélioration du modèle DL (stratégies typiques)** :
    - **Si High Bias (Underfitting)** :
        - **Utiliser un réseau plus grand/profond (Bigger network)** : Plus de couches, plus de neurones par couche.
        - Essayer une architecture différente.
        - Entraîner plus longtemps (si la perte d'entraînement n'a pas encore convergé).
        - Réduire la régularisation (si trop forte).
    - **Si High Variance (Overfitting)** :
        - **Obtenir plus de données (More data)** : Le meilleur remède, mais pas toujours possible.
        - **Techniques de Régularisation** :
            - L1/L2 regularization (Weight decay).
            - Dropout.
            - Data Augmentation.
            - Early Stopping.
        - Utiliser une architecture plus petite.
        - Feature selection (si applicable).

## G3. Fonctions d'Activation : Analyse Comparative et Impact
- **Rappel du Perceptron (pré-activation `z`, activation `a = g(z)`)** : La fonction d'activation introduit la non-linéarité.
- **Types de fonctions d'activation (récapitulatif)** :
    - **Linear (Identité)** : `g(z) = z`.
    - **Sigmoid** : `g(z) = 1 / (1 + e^(-z))`.
    - **Tanh (Tangente Hyperbolique)** : `g(z) = (e^z - e^(-z)) / (e^z + e^(-z))`.
    - **ReLU (Rectified Linear Unit)** : `g(z) = max(0, z)`.
    - **Leaky ReLU** : `g(z) = z` si `z > 0`, sinon `αz`.
- **Fonctions linéaires (limites)** :
    - Un réseau de neurones avec uniquement des fonctions d'activation linéaires (y compris dans les couches cachées) est équivalent à un réseau linéaire simple (une seule couche). Il ne peut pas apprendre de relations non linéaires complexes.
- **Fonction Softmax** :
    - Spécifiquement pour la **couche de sortie des problèmes de classification multi-classes**.
    - Transforme les scores bruts (logits) en une distribution de probabilités (valeurs entre 0 et 1, somme à 1).
    - `Softmax(zi) = e^zi / Σ(e^zj)`.
- **Analyse des dérivées des fonctions (impact sur l'apprentissage)** :
    - **Sigmoid et Tanh** :
        - Leurs dérivées sont proches de zéro pour des valeurs d'entrée très grandes (positives ou négatives) -> **Saturation**.
        - Cela conduit au problème du **Vanishing Gradient** lors de la rétropropagation dans les réseaux profonds (les gradients deviennent très petits, l'apprentissage des premières couches est lent ou nul).
    - **ReLU et ses variantes** :
        - Dérivée constante (1) pour `z > 0`, ce qui aide à atténuer le vanishing gradient.
        - Dérivée nulle pour `z < 0` (peut causer des "dead neurons" pour ReLU).
- **Démo interactive (TensorFlow Playground)** : Permet d'expérimenter avec différentes fonctions d'activation et d'observer leur impact sur la capacité du réseau à apprendre différentes frontières de décision.

## G4. Explainable Deep Learning (XAI - IA Explicable)
- **Facteurs de succès du Deep Learning (récapitulatif)** : Big Data, HPC/GPU, modèles/algorithmes avancés.
- **Problèmes / Défis majeurs du Deep Learning** :
    - **Explicabilité et Interprétabilité (Boîte Noire - Black Box)** : Les modèles DL sont souvent très complexes, rendant difficile de comprendre *pourquoi* ils prennent une décision spécifique. C'est un frein pour les applications critiques (médical, finance, justice).
    - **Déploiement sur systèmes embarqués (Edge AI)** : Les modèles profonds peuvent être volumineux et gourmands en calcul, ce qui pose des défis pour leur déploiement sur des appareils aux ressources limitées (smartphones, IoT). (Note: ce point est moins développé dans la suite de la section XAI mais reste un challenge important).
- **Explainable Deep Learning (XAI) : définition et objectifs** :
    - **Définition** : Ensemble de techniques et de méthodes visant à rendre les décisions des modèles d'IA (en particulier DL) compréhensibles par les humains.
    - **Objectifs** :
        - Augmenter la confiance dans les modèles.
        - Permettre le débogage et l'amélioration des modèles.
        - Identifier les biais potentiels.
        - Assurer la conformité réglementaire (ex: GDPR et le "droit à l'explication").
        - Faciliter l'interaction homme-machine.
- **Approches d'explicabilité (catégories et exemples)** :
    - **Méthodes basées sur la perturbation (Perturbation-based methods)** : Modifient des parties de l'entrée (ex: masquer des régions d'une image) et observent l'impact sur la sortie pour identifier les régions importantes.
        - **Exemple : Occlusion Visualization (ou Occlusion Sensitivity)** : On fait glisser un patch gris sur l'image et on mesure la chute de probabilité pour la classe prédite. Les zones où la probabilité chute le plus sont considérées comme importantes.
    - **Méthodes basées sur le gradient (Gradient-based methods)** : Utilisent les gradients de la sortie par rapport à l'entrée (ou aux activations internes) pour indiquer l'importance des caractéristiques d'entrée.
        - **Exemple : Gradient Backpropagation (ou Saliency Maps)** : Visualise le gradient de la classe prédite par rapport aux pixels d'entrée.
        - **Exemple : Integrated Gradients** : Attribue l'importance en intégrant les gradients le long d'un chemin depuis une baseline (ex: image noire) jusqu'à l'image d'entrée. Satisfait des propriétés d'axiomes d'attribution.
    - **Méthodes basées sur CAM (Class Activation Mapping)** : Utilisent les gradients qui affluent dans les dernières couches de convolution pour produire une carte de localisation grossière des régions importantes pour une classe spécifique.
        - **Exemple : Grad-CAM (Gradient-weighted Class Activation Mapping)** : Pondère les cartes d'activation de la dernière couche convolutive par les gradients de la classe cible. Ne nécessite pas de modification d'architecture. Produit des heatmaps.
- **TIS (Transformer Input Sampling) for Vision Transformers** : Méthode d'explicabilité spécifique aux ViT, qui échantillonne des sous-ensembles de patchs d'entrée pour évaluer leur contribution.
- **Exemple : Deep Learning pour la détection de COVID-19 (application médicale)**
    - **Données (radiographies pulmonaires)** :
        - **Classes** : COVID-19, Normal, Pneumonie Virale, Pneumonie Bactérienne.
        - **Datasets** : Diverses sources publiques.
        - **Augmentation de données** : Techniques pour augmenter artificiellement la taille du dataset (rotations, zooms, etc.).
        - **Taille totale** : Nombre d'images utilisées.
    - **Développement du modèle** :
        - **DNN (MLP), CNN (from scratch), Transfer Learning (VGG16, ResNet, etc.)**.
        - **Optimisation** : Choix de l'optimiseur, du taux d'apprentissage.
        - **Cross-validation (Validation croisée)** : Technique pour évaluer la performance du modèle de manière plus robuste en divisant les données en plusieurs plis (folds).
        - **Accuracy (Précision globale)** : Métrique de performance (pourcentage de classifications correctes).
- **Exemple : Explainable Deep Learning (XAI) pour COVID-19 (utilisation de Grad-CAM)**
    - **Input** : Radiographie.
    - **Classification** : Prédiction du modèle (ex: COVID-19).
    - **Explication (Visualisation Grad-CAM)** : Heatmap superposée à la radiographie, montrant les zones du poumon sur lesquelles le modèle s'est "concentré" pour prendre sa décision.
    - **Identification de Biais dans les données (grâce à XAI)** :
        - Le modèle peut apprendre des **artefacts non pertinents** si la base de données est biaisée.
        - **Exemple 1 : Lettres/annotations sur les radiographies** : Le modèle peut se baser sur des marqueurs textuels (ex: 'L' pour gauche, nom de l'hôpital) présents différemment selon les classes.
        - **Exemple 2 : Radiographies d'enfants pour la classe "normal"** : Si la plupart des images "normales" proviennent d'enfants (anatomie différente), le modèle peut apprendre à distinguer "enfant" de "adulte avec pathologie" au lieu de "normal" de "pathologie".
    - **Correction de la base de données** : Nettoyer les données pour enlever les biais (ex: masquer les annotations, équilibrer les sources de données).
    - **Choix du meilleur modèle pour interprétation (VGG16)** : Parfois, un modèle légèrement moins performant mais plus simple ou dont les activations sont plus facilement interprétables peut être préféré si l'explicabilité est cruciale. (Ici, VGG16 semble avoir été utilisé pour sa structure plus "classique" pour appliquer Grad-CAM).
- **Explainable DL for CT-scans COVID-19 images classification** : Application similaire de XAI sur des images de scanner (CT-scans), qui sont des volumes 3D.
- **CNN Layers visualisation (outils interactifs)** :
    - **CNN Explainer** (référence déjà vue).
    - **Ryerson ConvnetJS demo / TensorFlow Playground** : Outils qui permettent de visualiser les activations des différentes couches d'un CNN, aidant à comprendre ce que chaque couche apprend (des bords simples aux motifs plus complexes).
