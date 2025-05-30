# Chapitre 1 : Introduction à l’Intelligence Artificielle

## 1. Introduction : Qu'est-ce que l'IA et où la trouve-t-on ?
- **Questions de définition et d'applications** : L'IA cherche à créer des machines capables de simuler l'intelligence humaine pour effectuer des tâches variées. Ses applications sont vastes et en pleine expansion.
- **Exemples d'applications concrètes** :
    - **Voiture autonome** : Prise de décision, perception de l'environnement.
    - **Assistant Alexa** : Compréhension du langage naturel, réponse aux requêtes.
    - **Robocup 2018** : Coordination d'agents autonomes (robots footballeurs).
    - **Robot Boston Dynamics** : Mobilité avancée, équilibre, interaction physique.

## 2. Prérequis de l’IA : Composants Essentiels et Fondations (incluant Vision de Turing)
- **Concepts fondamentaux** : Nécessité de bases en logique, probabilités, statistiques, théorie de l'information.
- **Algorithmes** : Séquences d'instructions précises pour résoudre un problème ou effectuer une tâche. Au cœur de l'IA.
- **Big Data** : Vastes ensembles de données nécessaires pour entraîner et améliorer les modèles d'IA, notamment en apprentissage machine.
- **Cloud computing** : Fournit la puissance de calcul et le stockage nécessaires pour traiter le Big Data et exécuter des algorithmes d'IA complexes.
- **Principales étapes du parcours de l’IA (vision d'Alan Turing)** : Référence aux idées pionnières de Turing sur la possibilité pour les machines de "penser" et les capacités qu'elles devraient démontrer.

## 3. Définitions de l’IA : Quatre Approches et Leurs Caractéristiques
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

## 4. Histoire de l’IA : Des Origines aux Développements Récents
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

## 5. Approches de l’IA : Méthodologies (Symbolique, Apprentissage) et Formes (Générative, Augmentée)
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

## 6. Applications de l’IA : Domaines d'Impact, Exemples Concrets et Paradoxe de Moravec
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

## 7. Conclusion (Récapitulatif du Chapitre 1)
- Ce chapitre a introduit les concepts clés de l'IA, ses définitions, son histoire, ses approches méthodologiques et ses principales applications, posant les bases pour la suite du cours.
