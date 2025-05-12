# **Script de Présentation PC4 – Projet "Tick"**

---

## **(0:30) Introduction générale - Mohammed**

**Mohammed :** "Bonjour à tous. Nous sommes ravis de vous présenter aujourd'hui l'aboutissement de notre projet d'informatique : "Tick", un module intelligent d’alerte et de localisation pour véhicules.

---

## **(2:00) Problème et la limite des solutions actuelles - Mohammed**

**Mohammed :** "Le problème est concret : en Belgique, rien qu'en 2023, la Police Fédérale a enregistré plus de 3600 vols de deux-roues motorisés. Pour les motos, qui représentent un investissement financier conséquent, souvent au-delà de 10 000 euros, la perte est d'autant plus douloureuse.

Face à ce risque, quelles sont les solutions existantes ? Les dispositifs de sécurité traditionnels, comme les robustes cadenas en U, les bloque-disques, ou les chaînes, sont bien entendu indispensables. Ils constituent le premier rempart physique contre le vol. Cependant, malgré leur utilité, ils présentent des limites intrinsèques importantes :
1.  Tout d'abord, aussi solides soient-ils, ils peuvent être neutralisés par des voleurs déterminés et équipés. Cela peut être une question de temps, mais ils ne sont pas infaillibles.
2.  Surtout, elles n'offrent **aucune alerte en temps réel** au propriétaire lors d'une tentative de vol. Lors d'une tentative de vol, ces dispositifs ne vous préviennent pas. Le propriétaire ne découvre le méfait que bien plus tard, souvent le lendemain matin, lorsqu'il est déjà trop tard pour réagir rapidement. L'effet de surprise est total pour le voleur.
3.  Et une fois le vol commis, ces dispositifs ne fournissent **aucune aide active** pour localiser ou récupérer le véhicule.

Néanmoins, c'est important de le souligner : **Tick n'est pas un antivol au sens strict**. Il ne va pas physiquement empêcher le vol comme le ferait un cadenas. Notre objectif n'est pas de remplacer ces dispositifs, mais de les **compléter** en apportant une couche de sécurité active et informative."

---

## **(1:30) Présentation de la solution "Tick" - Mohammed**

**Mohammed :** "C'est pour pallier ces manques que nous avons développé "Tick". Il s'agit d'un système intelligent et connecté, conçu pour être discrètement installé sur votre véhicule.

L'objectif principal de Tick est double :
1.  **Alerter immédiatement** le propriétaire en cas de mouvements suspects ou non autorisés de son véhicule. Grâce à son système de détection de mouvement intégré, le Tick est capable d'identifier des mouvements suspects ou non autorisés de votre véhicule. Si une telle activité est détectée, le propriétaire est instantanément alerté via une notification sur son smartphone. Cette réactivité peut faire toute la différence, permettant potentiellement d'interrompre le vol en cours.
2.  Permettre un **suivi de localisation GPS en temps réel** en cas de vol, augmentant ainsi significativement les chances de récupération. Si malheureusement le vol ne peut être évité, Tick ne vous laisse pas sans ressources. Grâce à son module GPS, il permet au propriétaire de suivre la position de son véhicule en temps réel via l'application mobile. Cette information est cruciale pour les forces de l'ordre et augmente considérablement les chances de retrouver le véhicule.

Le Tick vise donc à offrir une surveillance continue et des informations cruciales, là où les antivols traditionnels sont passifs. Il apporte une réactivité et une capacité de suivi post-vol.

*(Transition vers Walid)*
Pour vous montrer comment nous avons concrétisé cette solution, Walid va commencer par vous présenter la partie matérielle et embarquée du système."

---

## **(3:00) Technique : ESP32 – Walid**

**Walid :** "Merci Mohammed. Bonjour à tous. La partie matérielle, le 'Tick' physique, est au cœur de notre système.

*   **Présenter les modules :**
*   *(Distribuer la première page aux profs avec les images des modules)*
    Notre prototype s'articule autour de la carte **LilyGO T-SIM7000G**. C'est une solution intégrée performante qui combine un microcontrôleur **ESP32** (avec Wi-Fi et Bluetooth), un module de communication cellulaire **SIM7000G** (pour la connectivité 4G LTE-M/NB-IoT) et un récepteur **GPS** intégré.
    Pour la détection de mouvement, nous utilisons un capteur **MPU6050**, une unité de mesure inertielle avec accéléromètre et gyroscope, communiquant en I2C avec l'ESP32.
    L'alimentation est assurée par une **batterie LiPo rechargeable**, gérée par la carte LilyGO.

*   **Présenter le boîtier :**
    *(Montrer le boitier)*
    Le prototype du boitier, en forme de tique, est pour le moment imprimé en 3D en PLA. Les composants sont assemblés de manière compacte à l'intérieur pour faciliter une intégration discrète. L'un des axes d'amélioration futurs serait un boîtier étanche et encore plus compact.

*   **Parler des fonctionnalités :**
    L'ESP32 est programmé pour :
    *   Lire en continu les données de l'accéléromètre MPU6050.
    *   Détecter les mouvements suspects selon des seuils que nous avons définis.
    *   Acquérir les coordonnées GPS.
    *   Gérer différents états de fonctionnement : un mode **veille** pour économiser l'énergie où seul l'accéléromètre est actif à basse consommation, un mode **actif** où tous les modules sont réveillés lors d'une détection, et un mode **alerte** pour la transmission des données.
    *   Communiquer avec notre backend cloud AWS via MQTT sur le réseau cellulaire pour envoyer les alertes et les données de localisation. Il gère également la réception de commandes depuis l'application, comme la demande de localisation.

*   **Parler de l’autonomie (avec le graphe) :**
    L'autonomie est un point crucial. Nous avons travaillé sur l'optimisation de la consommation, notamment avec le mode veille profonde de l'ESP32 et la gestion de l'alimentation des modules.
*   *(Distribuer la deuxième page aux profs avec les graphes d'autonomie)*
    Nos tests préliminaires montrent une autonomie de X heures en conditions typiques, mais c'est un domaine où des améliorations sont possibles, notamment avec un choix de batterie plus conséquent ou une optimisation logicielle plus poussée. Par exemple, le mode veille est conçu pour ne réveiller le GPS et le module cellulaire que lorsque c'est strictement nécessaire. Grace à ces premières optimisations, le Tick passe à une autonomie de X heures.

*(Transition vers Aymeric)*
Maintenant qu'on a vu le dispositif physique, Aymeric va vous expliquer comment toutes ces données sont gérées et traitées dans le cloud."

---

## **(3:00) Technique : Cloud – Aymeric**

**Aymeric :** "Merci Walid. Bonjour à tous. Pour la partie backend, nous avons fait le choix stratégique d'utiliser Amazon Web Services (AWS).

*   **C'est quoi AWS et pourquoi ?**
    AWS est une plateforme cloud offrant une vaste gamme de services. Nous l'avons choisie pour plusieurs raisons :
    *   **Son écosystème IoT mature :** AWS IoT Core est spécifiquement conçu pour les appareils connectés, simplifiant la gestion des connexions et la communication.
    *   **Son modèle serverless :** Avec des services comme AWS Lambda, nous payons uniquement pour ce que nous consommons et n'avons pas à gérer de serveurs, ce qui est idéal pour un prototype et pour la scalabilité.
    *   **Sa scalabilité et fiabilité :** AWS permet de gérer un grand nombre d'appareils et de requêtes sans effort.
    *   **Son coût :** Le niveau gratuit d'AWS nous a permis de développer et de profiter de toutes les fonctionnalités d'AWS sans coût. Par exemple, avec l'offre gratuite, la limite mensuelle s'élève à 1 million de requête API, 500 milles messages MQTT etc.

*   **Limites quasi illimitées :**
    L'architecture serverless et les services managés d'AWS signifient que notre solution peut théoriquement s'adapter à des milliers, voire des millions d'utilisateurs et de dispositifs Tick sans avoir à re-architecturer en profondeur. La scalabilité est gérée par AWS.

*   **Technologies utilisées :**
*   *(Distribuer la troisième page aux profs avec le diagramme AWS d'Aymeric)*
    Notre infrastructure sur AWS repose sur plusieurs services clés :
    *   **AWS IoT Core :** C'est le hub central. Nos Ticks s'y connectent de manière sécurisée via MQTT. IoT Core reçoit les messages de statut, de position, d'alerte, et peut aussi transmettre des commandes de l'application vers les Ticks.
    *   **Amazon DynamoDB :** C'est notre base de données NoSQL. Nous y stockons les informations des Ticks (ID, nom, dernière position, statut, etc.) dans une table 'Ticks', et l'historique des alertes et des positions dans une table 'TickAlertHistory'. C'est scalable et performant.
    *   **AWS Lambda :** Ce sont nos fonctions backend sans serveur. Elles sont déclenchées soit par des requêtes HTTP depuis l'application mobile (via des Function URLs sécurisées par IAM) pour des actions comme associer un Tick, récupérer l'historique, ou demander une localisation, soit par des règles définies dans IoT Core pour traiter les messages entrants des Ticks (par exemple, mettre à jour DynamoDB).
    *   **(Firebase pour les notifications push) :** Bien que Firebase ne soit pas un service AWS, il s'intègre. Lorsqu'un événement important se produit (comme une alerte de mouvement), une fonction Lambda peut déclencher l'envoi d'une notification push via AWS SNS (Simple Notification Service) qui, à son tour, communique avec Firebase Cloud Messaging (FCM) pour notifier l'utilisateur sur son application mobile.

*(Transition vers la démo)*
Maintenant que vous avez une vue d'ensemble de l'architecture matérielle et cloud, nous allons passer à une démonstration live pour vous montrer comment tout cela fonctionne ensemble."

---

## **(5:00) Démonstration live - Tous**

**Aymeric :** "Pour cette démonstration, nous allons simuler une situation concrète. Ayoub n'est pas avec nous physiquement ici... car il est actuellement à l'extérieur, près d'une moto équipée de notre système Tick. Nous avons installé une 'caméra de sécurité' – en réalité son téléphone en appel Teams – qui pointe vers la moto.
Maximilien va projeter l'écran de son téléphone avec l'application Tick.

*(Mohammed/Maximilien projette l'application Tick sur l'écran principal, et met en place l'appel Teams avec la caméra d'Ayoub en Picture-in-Picture ou sur un écran partagé.)* "

**Maximilien :** "Voilà, vous pouvez voir sur l'application l'état actuel du Tick sur la moto. Il est actuellement 'Actif' et 'Immobile', avec sa dernière position connue."

**Aymeric :** "Ayoub, tu peux commencer la simulation. Imagine que tu es une personne mal intentionnée."

*(Ayoub, visible sur la caméra externe, s'approche de la moto et commence à la manipuler, la secouer un peu.)*

**Walid :** "Sur l'ESP32, le MPU6050 détecte cette vibration. Si le mouvement dépasse le seuil que nous avons configuré, il passe en mode 'Alerte Mouvement'."

**Maximilien :** *(Montrant l'application)* "Et voilà ! Regardez l'application. Nous venons de recevoir une notification push. Le statut du Tick est passé à 'Mouvement détecté'. La couleur a changé pour attirer l'attention. L'heure de l'alerte est enregistrée."

**Aymeric :** "Maintenant, Ayoub, simule le vol. Déplace la moto sur quelques mètres."

*(Ayoub commence à déplacer la moto, s'éloignant de sa position initiale.)*

**Walid :** "Lorsque le Tick détecte un mouvement prolongé ou un déplacement significatif après une alerte initiale, l'ESP32 active le GPS pour obtenir une nouvelle position et l'envoie au cloud."

**Maximilien :** *(Montrant l'application)* "Regardez la carte sur l'application. La position du Tick vient de se mettre à jour ! Nous pouvons suivre en temps réel le déplacement de la moto. Si Ayoub continue de se déplacer, la position continuera d'être actualisée à intervalle régulier."

**Mohammed :** "À partir de là, l'utilisateur a plusieurs options : il peut consulter l'historique des alertes et des positions, et bien sûr, contacter les forces de l'ordre en leur fournissant la localisation précise du véhicule."

**Aymeric :** "Merci Ayoub, tu peux arrêter la simulation. Cette démonstration illustre le flux complet : détection par l'ESP32, communication vers AWS, traitement par Lambda, mise à jour de DynamoDB, notification push vers l'application Flutter et visualisation en temps réel.

*(Transition vers Max)*
Au-delà de cette fonctionnalité principale de suivi et d'alerte, l'application offre d'autres possibilités de gestion que Maximilien va maintenant vous présenter."

---

## **(2:00) Technique : Autres fonctionnalités application – Max**

**Maximilien :** "Merci Aymeric. Effectivement, l'application mobile, développée avec Flutter, est l'interface principale pour l'utilisateur. Outre la visualisation des alertes et de la carte que vous venez de voir, elle permet de gérer l'ensemble du cycle de vie du Tick et du compte utilisateur.

*(Naviguer dans l'application pour montrer brièvement chaque point)*

*   **Créer un compte / Se connecter / Se déconnecter :**
    Nous utilisons AWS Cognito pour une authentification sécurisée des utilisateurs. On peut créer un compte avec email et mot de passe, se connecter, et bien sûr se déconnecter. *(Montrer les écrans respectifs)*

*   **Associer un Tick (Appairage) :**
    Lorsqu'un utilisateur acquiert un nouveau Tick, il doit l'associer à son compte. Cela se fait via Bluetooth Low Energy (BLE). L'application scanne les Ticks à proximité, l'utilisateur sélectionne le sien, lui donne un nom, et les informations d'identification sont échangées de manière sécurisée pour lier le Tick au compte utilisateur et à notre backend AWS. *(Montrer l'écran d'ajout de Tick et le scan BLE si possible, ou une capture d'écran)*

*   **Dissocier un Tick :**
    Inversement, un utilisateur peut dissocier un Tick de son compte s'il ne l'utilise plus ou s'il le vend. Cela supprime le lien dans notre base de données.

*   **Désactiver temporairement / Réactiver la surveillance :**
    L'utilisateur peut vouloir désactiver temporairement la surveillance, par exemple s'il prête sa moto à un ami ou pour une maintenance. L'application permet d'envoyer une commande au Tick pour le mettre en mode 'Inactif'. Il arrêtera alors d'envoyer des alertes de mouvement, tout en restant joignable pour une réactivation. On peut ensuite le réactiver de la même manière. *(Montrer le bouton ou l'option pour changer l'état du Tick)*

Toutes ces interactions transitent par nos fonctions Lambda sur AWS, qui communiquent ensuite avec DynamoDB ou AWS IoT Core pour interagir avec le dispositif Tick.

*(Transition vers Ayoub pour la conclusion)*
Voilà pour les principales fonctionnalités de l'application. Ayoub va maintenant conclure cette présentation."

---

## **(3:00) Conclusion & vision future – Ayoub**

**Ayoub :** "Merci Maximilien. Pour conclure :

*   **Rappeler très vite fait le Tick en une phrase :**
    Tick est donc un système connecté de surveillance et de localisation qui vient compléter la sécurité passive de votre véhicule en vous alertant et en vous aidant à le retrouver en cas de vol.

*   **Solutions apportées par le Tick :**
    Comme nous l'avons démontré, Tick apporte une réponse aux limites des antivols classiques en offrant :
    *   Des alertes en temps réel en cas de mouvement suspect.
    *   Un suivi GPS pour la localisation post-vol.
    *   Une interface utilisateur intuitive pour gérer et surveiller son véhicule à distance.

*   **Vision future :**
    Ce projet a été une formidable expérience d'apprentissage, et nous avons identifié de nombreuses pistes d'amélioration et d'évolution pour Tick :
    *   **Utiliser nos propres composants / Améliorer le matériel :** Actuellement, nous utilisons des modules comme la carte LilyGO. À terme, concevoir notre propre PCB (carte de circuit imprimé) nous permettrait d'optimiser la taille, la consommation et d'intégrer des composants spécifiques à nos besoins, comme un GPS plus performant (pour une meilleure précision et un Time To First Fix plus rapide).
    *   **Améliorer l’autonomie :** C'est un enjeu majeur. En plus d'une optimisation matérielle, nous pourrions explorer des batteries de plus grande capacité, des modes de veille encore plus poussés, voire une petite recharge solaire pour certains cas d'usage.
    *   **Design beaucoup plus compact et robuste :** Un boîtier plus petit, étanche (IP67 par exemple), et résistant aux chocs rendrait le Tick encore plus discret et adapté à toutes les conditions.
    *   **Intelligence Artificielle pour les alertes faux positifs :** Pour affiner la détection de mouvement, on pourrait intégrer de l'IA embarquée (TinyML) ou côté cloud pour mieux distinguer un vrai vol de vibrations normales ou d'une simple bousculade accidentelle, réduisant ainsi les fausses alertes.
    *   **Haut-parleur :** *(Montrer le haut-parleur et l'ampli PAM8403 si disponibles et non intégrés dans la démo)* Nous avions prévu d'intégrer ce haut-parleur. Il pourrait servir de sirène dissuasive activable à distance via l'application, ou pour émettre des signaux sonores indiquant l'état du Tick (appairage, batterie faible). Cela n'a pas pu être finalisé par manque de temps, mais c'est une évolution logique.
    *   **Exporter le produit pour tout type de véhicule + les flottes d’entreprise :** Bien que notre focus initial ait été les deux-roues personnels, la technologie Tick est adaptable pour sécuriser des voitures, des vélos, des engins de chantier, et même pour la gestion de flottes d'entreprise, avec des fonctionnalités spécifiques (suivi multiple, rapports, etc.).

*   **Conclusion :**
    Le projet Tick nous a permis de mettre en pratique l'ensemble du cycle de développement d'un produit IoT, de la conception matérielle à l'application mobile, en passant par l'infrastructure cloud. Nous avons rencontré des défis, notamment en termes d'intégration des différents composants et d'optimisation de l'autonomie, mais nous sommes fiers du système fonctionnel que nous vous avons présenté aujourd'hui. Il y a un réel potentiel pour que Tick devienne une solution de choix pour la sérénité des propriétaires de véhicules.

Nous vous remercions pour votre attention et nous sommes maintenant prêts à répondre à vos questions et à discuter plus en détail des aspects techniques ou des choix que nous avons faits."

---

**Quelques remarques supplémentaires pour vous :**

*   **Répartition du travail :** Bien que le script le suggère par qui parle de quoi, soyez prêts à répondre à la question "qui a fait quoi" plus explicitement si elle est posée.
*   **Difficultés rencontrées :** Elles sont subtilement intégrées (autonomie, intégration, faux positifs à améliorer). Soyez prêts à en parler plus si on vous le demande. Le PC3 liste pas mal de défis d'intégration qui peuvent être évoqués.
*   **Code :** Ayez quelques extraits de code clés prêts à être montrés si besoin (ex: la logique de détection sur ESP32, une fonction Lambda importante, un service Flutter).
*   **Budget :** Si vous avez eu un budget et que vous devez en parler, Ayoub pourrait l'intégrer brièvement avant la vision future.
*   **Fluidité :** Entraînez-vous pour que les transitions soient naturelles.
*   **La démo :** C'est le clou du spectacle. Assurez-vous que la technique (projection, appel Teams, connexion internet) est irréprochable. Ayez un plan B (vidéo de la démo) au cas où quelque chose tournerait mal en live.
