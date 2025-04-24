# Rapport PC3

| N° Section | Section | Assigné | Livrables attendus |
| :-- | :-- | :-- | :-- |
| **1** | Abstract | Marmoush | - 1 page récapitulant le projet <br>- Rappel des objectifs initiaux <br>- État actuel d'avancement global |
| **2.1** | Réalisation matérielle | Debbouze | - 2 pages <br>- Schémas électroniques <br>- Photos du hardware sans boitier <br>- Description des capteurs (1 paragraphe) |
| **2.2** | Programmation ESP32 | Messiha | - 3 pages <br>- Diagramme d'état <br>- 3 captures d'écran de bout de code Arduino  <br>- Description des 3 codes |
| **2.3** | Développement Flutter | Dictateur | - 3 pages <br>- Architecture logicielle <br>- 3 captures d'écran de l'app avec description <br>- Diagramme de séquence utilisateur |
| **2.4** | Infrastructure AWS | Vertonghen | - 2 pages<br>- Pourquoi AWS ?<br>- Diagramme d'architecture cloud <br>- Description des services utilisés |
| **2.5** | Intégration système | Marmoush | - 1 page<br>- Photo du prototype final <br>- Description des défis d'intégration <br>- Lien vidéo de démonstration complète (à faire) |
| **3** | Gestion de projet | Dictateur | - 1 page avec diagramme de Gantt actualisé <br>- Répartition initiale vs réelle des tâches <br>- Retour sur la méthodologie utilisée |
| **4** | Retour formation "Machine Virtuelle" | TOUS | - 5 paragraphes distincts (1 par membre) <br>- Critique constructive de max. 10 lignes chacun |

# PowerPoint PC3 (Assigné : Marmoush)

| N° Slide | Section PowerPoint | Contenu demandé | Visuels requis |
| :-- | :-- | :-- | :-- |
| **1** | Page de titre | - Logo projet<br>- Titre complet<br>- Noms des membres<br>- Date de soutenance | Photo 3D du boîtier en fond |
| **2** | Contexte projet | - Diagramme "Avant/Après"<br>- Objectifs sous forme de checklist<br>- Carte mentale du système complet | Capture écran rapport section 1 |
| **3** | Hardware (2.1) | - Comparatif schéma théorique/réel<br>- Timeline des prototypes<br>- Spécifications techniques clés | 4 photos hardware + 1 vidéo 15s (sans boîtier) |
| **4** | ESP32 (2.2) | - Diagramme d'état interactif<br>- Code commenté (extraits)<br>- Métriques de performance | Capture IDE Arduino + graphes serial monitor |
| **5** | Flutter (2.3) | - Démo vidéo app (30s)<br>- Architecture modulaire<br>- UX avantages/limites | 3 captures écran annotées + mockup |
| **6** | AWS (2.4) | - Dashboard personnalisé<br>- Diagramme architecture dynamique<br>- Coûts mensuels estimés | Capture CloudWatch + schéma MermaidJS |
| **7** | Intégration (2.5) | - Comparatif boîtier v1/v2<br>- Diagramme d'assemblage<br>- Test d'étanchéité | 2 photos techniques + vidéo 20s (montage) |
| **8** | Conclusion | - Roadmap finale<br>- Budget réel vs prévu<br>- Perspectives commerciales |

# Rapport final

| N° Section | Section | Assigné | Livrables attendus |
| :-- | :-- | :-- | :-- |
| **0** | Abstract | Vertonghen | - 1 paragraphe (max 250 mots) <br>- Synthèse du problème, solution, résultats clés |
| **1** | Introduction | Marmoush | - 1-2 pages <br>- Contexte (marché, besoin utilisateur) <br>- Problématique détaillée <br>- Objectifs SMART du projet <br>- Présentation de la solution globale <br>- Structure du rapport |
| **2** | État de l'art et Analyse fonctionnelle | Dictateur | - 2-3 pages <br>- Revue des solutions existantes (trackers GPS moto) <br>- Analyse des besoins fonctionnels et non-fonctionnels <br>- Justification des choix technologiques majeurs (ESP32 vs autres, Flutter, AWS) |
| **3** | Conception Détaillée |  |  |
| 3.1 | Architecture Matérielle | Debbouze | - 2-3 pages <br>- Schéma bloc fonctionnel du hardware <br>- Choix et justification des composants (ESP32, GPS, SIM, batterie, etc.) <br>- Schéma électronique final (KiCad/Eagle) <br>- Conception du PCB (si applicable) ou prototype sur breadboard/veroboard (photos détaillées) |
| 3.2 | Conception du Logiciel Embarqué | Messiha | - 2-3 pages <br>- Architecture logicielle (RTOS ? Boucle principale ?) <br>- Diagramme d'états final <br>- Algorithmes clés (lecture GPS, communication serveur, gestion énergie) <br>- Description des bibliothèques utilisées |
| 3.3 | Conception de l'Application Mobile | Vertonghen | - 2-3 pages <br>- Architecture de l'application (MVVM, Bloc, etc.) <br>- Maquettes UI/UX finales (Figma/Adobe XD si utilisé, sinon captures d'écran clés) <br>- Diagramme de navigation <br>- Description de l'API REST utilisée côté client |
| 3.4 | Conception de l'Infrastructure Cloud | Dictateur | - 2 pages <br>- Architecture AWS détaillée finale (diagramme) <br>- Choix et configuration des services (IoT Core, Lambda, DynamoDB, API Gateway, etc.) <br>- Modèle de données (DynamoDB) <br>- Description de l'API REST côté serveur |
| 3.5 | Conception du Boîtier | Marmoush | - 1-2 pages <br>- Processus de design (croquis, CAO) <br>- Choix des matériaux <br>- Contraintes (étanchéité, fixation, taille) <br>- Plans/Vues 3D du boîtier final <br>- Photos du boîtier imprimé/réalisé |
| **4** | Implémentation et Intégration |  |  |
| 4.1 | Réalisation et Assemblage Hardware | Debbouze | - 1 page <br>- Photos du montage finalisé (avec et sans boîtier) <br>- Défis rencontrés lors de l'assemblage |
| 4.2 | Développement Logiciel Embarqué | Messiha | - 1-2 pages <br>- Extraits de code pertinents et commentés (5 max) <br>- Difficultés et solutions (debug, optimisations) |
| 4.3 | Développement Application Mobile | Vertonghen | - 1 page <br>- Captures d'écran finales de l'application <br>- Liens vers démo vidéo de l'app (si disponible) |
| 4.4 | Déploiement Infrastructure Cloud | Dictateur | - 1 page <br>- Étapes de déploiement (Serverless framework, Console AWS...) <br>- Enjeux de sécurité mis en place |
| 4.5 | Intégration et Système Complet | Marmoush | - 1 page <br>- Description du processus d'intégration (Hardware + Software + Cloud) <br>- Photo/Vidéo du système complet en fonctionnement <br>- Principaux points de blocage et comment ils ont été résolus |
| **5** | Tests et Validation | Messiha | - 2-3 pages <br>- Stratégie de test (unitaire, intégration, système, utilisateur) <br>- Description des cas de test <br>- Résultats des tests (GPS précision, autonomie batterie, temps de réponse serveur/app) <br>- Tableau comparatif : Objectifs vs Résultats |
| **6** | Analyse des Résultats et Discussion | Debbouze | - 1-2 pages <br>- Interprétation des résultats de tests <br>- Performances globales du système <br>- Limitations identifiées <br>- Comparaison aux objectifs initiaux et à l'état de l'art |
| **7** | Gestion de Projet | Vertonghen | - 1-2 pages <br>- Planning prévisionnel vs réalisé (Gantt final) <br>- Répartition des tâches et collaboration <br>- Outils utilisés (Trello, Git, Discord...) <br>- Analyse des risques et gestion des imprévus <br>- Budget prévisionnel vs dépenses réelles |
| **8** | Conclusion et Perspectives | Marmoush | - 1 page <br>- Rappel des objectifs et synthèse des réalisations <br>- Bilan du projet (points forts, points faibles) <br>- Pistes d'améliorations futures <br>- Ouverture (potentiel commercial, développements futurs) |
| **9** | Bibliographie | Dictateur | - Liste des sources (articles, datasheets, tutoriels, bibliothèques) <br>- Format standard (IEEE, APA...) |
| **10** | Annexes | TOUS (chacun fournit sa partie) | - A : Schémas électroniques complets (Debbouze) <br>- B : Code source principal ESP32 (commenté) (Messiha) <br>- C : Manuel utilisateur simplifié (Vertonghen) <br>- D : Configuration / Scripts clés AWS (Dictateur) <br>- E : Plans détaillés du boîtier (Marmoush) <br>- F : Fiches techniques composants principaux (Debbouze) |
