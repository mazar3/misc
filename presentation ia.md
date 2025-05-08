**(Max)**

*   **Slide 11 : Tâche 4 - Application Vidéo en Temps Réel**
    *   Titre : En Action : Traitement Vidéo
    *   Objectif : Appliquer les modèles sur une séquence vidéo.
    *   Méthode : Capture frame par frame, inférence, annotation.
    *   **Démonstration (très courts clips ou GIFs) :**
        *   Extrait 1 : Classification en temps réel (texte de la classe sur la vidéo).
        *   Extrait 2 : Détection en temps réel (BBoxes sur la vidéo).
        *   Extrait 3 : Segmentation en temps réel (Masques sur la vidéo).
    *   Commentaire : "Les modèles montrent leur capacité à généraliser sur des séquences animées."

*   **Slide 12 : Tâche 5 - Analyse des Performances (PyTorch bench)**
    *   Titre : Sous le Capot : Benchmarking des Modèles
    *   Objectif : Évaluer la consommation de ressources (calcul, mémoire, énergie).
    *   Outil : `pytorch-bench`.
    *   Visuel : **Le tableau récapitulatif généré par votre script.**
        *   Mettre en évidence (ex: avec des flèches ou en gras) une ou deux lignes/colonnes intéressantes.
        *   Exemple de commentaire : "On observe que [Modèle X] est le plus rapide en FPS, tandis que [Modèle Y] est plus léger en taille."

*   **Slide 13 : Conclusion & Perspectives**
    *   Titre : Bilan et Prochaines Étapes
    *   **Réalisations :**
        *   Succès dans les 5 tâches assignées.
        *   Développement d'un pipeline complet (données -> entraînement -> évaluation -> application).
        *   Utilisation efficace de W&B pour le suivi.
    *   **(Optionnel) Défis :** Brièvement, si un défi notable a été surmonté (ex: gestion des données, temps de calcul).
    *   **Perspectives :**
        *   Optimisation des hyperparamètres (Optuna).
        *   Participation à la compétition Kaggle.
        *   Test d'architectures plus larges ou plus récentes.
        *   Déploiement sur un système embarqué (si pertinent).

*   **Slide 14 : Questions & Remerciements**
    *   Titre : Merci de votre attention !
    *   "Des questions ?"
    *   Peut-être le lien vers votre repo GitHub/WanDB si vous voulez le partager.

---

**Conseils Généraux pour la Présentation :**

1.  **Répétez ensemble !** Surtout les transitions entre les orateurs.
2.  **Soyez dynamiques :** Variez le ton, regardez l'audience.
3.  **Maîtrisez votre temps :** 10-15 minutes, c'est court. Chaque slide doit être pertinente. Ne lisez pas vos slides.
4.  **Visuels > Texte :** Privilégiez les graphiques, images, démos. Les slides doivent supporter votre discours, pas le remplacer.
5.  **Simplicité :** N'entrez pas dans des détails trop techniques sauf si on vous le demande. L'idée est de montrer que vous avez fait le travail et obtenu des résultats.
6.  **Anticipez les questions :** Qui répond à quoi ? Préparez-vous sur les choix de modèles, les hyperparamètres, les difficultés.
7.  **WandB :** N'hésitez pas à mentionner que vous avez utilisé WandB pour logger vos expériences, c'est un bon point. Vous pouvez même avoir un onglet de navigateur ouvert sur un dashboard WandB pertinent en backup si une question pointue arrive.


---

Commençons par la **Tâche 4 : Application Vidéo en Temps Réel** (ta Slide 11).

---


"Je vais vous présenter la quatrième tâche de notre projet : **l'application de nos modèles sur une vidéo en temps réel.**"

**(En affichant la Slide 11)**

"L'objectif ici était de démontrer la capacité de nos modèles, que ce soit pour la classification, la détection ou la segmentation, à fonctionner sur des séquences dynamiques, simulant une application de surveillance vidéo sur un chantier."

"Pour cela, nous avons utilisé la vidéo `unity_video.mp4` fournie. Notre approche a été de traiter cette vidéo image par image. Pour chaque image :
*   Nous avons d'abord appliqué notre **meilleur modèle de classification** – dans notre cas, c'était le [mentionne le nom de ton meilleur modèle de classif, ex: ConvNeXt que nous avons entraîné] – pour obtenir une prédiction globale de la scène.
*   Ensuite, nous avons utilisé notre **modèle YOLOv11 entraîné pour la détection** afin d'identifier et de localiser les objets avec des boîtes englobantes.
*   Et enfin, notre **modèle YOLOv11-seg pour la segmentation**, pour délimiter précisément la forme de ces objets."

"Chaque modèle a produit des annotations sur les images, et nous avons ensuite reconstruit trois vidéos distinctes pour visualiser ces résultats."

**(Pendant que les GIFs/vidéos tournent, pointe-les ou décris-les brièvement)**

"Vous pouvez voir ici :
*   **À gauche/En haut (selon ta disposition), la vidéo de classification :** le label de la classe prédite pour l'ensemble de la scène est affiché sur chaque image.
*   **Au centre/Milieu, la détection d'objets :** le modèle YOLOv11 identifie et encadre les différents objets d'intérêt comme les ouvriers ou les engins, en affichant leur classe.
*   **Et à droite/En bas, la segmentation :** les masques de segmentation, générés par YOLOv11-seg, délimitent plus finement la forme des objets détectés, offrant une compréhension plus détaillée de la scène."

"Ces démonstrations illustrent comment nos différents modèles peuvent être intégrés pour une analyse vidéo complète. Bien sûr, pour une véritable application en temps réel, des optimisations supplémentaires seraient nécessaires pour assurer une haute cadence d'images par seconde, mais cela donne un bon aperçu du potentiel."

**(Transition vers ta prochaine slide/partie - Tâche 5 Benchmarking)**

"Après avoir vu nos modèles en action, il est important d'évaluer leur efficacité non seulement en termes de précision, mais aussi en termes de ressources consommées. C'est l'objet de la cinquième tâche : l'analyse des performances avec PyTorch bench, que je vais vous présenter maintenant."

---

### 3. Explication du Code pour la Tâche 4 (Vidéo)

Tu as trois blocs principaux de code pour générer les vidéos. Voici les points clés pour chacun :

**Code Commun (Setup) :**
*   `video_input_path = "/content/unity_video.mp4"` : Chemin vers la vidéo source.
*   `video_output_cls_path`, `_det_path`, `_seg_path`: Chemins pour les vidéos de sortie.
*   `os.makedirs("/content/output_videos", exist_ok=True)`: Crée le dossier de sortie s'il n'existe pas.
*   `cv2.VideoCapture(video_input_path)`:
    *   **Explication :** "Cette fonction d'OpenCV ouvre le fichier vidéo. L'objet `cap` retourné nous permet ensuite de lire les images (frames) une par une."
*   `fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))`, `fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))`, `fps = int(cap.get(cv2.CAP_PROP_FPS))`:
    *   **Explication :** "On récupère les propriétés de la vidéo d'origine : sa largeur (`fw`), sa hauteur (`fh`), et son nombre d'images par seconde (`fps`). Ces informations sont essentielles pour créer les vidéos de sortie avec les mêmes caractéristiques."
*   `cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))`:
    *   **Explication :** "Ceci initialise un objet `VideoWriter` qui va nous permettre d'écrire les images traitées dans un nouveau fichier vidéo.
        *   `output_path`: Le nom du fichier de sortie.
        *   `cv2.VideoWriter_fourcc(*'mp4v')`: C'est le codec utilisé pour encoder la vidéo. `'mp4v'` est un identifiant FourCC pour le format MPEG-4, compatible avec les fichiers .mp4.
        *   `fps`: Le nombre d'images par seconde de la vidéo de sortie (on réutilise celui de l'entrée).
        *   `(fw, fh)`: La taille des images de la vidéo de sortie."
*   `while True: ret, frame = cap.read(); if not ret: break`:
    *   **Explication :** "C'est la boucle principale qui parcourt la vidéo. `cap.read()` tente de lire la prochaine image. `ret` est un booléen qui est `True` si une image a été lue correctement, et `frame` est l'image elle-même (un tableau NumPy). Si `ret` est `False`, cela signifie qu'on a atteint la fin de la vidéo et on sort de la boucle."
*   `cap.release(); out_XYZ.release()`:
    *   **Explication :** "Après avoir traité toute la vidéo, il est crucial de libérer les ressources. `cap.release()` ferme le fichier vidéo d'entrée, et `out_XYZ.release()` finalise et ferme le fichier vidéo de sortie."

**7.1 Génération de la vidéo de classification :**
*   `best_cls_model_for_video_path = ...`:
    *   **Explication :** "Pour la vidéo de classification, on ne prend pas n'importe quel modèle. On a une logique qui parcourt les métriques de nos modèles de classification (CNN, ViT, ConvNeXt) stockées précédemment (`stored_model_metrics`) et sélectionne celui qui a obtenu la meilleure accuracy de validation. C'est ce modèle 'champion' qu'on utilise pour la vidéo."
*   `if "CNN_ResNet18" in best_cls_model_name_for_video: ... elif "ViT_base_patch16_224" ... elif "ConvNeXt_tiny" ...`:
    *   **Explication :** "En fonction du nom du meilleur modèle, on recrée son architecture. Par exemple, si c'est un ResNet18, on instancie `torchvision.models.resnet18` et on adapte sa couche de sortie `fc` au nombre de classes de notre dataset."
*   `loaded_model_for_video_cls.load_state_dict(torch.load(best_cls_model_for_video_path, map_location=device))`:
    *   **Explication :** "On charge les poids (les paramètres appris) du meilleur modèle sauvegardé dans l'architecture qu'on vient de recréer. `map_location=device` assure que le modèle est chargé sur le bon device (CPU ou GPU)."
*   `loaded_model_for_video_cls.eval()`:
    *   **Explication :** "Très important ! On passe le modèle en mode évaluation. Cela désactive certaines couches spécifiques à l'entraînement comme le Dropout ou la Batch Normalization qui se comportent différemment en inférence."
*   `img_pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))`:
    *   **Explication :** "OpenCV lit les images au format BGR (Bleu, Vert, Rouge). Les modèles PyTorch sont généralement entraînés avec des images RGB (Rouge, Vert, Bleu). Donc, on convertit l'image (`frame`) de BGR à RGB, puis on la transforme en objet PIL Image, car notre pipeline de transformation (`classification_transform`) l'attend sous ce format."
*   `input_tensor = classification_transform(img_pil).unsqueeze(0).to(device)`:
    *   **Explication :** "L'image PIL est passée à travers les mêmes transformations que celles utilisées pendant l'entraînement (redimensionnement, normalisation des pixels). `unsqueeze(0)` ajoute une dimension de batch (taille 1) car les modèles PyTorch attendent des batches d'images. Enfin, `.to(device)` envoie le tenseur sur le GPU si disponible."
*   `with torch.no_grad(): outputs = loaded_model_for_video_cls(input_tensor); _, pred_idx = torch.max(outputs,1)`:
    *   **Explication :** "`torch.no_grad()` est un gestionnaire de contexte qui désactive le calcul des gradients. C'est crucial en inférence pour économiser de la mémoire et accélérer les calculs. On passe le tenseur d'entrée au modèle pour obtenir les `outputs` (scores pour chaque classe). `torch.max(outputs, 1)` trouve la classe avec le score le plus élevé et retourne sa valeur et son index (`pred_idx`)."
*   `last_label = class_names_cls[pred_idx.item()]`:
    *   **Explication :** "On utilise l'index de la classe prédite pour récupérer son nom lisible à partir de notre liste `class_names_cls`."
*   `cv2.putText(frame, f"Class: {last_label}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)`:
    *   **Explication :** "Cette fonction d'OpenCV dessine le texte (`f"Class: {last_label}"`) sur l'image (`frame`) à la position (50,50), avec une police, une taille, une couleur (vert ici), une épaisseur et un type de ligne spécifiés."
*   `out_cls.write(frame)`:
    *   **Explication :** "L'image (`frame`), maintenant annotée avec le label de la classe, est écrite dans le fichier vidéo de classification de sortie."
*   `if frame_count % 1 == 0:` (dans ton code) / `process_every_n_frames_det = 1`
    *   **Explication :** "Ici, on traite chaque image (`frame_count % 1 == 0`). Si on voulait accélérer le traitement (au détriment de la fluidité des annotations), on pourrait augmenter ce nombre (ex: `frame_count % 5 == 0` pour traiter 1 image sur 5). Pour la démo, traiter chaque image est bien."

**7.2 (Détection) & 7.3 (Segmentation) Génération des vidéos :**
*   `model_det_for_video = YOLO(path_det_best)` ou `model_seg_for_video = YOLO(path_seg_best)`:
    *   **Explication :** "Pour la détection et la segmentation, on utilise directement la classe `YOLO` d'Ultralytics en lui passant le chemin vers les poids du meilleur modèle entraîné (`best.pt`) que YOLO a sauvegardé. Si ce fichier n'existe pas (par exemple si l'entraînement a été interrompu ou n'a pas été fait), on charge un modèle pré-entraîné par défaut (`yolo11n.pt` ou `yolo11n-seg.pt`) ou le modèle en mémoire s'il existe."
*   `model_det_for_video.to(device)`:
    *   **Explication :** "On s'assure que le modèle YOLO est chargé sur le bon device (CPU/GPU)."
*   `results = model_det_for_video.predict(source=frame, device=device, verbose=False, stream=False)` (similaire pour segmentation):
    *   **Explication :** "La méthode `.predict()` du modèle YOLO prend l'image brute (`frame`) en entrée.
        *   `source=frame`: l'image à analyser.
        *   `device=device`: spécifie si l'inférence doit se faire sur CPU ou GPU.
        *   `verbose=False`: pour éviter que YOLO n'imprime des informations détaillées pour chaque frame dans la console.
        *   `stream=False`: indique qu'on traite une image à la fois. Si c'était `True`, YOLO optimiserait pour un flux continu, mais ici on gère la boucle nous-mêmes.
        L'objet `results` contient toutes les informations sur les détections/segmentations."
*   `annotated_frame = results[0].plot()`:
    *   **Explication :** "C'est une fonctionnalité très pratique de la librairie Ultralytics YOLO. `results[0]` correspond aux prédictions pour la première (et unique dans ce cas) image du batch. La méthode `.plot()` prend ces résultats et dessine automatiquement les boîtes englobantes, les labels et les scores (pour la détection) ou les masques (pour la segmentation) directement sur une copie de l'image d'origine. Elle retourne cette image annotée."
*   `out_det.write(annotated_frame)` (similaire pour segmentation):
    *   **Explication :** "L'image, maintenant annotée par la méthode `.plot()` de YOLO, est écrite dans le fichier vidéo de sortie correspondant."

---

Prépare-toi à ce que le prof puisse te demander pourquoi tu as choisi un modèle spécifique pour la vidéo de classification si plusieurs ont été entraînés, ou des questions sur le compromis vitesse/précision du traitement vidéo.

---

### 2. Ton Speech pour la Slide 12

**(Transition depuis ta slide précédente - Traitement Vidéo)**

"Après avoir vu nos modèles en action sur la vidéo, il est essentiel de comprendre non seulement leur précision, mais aussi leur efficacité en termes de ressources. C'est l'objectif de notre cinquième tâche : **l'analyse des performances à l'aide de la librairie `pytorch-bench`.**"

**(En affichant la Slide 12 avec le tableau)**

"Pour cette analyse, nous avons utilisé `pytorch-bench`, une librairie développée ici à l'UMONS par Maxime Gloesener. Cet outil nous permet de mesurer des métriques clés comme le temps d'inférence – que nous exprimons en FPS (images par seconde sur GPU) –, l'utilisation maximale de la mémoire, mais aussi d'estimer l'empreinte environnementale via les émissions de CO2 et la consommation d'énergie, grâce à son intégration avec `codecarbon`."

"La méthode consiste à faire tourner chaque modèle en mode inférence sur des données aléatoires (des "dummy inputs") pendant un certain nombre d'itérations, après une phase de chauffe pour stabiliser les mesures."

"Voici un tableau récapitulatif des performances que nous avons obtenues lors d'une de nos séries de tests."
**(Tu commentes maintenant le tableau en pointant les chiffres clés. Adapte les commentaires suivants aux VRAIES valeurs de TON tableau. Je vais utiliser des exemples basés sur l'image que tu as envoyée, mais tu dois les vérifier.)**

"Si nous regardons les modèles de **classification** :"
*   "En termes de **précision de test**, le modèle **ViT** se distingue nettement avec environ **96,6%** d'accuracy, surpassant le CNN (ResNet18) à 88,5% et le ConvNeXt qui, dans cette exécution, a eu plus de difficultés avec environ 61,8%."
*   "Concernant le **nombre de paramètres**, le **CNN (ResNet18)** est le plus léger avec environ **11 millions de paramètres**, tandis que le ViT est le plus gourmand avec près de **86 millions**."
*   "Cela se reflète dans la **taille du modèle sur disque** : **42 Mo pour le CNN** contre **327 Mo pour le ViT**."
*   "Au niveau de la **vitesse d'inférence (FPS GPU)**, le **CNN** est le plus rapide avec environ **50 FPS**, suivi de près par ConvNeXt à 49 FPS et ViT à 45 FPS. C'est intéressant de noter que malgré sa taille, ViT reste compétitif en vitesse."
*   "Pour l'**utilisation mémoire maximale**, les trois modèles de classification se situent autour de **1500 Mo** pendant l'inférence sur notre machine de test."
*   "Enfin, pour l'**impact environnemental**, les différences sont minimes pour une seule inférence, mais le **ConvNeXt** semble être légèrement plus économe en énergie et en CO2 pour cette tâche."

"Passons maintenant aux modèles **YOLOv11 pour la détection et la segmentation** :"
*   "Pour la **détection**, notre YOLOv11 a une taille de modèle d'environ **6 Mo** et atteint **25 FPS**. La **segmentation**, avec une taille de **6,7 Mo**, est un peu plus rapide à **35 FPS**."
*   "On note que le modèle de **segmentation (YOLOv11-seg)** utilise significativement plus de mémoire maximale, près de **3000 Mo**, comparé aux **1500 Mo** pour la détection. C'est logique car la tâche de prédire des masques pixel par pixel est plus intensive."
*   "L'impact environnemental reste très faible pour ces modèles optimisés pour la vitesse."

**(Conclusion du commentaire du tableau)**https://github.com/mazar3/misc/tree/main
"Ce type de benchmarking est crucial. Il nous aide à choisir le bon modèle non seulement pour sa précision, mais aussi en fonction des contraintes de déploiement : si on a besoin de vitesse, si la mémoire est limitée, ou si l'efficacité énergétique est une priorité."

"Il est important de noter que ces chiffres peuvent varier légèrement en fonction de la machine de test et de la configuration exacte, mais ils donnent une bonne indication des compromis entre les différentes architectures."

**(Transition vers la conclusion générale du projet)**
"Voilà pour l'analyse des performances. Je vais maintenant laisser [Nom du prochain orateur, ou si c'est toi qui conclut, enchaîne directement] conclure notre présentation."

---

### 3. Explication du Code pour la Tâche 5 (Benchmarking)

Voici les parties importantes de ton code de benchmarking que le prof pourrait questionner :

*   `models_to_benchmark_config = { ... }`:
    *   **Explication :** "Ce dictionnaire définit les modèles que nous voulons benchmarker. Pour chaque modèle, on spécifie un `id` (un nom qu'on a défini pour le retrouver), la `input_shape` attendue par le modèle (batch_size, canaux, hauteur, largeur), et un `type` ('cls' pour classification, 'yolo' pour les modèles YOLO). Cette configuration centralise les informations nécessaires pour charger et tester chaque modèle."

*   `get_model_for_benchmarking(model_id_str, num_classes_cls_param, device_param)`:
    *   **Explication :** "Cette fonction est responsable de charger le modèle spécifié par `model_id_str`.
        *   Pour les modèles de classification, elle utilise les informations stockées dans `stored_model_metrics` (qui contient les chemins vers les poids sauvegardés, `.pth`). Elle recrée l'architecture du modèle (ResNet18, ViT, ConvNeXt) puis charge les poids.
        *   Pour les modèles YOLO, si un `best.pt` a été généré pendant l'entraînement, elle le charge. Sinon, elle se rabat sur le modèle YOLO en mémoire (s'il a été entraîné dans la session) ou sur un modèle pré-entraîné de base (`.pt` fourni par Ultralytics).
        *   Elle s'assure que le modèle est sur le bon `device` et en mode `eval()` pour l'inférence."

*   `dummy_input = torch.randn(config["input_shape"]).to(device)`:
    *   **Explication :** "Pour benchmarker l'inférence, on a besoin d'une donnée d'entrée. On crée ici un tenseur aléatoire (`torch.randn`) qui a la même forme (`config["input_shape"]`) que ce que le modèle attend. On l'envoie sur le `device` (GPU/CPU). Utiliser des données aléatoires est standard pour le benchmarking d'inférence pure car on se concentre sur la vitesse et la consommation du modèle, pas sur la qualité de sa prédiction sur une image spécifique."

*   `benchmark(model_to_bench, yolo_inference_func, dummy_input, benchmark_type="inference_time", n_warmup=5, n_loops=20, use_codecarbon=True, codecarbon_report_dir='benchmarking/codecarbon_reports')`:
    *   **Explication :** "C'est l'appel principal à la fonction `benchmark` de la librairie `pytorch-bench`.
        *   `model_to_bench`: Le modèle à tester.
        *   `dummy_input`: L'entrée factice. Pour les modèles YOLO, on passe une fonction `yolo_inference_func` car leur API `.predict()` est un peu différente d'un simple appel `model(input)`.
        *   `benchmark_type="inference_time"`: On spécifie qu'on veut mesurer le temps d'inférence.
        *   `n_warmup=5`: Le nombre d'inférences initiales qui ne sont pas comptées dans les mesures. C'est pour "chauffer" le GPU, s'assurer que tout est chargé en mémoire, et que les premières exécutions (souvent plus lentes) ne faussent pas les résultats.
        *   `n_loops=20`: Le nombre d'inférences réelles sur lesquelles les mesures de temps moyen, FPS, etc., seront calculées.
        *   `use_codecarbon=True`: Active le suivi de la consommation énergétique et des émissions de CO2 via la librairie `codecarbon`.
        *   `codecarbon_report_dir`: Le dossier où les rapports de `codecarbon` sont sauvegardés."

*   Remplissage du `performance_data_for_table`:
    *   **Explication :** "Après l'appel à `benchmark`, la fonction retourne un dictionnaire `bench_results` contenant les métriques (FPS, mémoire, CO2, énergie). On utilise ces valeurs pour remplir notre tableau `performance_data_for_table`. Pour les modèles de classification, on récupère aussi les métriques de précision, de perte, le nombre de paramètres et la taille du modèle qui avaient été stockées précédemment dans `stored_model_metrics` lors de leur entraînement."
    *   "Pour les modèles YOLO, le nombre de paramètres et la taille du modèle sont extraits directement du modèle chargé ou du fichier `.pt`."

*   `df_final_metrics_bench = pd.DataFrame(performance_data_for_table)` et affichage/sauvegarde :
    *   **Explication :** "Enfin, on convertit notre dictionnaire de données en un DataFrame Pandas pour un affichage tabulaire plus propre et pour pouvoir le sauvegarder en CSV."

**Concernant le fait que le tableau n'a pas été généré dans le notebook final :**
*   **Si le prof demande pourquoi :** "Nous avons rencontré quelques instabilités avec la librairie `pytorch-bench` dans l'environnement final du notebook au moment de la soumission, possiblement liées à des dépendances ou à la configuration spécifique de Colab/l'environnement d'exécution. Cependant, nous avions pu la faire fonctionner correctement lors de développements antérieurs. Le tableau que nous présentons ici est issu d'une de ces exécutions de référence et illustre bien la démarche et le type de résultats que l'outil permet d'obtenir. Le code pour le benchmarking est bien présent dans le notebook soumis."

Cette approche te permet de présenter le travail de manière honnête tout en montrant que tu as compris la tâche.

Vérifie bien les chiffres du tableau que tu vas présenter et adapte ton speech en conséquence. Prêt pour la conclusion ou une autre partie ?
