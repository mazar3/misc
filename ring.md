Okay, let's integrate the "Ring Tick" functionality.

This involves modifications in Flutter (UI, state, API calls) and creating a new AWS Lambda function.

**I. Flutter Implementation**

1.  **Dependencies & Assets:**
    *   Add the `audioplayers` package to your `pubspec.yaml`:
        ```yaml
        dependencies:
          # ... other dependencies
          audioplayers: ^6.0.0 # Or latest version
          # ...
        ```
    *   Run `flutter pub get`.
    *   Create an `assets/sounds/` directory in your project root.
    *   Place your sound files there (e.g., `1.mp3`, `2.mp3`, `3.mp3`). Make sure the filenames match the integers you'll use.
    *   Declare the assets folder in `pubspec.yaml`:
        ```yaml
        flutter:
          uses-material-design: true
          assets:
            - assets/
            - assets/sounds/ # Add this line
        ```

2.  **Constants (`lib/utils/constants.dart`):**
    *   Add the new Lambda URL to `ApiConfigURLs`. **Replace the placeholder URL with your actual Function URL once created.**
    *   Add relevant texts to `AppTexts`.

    ```dart
    // lib/utils/constants.dart

    import 'package:flutter/material.dart';
    import 'package:tick_app/utils/theme.dart'; // Pour AppTheme.errorColor, etc.
    import '../models/tick_model.dart'; // Pour TickStatus

    // ignore_for_file: constant_identifier_names

    /// Contient les configurations liées à l'API backend.
    class ApiConfigURLs {
      /// URL de la fonction Lambda pour enregistrer le token FCM de l'appareil.
      /// Retourne { "success": true } ou { "success": false, "error": "..." }
      static const String registerDeviceTokenFunctionUrl = 'https://jtslen53males67uck3ogfrekq0snomp.lambda-url.eu-north-1.on.aws/';

      /// URL de la fonction Lambda pour récupérer les Ticks de l'utilisateur.
      /// Doit retourner un JSON: { "success": true, "data": [...] } ou { "success": false, "error": "..." }
      static const String getMyTicksFunctionUrl = 'https://g2wgiv7n5vy4b4xdjbisbvgsny0asnki.lambda-url.eu-north-1.on.aws/';

      /// URL de la fonction Lambda pour demander la localisation d'un Tick.
      /// Doit retourner un JSON: { "success": true, "data": { ... } } ou { "success": false, "error": "..." }
      static const String requestLocationFunctionUrl = 'https://khrjd2jrqsyubgq3vad3o7z4li0eezcb.lambda-url.eu-north-1.on.aws/';

      /// URL de la fonction Lambda pour associer un nouveau Tick.
      /// Doit retourner un JSON: { "success": true, "data": { ... } } ou { "success": false, "error": "..." }
      static const String associateTickFunctionUrl = 'https://sxnfyrupf2navy3327vv7uvjvy0phdas.lambda-url.eu-north-1.on.aws/';

      /// URL de la fonction Lambda pour récupérer l'historique d'un Tick.
      /// Doit retourner un JSON: { "success": true, "data": [...] } ou { "success": false, "error": "..." }
      static const String getTickHistoryFunctionUrl = 'https://svhswmtmjywgi35alhomkjwhwm0yhngo.lambda-url.eu-north-1.on.aws/';

      /// URL de la fonction Lambda pour dissocier un Tick.
      /// Attend probablement un POST ou DELETE avec { "tickId": "..." } dans le body ou query parameter.
      static const String removeTickFunctionUrl = 'https://6s2qj5fs65hvbqxgyoqw67i6ye0qhtcf.lambda-url.eu-north-1.on.aws/';

      /// URL de la fonction Lambda pour mettre à jour les paramètres d'un Tick (nom, etc.).
      static const String updateTickSettingsFunctionUrl = 'https://tretbu4cr3vuwqkxge2nfai7me0cykak.lambda-url.eu-north-1.on.aws/';

      /// URL de la fonction Lambda pour faire sonner un Tick.
      /// Attend POST avec { "tickId": "...", "soundIndex": N }
      /// !! REMPLACER PAR L'URL RÉELLE !!
      static const String ringTickFunctionUrl = 'https://YOUR_NEW_RING_TICK_FUNCTION_URL.lambda-url.YOUR_REGION.on.aws/'; // <-- AJOUTER L'URL RÉELLE ICI

      /// URL de la fonction Lambda pour désactiver temporairement un Tick.
      static const String disableTickFunctionUrl = 'https://glsmok4bup253nhrzoapgradei0wzxpw.lambda-url.eu-north-1.on.aws/';

      /// URL de la fonction Lambda pour sortir le Tick du mode veille
      static const String reactivateTickFunctionUrl= 'https://ekfunfrncrbqkn276ob25spvnu0zhsvq.lambda-url.eu-north-1.on.aws/';
    }


    /// Messages d'erreur standardisés affichés à l'utilisateur.
    class ErrorMessages {
      // Erreurs réseau et serveur
      static const String connectionFailed = 'Impossible de se connecter au serveur. Vérifiez votre connexion internet.';
      static const String networkError = 'Erreur de réseau, veuillez réessayer.';
      static const String apiError = 'Erreur de communication avec le serveur.'; // Erreur générique API
      static const String unknownError = 'Une erreur inconnue est survenue.';

      // Erreurs d'authentification (certains sont gérés par Amplify)
      static const String invalidCredentials = 'Email ou mot de passe incorrect.';
      static const String emailInUse = 'Cette adresse email est déjà utilisée.';
      static const String userNotConfirmed = 'Veuillez confirmer votre compte avant de vous connecter.'; // Utilisé par AuthService

      // Erreurs Bluetooth et Association
      static const String deviceNotFound = 'Aucun appareil Tick compatible trouvé à proximité.';
      static const String bluetoothScanError = 'Erreur lors de la recherche Bluetooth.';
      static const String associationFailed = 'Échec de l\'association du Tick.';
      static const String bluetoothNotEnabled = 'Veuillez activer le Bluetooth.';
      static const String bleNotSupported = 'Le Bluetooth Low Energy n\'est pas supporté sur cet appareil.';

      // Erreurs de permissions
      static const String permissionDenied = 'Permission nécessaire refusée.';
      static const String permissionDeniedLocation = 'La permission de localisation est requise.'; // Plus spécifique
      static const String permissionDeniedBluetooth = 'Les permissions Bluetooth sont requises.'; // Plus spécifique
      static const String permissionDeniedForever = 'Permission refusée définitivement.'; // Non utilisé directement
      static const String permissionDeniedLocationExplain = 'Permission de localisation refusée définitivement. Veuillez l\'activer manuellement dans les paramètres de l\'application pour utiliser cette fonctionnalité.';
      static const String locationServiceDisabled = 'Le service de localisation doit être activé.'; // Ajouté
      static const String unauthorizedAccess = 'Accès refusé.';

      // Erreurs de formulaire
      static const String invalidInput = 'Veuillez vérifier les informations saisies.';
    }


    /// Textes utilisés dans l'interface utilisateur.
    class AppTexts {
      static const String appName = 'MyTick'; // Peut être mis à jour ici
      static const String welcome = 'Bienvenue';
      static const String tagline = 'Ne perdez plus jamais vos objets';
      static const String description = 'Localisez vos appareils Tick en temps réel et recevez des alertes instantanées.';

      // --- Boutons Communs ---
      static const String login = 'Se connecter';
      static const String register = 'Créer un compte';
      static const String continueWithoutAccount = 'Continuer sans compte'; // Optionnel
      static const String logout = 'Se déconnecter';
      static const String save = 'Enregistrer';
      static const String cancel = 'Annuler';
      static const String confirm = 'Confirmer';
      static const String next = 'Suivant';
      static const String back = 'Retour';
      static const String retry = 'Réessayer';
      static const String close = 'Fermer';
      static const String add = 'Ajouter';
      static const String delete = 'Supprimer';
      static const String edit = 'Modifier';
      static const String error = 'Erreur';
      static const String done = 'Terminé';

      // --- Authentification ---
      static const String forgotPassword = 'Mot de passe oublié ?';
      static const String noAccount = 'Pas encore de compte ?';
      static const String alreadyAccount = 'Déjà un compte ?';
      static const String passwordRecovery = 'Récupération de mot de passe';
      static const String passwordRecoveryInstructions = 'Entrez votre email pour recevoir les instructions de réinitialisation.';
      static const String sendRecoveryLink = 'Envoyer le lien';
      static const String recoveryLinkSent = 'Email de réinitialisation envoyé.';
      static const String confirmAccount = "Confirmer l'inscription";
      static const String confirmationCode = "Code de confirmation";
      static const String resendCode = "Renvoyer le code";
      static const String codeSentTo = "Un code de confirmation a été envoyé à :";
      static const String enterConfirmationCode = "Entrez le code à 6 chiffres";
      static const String checkEmailForCode = "Vérifiez votre email pour le code";
      static const String resetPassword = "Réinitialiser Mot de Passe";
      static const String newPassword = "Nouveau mot de passe";
      static const String enterResetCode = "Code de réinitialisation";

      // --- Formulaires ---
      static const String email = 'Adresse Email';
      static const String password = 'Mot de passe';
      static const String confirmPassword = 'Confirmer le mot de passe';
      static const String name = 'Nom complet';
      static const String firstName = 'Prénom'; // Si utilisé séparément
      static const String lastName = 'Nom'; // Si utilisé séparément
      static const String tickName = 'Nom du Tick';
      static const String tickNameHint = 'Ex: Clés, Sac à dos, Vélo...';

      // --- Validation Formulaires ---
      static const String requiredField = 'Ce champ est requis';
      static const String invalidEmail = 'Adresse email invalide';
      static const String passwordTooShort = 'Le mot de passe doit faire au moins 8 caractères'; // Maj selon politique Cognito
      static const String passwordsDoNotMatch = 'Les mots de passe ne correspondent pas';
      static const String invalidCode = 'Code invalide';

      // --- Écrans & Sections ---
      static const String myTicks = 'Mes Ticks';
      static const String addTick = 'Ajouter un Tick';
      static const String tickDetails = 'Détails du Tick';
      static const String noTicksAvailable = 'Aucun Tick associé';
      static const String addFirstTick = 'Ajoutez votre premier Tick pour commencer';
      static const String map = 'Carte';
      static const String settings = 'Paramètres';
      static const String profile = 'Profil';
      static const String history_map_page = 'Alertes';
      static const String history = 'Historique des alertes';
      static const String alerts = 'Alertes récentes';
      static const String information = 'Informations';
      static const String general = 'Général';
      static const String features = 'Fonctionnalités';
      static const String dangerZone = 'Zone de Danger';
      static const String appearance = 'Apparence';
      static const String notifications = 'Notifications';
      static const String security = 'Sécurité';

      // --- Association Tick (Refactorisée) ---
      static const String associateNewTick = 'Associer un nouveau Tick';
      static const String associationSteps = 'Suivez les étapes pour connecter votre appareil.';
      static const String step1_Naming = '1. Nommez votre Tick';
      static const String step2_Scanning = '2. Recherche du Tick';
      static const String step3_Sending = '3. Association';
      static const String step4_Done = '4. Terminé';
      static const String enableBluetoothPrompt = 'Le Bluetooth est nécessaire pour trouver votre Tick.';
      static const String enableBluetoothButton = 'Activer le Bluetooth';
      static const String bluetoothEnabled = 'Bluetooth activé';
      static const String activateTickPrompt = 'Assurez-vous que votre Tick est allumé et à proximité.';
      static const String searchTickButton = 'Rechercher mon Tick';
      static const String searchingTick = 'Recherche du Tick en cours...';
      static const String connectingTick = 'Connexion au Tick...';
      static const String tickFound = 'Tick trouvé !';
      static const String tickMacAddress = 'Adresse MAC';
      static const String associatingTick = 'Association en cours...';
      static const String associateTickButton = 'Associer ce Tick';
      static const String tickAssociatedSuccess = 'Tick associé avec succès !';
      static const String tickIdExtracted = 'ID du Tick détecté';

      // --- Détails Tick & Carte ---
      static const String currentStatus = 'État actuel';
      static const String lastPosition = 'Dernière position';
      static const String battery = 'Batterie';
      static const String lastUpdate = 'Dernière MàJ';
      static const String locate = 'Localiser';
      static const String ring = 'Faire sonner';
      static const String centerOnTick = 'Centrer sur le Tick';
      static const String centerOnMe = 'Centrer sur ma position';
      static const String fetchingLocation = 'Récupération de la position...';
      static const String ringingTick = 'Sonnerie en cours...'; // <-- MODIFIÉ
      static const String ringingTickCommandSent = 'Commande de sonnerie envoyée.'; // <-- AJOUTÉ
      static const String ringingTickError = 'Erreur lors de la sonnerie.'; // <-- AJOUTÉ
      static const String updating = 'Mise à jour...';
      static const String errorFetchingLocation = 'Erreur de localisation';
      static const String locationUpdated = 'Position mise à jour';
      static const String locationRequestSent = 'Demande de localisation envoyée...';
      static const String noLocationAvailable = 'Position non disponible';
      static const String tryToLocate = 'Tenter de localiser';

      // --- Paramètres Tick ---
      static const String tickSettings = 'Paramètres du Tick';
      static const String changeName = 'Changer le nom';
      static const String soundSettings = 'Sonnerie d\'alarme'; // <-- MODIFIÉ
      static const String selectAlarmSound = 'Sélectionner la sonnerie'; // <-- AJOUTÉ
      static const String preview = 'Écouter'; // <-- AJOUTÉ
      static const String noSoundSelected = 'Aucune sélectionnée'; // <-- AJOUTÉ
      static const String disableDevice = "Désactiver l'appareil";
      static const String reactivateDevice = "Réactiver l'appareil";
      static const String disableDuration = 'Durée de désactivation';
      static const String untilReactivation = "Jusqu'à réactivation";
      static const String unlinkDevice = 'Désassocier cet appareil';
      static const String unlinkDeviceConfirm = 'Êtes-vous sûr de vouloir désassocier ce Tick ? Cette action est irréversible.';
      static const String unlinkSuccess = 'Tick désassocié avec succès.';
      static const String featureComingSoon = 'Fonctionnalité bientôt disponible';

      // --- Autres ---
      static const String loading = 'Chargement...';
      static const String noHistoryAvailable = 'Aucun historique disponible';
      static const String loadingHistoryError = 'Erreur de chargement de l\'historique';
      static const String featureNotAvailableOnPlatform = 'Fonctionnalité non disponible sur cette plateforme.';
      static const String openSettings = 'Ouvrir les paramètres';
      static const String unknownUser = 'Utilisateur inconnu';
      static const String notConnected = 'Non connecté';
      static const String updateSuccess = 'Mise à jour réussie.';
      static const String updateError = 'Erreur lors de la mise à jour.';
    }


    /// Routes nommées pour la navigation GoRouter/Navigator 2.0.
    class Routes {
      static const String splash = '/splash'; // Écran initial
      static const String welcome = '/welcome'; // Écran si non connecté
      static const String login = '/login';
      static const String register = '/register';
      static const String confirmSignUp = '/confirm'; // Route pour la page de confirmation
      static const String passwordRecovery = '/password-recovery'; // Route pour page de réinitialisation
      static const String tickList = '/ticks'; // Écran principal post-connexion
      static const String tickMap = '/ticks/map'; // Devrait utiliser un paramètre ex: /ticks/map/{tickId}
      static const String addTick = '/ticks/add';
      static const String tickSettings = 'tick_settings_page'; // Devrait utiliser un paramètre ex: /ticks/settings/{tickId}
      static const String tickHistory = '/ticks/history'; // Devrait utiliser un paramètre ex: /ticks/history/{tickId}
      static const String profile = '/profile'; // Page profil utilisateur
      static const String settings = '/settings'; // Page paramètres généraux de l'app
      static const String changePassword = '/profile/change-password';
    }


    /// Configuration spécifique au Bluetooth.
    class BluetoothConfig {
      /// UUID du service principal exposé par les Ticks ESP32 en BLE.
      /// Doit correspondre exactement à celui défini dans le firmware ESP32.
      static const String tickServiceUuid = "7a8274fc-0723-44da-ad07-2293d5a5212a";

      /// Préfixe attendu pour le nom des appareils Tick diffusé en BLE.
      /// Le format attendu est "Tick-ID_UNIQUE_DU_TICK".
      static const String tickNamePrefix = "Tick-";

      /// Durée maximale (en secondes) pour un scan Bluetooth.
      static const int scanTimeoutSeconds = 15;
    }


    /// Couleurs spécifiques liées à la logique métier (statut, batterie).
    class AppColors {
      /// Retourne la couleur associée à un statut [TickStatus].
      /// Prend en compte le [BuildContext] pour adapter au thème clair/sombre si nécessaire.
      static Color getStatusColor(TickStatus status, BuildContext context) {
        final bool isDark = Theme.of(context).brightness == Brightness.dark;
        switch (status) {
          case TickStatus.active: return AppTheme.successColor;
          case TickStatus.inactive:
            return isDark ? AppTheme.textSecondaryColorDark : AppTheme.textSecondaryColorLight;
          case TickStatus.lowBattery: return AppTheme.warningColor;
          case TickStatus.moving: return AppTheme.primaryColor; // Couleur neutre/info pour mouvement simple
          case TickStatus.theftAlert: return AppTheme.errorColor; // Rouge pour alerte vol
          case TickStatus.disabled: return AppTheme.errorColor;
          case TickStatus.unknown:
          default: return Colors.grey; // Gris neutre pour inconnu
        }
      }

      /// Retourne la couleur associée à un niveau de batterie.
      static Color getBatteryColor(int? level) {
        // Utilise les couleurs sémantiques définies dans AppTheme
        if (level == null) return Colors.grey; // Gris si inconnu
        if (level > 60) return AppTheme.successColor; // Vert si > 60%
        if (level > 20) return AppTheme.warningColor; // Orange si > 20%
        return AppTheme.errorColor; // Rouge si <= 20%
      }
    }

    /// Durées standard utilisées dans l'application (animations, timeouts).
    class AppDurations {
      static const Duration shortFade = Duration(milliseconds: 200);
      static const Duration mediumFade = Duration(milliseconds: 400);
      static const Duration longFade = Duration(milliseconds: 600);
      static const Duration snackbarDuration = Duration(seconds: 4);
      static const Duration apiTimeout = Duration(seconds: 20);
      static const Duration shortDelay = Duration(milliseconds: 500);
      static const Duration mediumDelay = Duration(seconds: 1);
      static const Duration longDelay = Duration(seconds: 3);
    }

    class MqttConfig {
       static const String awsIotEndpoint = 'apflqqfc1rv3u-ats.iot.eu-north-1.amazonaws.com';
       static const String awsRegion = 'eu-north-1';
       static const int port = 443;
       static String getClientId(String userId) => 'flutter_app_${userId}_${DateTime.now().millisecondsSinceEpoch}'; // Assure l'unicité
    }

    /// --- AJOUT : Liste des Sons ---
    /// Mappage entre l'index (utilisé pour la valeur MQTT) et le nom/chemin du fichier.
    class AppSounds {
        static const Map<int, String> alarmSounds = {
            1: 'Doux',
            2: 'Classique',
            3: 'Urgence',
            4: 'Bip répétitif',
            // Ajoutez d'autres sons ici si nécessaire
        };

        /// Retourne le chemin d'accès au fichier son dans les assets.
        /// Ex: `assets/sounds/1.mp3`
        static String getSoundPath(int index) {
            // Adapter l'extension si vous utilisez .wav
            return 'sounds/$index.mp3'; // Chemin relatif DANS assets/
        }
    }
    /// --- FIN AJOUT ---
    ```

3.  **Model (`lib/models/tick_model.dart`):**
    *   Add `selectedAlarmSoundIndex` field.
    *   Update `fromJson`, `toJson`, `copyWith`, `==`, and `hashCode`.

    ```dart
    // lib/models/tick_model.dart
    import 'package:flutter/foundation.dart' show immutable;
    import 'package:intl/intl.dart'; // Pour formater les dates/heures

    /// Représente les différents états possibles d'un Tick.
    enum TickStatus {
      active,       /// Fonctionnement normal, communication récente.
      inactive,     /// Pas de communication récente.
      lowBattery,   /// Niveau de batterie bas détecté.
      moving,       /// Mouvement détecté (sans changement de position majeur récent).
      theftAlert,   /// Déplacement significatif détecté (alerte vol).
      unknown,      /// Statut initial ou inconnu.
      disabled,     /// Appareil désactivé
    }

    /// Représente un appareil Tick GPS.
    @immutable // Marque la classe comme immuable
    class Tick {
      /// ID unique du Tick (probablement l'ID extrait du nom BLE).
      final String id;

      /// Nom personnalisé donné par l'utilisateur.
      final String name;

      /// Dernière latitude connue (peut être null).
      final double? latitude;

      /// Dernière longitude connue (peut être null).
      final double? longitude;

      /// Dernier niveau de batterie connu en pourcentage (peut être null).
      final int? batteryLevel;

      /// Date et heure de la dernière mise à jour reçue du Tick (peut être null).
      final DateTime? lastUpdate;

      /// Statut actuel déduit du Tick.
      final TickStatus status;

      /// ID de l'utilisateur propriétaire du Tick.
      final String ownerId;

      /// Adresse MAC Bluetooth du Tick (peut être null ou non pertinente après association).
      final String? macAddress;

      /// Index du son d'alarme sélectionné (peut être null si non défini).
      final int? selectedAlarmSoundIndex; // <-- AJOUTÉ

      /// Constructeur principal.
      const Tick({
        required this.id,
        required this.name,
        required this.ownerId, // ownerId est requis
        this.latitude,
        this.longitude,
        this.batteryLevel,
        this.lastUpdate,
        this.status = TickStatus.unknown,
        this.macAddress,
        this.selectedAlarmSoundIndex, // <-- AJOUTÉ
      });

      /// Crée une instance `Tick` à partir d'une Map JSON (venant de l'API/DB).
      factory Tick.fromJson(Map<String, dynamic> json) {
        // Extraction prudente des données, gérant plusieurs noms de clés possibles
        final String tickId = json['id'] as String? ?? json['tickId'] as String? ?? json['tick_id'] as String? ?? '';
        final String tickName = json['name'] ?? json['tickName'] ?? 'Tick Sans Nom';
        final String userId = json['ownerId'] ?? json['userId'] ?? '';
        final String? mac = json['macAddress'] as String?;

        double? lat;
        if (json['latitude'] is num) lat = (json['latitude'] as num).toDouble();
        if (lat == null && json['lastPosition']?['lat'] is num) lat = (json['lastPosition']['lat'] as num).toDouble();
        if (lat == null && json['lat'] is num) lat = (json['lat'] as num).toDouble();


        double? lng;
        if (json['longitude'] is num) lng = (json['longitude'] as num).toDouble();
        if (lng == null && json['lastPosition']?['lng'] is num) lng = (json['lastPosition']['lng'] as num).toDouble();
        if (lng == null && json['lng'] is num) lng = (json['lng'] as num).toDouble();

        int? bat;
        if (json['batteryLevel'] is num) bat = (json['batteryLevel'] as num).toInt();
        if (bat == null && json['battery'] is num) bat = (json['battery'] as num).toInt();
        if (bat == null && json['bat'] is num) bat = (json['bat'] as num).toInt();

        DateTime? lastUpdt;
        if (json['lastUpdate'] is String) lastUpdt = DateTime.tryParse(json['lastUpdate']);
        if (lastUpdt == null && json['lastUpdated'] is String) lastUpdt = DateTime.tryParse(json['lastUpdated']);
        // Essayer de parser depuis un timestamp epoch (millisecondes ou secondes)
        if (lastUpdt == null && json['timestamp'] is num) {
           final tsNum = json['timestamp'] as num;
           if (tsNum > 1000000000000) { // Probablement millisecondes
              lastUpdt = DateTime.fromMillisecondsSinceEpoch(tsNum.toInt());
           } else { // Probablement secondes
              lastUpdt = DateTime.fromMillisecondsSinceEpoch(tsNum.toInt() * 1000);
           }
        }

        // --- AJOUT : Parsing de selectedAlarmSoundIndex ---
        int? alarmIndex;
        // Vérifie si la clé existe et si la valeur est un nombre
        if (json['selectedAlarmSoundIndex'] is num) {
            alarmIndex = (json['selectedAlarmSoundIndex'] as num).toInt();
        } else if (json['selectedAlarmSoundIndex'] is String) {
            // Essayer de parser depuis une chaîne si nécessaire
            alarmIndex = int.tryParse(json['selectedAlarmSoundIndex'] as String);
        }
        // ---------------------------------------------

        // Déterminer le statut basé sur les données disponibles (logique simplifiée)
        // La logique de statut pourrait être plus complexe côté backend
        TickStatus currentStatus = TickStatus.unknown;
        String? lastEventType = json['lastEventType'] as String?; // Lire un potentiel dernier type d'événement

        if (lastEventType == 'theft_alert') {
           currentStatus = TickStatus.theftAlert;
        } else if (lastEventType == 'movement_alert') {
           currentStatus = TickStatus.moving;
        } else if (bat != null && bat <= 20) {
           currentStatus = TickStatus.lowBattery;
        } else if (lastUpdt != null) {
           // Considérer actif si communication récente (ex: moins de 24h)
           if (DateTime.now().difference(lastUpdt).inHours < 24) {
              currentStatus = TickStatus.active;
           } else {
              currentStatus = TickStatus.inactive; // Communication ancienne
           }
        }
        // Lire un champ 'status' explicite envoyé par l'API peut surcharger la logique ci-dessus
         if (json['status'] is String) {
            currentStatus = TickStatus.values.firstWhere(
               (e) => e.name == json['status'],
               orElse: () => currentStatus // Garder statut déduit si inconnu
            );
         }

        return Tick(
          id: tickId,
          name: tickName,
          latitude: lat,
          longitude: lng,
          batteryLevel: bat,
          lastUpdate: lastUpdt,
          status: currentStatus,
          ownerId: userId,
          macAddress: mac,
          selectedAlarmSoundIndex: alarmIndex, // <-- AJOUTÉ
        );
      }

      /// Convertit l'objet `Tick` en une Map JSON.
      /// Utile si l'objet doit être envoyé à une API.
      Map<String, dynamic> toJson() {
        return {
          'id': id,
          'name': name,
          'latitude': latitude,
          'longitude': longitude,
          'batteryLevel': batteryLevel,
          'lastUpdate': lastUpdate?.toIso8601String(), // Format ISO 8601 standard
          'status': status.name, // Envoyer le nom de l'enum (ex: 'active')
          'ownerId': ownerId,
          'macAddress': macAddress,
          'selectedAlarmSoundIndex': selectedAlarmSoundIndex, // <-- AJOUTÉ
        };
      }

      /// Crée une copie de l'objet `Tick` avec certaines valeurs modifiées.
      /// Gère explicitement la possibilité de passer `null` pour écraser une valeur existante.
      Tick copyWith({
        String? id,
        String? name,
        Object? latitude = _sentinel, // Utilise Object? et sentinel pour gérer null explicitement
        Object? longitude = _sentinel,
        Object? batteryLevel = _sentinel,
        Object? lastUpdate = _sentinel,
        TickStatus? status,
        String? ownerId,
        Object? macAddress = _sentinel,
        Object? selectedAlarmSoundIndex = _sentinel, // <-- AJOUTÉ
      }) {
        return Tick(
          id: id ?? this.id,
          name: name ?? this.name,
          ownerId: ownerId ?? this.ownerId,
          // Si la valeur sentinelle est passée, on garde l'ancienne valeur.
          // Sinon, on prend la nouvelle valeur (qui peut être null si Object? est null).
          latitude: latitude == _sentinel ? this.latitude : latitude as double?,
          longitude: longitude == _sentinel ? this.longitude : longitude as double?,
          batteryLevel: batteryLevel == _sentinel ? this.batteryLevel : batteryLevel as int?,
          lastUpdate: lastUpdate == _sentinel ? this.lastUpdate : lastUpdate as DateTime?,
          status: status ?? this.status,
          macAddress: macAddress == _sentinel ? this.macAddress : macAddress as String?,
          selectedAlarmSoundIndex: selectedAlarmSoundIndex == _sentinel // <-- AJOUTÉ
              ? this.selectedAlarmSoundIndex
              : selectedAlarmSoundIndex as int?, // <-- AJOUTÉ
        );
      }

      /// Retourne une chaîne de caractères formatée représentant la dernière mise à jour.
      /// Utilise `intl` pour une meilleure localisation et formatage relatif.
      String get formattedLastUpdate {
        if (lastUpdate == null) return 'jamais';

        // Récupère le DateTime (qui est stocké en UTC)
        final DateTime utcTimestamp = lastUpdate!;
        // Convertit en local pour les comparaisons et le formatage final de l'heure
        final DateTime localTimestamp = utcTimestamp.toLocal();
        final now = DateTime.now(); // Est déjà local
        final difference = now.difference(localTimestamp); // Compare avec le temps local

        // Formats (utilisent la locale 'fr_FR' définie dans main.dart)
        final timeFormat = DateFormat('HH:mm', 'fr_FR');
        final dateTimeFormat = DateFormat('le dd/MM/yy à HH:mm', 'fr_FR');
        final weekdayFormat = DateFormat('EEEE', 'fr_FR');

        // Logique pour "à l'instant", "il y a X min", "auj.", "hier", etc.
        // basée sur la différence de temps calculée avec l'heure LOCALE.
        if (difference.inSeconds < 60) {
          return "à l'instant";
        } else if (difference.inMinutes < 60) {
          return "il y a ${difference.inMinutes} min";
        } else if (_isSameDay(localTimestamp, now)) { // Utilise _isSameDay (qui doit comparer en local)
          return "auj. à ${timeFormat.format(localTimestamp)}";
        } else if (_isSameDay(localTimestamp, now.subtract(const Duration(days: 1)))) {
          return "hier à ${timeFormat.format(localTimestamp)}";
        } else if (difference.inDays < 7) {
          return "${weekdayFormat.format(localTimestamp)} à ${timeFormat.format(localTimestamp)}";
        } else {
          return dateTimeFormat.format(localTimestamp);
        }
      }

      // Helper (doit être dans la classe Tick ou accessible) pour comparer en local
      bool _isSameDay(DateTime dt1, DateTime dt2) {
        final localDt1 = dt1.toLocal();
        final localDt2 = dt2.toLocal();
        return localDt1.year == localDt2.year &&
            localDt1.month == localDt2.month &&
            localDt1.day == localDt2.day;
      }

      /// Retourne une description textuelle du statut actuel.
      String get statusDescription {
        switch (status) {
          case TickStatus.active: return 'Actif';
          case TickStatus.inactive: return 'Inactif';
          case TickStatus.lowBattery: return 'Batterie faible';
          case TickStatus.moving: return 'Mouvement détecté';
          case TickStatus.theftAlert: return 'Déplacement détecté';
          case TickStatus.disabled: return 'Désactivé';
          case TickStatus.unknown:
          default: return 'Inconnu';
        }
      }

      // Pour faciliter la comparaison et l'utilisation dans les Sets/Maps
      @override
      bool operator ==(Object other) =>
          identical(this, other) ||
          other is Tick &&
              runtimeType == other.runtimeType &&
              id == other.id &&
              name == other.name &&
              latitude == other.latitude &&
              longitude == other.longitude &&
              batteryLevel == other.batteryLevel &&
              lastUpdate == other.lastUpdate &&
              status == other.status &&
              ownerId == other.ownerId &&
              macAddress == other.macAddress &&
              selectedAlarmSoundIndex == other.selectedAlarmSoundIndex; // <-- AJOUTÉ

      @override
      int get hashCode =>
          id.hashCode ^
          name.hashCode ^
          latitude.hashCode ^
          longitude.hashCode ^
          batteryLevel.hashCode ^
          lastUpdate.hashCode ^
          status.hashCode ^
          ownerId.hashCode ^
          macAddress.hashCode ^
          selectedAlarmSoundIndex.hashCode; // <-- AJOUTÉ

       @override
       String toString() {
          return 'Tick(id: $id, name: "$name", status: $status, lat: $latitude, lng: $longitude, bat: $batteryLevel, lastUpdate: $lastUpdate, ownerId: $ownerId, alarmIdx: $selectedAlarmSoundIndex)'; // <-- AJOUTÉ
       }
    }

    /// Valeur sentinelle interne utilisée par `copyWith` pour différencier
    /// une valeur non fournie (garder l'ancienne) d'une valeur explicitement `null`.
    const Object _sentinel = Object();
    ```

4.  **Tick Service (`lib/services/tick_service.dart`):**
    *   Update `updateTickSettings` signature and logic.
    *   Update `ringTick` logic to use the new Lambda and include the sound index.

    ```dart
    // lib/services/tick_service.dart
    import 'dart:async'; // Pour StreamSubscription si écoute MQTT directe
    import 'package:flutter/foundation.dart'; // Pour ChangeNotifier

    import '../models/tick_model.dart';
    import 'api_service.dart'; // Service pour appeler les Lambdas
    import 'auth_service.dart'; // Pour vérifier l'authentification et obtenir l'ID user
    import '../utils/constants.dart'; // Pour URLs et ErrorMessages
    // import 'package:collection/collection.dart'; // Pour firstWhereOrNull (alternative)

    // ignore_for_file: avoid_print

    /// Service gérant les données et les opérations liées aux appareils Tick.
    /// Interagit avec [ApiService] pour communiquer avec le backend (Lambdas).
    class TickService with ChangeNotifier {
      final ApiService _apiService;
      AuthService _authService; // Référence au service d'authentification

      List<Tick> _ticks = []; // Liste locale des Ticks de l'utilisateur
      bool _isLoading = false; // Indicateur de chargement global pour le service
      String? _error; // Dernier message d'erreur

      // Getters publics
      List<Tick> get ticks => List.unmodifiable(_ticks); // Copie immuable pour l'UI
      bool get isLoading => _isLoading;
      String? get error => _error;

      /// Constructeur: Initialise avec les services requis et configure les listeners.
      TickService(this._apiService, this._authService) {
        print("TickService: Initializing...");
        // Écouter les changements d'état d'authentification
        _authService.addListener(_handleAuthChange);
        // Charger les ticks initiaux si l'utilisateur est déjà connecté au démarrage
        if (_authService.isAuthenticated) {
          fetchTicks();
        }
      }

      /// Met à jour la référence [AuthService] (utilisé par ChangeNotifierProxyProvider).
      void updateAuth(AuthService authService) {
        // Éviter mise à jour inutile si la référence est la même
        if (_authService == authService) return;
        print("TickService: Updating AuthService reference.");
        // Supprimer l'ancien listener avant d'ajouter le nouveau
        _authService.removeListener(_handleAuthChange);
        _authService = authService;
        _authService.addListener(_handleAuthChange);
        // Gérer l'état actuel après la mise à jour de la référence
        _handleAuthChange();
      }

      /// Réagit aux changements d'état d'authentification.
      void _handleAuthChange() {
        print("TickService: Auth state changed. User Authenticated: ${_authService.isAuthenticated}");
        if (_authService.isAuthenticated) {
          // Si l'utilisateur est connecté et que la liste est vide (ou après reconnexion)
          if (_ticks.isEmpty) {
            fetchTicks(); // Charger les Ticks
          }
        } else {
          // Si l'utilisateur est déconnecté, effacer les données locales
          if (_ticks.isNotEmpty || _error != null || _isLoading) {
            print("TickService: Clearing local tick data due to logout.");
            _ticks = [];
            _error = null;
            _isLoading = false; // Stopper tout chargement en cours
            notifyListeners(); // Notifier l'UI de l'effacement
          }
        }
      }

      // --- Méthodes Publiques pour l'UI ---

      /// Récupère la liste des Ticks de l'utilisateur depuis le backend.
      Future<void> fetchTicks() async {
        // Vérifier si l'utilisateur est connecté et si une opération n'est pas déjà en cours
        if (!_checkAuthAndLoading()) return;

        _setLoading(true);
        _clearError(); // Effacer l'erreur précédente

        print("TickService: Fetching ticks from URL: ${ApiConfigURLs.getMyTicksFunctionUrl}");

        try {
          // Appel API via ApiService
          final response = await _apiService.get(ApiConfigURLs.getMyTicksFunctionUrl);

          if (response['success']) {
            final List<dynamic>? tickDataList = response['data'] as List<dynamic>?;
            print("TickService: RAW Tick Data Received: $tickDataList");
            if (tickDataList != null) {
              // Parser les données JSON en objets Tick
              _ticks = tickDataList.map((data) {
                try {
                  return Tick.fromJson(data as Map<String, dynamic>);
                } catch (e) {
                  print("TickService: Error parsing Tick JSON: $e \nData: $data");
                  return null; // Ignorer les données mal formées
                }
              }).whereType<Tick>().toList(); // Filtrer les nulls
              print("TickService: Ticks fetched successfully: ${_ticks.length} ticks loaded.");
            } else {
               // La clé 'data' manquait ou était null, mais succès=true ?
               print("TickService: Ticks fetch API success but data is null or not a list.");
               _ticks = []; // Assurer que la liste est vide
            }
          } else {
            // L'API a renvoyé une erreur
            _setError(response['error'] ?? ErrorMessages.apiError);
            print("TickService: Error fetching ticks from Lambda: $_error");
          }
        } catch (e) {
          // Erreur de connexion ou autre exception
          print("TickService: Exception fetching ticks: $e");
          _setError(ErrorMessages.connectionFailed);
        } finally {
          _setLoading(false); // Assurer que le chargement s'arrête
        }
      }

      /// Associe un nouveau Tick au compte de l'utilisateur.
      /// [nickname]: Nom donné par l'utilisateur.
      /// [extractedTickId]: ID unique du Tick extrait lors du scan BLE.
      Future<bool> associateTick(String nickname, String extractedTickId) async {
        if (!_checkAuthAndLoading()) return false;

        _setLoading(true);
        _clearError();

        try {
          // Corps de la requête pour la Lambda d'association
          final body = {
            'tickName': nickname,
            'tickId': extractedTickId, // ID unique du matériel
            // L'ID utilisateur est extrait du token JWT par la Lambda
          };

          print("TickService: Associating tick via URL: ${ApiConfigURLs.associateTickFunctionUrl}");
          final response = await _apiService.post(ApiConfigURLs.associateTickFunctionUrl, body);

          if (response['success']) {
            print("TickService: Tick associated successfully via Lambda. Response: ${response['data']}");

            // 1. Parser les données du nouveau Tick retournées par la Lambda
            final dynamic newTickData = response['data'];
            if (newTickData is Map<String, dynamic>) {
              try {
                final newTick = Tick.fromJson(newTickData);
                // 2. Ajouter le nouveau Tick à la liste locale
                _ticks.add(newTick);
                // Optionnel: Trier la liste si nécessaire
                 _ticks.sort((a, b) => a.name.compareTo(b.name));
                print("TickService: New tick '${newTick.name}' added locally.");
                _setLoading(false); // Mettre fin au chargement
                notifyListeners(); // Notifier l'UI de l'ajout
                return true; // Succès !
              } catch (e) {
                 print("TickService: Error parsing new tick data after association: $e \nData: $newTickData");
                 _setError("Erreur lors de la lecture des données du nouveau Tick.");
                 _setLoading(false);
                 return false; // Échec du parsing des données retournées
              }
            } else {
               print("TickService: Association API success but returned data is not a valid Tick map. Response: $newTickData");
               // Optionnel : On pourrait quand même faire un fetchTicks ici en fallback,
               // mais c'est mieux si l'API retourne les données correctes.
               _setError("Réponse invalide du serveur après association.");
               _setLoading(false);
               return false;
            }

          } else {
            _setError(response['error'] ?? ErrorMessages.associationFailed);
            print("TickService: Failed to associate tick. API Error: $_error");
            _setLoading(false);
            return false;
          }
        } catch (e) {
          print("TickService: Exception associating tick: $e");
          _setError(ErrorMessages.connectionFailed);
          _setLoading(false);
          return false;
        }
      }

      Future<bool> temporaryDisableTick(String tickId, Duration duration) async {
        if (!_checkAuthAndLoading()) return false;
        if (duration.inMinutes <= 0) {
          _setError("La durée de désactivation doit être positive.");
          notifyListeners();
          return false;
        }

        _setLoading(true);
        _clearError();

        try {
          // Vérifier si l'URL est configurée
          if (ApiConfigURLs.disableTickFunctionUrl.isEmpty) {
            throw Exception("Temporary Disable URL not configured in constants.dart");
          }

          // Préparer le corps de la requête attendu par la Lambda
          final body = {
            'tickId': tickId,
            'duration': duration.inMinutes.toString(), // Envoyer la durée en minutes comme chaîne
          };
          print("TickService: Calling temporaryDisable Function URL: ${ApiConfigURLs.disableTickFunctionUrl}");

          // Appeler ApiService.post
          final response = await _apiService.post(ApiConfigURLs.disableTickFunctionUrl, body);

          if (response['success']) {
            print("TickService: Temporary disable command sent successfully for $tickId. Lambda response: ${response['data']}");
            // Pas de mise à jour locale immédiate du statut nécessaire ici,
            // car la confirmation viendra (ou pas) de l'ESP32 plus tard.
            _setLoading(false);
            return true; // Indique que la *demande* a été envoyée
          } else {
            _setError((response['error'] as String?) ?? "Erreur lors de la demande de désactivation");
            print("TickService: Temporary disable failed. API Error: $_error");
            _setLoading(false);
            return false;
          }
        } catch (e) {
          print("TickService: Exception during temporary disable request for $tickId: $e");
          _setError(e is Exception ? e.toString() : ErrorMessages.connectionFailed);
          _setLoading(false);
          return false;
        }
      }

      Future<bool> disableTickPermanently(String tickId) async {
        if (!_checkAuthAndLoading()) return false;
        _setLoading(true);
        _clearError();

        try {
          if (ApiConfigURLs.disableTickFunctionUrl.isEmpty) { // Réutilise l'URL disable/temp
            throw Exception("Temporary Disable URL not configured in constants.dart");
          }
          // Body SANS duration pour indiquer permanent (la Lambda doit gérer ça)
          final body = {'tickId': tickId};
          print("TickService: Calling temporaryDisable Function URL for PERMANENT disable: ${ApiConfigURLs.disableTickFunctionUrl}");
          final response = await _apiService.post(ApiConfigURLs.disableTickFunctionUrl, body);

          if (response['success']) {
            print("TickService: Permanent disable command sent successfully for $tickId.");
            // Optionnel : Mettre à jour l'état local immédiatement ?
            //updateTickDataLocally(getTickById(tickId)!.copyWith(status: TickStatus.disabled));
            _setLoading(false);
            return true;
          } else {
            _setError((response['error'] as String?) ?? "Erreur lors de la désactivation");
            _setLoading(false);
            return false;
          }
        } catch (e) {
          print("TickService: Exception during permanent disable request for $tickId: $e");
          _setError(e is Exception ? e.toString() : ErrorMessages.connectionFailed);
          _setLoading(false);
          return false;
        }
      }

      /// Demande la réactivation d'un Tick.
      Future<bool> reactivateTick(String tickId) async {
        if (!_checkAuthAndLoading()) return false;
        _setLoading(true);
        _clearError();

        try {
          if (ApiConfigURLs.reactivateTickFunctionUrl.isEmpty) {
            throw Exception("Reactivate Tick URL not configured in constants.dart");
          }
          final body = {'tickId': tickId};
          print("TickService: Calling reactivateTick Function URL: ${ApiConfigURLs.reactivateTickFunctionUrl}");
          final response = await _apiService.post(ApiConfigURLs.reactivateTickFunctionUrl, body);

          if (response['success']) {
            print("TickService: Reactivate command sent successfully for $tickId.");
            // Optionnel : Mettre à jour l'état local immédiatement ?
            //updateTickDataLocally(getTickById(tickId)!.copyWith(status: TickStatus.active)); // Ou 'inactive' si on attend confirmation
            _setLoading(false);
            return true;
          } else {
            _setError((response['error'] as String?) ?? "Erreur lors de la réactivation");
            _setLoading(false);
            return false;
          }
        } catch (e) {
          print("TickService: Exception during reactivate request for $tickId: $e");
          _setError(e is Exception ? e.toString() : ErrorMessages.connectionFailed);
          _setLoading(false);
          return false;
        }
      }



      /// Récupère l'historique d'un Tick spécifique.
      /// Retourne la réponse brute de l'API pour que l'UI (HistoryPage) la parse.
      Future<Map<String, dynamic>> getTickHistory(String tickId) async {
        if (!_authService.isAuthenticated) {
          return {'success': false, 'error': "Utilisateur non authentifié"};
        }
        // Pas de gestion de _isLoading ici, la page d'historique a son propre état

        // Construire l'URL avec le paramètre tickId
        try {
          if (ApiConfigURLs.getTickHistoryFunctionUrl.isEmpty) throw Exception("Get Tick History URL not configured");
           final urlWithParam = Uri.parse(ApiConfigURLs.getTickHistoryFunctionUrl).replace(
              queryParameters: {'tickId': tickId}
           ).toString();
           print("TickService: Getting history for $tickId from URL: $urlWithParam");
           // Appeler l'API
           final response = await _apiService.get(urlWithParam);
           return response; // Retourner la réponse brute (contient success/data ou success/error)
        } catch (e) {
            print("TickService: Exception getting tick history: $e");
            return {'success': false, 'error': e is Exception ? e.toString() : ErrorMessages.connectionFailed};
        }
      }


      /// Demande une mise à jour de localisation pour un Tick spécifique.
      /// Retourne `true` si la *demande* a été envoyée avec succès au backend.
      Future<bool> requestTickLocation(String tickId) async {
        if (!_checkAuthAndLoading()) return false;
        // Utiliser un indicateur de chargement spécifique à l'action serait mieux,
        // mais pour l'instant on utilise le global.
        _setLoading(true);
        _clearError();

        try {
          if (ApiConfigURLs.requestLocationFunctionUrl.isEmpty) throw Exception("Request Location URL not configured");
          final body = {'tickId': tickId};
          print("TickService: Requesting location for $tickId via URL: ${ApiConfigURLs.requestLocationFunctionUrl}");
          final response = await _apiService.post(ApiConfigURLs.requestLocationFunctionUrl, body);

          if (response['success']) {
            print("TickService: Location request sent successfully for $tickId. Lambda response: ${response['data']}");
            // La position réelle sera mise à jour via un autre mécanisme (MQTT -> Backend -> App ?)
            _setLoading(false);
            return true;
          } else {
            _setError(response['error'] ?? "Erreur lors de la demande de localisation");
            print("TickService: Failed location request for $tickId. API Error: $_error");
            _setLoading(false);
            return false;
          }
        } catch (e) {
          print("TickService: Exception requesting location for $tickId: $e");
          _setError(e is Exception ? e.toString() : ErrorMessages.connectionFailed);
          _setLoading(false);
          return false;
        }
      }

      /// Demande à faire sonner un Tick avec le son sélectionné.
      Future<bool> ringTick(String tickId) async {
        if (!_checkAuthAndLoading()) return false;

        // Récupérer le Tick pour obtenir le son sélectionné
        final tick = getTickById(tickId);
        if (tick == null) {
          _setError("Tick non trouvé pour la sonnerie.");
          notifyListeners();
          return false;
        }
        // Utiliser le son sélectionné ou un son par défaut (ex: 1)
        final int soundIndex = tick.selectedAlarmSoundIndex ?? 1;

        _setLoading(true);
        _clearError();

        try {
           if (ApiConfigURLs.ringTickFunctionUrl.isEmpty || ApiConfigURLs.ringTickFunctionUrl.startsWith('YOUR_')) {
               throw Exception("Ring Tick URL not configured");
           }
           // Corps de la requête incluant l'index du son
           final body = {
               'tickId': tickId,
               'soundIndex': soundIndex, // Utilise la valeur récupérée ou par défaut
           };
           print("TickService: Ringing tick $tickId with sound $soundIndex via URL: ${ApiConfigURLs.ringTickFunctionUrl}");

           // Appel API Réel
           final response = await _apiService.post(ApiConfigURLs.ringTickFunctionUrl, body);

           if ((response['success'] as bool? ?? false)) {
              print("TickService: Ring command sent for $tickId (Sound $soundIndex). Response: ${response['data']}");
              _setLoading(false);
              // Message succès affiché par l'UI (MapPage) si besoin
              return true;
           } else {
              _setError((response['error'] as String?) ?? AppTexts.ringingTickError);
              _setLoading(false);
              return false;
           }
        } catch (e) {
          print("TickService: Exception ringing tick $tickId: $e");
          _setError(e is Exception ? e.toString() : AppTexts.ringingTickError);
          _setLoading(false);
          return false;
        }
      }

      /// Met à jour les paramètres d'un Tick (nom et/ou index du son d'alarme).
      Future<bool> updateTickSettings(
          String tickId, {
          String? name, // Nom optionnel
          int? alarmSoundIndex, // Index du son optionnel
      }) async {
        if (!_checkAuthAndLoading()) return false;
        // Vérifier qu'au moins un paramètre est fourni
        if (name == null && alarmSoundIndex == null) {
            print("TickService: updateTickSettings called without any parameters to update.");
            return true; // Aucune mise à jour nécessaire, considérer comme un succès
        }
        // Valider le nom si fourni
        if (name != null && name.trim().isEmpty) {
            _setError("Le nom ne peut pas être vide.");
            notifyListeners();
            return false;
        }
        // Valider l'index du son si fourni (doit exister dans AppSounds)
        if (alarmSoundIndex != null && !AppSounds.alarmSounds.containsKey(alarmSoundIndex)) {
             _setError("Son sélectionné invalide.");
             notifyListeners();
             return false;
        }

        _setLoading(true);
        _clearError();

        try {
          // Vérifier si l'URL est configurée
          if (ApiConfigURLs.updateTickSettingsFunctionUrl.isEmpty || ApiConfigURLs.updateTickSettingsFunctionUrl.startsWith('YOUR_')) {
            throw Exception("Update Tick Settings URL not configured in constants.dart");
          }

          // Préparer le corps de la requête - N'inclure que les champs qui changent
          final body = <String, dynamic>{'tickId': tickId};
          if (name != null) body['name'] = name.trim();
          if (alarmSoundIndex != null) body['alarmSoundIndex'] = alarmSoundIndex;

          print("TickService: Updating settings for $tickId via URL: ${ApiConfigURLs.updateTickSettingsFunctionUrl} with body: $body");

          // Appel API via ApiService.put (ou .post si ta Lambda attend POST)
          final response = await _apiService.put(ApiConfigURLs.updateTickSettingsFunctionUrl, body);

          if (response['success']) {
            print("TickService: Tick settings updated successfully for $tickId. Response: ${response['data']}");
            // Mettre à jour l'objet Tick dans la liste locale
            final index = _ticks.indexWhere((t) => t.id == tickId);
            if (index != -1) {
              // Utiliser copyWith pour mettre à jour seulement les champs modifiés
              _ticks[index] = _ticks[index].copyWith(
                  // Met à jour le nom seulement si fourni dans l'appel initial
                  name: name != null ? (response['data']?['name'] as String? ?? name.trim()) : null,
                  // Met à jour l'index du son seulement si fourni dans l'appel initial
                  selectedAlarmSoundIndex: alarmSoundIndex != null
                      // Utilise Object? et _sentinel pour permettre de passer null explicitement si besoin
                      ? (Object.compare(alarmSoundIndex, _ticks[index].selectedAlarmSoundIndex) != 0
                           ? alarmSoundIndex
                           : _sentinel) // Ne passe que si différent ou si on veut forcer null
                      : _sentinel, // Ne pas passer si non fourni dans l'appel
              );
              notifyListeners(); // Notifier l'UI du changement local
              print("TickService: Local state updated for tick $tickId.");
            } else {
              print("TickService: Updated tick $tickId not found locally, fetching all ticks.");
              await fetchTicks(); // Fallback: Recharger tout si non trouvé localement
            }
            _setLoading(false); // Fin du chargement
            return true;
          } else {
            _setError((response['error'] as String?) ?? AppTexts.updateError);
            print("TickService: Update tick settings failed. API Error: $_error");
            _setLoading(false);
            return false;
          }
        } catch (e) {
          print("TickService: Exception updating tick settings for $tickId: $e");
          _setError(e is Exception ? e.toString() : ErrorMessages.connectionFailed);
          _setLoading(false);
          return false;
        }
      }


      /// Désassocie un Tick du compte utilisateur.
      /// Retourne `true` si l'opération a réussi côté backend et localement.
      Future<bool> unlinkTick(String tickId) async {
        if (!_checkAuthAndLoading()) return false;
        _setLoading(true);
        _clearError();

        try {
          if (ApiConfigURLs.removeTickFunctionUrl.isEmpty) {
              throw Exception("Remove Tick URL not configured");
          }

          // Construire l'URL avec le paramètre tickId pour la requête DELETE
          final urlWithParam = Uri.parse(ApiConfigURLs.removeTickFunctionUrl).replace(
              queryParameters: {'tickId': tickId}
          ).toString();
          print("TickService: Unlinking tick $tickId via DELETE URL: $urlWithParam");

          // Appel API via ApiService.delete
          final response = await _apiService.delete(urlWithParam);

          if (response['success']) {
            print("TickService: Tick unlinked successfully via Lambda: $tickId");
            // Supprimer le Tick de la liste locale immédiatement
            final initialLength = _ticks.length;
            _ticks.removeWhere((tick) => tick.id == tickId);
            // Notifier seulement si un élément a effectivement été supprimé
            if (_ticks.length < initialLength) {
              notifyListeners();
            } else {
              print("TickService: WARNING - Unlinked tick $tickId was not found in local list.");
            }
            _setLoading(false);
            return true;
          } else {
            // Gérer l'erreur renvoyée par la Lambda
            _setError(response['error'] ?? "Erreur lors de la désassociation");
            print("TickService: Failed to unlink tick $tickId. API Error: $_error");
            _setLoading(false);
            return false;
          }
        } catch (e) {
          // Gérer les exceptions (connexion, parsing URL, etc.)
          print("TickService: Exception unlinking tick $tickId: $e");
          _setError(e is Exception ? e.toString() : ErrorMessages.connectionFailed);
          _setLoading(false);
          return false;
        }
      }


      // --- Méthodes Utilitaires Internes ---

      /// Retourne un Tick de la liste locale par son ID, ou `null` si non trouvé.
      Tick? getTickById(String id) {
        try {
          // Utilise firstWhere pour trouver l'élément. Lance une exception si non trouvé.
          return _ticks.firstWhere((tick) => tick.id == id);
        } catch (e) {
          // L'exception StateError est lancée par firstWhere si aucun élément ne correspond.
          return null; // Retourne null si non trouvé
        }
        // Alternative plus sûre avec package:collection :
        // return _ticks.firstWhereOrNull((tick) => tick.id == id);
      }

      /// Met à jour les données d'un Tick dans la liste locale.
      /// Utile pour appliquer des mises à jour reçues (ex: via MQTT ou après une action).
      void updateTickDataLocally(Tick updatedTick) {
        final index = _ticks.indexWhere((t) => t.id == updatedTick.id);
        if (index != -1) {
          // Vérifier si les données ont réellement changé pour éviter notif inutile
          if (_ticks[index] != updatedTick) {
             _ticks[index] = updatedTick;
             print("TickService: Tick data updated locally for ${updatedTick.id}");
             notifyListeners();
          }
        } else {
          // Si le Tick n'existait pas (cas rare, ex: reçu via MQTT avant fetch), l'ajouter
          print("TickService: Adding new tick locally ${updatedTick.id} via updateTickDataLocally.");
          _ticks.add(updatedTick);
           // Trier la liste si nécessaire après ajout (ex: par nom)
           // _ticks.sort((a, b) => a.name.compareTo(b.name));
          notifyListeners();
        }
      }

      /// Vérifie si l'utilisateur est authentifié et si une opération est déjà en cours.
      /// Retourne `false` et gère l'erreur si l'une des conditions n'est pas remplie.
      bool _checkAuthAndLoading() {
        if (!_authService.isAuthenticated) {
          print("TickService: Operation prevented. User not authenticated.");
          _setError(ErrorMessages.unauthorizedAccess); // Utiliser une erreur appropriée
          notifyListeners(); // Notifier l'erreur
          return false;
        }
        if (_isLoading) {
          print("TickService: Operation skipped, another operation is in progress.");
          // Ne pas définir d'erreur ici, c'est juste une opération ignorée
          return false;
        }
        return true;
      }

      /// Met à jour l'état de chargement et notifie les listeners.
      void _setLoading(bool loading) {
        if (_isLoading == loading) return; // Éviter notifications inutiles
        _isLoading = loading;
        notifyListeners();
      }

      /// Définit un message d'erreur et notifie les listeners.
    void _setError(String? message) {
      if (_error == message) return;
      _error = message;
      notifyListeners();
    }

      /// Efface le message d'erreur actuel et notifie si nécessaire.
      void _clearError() {
        if (_error != null) {
          _error = null;
          notifyListeners(); // Notifier seulement si l'erreur est effacée
        }
      }

      /// Libère les ressources (listener d'authentification).
      @override
      void dispose() {
        print("TickService: Disposing...");
        _authService.removeListener(_handleAuthChange); // Très important !
        super.dispose();
      }
    }
    ```

5.  **Tick Settings Page (`lib/screens/tick/tick_settings_page.dart`):**
    *   Add imports for `audioplayers` and `AppSounds`.
    *   Add state variables for the audio player and currently playing sound.
    *   Add a `_buildSoundSettingTile` method.
    *   Add the new tile in the `build` method.
    *   Implement `_showSoundSelectionDialog`.
    *   Implement audio preview logic (`_playPreview`, `_stopPreview`).
    *   Dispose the audio player in `dispose`.

    ```dart
    // lib/screens/tick/tick_settings_page.dart
    import 'package:flutter/material.dart';
    import 'package:provider/provider.dart';
    import 'package:audioplayers/audioplayers.dart'; // <-- AJOUTÉ

    import '../../models/tick_model.dart';
    import '../../services/tick_service.dart';
    import '../../utils/constants.dart'; // <-- AJOUTÉ pour AppSounds
    import '../../utils/theme.dart';
    import '../../utils/validators.dart';
    import '../../widgets/alert_card.dart'; // Pour CustomSnackBar
    import '../../widgets/loading_indicator.dart';

    class TickSettingsPage extends StatefulWidget {
      /// ID du Tick dont on affiche les paramètres. Requis.
      final String tickId;

      const TickSettingsPage({Key? key, required this.tickId}) : super(key: key);

      @override
      State<TickSettingsPage> createState() => _TickSettingsPageState();
    }

    class _TickSettingsPageState extends State<TickSettingsPage> {
      // Contrôleurs et clés pour les champs éditables
      final _nameController = TextEditingController();
      final _nameFormKey = GlobalKey<FormState>();

      // États locaux pour gérer l'UI
      bool _isEditingName = false; // Mode édition du nom activé ?
      bool _isSavingName = false; // Sauvegarde du nom en cours ?
      bool _isUnlinking = false; // Désassociation en cours ?
      bool _isRinging = false; // Sonnerie en cours ? (Inchangé, géré par TickService)
      bool _isDisabling = false; // Désactivation temporaire en cours ?
      bool _isEnabling = false;

      // --- AJOUT : États pour la prévisualisation audio ---
      final AudioPlayer _audioPlayer = AudioPlayer();
      int? _currentlyPlayingSoundIndex; // Pour savoir quel son est joué
      bool _isSavingSound = false; // Pour le loader de sauvegarde du son
      // ---------------------------------------------------

      // Stocke les données actuelles du Tick (mis à jour via Provider)
      late Tick? _tick;

      @override
      void initState() {
        super.initState();
        // Récupérer les données initiales du Tick depuis le service
        _tick = Provider.of<TickService>(context, listen: false).getTickById(widget.tickId);
        // Initialiser le contrôleur de nom avec la valeur actuelle
        _nameController.text = _tick?.name ?? '';

        // --- AJOUT : Configurer l'AudioPlayer ---
        // Écouter la fin de la lecture pour réinitialiser l'état
        _audioPlayer.onPlayerComplete.listen((event) {
          if (mounted) {
              setState(() {
                _currentlyPlayingSoundIndex = null;
              });
          }
        });
        _audioPlayer.setReleaseMode(ReleaseMode.stop); // Arrête le son à la fin
        // -----------------------------------------
      }

       @override
      void didChangeDependencies() {
         super.didChangeDependencies();
         // S'assurer que _tick est à jour si le Provider notifie un changement
         final updatedTick = context.watch<TickService>().getTickById(widget.tickId);
         if (updatedTick == null && mounted && ModalRoute.of(context)?.isCurrent == true) {
             // Le Tick a été supprimé (désassocié?), fermer la page si elle est visible
             print("TickSettingsPage: Tick ${widget.tickId} no longer found. Popping.");
              WidgetsBinding.instance.addPostFrameCallback((_) {
                 if (mounted) {
                    // Pas de SnackBar ici car la page est en cours de fermeture
                    Navigator.of(context).pop();
                 }
              });
         } else if (_tick != updatedTick && updatedTick != null) {
              _tick = updatedTick;
              // Mettre à jour le contrôleur seulement si on n'est pas en mode édition
              if (!_isEditingName) {
                 _nameController.text = _tick!.name;
              }
              // Mettre à jour l'état ici si nécessaire (pour affichage son sélectionné par ex)
              if(mounted) setState(() {});
         }
      }


      @override
      void dispose() {
        _nameController.dispose();
        _audioPlayer.dispose(); // <-- AJOUTÉ : Libérer l'AudioPlayer
        super.dispose();
      }

      // --- Actions (Nom, Disable/Enable, Unlink - INCHANGÉES) ---

      /// Sauvegarde le nouveau nom du Tick via le service.
      Future<void> _saveName() async {
        if (!(_nameFormKey.currentState?.validate() ?? false)) {
          return; // Ne pas sauvegarder si le formulaire est invalide
        }
        FocusScope.of(context).unfocus(); // Masquer le clavier
        setState(() => _isSavingName = true);
        final newName = _nameController.text.trim();
        final tickService = Provider.of<TickService>(context, listen: false);


        // Utiliser la méthode mise à jour pour passer seulement le nom
        final success = await tickService.updateTickSettings(widget.tickId, name: newName);


        if (mounted) {
          setState(() {
            _isSavingName = false;
            if (success) {
              _isEditingName = false; // Quitter le mode édition
              CustomSnackBar.showSuccess(context, "Nom du Tick mis à jour.");
              // Le Provider mettra à jour _tick via didChangeDependencies
            } else {
              CustomSnackBar.showError(context, tickService.error ?? AppTexts.updateError);
            }
          });
        }
      }

      /// Désactive temporairement la surveillance du Tick.
      Future<void> _handleToggleActiveState() async {
        // Empêcher double clic si une action est déjà en cours
        if (_isDisabling || _isEnabling) return;

        final tickService = Provider.of<TickService>(context, listen: false);
        bool success = false;
        String tickId = widget.tickId; // Utiliser l'ID du widget

        // Logique 1 : Si le Tick est DÉSACTIVÉ, on le RÉACTIVE
        if (_tick?.status == TickStatus.disabled) {
          setState(() => _isEnabling = true); // Active le loader de réactivation
          print("TickSettingsPage: Attempting to reactivate Tick $tickId");
          success = await tickService.reactivateTick(tickId);

          // Vérifier si le widget est toujours monté après l'appel asynchrone
          if (!mounted) return;
          setState(() => _isEnabling = false); // Désactive le loader

          if (success) {
            CustomSnackBar.showSuccess(context, "Demande de réactivation envoyée.");
            // Si tu as décommenté _updateLocalTickStatus dans le service:
            // Provider.of<TickService>(context, listen: false)._updateLocalTickStatus(tickId, TickStatus.inactive); // Ou active?
          } else {
            CustomSnackBar.showError(context, tickService.error ?? "Erreur de réactivation.");
          }

        }
        // Logique 2 : Si le Tick est ACTIF (ou autre état), on propose de le DÉSACTIVER
        else {
          final Duration? selectedDuration = await showDialog<Duration>(
              context: context,
              builder: (context) => _buildDurationPickerDialog()
          );

          // Si l'utilisateur annule le dialogue
          if (selectedDuration == null) return;

          setState(() => _isDisabling = true); // Active le loader de désactivation

          // Cas 2a : Désactivation PERMANENTE (Duration.zero)
          if (selectedDuration == Duration.zero) {
            print("TickSettingsPage: Attempting permanent disable for Tick $tickId");
            success = await tickService.disableTickPermanently(tickId);
          }
          // Cas 2b : Désactivation TEMPORAIRE (autre durée)
          else {
            print("TickSettingsPage: Attempting temporary disable for Tick $tickId (${selectedDuration.inMinutes} mins)");
            success = await tickService.temporaryDisableTick(tickId, selectedDuration);
          }

          if (!mounted) return;
          setState(() => _isDisabling = false); // Désactive le loader

          if (success) {
            String message = selectedDuration == Duration.zero
                ? "Demande de désactivation envoyée."
                : "Demande de désactivation envoyée (${selectedDuration.inMinutes} minutes).";
            CustomSnackBar.showSuccess(context, message);
            // Si tu as décommenté _updateLocalTickStatus dans le service:
            // Provider.of<TickService>(context, listen: false)._updateLocalTickStatus(tickId, TickStatus.disabled);
          } else {
            CustomSnackBar.showError(context, tickService.error ?? "Erreur de désactivation.");
          }
        }
      }

       /// Lance le processus de désassociation du Tick.
      Future<void> _unlinkDevice() async {
        if (widget.tickId.isEmpty) {
         print("TickSettingsPage: Cannot unlink, widget.tickId is empty.");
         CustomSnackBar.showError(context, "Impossible de désassocier (ID manquant).");
         setState(() => _isUnlinking = false); // Assurer que le loader s'arrête s'il était actif
         return;
        }

        // Afficher une boîte de dialogue de confirmation
        final confirm = await showDialog<bool>(
          context: context,
          barrierDismissible: false, // Empêche de fermer en cliquant à côté
          builder: (context) => AlertDialog(
            title: const Text(AppTexts.unlinkDevice),
            content: const Text(AppTexts.unlinkDeviceConfirm),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: const Text(AppTexts.cancel),
              ),
              // Bouton rouge pour la suppression
              TextButton(
                style: TextButton.styleFrom(foregroundColor: AppTheme.errorColor),
                onPressed: () => Navigator.pop(context, true),
                child: const Text(AppTexts.unlinkDevice),
              ),
            ],
          ),
        );

        // Si l'utilisateur a confirmé
        if (confirm == true) {
          setState(() => _isUnlinking = true);
          final tickService = Provider.of<TickService>(context, listen: false);

          final success = await tickService.unlinkTick(widget.tickId);

          // Ne pas vérifier `mounted` ici car on va pop la page si succès
          if (!mounted) return;

          if (success) {
            // Afficher un message de succès AVANT de quitter la page
            CustomSnackBar.showSuccess(context, AppTexts.unlinkSuccess);
            // Attendre un court instant pour que l'utilisateur voie le SnackBar
            await Future.delayed(AppDurations.shortDelay);
            // Puis fermer la page explicitement
            if (mounted) { // Re-vérifier au cas où
              Navigator.pop(context);
            }
          } else {
            // Gérer l'erreur de désassociation (reste sur la page)
            setState(() => _isUnlinking = false);
            CustomSnackBar.showError(context, tickService.error ?? "Erreur de désassociation.");
          }
        }
      }


      // --- NOUVELLES Actions pour le son ---

      /// Sauvegarde l'index du son d'alarme sélectionné.
      Future<void> _saveSelectedSound(int soundIndex) async {
          setState(() => _isSavingSound = true);
          final tickService = Provider.of<TickService>(context, listen: false);

          final success = await tickService.updateTickSettings(
              widget.tickId,
              alarmSoundIndex: soundIndex // Passe seulement l'index du son
          );

          if (!mounted) return;

          setState(() => _isSavingSound = false);
          if (success) {
             CustomSnackBar.showSuccess(context, "Sonnerie mise à jour.");
             // L'UI se mettra à jour via le Provider
          } else {
             CustomSnackBar.showError(context, tickService.error ?? AppTexts.updateError);
          }
      }

      /// Joue un aperçu du son sélectionné.
      Future<void> _playPreview(int soundIndex) async {
        // Arrêter la lecture précédente si elle existe
        await _stopPreview();

        final soundPath = AppSounds.getSoundPath(soundIndex);
        print("Playing preview: $soundPath (Index: $soundIndex)");
        try {
           // Jouer depuis les assets
           await _audioPlayer.play(AssetSource(soundPath));
           if (mounted) {
              setState(() {
                 _currentlyPlayingSoundIndex = soundIndex; // Marquer comme en lecture
              });
           }
        } catch (e) {
           print("Error playing sound preview: $e");
           if (mounted) {
              CustomSnackBar.showError(context, "Erreur lors de la lecture de l'aperçu.");
              setState(() {
                 _currentlyPlayingSoundIndex = null;
              });
           }
        }
      }

      /// Arrête la lecture de l'aperçu en cours.
      Future<void> _stopPreview() async {
        if (_currentlyPlayingSoundIndex != null) {
           print("Stopping preview...");
           await _audioPlayer.stop();
           if (mounted) {
              setState(() {
                 _currentlyPlayingSoundIndex = null;
              });
           }
        }
      }

      /// Affiche la boîte de dialogue pour sélectionner le son.
      void _showSoundSelectionDialog() {
        // Arrêter l'aperçu si on ouvre la sélection
        _stopPreview();
        // Récupérer l'index actuel
        final int? currentSelection = _tick?.selectedAlarmSoundIndex;

        showDialog<int>(
          context: context,
          builder: (BuildContext context) {
            // Utiliser un StatefulWidget pour gérer l'état de la lecture dans le dialogue
            return StatefulBuilder(
              builder: (context, setDialogState) {
                return AlertDialog(
                  title: const Text(AppTexts.selectAlarmSound),
                  contentPadding: const EdgeInsets.only(top: 10, bottom: 0, left: 0, right: 0),
                  content: SizedBox( // Limite la hauteur
                    width: double.maxFinite,
                    child: ListView.separated(
                      shrinkWrap: true,
                      itemCount: AppSounds.alarmSounds.length,
                      separatorBuilder: (context, index) => const Divider(height: 1),
                      itemBuilder: (context, index) {
                        final soundIndex = AppSounds.alarmSounds.keys.elementAt(index);
                        final soundName = AppSounds.alarmSounds[soundIndex]!;
                        final bool isSelected = soundIndex == currentSelection;
                        final bool isPlaying = _currentlyPlayingSoundIndex == soundIndex;

                        return ListTile(
                          dense: true,
                          title: Text(soundName),
                          // Afficher une coche si sélectionné
                          leading: Radio<int>(
                            value: soundIndex,
                            groupValue: currentSelection,
                            onChanged: (value) {
                              // Sélectionner et fermer (ou juste sélectionner ?)
                              Navigator.pop(context, value); // Ferme et retourne la valeur
                            },
                           activeColor: Theme.of(context).colorScheme.primary,
                          ),
                          // Bouton Play/Stop pour l'aperçu
                          trailing: IconButton(
                            icon: Icon(isPlaying ? Icons.stop_circle_outlined : Icons.play_circle_outline, size: 24),
                            color: Theme.of(context).colorScheme.secondary,
                            tooltip: isPlaying ? 'Arrêter' : AppTexts.preview,
                            onPressed: () {
                              // Utiliser setDialogState pour rafraîchir l'icône du dialogue
                              if (isPlaying) {
                                _stopPreview().then((_) => setDialogState(() {}));
                              } else {
                                _playPreview(soundIndex).then((_) => setDialogState(() {}));
                              }
                            },
                          ),
                          onTap: () {
                            // Sélectionner et fermer au clic sur la ligne
                            Navigator.pop(context, soundIndex);
                          },
                        );
                      },
                    ),
                  ),
                  actions: [
                    TextButton(
                      onPressed: () {
                         _stopPreview(); // Arrêter l'aperçu si on annule
                         Navigator.pop(context); // Ferme sans valeur
                       },
                      child: const Text(AppTexts.cancel),
                    ),
                    // Optionnel : Bouton "OK" si on ne ferme pas au clic
                    // ElevatedButton(
                    //   onPressed: () {
                    //     _stopPreview();
                    //     Navigator.pop(context, /* valeur sélectionnée */);
                    //   },
                    //   child: const Text(AppTexts.confirm),
                    // ),
                  ],
                );
              },
            );
          },
        ).then((selectedValue) {
            // Après fermeture du dialogue
            _stopPreview(); // S'assurer que l'aperçu est arrêté
            if (selectedValue != null && selectedValue != _tick?.selectedAlarmSoundIndex) {
               // Si une nouvelle valeur a été sélectionnée, la sauvegarder
               _saveSelectedSound(selectedValue);
            }
        });
      }

      // --- Construction de l'UI ---

      @override
      Widget build(BuildContext context) {
        // Si _tick devient null (suite à une mise à jour du Provider pendant le build),
        // afficher un état vide pour éviter un crash avant que didChangeDependencies ne pop.
        if (_tick == null) {
          return Scaffold(
            appBar: AppBar(title: const Text("")), // Titre vide
            body: const Center(child: LoadingIndicator()), // Ou un message d'erreur simple
          );
        }

        return Scaffold(
          appBar: AppBar(
            // Afficher le nom actuel du Tick dans l'AppBar
            title: Text("Paramètres - ${_tick!.name}"),
          ),
          // Utiliser AbsorbPointer pour désactiver toute la page pendant la désassociation
          body: AbsorbPointer(
            absorbing: _isUnlinking || _isSavingSound, // Aussi pendant sauvegarde son
            child: Stack( // Stack pour superposer le loader de désassociation
              children: [
                ListView(
                  padding: const EdgeInsets.all(16.0),
                  children: [
                    // --- Section Nom ---
                    _buildSectionTitle(context, AppTexts.general),
                    _buildNameTile(), // Widget pour afficher/éditer le nom
                    const SizedBox(height: 8),
                    // Afficher l'adresse MAC (non modifiable)
                     _buildInfoTile(
                        icon: Icons.bluetooth,
                        title: AppTexts.tickMacAddress,
                        subtitle: _tick!.macAddress ?? 'Non disponible',
                     ),

                    const Divider(height: 32),

                    // --- Section Fonctionnalités ---
                    _buildSectionTitle(context, AppTexts.features),
                    Builder( // Utilise Builder pour obtenir un context à jour pour le statut
                        builder: (context) {
                          // Récupérer le statut le plus récent du Tick depuis le Provider
                          final currentTickStatus = Provider.of<TickService>(context).getTickById(widget.tickId)?.status;
                          bool isCurrentlyDisabled = currentTickStatus == TickStatus.disabled;
                          bool isActionRunning = _isDisabling || _isEnabling; // Utilise les deux flags

                          return _buildFeatureTile(
                            icon: isCurrentlyDisabled
                                ? Icons.play_circle_outline// Icône Réactiver
                                : Icons.pause_circle_outline, // Icône Désactiver
                            title: isCurrentlyDisabled
                                ? AppTexts.reactivateDevice
                                : AppTexts.disableDevice,
                            subtitle: isCurrentlyDisabled
                                ? 'Réactiver la surveillance et les alertes'
                                : 'Désactiver la surveillance et les alertes', // Simplifié
                            onTap: isActionRunning ? null : _handleToggleActiveState, // Appelle la nouvelle fonction
                            trailing: isActionRunning ? const LoadingIndicator(size: 18) : null,
                            // Changer la couleur si désactivé pour attirer l'attention ?
                            color: isCurrentlyDisabled ? AppTheme.warningColor : null, // Orange quand désactivé ?
                          );
                        }
                    ),
                    // --- MODIFICATION : Remplacement du Tile Sonnerie ---
                    _buildSoundSettingTile(),
                    // --------------------------------------------------

                    const Divider(height: 32),

                    // --- Section Danger Zone ---
                    _buildSectionTitle(context, AppTexts.dangerZone, color: AppTheme.errorColor),
                    _buildFeatureTile(
                      icon: Icons.link_off,
                      title: AppTexts.unlinkDevice,
                      subtitle: 'Supprimer ce Tick de votre compte (irréversible)',
                      color: AppTheme.errorColor, // Couleur rouge pour le titre/icône
                      onTap: _isUnlinking ? null : _unlinkDevice, // Désactiver pendant désassociation
                      trailing: Container( // Enveloppe le trailing dans un Container
                         alignment: Alignment.centerRight, // Alignement à droite (optionnel)
                         width: 48, // Largeur fixe (taille typique d'un IconButton)
                         child: _isUnlinking
                             ? const LoadingIndicator(size: 18, color: AppTheme.errorColor)
                             : Icon(Icons.delete_forever_outlined, color: AppTheme.errorColor),
                       ),
                    ),
                  ],
                ),
                // Loader global superposé pendant la désassociation ou sauvegarde son
                if (_isUnlinking || _isSavingSound)
                   Container(
                     color: Colors.black.withOpacity(0.3),
                     child: const Center(child: LoadingIndicator(size: 40)),
                   ),
              ],
            ),
          ),
        );
      }

      /// Construit le ListTile pour afficher/éditer le nom. (INCHANGÉ)
      Widget _buildNameTile() {
        if (_isEditingName) {
          // Mode édition: affiche un TextFormField avec boutons Save/Cancel
          return Form(
            key: _nameFormKey,
            child: ListTile(
              contentPadding: EdgeInsets.zero, // Pas de padding pour aligner avec les autres ListTile
              leading: const Icon(Icons.label_outline), // Garder l'icône
              title: TextFormField(
                controller: _nameController,
                decoration: const InputDecoration(
                   labelText: AppTexts.tickName,
                   isDense: true, // Réduit la hauteur
                   contentPadding: EdgeInsets.symmetric(vertical: 8), // Ajuste padding interne
                ),
                validator: (value) => Validators.validateNotEmpty(value, "Le nom ne peut pas être vide"),
                textInputAction: TextInputAction.done,
                onFieldSubmitted: (_) => _saveName(),
                autofocus: true,
                enabled: !_isSavingName, // Désactiver pendant la sauvegarde
                textCapitalization: TextCapitalization.sentences,
              ),
              // Boutons d'action pour sauvegarder ou annuler
              trailing: Row(
                 mainAxisSize: MainAxisSize.min,
                 children: [
                    IconButton(
                      icon: _isSavingName
                          ? const LoadingIndicator(size: 18)
                          : const Icon(Icons.check, color: AppTheme.successColor),
                      tooltip: AppTexts.save,
                      onPressed: _isSavingName ? null : _saveName,
                    ),
                    IconButton(
                      icon: const Icon(Icons.close),
                      tooltip: AppTexts.cancel,
                      onPressed: _isSavingName ? null : () => setState(() {
                         _isEditingName = false;
                         _nameController.text = _tick?.name ?? ''; // Rétablir la valeur initiale
                         _nameFormKey.currentState?.reset(); // Reset validation state
                      }),
                    ),
                 ],
              ),
            ),
          );
        } else {
          // Mode affichage: affiche le nom actuel avec un bouton Edit
          return ListTile(
            leading: const Icon(Icons.label_outline),
            title: const Text(AppTexts.tickName),
            subtitle: Text(_tick?.name ?? ''), // Utiliser valeur de _tick (peut être null brièvement)
            trailing: IconButton(
              icon: const Icon(Icons.edit_outlined, size: 20),
              tooltip: AppTexts.edit,
              onPressed: () => setState(() => _isEditingName = true),
            ),
            onTap: () => setState(() => _isEditingName = true), // Permet de cliquer sur toute la ligne
          );
        }
      }

      /// Construit un ListTile simple pour afficher une information (non éditable). (INCHANGÉ)
      Widget _buildInfoTile({required IconData icon, required String title, required String subtitle}) {
         return ListTile(
           leading: Icon(icon),
           title: Text(title),
           subtitle: Text(subtitle),
           dense: true,
         );
      }

      /// Construit un ListTile pour une fonctionnalité ou une action. (INCHANGÉ)
      Widget _buildFeatureTile({
        required IconData icon,
        required String title,
        required String subtitle,
        VoidCallback? onTap,
        Color? color, // Pour le titre/icône
        Widget? trailing,
      }) {
        final theme = Theme.of(context); // Récupérer le thème
        return ListTile(
          leading: Icon(icon, color: onTap == null ? theme.disabledColor : color),
          title: Text(title, style: TextStyle(color: onTap == null ? theme.disabledColor : color)),
          subtitle: Text(
            subtitle,
            // Ajout de maxLines et overflow pour éviter les problèmes de layout
            maxLines: 2, // Permet 2 lignes si besoin, sinon utiliser 1
            overflow: TextOverflow.ellipsis, // Ajoute '...' si le texte est trop long
          ),
          onTap: onTap,
          trailing: trailing ?? (onTap != null ? Icon(Icons.arrow_forward_ios, size: 16, color: theme.textTheme.bodySmall?.color) : null), // Utiliser couleur du thème pour la flèche
          dense: true, // Optionnel: Rendre un peu plus compact
        );
      }

      /// --- NOUVEAU : Construit le ListTile pour la sélection du son ---
      Widget _buildSoundSettingTile() {
          // Obtenir le nom du son sélectionné, ou un texte par défaut
          final selectedIndex = _tick?.selectedAlarmSoundIndex;
          final soundName = selectedIndex != null
                           ? AppSounds.alarmSounds[selectedIndex] ?? AppTexts.noSoundSelected
                           : AppTexts.noSoundSelected;

          return ListTile(
              leading: const Icon(Icons.music_note_outlined),
              title: const Text(AppTexts.soundSettings),
              subtitle: Text('Actuelle: $soundName'), // Affiche le nom du son
              trailing: _isSavingSound
                  ? const LoadingIndicator(size: 18)
                  : Icon(Icons.arrow_forward_ios, size: 16, color: Theme.of(context).textTheme.bodySmall?.color),
              onTap: _isSavingSound ? null : _showSoundSelectionDialog,
          );
      }
      // -----------------------------------------------------------

      /// Construit le titre d'une section dans la liste. (INCHANGÉ)
      Widget _buildSectionTitle(BuildContext context, String title, {Color? color}) {
        return Padding(
          padding: const EdgeInsets.only(top: 24.0, bottom: 8.0, left: 16.0, right: 16.0), // Ajuster padding
          child: Text(
            title.toUpperCase(),
            style: TextStyle(
              color: color ?? Theme.of(context).colorScheme.primary,
              fontWeight: FontWeight.bold,
              fontSize: 12,
              letterSpacing: 0.8,
            ),
          ),
        );
      }

      /// Widget pour choisir la durée de désactivation (placeholder). (INCHANGÉ)
      Widget _buildDurationPickerDialog() {
        return SimpleDialog(
          title: const Text("Désactiver l'appareil"), // Titre mis à jour
          children: [
            SimpleDialogOption( // Option Permanente en premier
              onPressed: () => Navigator.pop(context, Duration.zero), // Utilise Duration.zero comme indicateur
              child: const Text(AppTexts.untilReactivation), // "Jusqu'à réactivation"
            ),
            const Divider(), // Séparateur visuel
            SimpleDialogOption(
              onPressed: () => Navigator.pop(context, const Duration(minutes: 30)),
              child: const Text('30 minutes'),
            ),
            SimpleDialogOption(
              onPressed: () => Navigator.pop(context, const Duration(hours: 1)),
              child: const Text('1 heure'),
            ),
            SimpleDialogOption(
              onPressed: () => Navigator.pop(context, const Duration(hours: 2)),
              child: const Text('2 heures'),
            ),
            SimpleDialogOption(
              onPressed: () => Navigator.pop(context, const Duration(hours: 4)),
              child: const Text('4 heures'),
            ),
            const Divider(), // Séparateur avant Annuler
            SimpleDialogOption(
              onPressed: () => Navigator.pop(context), // Annuler
              child: const Text(AppTexts.cancel, style: TextStyle(color: AppTheme.errorColor)),
            ),
          ],
        );
      }
    }
    ```

6.  **Map Page (`lib/screens/tick/map_page.dart`):**
    *   Update the `_ringTick` method to use the modified `TickService.ringTick` and provide better user feedback.

    ```dart
    // lib/screens/tick/map_page.dart

    import 'package:flutter/material.dart';
    import 'package:flutter/services.dart';
    import 'package:url_launcher/url_launcher.dart';
    import 'package:google_maps_flutter/google_maps_flutter.dart';
    import 'package:provider/provider.dart';
    import 'package:geolocator/geolocator.dart'; // Pour la localisation utilisateur
    import 'package:geocoding/geocoding.dart';
    import 'package:permission_handler/permission_handler.dart' as ph; // Pour ouvrir les paramètres
    import 'package:tick_app/screens/tick/tick_settings_page.dart';
    import 'dart:async';

    import '../../models/tick_model.dart';
    import '../../services/theme_service.dart';
    import '../../services/tick_service.dart';
    import '../../widgets/action_button.dart'; // Utiliser ActionButton
    import '../../widgets/alert_card.dart'; // Pour CustomSnackBar
    import '../../widgets/theme_toggle_button.dart';
    import '../../widgets/loading_indicator.dart';
    import '../../utils/map_styles.dart';
    import '../../utils/constants.dart';
    import '../../utils/theme.dart'; // Pour AppTheme et AlertType
    import 'history_page.dart'; // Pour la navigation vers l'historique

    class MapPage extends StatefulWidget {
      /// L'objet Tick à afficher sur la carte. Requis.
      final Tick tick;

      const MapPage({
        Key? key,
        required this.tick,
      }) : super(key: key);

      @override
      State<MapPage> createState() => _MapPageState();
    }

    class _MapPageState extends State<MapPage> {
      final Completer<GoogleMapController> _mapControllerCompleter = Completer();
      GoogleMapController? _mapController;

      // État local pour suivre les chargements spécifiques
      bool _isMapLoading = true; // Chargement initial de Google Maps
      bool _isLocateActionLoading = false; // NOUVEAU : Pour 'Localiser'
      bool _isRingActionLoading = false; // NOUVEAU : Pour 'Faire sonner'
      bool _isUserLocationLoading = false; // Pour la récupération de la position utilisateur
      bool _isFetchingData = false;
      // La position de l'utilisateur (peut être null)
      LatLng? _userPosition;

      // Les marqueurs à afficher sur la carte (contiendra le Tick et potentiellement l'utilisateur)
      final Set<Marker> _markers = {};

      // Référence locale au Tick, mise à jour par Provider
      late Tick _currentTickData;


      String? _tickAddress;
      bool _isFetchingAddress = false;

      @override
      void initState() {
        super.initState();
        // Initialiser avec les données passées via le constructeur
        _currentTickData = widget.tick;
        // Mettre à jour le marqueur initial du Tick
        _updateTickMarker();
        // Essayer de récupérer la position initiale de l'utilisateur (sans bloquer)
        _getCurrentUserLocation(centerMap: false);
        if (_currentTickData.latitude != null && _currentTickData.longitude != null) {
          _fetchAddressFromCoordinates(
              _currentTickData.latitude!, _currentTickData.longitude!);
        }
      }

      Future<void> _fetchAddressFromCoordinates(double lat, double lon) async {
        // Prevent concurrent calls or calls if already fetching
        if (!mounted || _isFetchingAddress) return;

        setState(() {
          _isFetchingAddress = true;
          _tickAddress = null; // Clear previous address while loading new one
        });

        print("MapPage: Fetching address for $lat, $lon");

        try {
          // Use geolocator's placemarkFromCoordinates
          // This might take a moment as it may involve network requests
          List<Placemark> placemarks = await placemarkFromCoordinates(lat, lon);

          if (!mounted) return; // Check if widget is still mounted after async operation

          if (placemarks.isNotEmpty) {
            final Placemark place = placemarks.first;

            final street = place.street;       // Ex: "1600 Amphitheatre Pkwy" ou "Rue de la Gare 5" ou null
            final city = place.locality;       // Ex: "Mountain View" ou "Bruxelles" ou null

            // Construit la liste avec seulement ces deux éléments
            final addressParts = [street, city];

            // Filtre les éléments null ou vides et les joint avec une virgule et un espace
            _tickAddress = addressParts
                .where((part) => part != null && part.isNotEmpty) // Garde seulement les parties valides
                .join(', '); // Joint avec ", "

            // Gérer le cas où ni la rue ni la ville ne sont trouvées
            if (_tickAddress!.isEmpty) {
                _tickAddress = "Adresse non détaillée"; // Ou un autre message par défaut
            }
            print("MapPage: Fetched address: $_tickAddress");
          } else {
            print("MapPage: No address found for coordinates.");
            _tickAddress = "Adresse introuvable"; // Fallback message
          }
        } catch (e, stacktrace) {
          print("MapPage: Error fetching address: $e");
          print(stacktrace);
          if (mounted) {
            _tickAddress = "Erreur de géocodage"; // Error message for UI
          }
          // Optionally show a SnackBar error to the user
          // CustomSnackBar.showError(context, "Impossible de récupérer l'adresse.");
        } finally {
          // Ensure loading state is turned off even if errors occur
          if (mounted) {
            setState(() => _isFetchingAddress = false);
          }
        }
      }

      @override
      void didChangeDependencies() {
        super.didChangeDependencies();
        final updatedTick = context.watch<TickService>().getTickById(widget.tick.id);

        if (updatedTick == null && mounted && ModalRoute.of(context)?.isCurrent == true) {
          // Le Tick a été supprimé (désassocié?), revenir en arrière
          print("MapPage: Tick ${widget.tick.id} no longer found in service. Popping.");
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (mounted) {
              CustomSnackBar.showError(context, "Ce Tick n'est plus disponible.");
              Navigator.of(context).pop();
            }
          });
        } else if (updatedTick != null && updatedTick != _currentTickData) {
          print("MapPage: Received update for Tick ${_currentTickData.id}");

          // --- MODIFIED: Check if coordinates changed ---
          bool coordsChanged = (_currentTickData.latitude != updatedTick.latitude ||
              _currentTickData.longitude != updatedTick.longitude) &&
              updatedTick.latitude != null && // Ensure new coords are valid
              updatedTick.longitude != null;

          setState(() {
            _currentTickData = updatedTick;
            _updateTickMarker(); // Mettre à jour le marqueur avec les nouvelles données
          });

          // Fetch address only if coordinates are valid and have actually changed
          if (coordsChanged) {
            _fetchAddressFromCoordinates(updatedTick.latitude!, updatedTick.longitude!);
          } else if (updatedTick.latitude == null || updatedTick.longitude == null) {
            // Clear address if coordinates become null
            if (_tickAddress != null) {
              setState(() => _tickAddress = null);
            }
          }
          // --- End MODIFIED ---
        }
      }


      @override
      void dispose() {
        _mapController?.dispose(); // Libérer le contrôleur de carte
        super.dispose();
      }

      // --- Gestion des Marqueurs ---

      /// Met à jour le marqueur du Tick sur la carte.
      void _updateTickMarker() {
        if (_currentTickData.latitude != null && _currentTickData.longitude != null) {
          final tickPosition = LatLng(_currentTickData.latitude!, _currentTickData.longitude!);
          final marker = Marker(
            markerId: MarkerId(_currentTickData.id),
            position: tickPosition,
            infoWindow: InfoWindow(
              title: _currentTickData.name,
              snippet: '${_currentTickData.statusDescription} - ${_currentTickData.formattedLastUpdate}',
            ),
            icon: _getMarkerIcon(_currentTickData.status),
            // anchor: const Offset(0.5, 0.5), // Centre l'icône sur la position
          );
          // Remplacer l'ancien marqueur par le nouveau
          _markers.removeWhere((m) => m.markerId.value == _currentTickData.id);
          _markers.add(marker);

        } else {
          // Si pas de position, supprimer le marqueur
          _markers.removeWhere((m) => m.markerId.value == _currentTickData.id);
        }
        // Rafraîchir l'UI si le widget est toujours monté
        if (mounted) {
          setState(() {});
        }
      }

      /// Retourne l'icône de marqueur appropriée en fonction du statut du Tick.
      BitmapDescriptor _getMarkerIcon(TickStatus status) {
        // Utilise les teintes standard de BitmapDescriptor
        double hue;
        switch (status) {
          case TickStatus.active: hue = BitmapDescriptor.hueGreen; break;
          case TickStatus.moving: hue = BitmapDescriptor.hueAzure; break; // Bleu clair pour mouvement
          case TickStatus.lowBattery: hue = BitmapDescriptor.hueOrange; break;
          case TickStatus.theftAlert: hue = BitmapDescriptor.hueRed; break; // Rouge pour vol
          case TickStatus.disabled: hue = BitmapDescriptor.hueViolet; break; // Violet pour désactivé
          case TickStatus.inactive:
          case TickStatus.unknown:
          default: hue = BitmapDescriptor.hueGrey; break; // Gris pour inactif/inconnu
        }
        return BitmapDescriptor.defaultMarkerWithHue(hue);
      }

      // --- Gestion de la Localisation Utilisateur ---

      /// Récupère la position actuelle de l'utilisateur.
      /// Met à jour `_userPosition` et gère les permissions/erreurs.
      Future<void> _getCurrentUserLocation({bool centerMap = false}) async {
        if (!mounted) return;
        setState(() => _isUserLocationLoading = true);

        // Définir les paramètres de précision souhaités
        const locationSettingsHigh = LocationSettings(
          accuracy: LocationAccuracy.high,
          distanceFilter: 10, // Mettre à jour seulement si déplacé de 10m
          timeLimit: Duration(seconds: 10), // Timeout pour haute précision
        );
         const locationSettingsMedium = LocationSettings(
          accuracy: LocationAccuracy.medium,
          distanceFilter: 50,
        );

        String? errorMsg; // Stocker le message d'erreur potentiel

        try {
          // 1. Vérifier si le service de localisation est activé
          bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
          if (!serviceEnabled) {
            // Optionnel: Proposer à l'utilisateur d'activer les services via Geolocator
            // await Geolocator.openLocationSettings();
            throw Exception(ErrorMessages.locationServiceDisabled);
          }

          // 2. Vérifier et demander les permissions de localisation
          LocationPermission permission = await Geolocator.checkPermission();
          if (permission == LocationPermission.denied) {
            permission = await Geolocator.requestPermission();
            if (permission == LocationPermission.denied) {
              throw Exception(ErrorMessages.permissionDeniedLocation); // Refus simple
            }
          }
          if (permission == LocationPermission.deniedForever) {
            throw Exception(ErrorMessages.permissionDeniedLocationExplain); // Refus définitif
          }

          // 3. Obtenir la position
          Position position;
          try {
              // Tenter haute précision avec timeout
              position = await Geolocator.getCurrentPosition(locationSettings: locationSettingsHigh);
          } catch (e) {
              print("MapPage: High accuracy location failed ($e), falling back to medium...");
              // Si échec, tenter moyenne précision (sans timeout spécifique ici)
              position = await Geolocator.getCurrentPosition(locationSettings: locationSettingsMedium);
          }


          if (!mounted) return;

          _userPosition = LatLng(position.latitude, position.longitude);
          print("MapPage: User location updated: $_userPosition");

          // Centrer la carte si demandé et si la carte est prête
          if (centerMap && _mapController != null && _userPosition != null) {
            _mapController!.animateCamera(
              CameraUpdate.newLatLngZoom(_userPosition!, 16.0), // Zoom correct pour voir alentours
            );
          }

        } on Exception catch (e) { // Capturer toutes les exceptions (permission, service, timeout...)
          print("MapPage: Error getting user location: $e");
           // Utiliser le message de l'exception si disponible
          errorMsg = e.toString().replaceFirst('Exception: ', '');
          // Assurer que _userPosition est null en cas d'erreur
          _userPosition = null;
        } finally {
          if (mounted) {
            setState(() => _isUserLocationLoading = false);
            // Afficher le message d'erreur s'il y en a un
            if (errorMsg != null) {
               // Ajouter un bouton pour ouvrir les paramètres si permission refusée définitivement
               SnackBarAction? action;
               if (errorMsg == ErrorMessages.permissionDeniedLocationExplain) {
                  action = SnackBarAction(
                     label: AppTexts.openSettings.toUpperCase(),
                     onPressed: () async => await ph.openAppSettings(),
                  );
               }
               CustomSnackBar.showError(context, errorMsg, action: action);
            }
          }
        }
      }

      // --- Contrôle de la Carte ---

      /// Centre la carte sur la dernière position connue du Tick.
      Future<void> _centerOnTick() async {
        if (_currentTickData.latitude == null || _currentTickData.longitude == null) {
            CustomSnackBar.show(context, message: AppTexts.noLocationAvailable, type: AlertType.info);
            return;
        }
        // Attendre que le contrôleur soit prêt si ce n'est pas déjà le cas
        if (_mapController == null) await _mapControllerCompleter.future;

        _mapController?.animateCamera(
          CameraUpdate.newLatLngZoom(
            LatLng(_currentTickData.latitude!, _currentTickData.longitude!),
            16.0, // Zoom plus proche sur le Tick
          ),
        );
      }

      /// Centre la carte sur la position actuelle de l'utilisateur.
      /// Rafraîchit d'abord la position de l'utilisateur.
      Future<void> _centerOnUser() async {
        await _getCurrentUserLocation(centerMap: true);
      }

      // --- Actions du Tick ---

      /// Demande une mise à jour de la localisation du Tick via le service.
      Future<void> _requestLocationUpdate() async {
        if (_isLocateActionLoading || _isRingActionLoading) return; // Éviter les appels multiples
        setState(() => _isLocateActionLoading = true);

        final tickService = Provider.of<TickService>(context, listen: false);
        final success = await tickService.requestTickLocation(_currentTickData.id);

        if (!mounted) return;
        setState(() => _isLocateActionLoading = false);

        if (success) {
          CustomSnackBar.show(context, message: AppTexts.locationRequestSent, type: AlertType.info);
          print("MapPage: Location request sent, now triggering data refresh...");
          _refreshTickData();
        } else {
          CustomSnackBar.showError(context, tickService.error ?? ErrorMessages.apiError);
        }
      }

      /// Demande à faire sonner le Tick via le service.
      Future<void> _ringTick() async {
        if (_isLocateActionLoading || _isRingActionLoading) return; // Éviter les appels multiples
        setState(() => _isRingActionLoading = true);

        final tickService = Provider.of<TickService>(context, listen: false);
        final success = await tickService.ringTick(_currentTickData.id); // Utilise la méthode mise à jour du service

        if (!mounted) return;
        setState(() => _isRingActionLoading = false);

        if (success) {
          // Afficher un message de confirmation que la *commande* a été envoyée
          CustomSnackBar.showSuccess(context, AppTexts.ringingTickCommandSent);
        } else {
          // Afficher l'erreur renvoyée par TickService
          CustomSnackBar.showError(context, tickService.error ?? AppTexts.ringingTickError);
        }
      }

      // --- Navigation ---

      /// Navigue vers la page des paramètres du Tick.
      void _navigateToTickSettings() {
        // Vérifier si l'ID est valide avant de naviguer
        if (_currentTickData.id.isEmpty) {
          print("MapPage: Cannot navigate to settings, Tick ID is empty.");
          CustomSnackBar.showError(context, "Impossible d'ouvrir les paramètres (ID manquant).");
          return; // Ne pas naviguer
        }

        print("Navigating to settings for Tick ID: ${_currentTickData.id}"); // Ajout log pour confirmer
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => TickSettingsPage(tickId: _currentTickData.id),
            settings: const RouteSettings(name: Routes.tickSettings),
          ),
        );
      }

      /// Navigue vers la page d'historique du Tick.
      void _navigateToHistory() {
         // Utilise la route standard MaterialPageRoute pour passer des arguments complexes si besoin,
         // ou une route nommée si seulement l'ID est nécessaire.
          Navigator.push(
             context,
             MaterialPageRoute(
                builder: (context) => HistoryPage(
                   tickId: _currentTickData.id,
                   tickName: _currentTickData.name, // Passer aussi le nom
                ),
             ),
          );
         // Alternative avec route nommée (si configurée pour accepter ID):
         // Navigator.pushNamed(context, Routes.tickHistory, arguments: _currentTickData.id);
      }

      Future<void> _refreshTickData() async {
        if (_isFetchingData) return; // Avoid multiple calls
        setState(() => _isFetchingData = true);

        final tickService = Provider.of<TickService>(context, listen: false);
        // Call fetchTicks which will update the provider state
        await tickService.fetchTicks();

        // Check if mounted after async call
        if (!mounted) return;

        setState(() => _isFetchingData = false);

        // Optionally show feedback based on tickService.error
        if (tickService.error != null) {
          CustomSnackBar.showError(context, "Erreur de rafraîchissement: ${tickService.error}");
        } else {
          // Optional: Success feedback (might be redundant as UI updates)
          // CustomSnackBar.show(context, message: "Données actualisées.", type: AlertType.info);
          print("MapPage: Tick data refreshed via fetchTicks().");
        }
      }

      void _showAddressActionsMenu(BuildContext context, LongPressStartDetails details, String address) {
        final bool canOpenMap = _currentTickData.latitude != null && _currentTickData.longitude != null;
        final double latitude = _currentTickData.latitude ?? 0;
        final double longitude = _currentTickData.longitude ?? 0;

        final position = RelativeRect.fromLTRB(
          details.globalPosition.dx,
          details.globalPosition.dy - 60, // Remonter un peu plus pour la hauteur du "tooltip"
          details.globalPosition.dx + 1,
          details.globalPosition.dy - 59,
        );

        // Thème actuel pour adapter les couleurs
        final theme = Theme.of(context);
        final tooltipTheme = Theme.of(context).tooltipTheme; // Récupérer le thème global du Tooltip
        final bool isDark = theme.brightness == Brightness.dark;

        // Couleurs pour simuler le tooltip
        final tooltipBackgroundColor = tooltipTheme.decoration is BoxDecoration
            ? (tooltipTheme.decoration as BoxDecoration).color ?? (isDark ? Colors.white.withOpacity(0.9) : Colors.black.withOpacity(0.8))
            : (isDark ? Colors.white.withOpacity(0.9) : Colors.black.withOpacity(0.8));
        final tooltipTextColor = tooltipTheme.textStyle?.color ?? (isDark ? Colors.black : Colors.white);


        showMenu<String>(
          context: context,
          position: position,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)), // Coins pour le menu global
          color: theme.cardColor, // Couleur de fond standard pour les options du menu
          items: <PopupMenuEntry<String>>[

            // --- 1. Simulation du Tooltip (Item non cliquable) ---
            PopupMenuItem<String>(
              value: 'tooltip_display', // Valeur unique
              padding: EdgeInsets.zero, // Retirer le padding par défaut du MenuItem
              enabled: false, // Rendre non cliquable
              height: 0, // Hauteur minimale pour ne pas prendre trop de place verticale inutile
              child: Container(
                padding: tooltipTheme.padding ?? const EdgeInsets.symmetric(horizontal: 12, vertical: 8), // Utiliser padding du thème
                margin: const EdgeInsets.only(bottom: 6.0), // Marge sous le faux tooltip
                constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.8), // Limiter la largeur
                decoration: BoxDecoration( // Appliquer la décoration du thème tooltip
                  color: tooltipBackgroundColor,
                  borderRadius: tooltipTheme.decoration is BoxDecoration
                      ? (tooltipTheme.decoration as BoxDecoration).borderRadius ?? BorderRadius.circular(4)
                      : BorderRadius.circular(4),
                ),
                child: Text(
                  address, // Adresse complète
                  style: tooltipTheme.textStyle ?? TextStyle(color: tooltipTextColor, fontSize: 12), // Style du thème
                  textAlign: TextAlign.center,
                  softWrap: true,
                ),
              ),
            ),

            // --- Pas besoin de diviseur ici, la marge du container au-dessus suffit ---
            // const PopupMenuDivider(),

            // --- 2. Option Copier ---
            PopupMenuItem<String>(
              value: 'copy',
              height: 40, // Hauteur standard pour les items cliquables
              onTap: () {
                Clipboard.setData(ClipboardData(text: address));
                if (mounted) {
                  CustomSnackBar.show(context, message: 'Adresse copiée !', type: AlertType.success);
                }
              },
              child: const Row(
                children: [
                  Icon(Icons.copy_outlined, size: 20),
                  SizedBox(width: 12),
                  Text('Copier l\'adresse'),
                ],
              ),
            ),

            // --- 3. Option Ouvrir dans Maps (si coordonnées dispo) ---
            if (canOpenMap)
              PopupMenuItem<String>(
                value: 'open_map',
                height: 40, // Hauteur standard
                onTap: () async {
                  final Uri googleMapsUrl = Uri.parse(
                      'https://www.google.com/maps/search/?api=1&query=$latitude,$longitude'
                  );
                  try {
                    if (await canLaunchUrl(googleMapsUrl)) {
                      await launchUrl(googleMapsUrl, mode: LaunchMode.externalApplication);
                    } else {
                      if(mounted) CustomSnackBar.showError(context, 'Impossible d\'ouvrir Google Maps.');
                    }
                  } catch (e) {
                    print("Error launching map URL: $e");
                    if(mounted) CustomSnackBar.showError(context, 'Erreur lors de l\'ouverture de la carte.');
                  }
                },
                child: const Row(
                  children: [
                    Icon(Icons.map_outlined, size: 20),
                    SizedBox(width: 12),
                    Text('Ouvrir dans Maps'),
                  ],
                ),
              ),
          ],
        );
      }

      // --- Construction de l'UI ---

      @override
      Widget build(BuildContext context) {
        final themeService = context.watch<ThemeService>(); // Pour le style de la carte

        return Scaffold(
          appBar: AppBar(
            title: Text(_currentTickData.name), // Nom du Tick actuel
            actions: [
              const ThemeToggleButton(),
              // Bouton de rafraîchissement manuel (demande active de localisation)
              IconButton(
                // Use the new loading state variable
                icon: _isFetchingData
                    ? const LoadingIndicator(size: 18)
                    : const Icon(Icons.refresh),
                tooltip: "Actualiser les données", // Updated tooltip
                // Disable while fetching, call the new function
                onPressed: _isFetchingData ? null : _refreshTickData,
              ),
              IconButton(
                icon: const Icon(Icons.settings_outlined),
                tooltip: AppTexts.settings,
                onPressed: _navigateToTickSettings,
              ),
            ],
          ),
          body: Column(
            children: [
              // --- Carte Google Maps ---
              SizedBox(
                // Occupe la moitié supérieure de l'écran
                height: MediaQuery.of(context).size.height * 0.5,
                child: Stack(
                  alignment: Alignment.center, // Centre le LoadingIndicator
                  children: [
                    // Affiche la carte seulement si la position initiale est connue
                    if (_currentTickData.latitude != null && _currentTickData.longitude != null)
                      GoogleMap(
                        mapType: MapType.normal,
                        initialCameraPosition: CameraPosition(
                          target: LatLng(_currentTickData.latitude!, _currentTickData.longitude!),
                          zoom: 16.0,
                        ),
                        markers: _markers,
                        style: themeService.isDarkMode(context) ? MapStyles.darkStyle : MapStyles.lightStyle,
                        onMapCreated: (GoogleMapController controller) {
                          if (!_mapControllerCompleter.isCompleted) {
                            _mapControllerCompleter.complete(controller);
                            _mapController = controller;
                          }
                          // Indiquer que la carte est chargée (même si le style met du temps)
                          if (mounted && _isMapLoading) setState(() => _isMapLoading = false);
                        },
                        myLocationButtonEnabled: false, // On utilise nos propres boutons
                        myLocationEnabled: true, // Affiche le point bleu (nécessite permission)
                        zoomControlsEnabled: false, // Contrôles +/-
                        zoomGesturesEnabled: true,
                        mapToolbarEnabled: false, // Masque la barre d'outils Google (Directions, etc.)
                      )
                    else // Afficher message si pas de coordonnées initiales
                      Center(
                        child: Padding(
                          padding: const EdgeInsets.all(16.0),
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.location_off_outlined, size: 50, color: Theme.of(context).disabledColor),
                              const SizedBox(height: 16),
                              Text(AppTexts.noLocationAvailable, style: Theme.of(context).textTheme.titleMedium, textAlign: TextAlign.center,),
                              const SizedBox(height: 16),
                              ElevatedButton.icon(
                                // NOUVEAU: Utiliser _isLocateActionLoading
                                icon: _isLocateActionLoading ? const LoadingIndicator(size: 18, color: Colors.white) : const Icon(Icons.refresh, size: 18),
                                label: const Text(AppTexts.tryToLocate),
                                onPressed: _isLocateActionLoading || _isRingActionLoading ? null : _requestLocationUpdate,
                              )
                            ],
                          ),
                        ),
                      ),

                    // Indicateur de chargement PENDANT que la carte s'initialise
                    if (_isMapLoading && _currentTickData.latitude != null)
                      const Center(child: LoadingIndicator()),

                    // Boutons flottants pour centrer la carte
                    Positioned(
                      right: 16,
                      bottom: 90, // Positionné au-dessus du bouton utilisateur
                      child: FloatingActionButton.small(
                        heroTag: "centerTickBtn", // Tags uniques pour Hero animation
                        onPressed: _centerOnTick,
                        tooltip: AppTexts.centerOnTick,
                        backgroundColor: themeService.isDarkMode(context)
                            ? AppTheme.dividerColorDark // Utilise une couleur de fond adaptée au thème
                            : AppTheme.surfaceColorLight,
                        child: Icon(Icons.location_on, color: _getMarkerIcon(_currentTickData.status) == BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueRed) ? AppTheme.errorColor : AppTheme.primaryColor), // Adapte couleur icone à l'état
                      ),
                    ),
                    Positioned(
                      right: 16,
                      bottom: 20,
                      child: FloatingActionButton.small(
                        heroTag: "centerUserBtn",
                        onPressed: _centerOnUser,
                        tooltip: AppTexts.centerOnMe,
                        backgroundColor: themeService.isDarkMode(context)
                            ? AppTheme.dividerColorDark // Fond sombre (gris foncé) en mode sombre
                            : AppTheme.surfaceColorLight,
                        child: _isUserLocationLoading
                          ? const LoadingIndicator(size: 18) // Indicateur pendant la recherche utilisateur
                          : const Icon(Icons.my_location,
                          color: AppTheme.primaryColor,),
                      ),
                    ),
                  ],
                ),
              ),

              // --- Section Inférieure (Infos, Actions) ---
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.fromLTRB(16.0, 24.0, 16.0, 16.0), // Plus de padding en haut
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Carte d'informations du Tick
                      _buildTickInfoCard(),
                      const SizedBox(height: 24),

                      // Boutons d'action principaux
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          ActionButton( // Utiliser le widget personnalisé
                            icon: Icons.location_searching,
                            label: AppTexts.locate,
                            onPressed: _requestLocationUpdate,
                            isLoading: _isLocateActionLoading, // NOUVEAU
                            isDisabled: _isLocateActionLoading || _isRingActionLoading, // Désactiver si une action est en cours
                            color: AppTheme.accentColor, // Vert pour localiser
                          ),
                          ActionButton(
                            icon: Icons.volume_up_outlined,
                            label: AppTexts.ring,
                            onPressed: _ringTick,
                            isLoading: _isRingActionLoading, // NOUVEAU
                            isDisabled: _isLocateActionLoading || _isRingActionLoading, // Désactiver si une action est en cours
                            color: AppTheme.warningColor, // Orange pour sonner
                          ),
                           ActionButton(
                            icon: Icons.history_outlined,
                            label: AppTexts.history_map_page,
                            onPressed: _navigateToHistory,
                            isDisabled: _isLocateActionLoading || _isRingActionLoading, // Désactiver si une action est en cours
                          ),
                        ],
                      ),

                      // On pourrait ajouter l'historique récent ici aussi, mais la page dédiée est préférable
                      // const SizedBox(height: 24),
                      // Text(AppTexts.alerts, style: Theme.of(context).textTheme.titleLarge), ...
                    ],
                  ),
                ),
              ),
            ],
          ),
        );
      }

      /// Construit la carte affichant les informations détaillées du Tick.
      Widget _buildTickInfoCard() {
        return Card(
          // Utilise la CardTheme globale
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                _buildInfoRow(
                  icon: Icons.info_outline,
                  label: AppTexts.currentStatus,
                  value: _currentTickData.statusDescription,
                  valueColor: AppColors.getStatusColor(_currentTickData.status, context),
                ),
                const Divider(height: 16, thickness: 0.5), // Séparateur plus fin
                _buildInfoRow(
                  icon: Icons.location_on_outlined,
                  label: AppTexts.lastPosition,
                  value: _isFetchingAddress
                      ? 'Chargement adresse...' // Show loading text
                      : _tickAddress ?? // Use fetched address if available
                      (_currentTickData.latitude != null && _currentTickData.longitude != null
                          ? '${_currentTickData.latitude!.toStringAsFixed(5)}, ${_currentTickData.longitude!.toStringAsFixed(5)}' // Fallback to coordinates
                          : AppTexts.noLocationAvailable), // Fallback if no coordinates/address
                  // Optionally make the address text smaller if it tends to be long
                  // valueStyle: TextStyle(fontSize: 13), // Example
                ),
                const SizedBox(height: 8), // Pas de divider ici, juste un espace
                _buildInfoRow(
                  icon: Icons.access_time,
                  label: AppTexts.lastUpdate,
                  value: _currentTickData.formattedLastUpdate, // Utilise le getter formaté
                ),
                const SizedBox(height: 8),
                _buildInfoRow(
                  icon: _getBatteryIcon(_currentTickData.batteryLevel), // Icône dynamique
                  label: AppTexts.battery,
                  value: _currentTickData.batteryLevel != null
                      ? '${_currentTickData.batteryLevel}%'
                      : 'Inconnu',
                  valueColor: AppColors.getBatteryColor(_currentTickData.batteryLevel),
                ),
              ],
            ),
          ),
        );
      }

      /// Helper pour construire une ligne d'information (Icône - Label: Valeur).
      Widget _buildInfoRow({
        required IconData icon,
        required String label,
        required String value,
        Color? valueColor,
        TextStyle? valueStyle,
      }) {
        final textTheme = Theme.of(context).textTheme;
        final secondaryColor = textTheme.bodySmall?.color;

        // --- Logique pour le statut (inchangée) ---
        if (label == AppTexts.currentStatus) {
          final currentStatus = _currentTickData.status;
          if (currentStatus == TickStatus.disabled) {
            value = 'Désactivé';
            valueColor = AppTheme.errorColor;
          } else if (currentStatus == TickStatus.active) {
            value = 'Actif';
            valueColor = AppTheme.successColor;
          } else {
            value = _currentTickData.statusDescription;
            valueColor = AppColors.getStatusColor(currentStatus, context);
          }
        }
        // --- Fin logique statut ---

        // Widget affichant la valeur initiale (le Text)
        Widget valueWidget = Text(
          value,
          textAlign: TextAlign.right,
          style: valueStyle ??
              textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w500,
                color: valueColor ?? textTheme.bodyMedium?.color,
              ),
          overflow: TextOverflow.ellipsis, // Garder l'ellipse
          maxLines: 1,
        );

        // --- REMPLACEMENT TOOLTIP PAR GESTUREDETECTOR ---
        // Envelopper le Text de l'adresse avec GestureDetector si c'est la bonne ligne
        if (label == AppTexts.lastPosition && value != AppTexts.noLocationAvailable && value != 'Chargement adresse...')
        {
          // Créer une copie locale de 'value' pour l'utiliser dans le callback
          final String addressValue = value;
          valueWidget = GestureDetector(
            // Détecter le début du long press pour obtenir la position
            onLongPressStart: (details) {
              print("Long press detected on address: $addressValue");
              _showAddressActionsMenu(context, details, addressValue); // Appeler la fonction du menu
            },
            child: valueWidget, // L'enfant est le Text original
          );
        }
        // --- FIN REMPLACEMENT ---

        // Retourner la Row en utilisant la variable valueWidget
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 4.0),
          child: Row(
            children: [
              Icon(icon, size: 18, color: secondaryColor),
              const SizedBox(width: 12),
              Text('$label:', style: textTheme.bodyMedium?.copyWith(color: secondaryColor)),
              const SizedBox(width: 8),
              Expanded(
                child: valueWidget, // valueWidget est maintenant soit Text, soit GestureDetector(Text)
              ),
            ],
          ),
        );
      }

      /// Retourne l'icône de batterie appropriée en fonction du niveau.
      IconData _getBatteryIcon(int? level) {
        if (level == null) return Icons.battery_unknown_outlined;
        if (level > 95) return Icons.battery_full_outlined;
        if (level > 80) return Icons.battery_6_bar_outlined;
        if (level > 60) return Icons.battery_5_bar_outlined;
        if (level > 40) return Icons.battery_3_bar_outlined;
        if (level > 20) return Icons.battery_1_bar_outlined;
        return Icons.battery_alert_outlined; // <= 20%
      }
    }
    ```

**II. AWS Lambda Implementation (Python 3.x)**

1.  **Create the Lambda Function:**
    *   Go to the AWS Lambda console.
    *   Click "Create function".
    *   Choose "Author from scratch".
    *   Function name: e.g., `ringTickFunction`.
    *   Runtime: Python 3.11 (or your preferred Python 3 version).
    *   Architecture: `x86_64`.
    *   Permissions: Choose "Create a new role with basic Lambda permissions". We'll modify this role later.
    *   Click "Create function".

2.  **Add Permissions to the IAM Role:**
    *   Go to the IAM console -> Roles.
    *   Find the role created for your Lambda function (e.g., `ringTickFunction-role-xxxxx`).
    *   Click "Add permissions" -> "Attach policies".
    *   Search for and attach the `AWSIoTDataAccess` policy (provides `iot:Publish`).
    *   Click "Add permissions" -> "Create inline policy".
    *   Switch to the JSON tab and paste the following policy, replacing `YOUR_REGION`, `YOUR_ACCOUNT_ID`, and `YOUR_DYNAMODB_TABLE_NAME` with your actual values:

        ```json
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "dynamodb:GetItem",
                    "Resource": "arn:aws:dynamodb:YOUR_REGION:YOUR_ACCOUNT_ID:table/YOUR_DYNAMODB_TABLE_NAME"
                }
            ]
        }
        ```
    *   Click "Review policy", give it a name (e.g., `DynamoDBGetTickItemPolicy`), and click "Create policy".

3.  **Add Function URL:**
    *   Go back to your Lambda function configuration page.
    *   Go to the "Configuration" tab -> "Function URL".
    *   Click "Create function URL".
    *   Auth type: `NONE` (we handle auth via JWT in the code).
    *   Check "Configure cross-origin resource sharing (CORS)".
        *   Allow origin: `*` (or your specific frontend domain for production).
        *   Allow methods: `POST`, `OPTIONS`.
        *   Allow headers: `Content-Type`, `Authorization`, `X-Amz-Date`, `X-Api-Key`, `X-Amz-Security-Token`.
        *   Expose headers: `Content-Type`, `Authorization`.
        *   Max age: `600`.
        *   Allow credentials: Leave unchecked unless needed.
    *   Click "Save".
    *   Copy the generated Function URL. **Paste this URL into `ApiConfigURLs.ringTickFunctionUrl` in your `constants.dart` file.**

4.  **Lambda Code (`lambda_function.py`):** Replace the default code with the following:

    ```python
    import json
    import boto3
    import os
    import base64
    from decimal import Decimal # Juste pour create_response
    from botocore.exceptions import ClientError

    # Config DynamoDB
    # Assurez-vous que cette variable d'environnement est définie dans la config Lambda
    # ou remplacez par le nom de table en dur si nécessaire.
    TABLE_NAME = os.environ.get('TICKS_TABLE_NAME', 'Ticks')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(TABLE_NAME)

    # Config IoT Core
    # Assurez-vous que la région est correcte
    iot_client = boto3.client('iot-data', region_name=os.environ.get('AWS_REGION', 'eu-north-1'))

    # Helper JSON (inchangé)
    class DecimalEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Decimal):
                # Convertir en int si c'est un entier, sinon float
                return int(obj) if obj % 1 == 0 else float(obj)
            return super(DecimalEncoder, self).default(obj)

    # --- Fonction Principale ---
    def lambda_handler(event, context):
        print("Received event for ringTickFunction:", json.dumps(event, indent=2))

        # --- 1. Extraire User ID (sub) du Token JWT ---
        user_id = None
        try:
            # Chercher l'en-tête d'autorisation (insensible à la casse)
            auth_header = event.get('headers', {}).get('authorization', event.get('headers', {}).get('Authorization'))
            if not auth_header or not auth_header.startswith('Bearer '):
                print("Missing or invalid Authorization header")
                return create_response(401, {'error': 'Unauthorized - Missing Token'})

            token = auth_header.split(' ')[1]
            # Décoder le payload JWT (sans vérification de signature ici, Cognito le fait)
            payload_encoded = token.split('.')[1]
            # Ajouter padding si nécessaire pour base64
            payload_encoded += '=' * (-len(payload_encoded) % 4)
            decoded_payload = base64.b64decode(payload_encoded).decode('utf-8')
            token_claims = json.loads(decoded_payload)

            user_id = token_claims.get('sub')
            if not user_id:
                print("Could not extract 'sub' (user ID) from token payload")
                return create_response(401, {'error': 'Unauthorized - Invalid Token Payload'})
            print(f"Authenticated user ID (sub): {user_id}")
        except Exception as e:
            print(f"Error extracting user ID from token: {e}")
            return create_response(401, {'error': f'Unauthorized - Token Error: {str(e)}'})

        # --- 2. Extraire tickId et soundIndex du corps ---
        tick_id_to_ring = None
        sound_index = None
        try:
            if isinstance(event.get('body'), str):
                body = json.loads(event['body'])
            else:
                body = event.get('body', {}) # Accepter un dict si déjà parsé (ex: test console)

            tick_id_to_ring = body.get('tickId')
            sound_index_raw = body.get('soundIndex') # Peut être int ou string

            if not tick_id_to_ring:
                print("Missing 'tickId' in request body")
                return create_response(400, {'error': 'Bad Request - Missing required field: tickId'})

            if sound_index_raw is None:
                print("Missing 'soundIndex' in request body")
                return create_response(400, {'error': 'Bad Request - Missing required field: soundIndex'})

            # Valider soundIndex (doit être un entier positif)
            try:
                sound_index = int(sound_index_raw)
                if sound_index <= 0:
                    raise ValueError("Sound index must be positive")
            except (ValueError, TypeError):
                 print(f"Invalid 'soundIndex' value: {sound_index_raw}. Must be a positive integer.")
                 return create_response(400, {'error': 'Bad Request - Invalid soundIndex value. Must be a positive integer.'})

            print(f"Processing ring request for Tick ID: '{tick_id_to_ring}' (Sound: {sound_index}) by User='{user_id}'")

        except json.JSONDecodeError:
             print("Invalid JSON in request body")
             return create_response(400, {'error': 'Bad Request - Invalid JSON body'})
        except Exception as e:
             print(f"Error parsing request body: {e}")
             return create_response(400, {'error': f'Bad Request - Body Error: {str(e)}'})

        # --- 3. Vérifier la propriété du Tick ---
        # Nom de la clé primaire et de l'attribut user ID dans DynamoDB
        partition_key_name = 'tick_id'
        owner_id_key_name = 'userID'

        try:
            print(f"Verifying ownership for Tick '{tick_id_to_ring}'...")
            response = table.get_item(Key={partition_key_name: tick_id_to_ring})
            item = response.get('Item')

            if not item:
                print(f"Tick '{tick_id_to_ring}' not found.")
                return create_response(404, {'error': f'Tick not found: {tick_id_to_ring}'})

            owner_id_in_db = item.get(owner_id_key_name)
            if not owner_id_in_db:
                 print(f"ERROR: Owner ID key '{owner_id_key_name}' not found in item for Tick {tick_id_to_ring}.")
                 # Erreur de données, ne devrait pas arriver
                 return create_response(500, {'error': 'Internal Server Error - Invalid Tick data (missing owner)'})

            if owner_id_in_db != user_id:
                print(f"ACCESS DENIED: User '{user_id}' tried to ring Tick '{tick_id_to_ring}' owned by '{owner_id_in_db}'.")
                return create_response(403, {'error': 'Forbidden - You do not own this Tick.'})

            print("Ownership verified.")

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            print(f"DynamoDB GetItem ClientError during ownership check: {error_code} - {error_message}")
            return create_response(500, {'error': 'Database error during verification'})
        except Exception as e:
            print(f"Unexpected error during ownership check: {e}")
            return create_response(500, {'error': 'Internal server error during verification'})

        # --- 4. Construire et Publier le message MQTT ---
        try:
            mqtt_topic = f"Tick/{tick_id_to_ring}" # Topic racine du Tick

            # Construire le payload EXACTEMENT comme spécifié
            mqtt_payload_dict = {
                "id": tick_id_to_ring,
                "type": "alarm",
                "value": sound_index # Utiliser l'entier validé
            }
            mqtt_payload = json.dumps(mqtt_payload_dict)

            print(f"Publishing to MQTT topic '{mqtt_topic}' with payload: {mqtt_payload}")

            # Publier le message
            iot_response = iot_client.publish(
                topic=mqtt_topic,
                qos=1, # Quality of Service 1 (At least once)
                payload=mqtt_payload.encode('utf-8')
            )

            print(f"IoT Publish successful. Response metadata: {iot_response.get('ResponseMetadata')}")
            print(f"Successfully sent ring command (Sound {sound_index}) to Tick '{tick_id_to_ring}'.")

            # Retourner succès (202 Accepted car l'action est asynchrone)
            return create_response(202, {'message': f'Ring command (Sound {sound_index}) sent to Tick {tick_id_to_ring}.'})

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            print(f"IoT Publish ClientError: {error_code} - {error_message}")
            if error_code == 'ForbiddenException':
                 # Erreur de permission IAM pour la Lambda
                 return create_response(500, {'error': f"Internal Server Error - Lambda lacks permission to publish to IoT topic '{mqtt_topic}': {error_message}"})
            else:
                 return create_response(500, {'error': f"IoT error during publish: {error_message}"})
        except Exception as e:
            print(f"Unexpected error during MQTT publish: {e}")
            return create_response(500, {'error': f'Internal server error during MQTT publish: {str(e)}'})

    # --- Fonction Helper ---
    def create_response(status_code, body):
        """Crée une réponse HTTP formatée avec CORS."""
        return {
            'statusCode': status_code,
            'headers': {
                'Access-Control-Allow-Origin': '*', # Permettre toutes origines (ajuster pour prod)
                'Access-Control-Allow-Headers': 'Content-Type,Authorization,X-Amz-Date,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'POST,OPTIONS', # Méthodes permises
                'Content-Type': 'application/json'
            },
            'body': json.dumps(body, cls=DecimalEncoder) # Encodage final
        }

    ```

5.  **Deploy the Lambda:** Click the "Deploy" button in the Lambda code editor.

**III. Final Steps & Testing**

1.  **Replace Placeholder URL:** Make sure you replaced `'https://YOUR_NEW_RING_TICK_FUNCTION_URL...'` in `lib/utils/constants.dart` with the actual Function URL you copied.
2.  **Run the App:** Launch your Flutter application.
3.  **Test Sound Selection:**
    *   Go to a Tick's settings page.
    *   Tap on the "Sonnerie d'alarme" tile.
    *   The dialog should appear.
    *   Test the "Écouter" button for each sound. Check your console/device logs for audio player output or errors.
    *   Select a sound. A confirmation SnackBar should appear.
    *   Re-open the dialog to verify the selection is saved.
4.  **Test Ringing:**
    *   Go to the map page for the Tick.
    *   Tap the "Faire sonner" button.
    *   Observe the app:
        *   A loading indicator should appear briefly on the button.
        *   A SnackBar "Commande de sonnerie envoyée." should appear on success.
        *   An error SnackBar should appear on failure.
    *   Observe the AWS CloudWatch Logs for your `ringTickFunction` Lambda:
        *   Check for successful execution or any errors (authentication, DynamoDB lookup, IoT publish).
    *   Observe your Tick device: It should receive the MQTT message and play the selected sound file (`value.wav`).

This comprehensive guide should get the "Ring Tick" feature working in your application. Remember to adapt file paths, URLs, and potentially error messages based on your specific setup.
