Okay, here is the remainder of the `lib\services\tick_service.dart` file, including the helper methods and the `dispose` method.

--- START OF MODIFIED FILE lib\services\tick_service.dart ---

```dart
import 'dart:async'; // Pour StreamSubscription si écoute MQTT directe
import 'package:flutter/foundation.dart'; // Pour ChangeNotifier

import '../models/tick_model.dart';
import 'api_service.dart'; // Service pour appeler les Lambdas
import 'auth_service.dart'; // Pour vérifier l'authentification et obtenir l'ID user
// import 'mqtt_service.dart'; // Décommenter si l'app écoute MQTT directement
import '../utils/constants.dart'; // Pour URLs et ErrorMessages
// import 'package:collection/collection.dart'; // Pour firstWhereOrNull (alternative)

// ignore_for_file: avoid_print

/// Service gérant les données et les opérations liées aux appareils Tick.
/// Interagit avec [ApiService] pour communiquer avec le backend (Lambdas).
class TickService with ChangeNotifier {
  final ApiService _apiService;
  AuthService _authService; // Référence au service d'authentification
  // final MQTTService? _mqttService; // Optionnel: si l'app doit écouter MQTT

  List<Tick> _ticks = []; // Liste locale des Ticks de l'utilisateur
  bool _isLoading = false; // Indicateur de chargement global pour le service
  String? _error; // Dernier message d'erreur

  // StreamSubscription? _mqttSubscription; // Pour écouter les messages MQTT si nécessaire

  // Getters publics
  List<Tick> get ticks => List.unmodifiable(_ticks); // Copie immuable pour l'UI
  bool get isLoading => _isLoading;
  String? get error => _error;

  /// Constructeur: Initialise avec les services requis et configure les listeners.
  TickService(this._apiService, this._authService /*, {this._mqttService} */) {
    print("TickService: Initializing...");
    // Écouter les changements d'état d'authentification
    _authService.addListener(_handleAuthChange);
    // Charger les ticks initiaux si l'utilisateur est déjà connecté au démarrage
    if (_authService.isAuthenticated) {
      fetchTicks();
    }
    // S'abonner aux messages MQTT si le service est fourni et que l'app doit écouter
    // _setupMqttListener();
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
        // S'assurer que l'écoute MQTT est active (si utilisée)
        // _setupMqttListener();
      }
    } else {
      // Si l'utilisateur est déconnecté, effacer les données locales
      if (_ticks.isNotEmpty || _error != null || _isLoading) {
        print("TickService: Clearing local tick data due to logout.");
        _ticks = [];
        _error = null;
        _isLoading = false; // Stopper tout chargement en cours
        // Se désabonner des messages MQTT (si applicable)
        // _cancelMqttListener();
        notifyListeners(); // Notifier l'UI de l'effacement
      }
    }
  }

  /* --- Optionnel: Gestion MQTT directe dans l'app ---
  void _setupMqttListener() {
    if (_mqttService == null || _mqttSubscription != null) return; // Déjà écouté ou service non dispo
    if (!_authService.isAuthenticated) return; // Ne pas écouter si non connecté

    _mqttSubscription = _mqttService!.messageStream.listen((message) {
      print("TickService: Received MQTT message: ${message['topic']}");
      final payload = message['payload'];
      final topic = message['topic'] as String;

      // Exemple: mise à jour d'un Tick basée sur un message MQTT
      // Le topic pourrait être 'ticks/{tickId}/update'
      if (payload is Map<String, dynamic> && payload.containsKey('id')) {
         final tickId = payload['id'] as String;
         // Essayer de créer un Tick à partir du payload
         try {
            final updatedTick = Tick.fromJson(payload);
            updateTickDataLocally(updatedTick);
         } catch (e) {
            print("TickService: Error parsing MQTT payload for tick $tickId: $e");
         }
      }
    }, onError: (error) {
      print("TickService: Error in MQTT stream: $error");
      // Gérer l'erreur de stream MQTT
    });

    // S'abonner aux sujets pertinents (peut-être spécifique à l'utilisateur?)
    // Exemple: _mqttService!.subscribe('ticks/${_authService.currentUser?.uid}/#');
    print("TickService: MQTT Listener setup complete.");
  }

  void _cancelMqttListener() {
    if (_mqttSubscription != null) {
      print("TickService: Cancelling MQTT Listener.");
      _mqttSubscription!.cancel();
      _mqttSubscription = null;
      // Se désabonner des sujets MQTT si nécessaire
      // _mqttService?.unsubscribe('ticks/${_authService.currentUser?.uid}/#');
    }
  }
  --- Fin Optionnel MQTT --- */


  // --- Méthodes Publiques pour l'UI ---

  /// Récupère la liste des Ticks de l'utilisateur depuis le backend.
  Future<void> fetchTicks() async {
    // Vérifier si l'utilisateur est connecté et si une opération n'est pas déjà en cours
    if (!_checkAuthAndLoading()) return;

    _setLoading(true);
    _clearError(); // Effacer l'erreur précédente

    print("TickService: Fetching ticks from URL: ${ApiConfig.getMyTicksFunctionUrl}");

    try {
      // Appel API via ApiService
      final response = await _apiService.get(ApiConfig.getMyTicksFunctionUrl);

      if (response['success']) {
        final List<dynamic>? tickDataList = response['data'] as List<dynamic>?;
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

      print("TickService: Associating tick via URL: ${ApiConfig.associateTickFunctionUrl}");
      final response = await _apiService.post(ApiConfig.associateTickFunctionUrl, body);

      if (response['success']) {
        print("TickService: Tick associated successfully via Lambda. Response: ${response['data']}");
        // Recharger la liste des Ticks pour inclure le nouveau
        await fetchTicks(); // fetchTicks gèrera setLoading(false)
        return true;
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
    // Le finally de fetchTicks gère le setLoading(false) en cas de succès
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
      if (ApiConfig.getTickHistoryFunctionUrl.isEmpty) throw Exception("Get Tick History URL not configured");
       final urlWithParam = Uri.parse(ApiConfig.getTickHistoryFunctionUrl).replace(
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
      if (ApiConfig.requestLocationFunctionUrl.isEmpty) throw Exception("Request Location URL not configured");
      final body = {'tickId': tickId};
      print("TickService: Requesting location for $tickId via URL: ${ApiConfig.requestLocationFunctionUrl}");
      final response = await _apiService.post(ApiConfig.requestLocationFunctionUrl, body);

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

  /// Demande à faire sonner un Tick.
  Future<bool> ringTick(String tickId) async {
    if (!_checkAuthAndLoading()) return false;
    _setLoading(true);
    _clearError();

    try {
       if (ApiConfig.ringTickFunctionUrl.isEmpty) throw Exception("Ring Tick URL not configured");
       final body = {'tickId': tickId, /* 'soundType': 'standard' */}; // Ajouter type de son si besoin
       print("TickService: Ringing tick $tickId via URL: ${ApiConfig.ringTickFunctionUrl}");
       // Appel API Réel (à décommenter et adapter)
       // final response = await _apiService.post(ApiConfig.ringTickFunctionUrl, body);
       await Future.delayed(AppDurations.mediumDelay); // Simulation
       final response = {'success': true, 'data': 'Ring command sent'}; // Placeholder

       if (response['success']) {
          print("TickService: Ring command sent for $tickId. Response: ${response['data']}");
          _setLoading(false);
          return true;
       } else {
          _setError(response['error'] ?? "Erreur lors de la commande Sonnerie");
          _setLoading(false);
          return false;
       }
    } catch (e) {
      print("TickService: Exception ringing tick $tickId: $e");
      _setError(e is Exception ? e.toString() : ErrorMessages.connectionFailed);
      _setLoading(false);
      return false;
    }
  }

  /// Met à jour les paramètres d'un Tick (actuellement seulement le nom).
  Future<bool> updateTickSettings(String tickId, {required String name}) async {
    if (!_checkAuthAndLoading()) return false;
    if (name.trim().isEmpty) {
       _setError("Le nom ne peut pas être vide.");
       notifyListeners(); // Notifier l'erreur immédiatement
       return false;
    }
    _setLoading(true);
    _clearError();

    try {
      if (ApiConfig.updateTickSettingsFunctionUrl.isEmpty) throw Exception("Update Tick Settings URL not configured");
      final body = {
         'tickId': tickId,
         'name': name.trim(),
         // Ajouter d'autres champs si nécessaire (ex: 'icon', 'ringtone')
      };
      print("TickService: Updating settings for $tickId via URL: ${ApiConfig.updateTickSettingsFunctionUrl}");

      // Appel API Réel (PUT ou POST) - (à décommenter et adapter)
      // final response = await _apiService.put(ApiConfig.updateTickSettingsFunctionUrl, body);
      await Future.delayed(AppDurations.shortDelay); // Simulation
      final response = {'success': true, 'data': {'id': tickId, 'name': name.trim()}}; // Placeholder

      if (response['success']) {
        print("TickService: Tick settings updated for $tickId. Response: ${response['data']}");
        // Mettre à jour l'objet Tick dans la liste locale pour réactivité UI
        final index = _ticks.indexWhere((t) => t.id == tickId);
        if (index != -1) {
          // Utiliser copyWith pour mettre à jour seulement le nom
          _ticks[index] = _ticks[index].copyWith(name: name.trim());
          notifyListeners(); // Notifier l'UI du changement local
        } else {
           // Si le Tick n'était pas dans la liste (étrange), recharger tout
           print("TickService: Updated tick $tickId not found locally, fetching all ticks.");
           await fetchTicks();
        }
        _setLoading(false); // Fin du chargement (notifyListeners déjà fait ou sera fait par fetchTicks)
        return true;
      } else {
        _setError(response['error'] ?? AppTexts.updateError);
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
  Future<bool> unlinkTick(String tickId) async {
    if (!_checkAuthAndLoading()) return false;
    _setLoading(true);
    _clearError();

    try {
      if (ApiConfig.removeTickFunctionUrl.isEmpty) throw Exception("Remove Tick URL not configured");
      // Déterminer si l'ID est passé en query param ou body selon l'API
      // Exemple avec body (POST ou DELETE qui accepte un body)
      final body = {'tickId': tickId};
      print("TickService: Unlinking tick $tickId via URL: ${ApiConfig.removeTickFunctionUrl}");
      // Appel API Réel (DELETE ou POST) - (à décommenter et adapter)
      // final response = await _apiService.post(ApiConfig.removeTickFunctionUrl, body); // Ou .delete si API supporte body sur DELETE
       await Future.delayed(AppDurations.mediumDelay); // Simulation
      final response = {'success': true, 'data': 'Tick unlinked'}; // Placeholder


      if (response['success']) {
        print("TickService: Tick unlinked successfully: $tickId");
        // Supprimer le Tick de la liste locale immédiatement
        final removedCount = _ticks.removeWhere((tick) => tick.id == tickId);
        if (removedCount > 0) {
           notifyListeners(); // Notifier l'UI de la suppression
        } else {
            print("TickService: Unlinked tick $tickId was not found in local list.");
        }
        _setLoading(false);
        return true;
      } else {
        _setError(response['error'] ?? "Erreur de désassociation");
        _setLoading(false);
        return false;
      }
    } catch (e) {
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
    // _cancelMqttListener(); // Si MQTT est utilisé
    super.dispose();
  }
}
```

--- END OF MODIFIED FILE lib\services\tick_service.dart ---

--- START OF MODIFIED FILE lib\services\theme_service.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart'; // Pour obtenir platformBrightness
// import 'package:shared_preferences/shared_preferences.dart'; // Pour persistance (optionnel)

/// Service pour gérer le thème de l'application (Clair, Sombre, Système).
class ThemeService with ChangeNotifier {
  // Clé pour la persistance (si utilisée)
  // static const String _themePrefKey = 'app_theme_mode';

  ThemeMode _themeMode = ThemeMode.system; // Thème par défaut

  // Getter public pour le mode actuel
  ThemeMode get themeMode => _themeMode;

  /// Constructeur: peut charger le thème persisté au démarrage.
  ThemeService() {
    // Optionnel: Charger le thème depuis SharedPreferences
    // _loadThemePreference();
  }

  /* --- Optionnel: Persistance avec SharedPreferences ---
  Future<void> _loadThemePreference() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final themeIndex = prefs.getInt(_themePrefKey);
      if (themeIndex != null && themeIndex >= 0 && themeIndex < ThemeMode.values.length) {
        _themeMode = ThemeMode.values[themeIndex];
        print("ThemeService: Loaded theme preference: $_themeMode");
        notifyListeners(); // Notifier si chargé depuis les préférences
      }
    } catch (e) {
       print("ThemeService: Error loading theme preference: $e");
    }
  }

  Future<void> _saveThemePreference(ThemeMode mode) async {
     try {
       final prefs = await SharedPreferences.getInstance();
       await prefs.setInt(_themePrefKey, mode.index);
       print("ThemeService: Saved theme preference: $mode");
     } catch (e) {
        print("ThemeService: Error saving theme preference: $e");
     }
  }
  --- Fin Optionnel --- */

  /// Vérifie si le mode sombre est actuellement actif, en tenant compte du mode système.
  bool isDarkMode(BuildContext context) {
    if (_themeMode == ThemeMode.system) {
      // Utiliser platformBrightnessOf pour obtenir la luminosité système actuelle
      // Cela est plus fiable que MediaQuery pendant certaines phases du build.
      var brightness = SchedulerBinding.instance.platformDispatcher.platformBrightness;
      // var brightness = MediaQuery.platformBrightnessOf(context); // Alternative
      return brightness == Brightness.dark;
    }
    // Si le mode est explicitement Dark
    return _themeMode == ThemeMode.dark;
  }

  /// Met à jour le mode de thème de l'application.
  void setThemeMode(ThemeMode mode) {
    if (_themeMode == mode) return; // Pas de changement
    _themeMode = mode;
    print("ThemeService: Theme mode set to $_themeMode");
    // Optionnel: Sauvegarder la préférence
    // _saveThemePreference(mode);
    notifyListeners(); // Notifier l'UI du changement
  }

  /// Bascule entre les modes de thème disponibles (Clair <-> Sombre).
  /// Ignore le mode Système pour une bascule simple.
  void toggleThemeMode(BuildContext context) {
    // Détermine le mode actuel effectif (clair ou sombre)
    bool isCurrentlyDark = isDarkMode(context);
    // Bascule vers le mode opposé explicite
    setThemeMode(isCurrentlyDark ? ThemeMode.light : ThemeMode.dark);

    // --- Alternative pour un cycle Light -> Dark -> System -> Light ... ---
    // switch (_themeMode) {
    //   case ThemeMode.light:
    //     setThemeMode(ThemeMode.dark);
    //     break;
    //   case ThemeMode.dark:
    //     setThemeMode(ThemeMode.system);
    //     break;
    //   case ThemeMode.system:
    //   default:
    //     setThemeMode(ThemeMode.light);
    //     break;
    // }
    // --- Fin Alternative ---
  }

  /// Retourne le nom lisible du mode de thème actuel.
  String getThemeModeName() {
    switch (_themeMode) {
      case ThemeMode.light:
        return 'Clair';
      case ThemeMode.dark:
        return 'Sombre';
      case ThemeMode.system:
      default:
        return 'Système';
    }
  }
}

```

--- END OF MODIFIED FILE lib\services\theme_service.dart ---

--- START OF MODIFIED FILE lib\services\bluetooth_service.dart ---

```dart
import 'dart:async';
import 'dart:io' show Platform; // Import explicite pour Platform
import 'package:flutter/foundation.dart'; // Pour kIsWeb et ChangeNotifier
import 'package:flutter_blue_plus/flutter_blue_plus.dart' as blue; // Alias pour clarté
import 'package:permission_handler/permission_handler.dart' as ph; // Alias pour clarté
import 'package:location/location.dart' as loc; // Alias pour clarté

import '../utils/constants.dart'; // Pour BluetoothConfig, ErrorMessages

// Alias pour les types d'état et de permission pour améliorer la lisibilité
typedef BluetoothState = blue.BluetoothAdapterState;
typedef LocationPermissionStatus = loc.PermissionStatus;
typedef HandlerPermissionStatus = ph.PermissionStatus;

/// Service gérant la logique Bluetooth Low Energy (BLE), y compris l'état de l'adaptateur,
/// la gestion des permissions et le scan des appareils Tick.
class BluetoothService with ChangeNotifier {
  // Service de localisation (utilisé pour vérifier/demander service et permission)
  final loc.Location _location = loc.Location();

  // État interne du service
  BluetoothState _state = BluetoothState.unknown;
  bool _isScanning = false;
  bool _isInitialized = false;
  String? _scanError; // Stocke la dernière erreur liée au scan ou aux permissions
  blue.ScanResult? _foundTickResult; // Dernier résultat de scan correspondant à un Tick
  String? _extractedTickId; // ID extrait du nom du Tick trouvé

  // Abonnements aux streams FlutterBluePlus (gérés en interne)
  StreamSubscription<BluetoothState>? _adapterStateSubscription;
  StreamSubscription<List<blue.ScanResult>>? _scanResultsSubscription;

  // Getters publics pour accéder à l'état
  BluetoothState get state => _state;
  bool get isScanning => _isScanning;
  bool get isInitialized => _isInitialized;
  String? get scanError => _scanError;
  blue.ScanResult? get foundTickResult => _foundTickResult;
  String? get extractedTickId => _extractedTickId;
  /// Indique si le BLE est supporté sur la plateforme actuelle.
  bool get isBleSupported => !kIsWeb && (Platform.isAndroid || Platform.isIOS || Platform.isMacOS);

  // --- Initialisation ---

  BluetoothService() {
    // L'initialisation réelle est déclenchée après la création via Provider
  }

  /// Initialise le service Bluetooth. Doit être appelé une fois.
  /// Configure l'écoute de l'état de l'adaptateur et met à jour l'état initial.
  Future<void> initialize() async {
    // Éviter ré-initialisation ou initialisation sur plateforme non supportée
    if (_isInitialized) return;
    if (!isBleSupported) {
      print("BluetoothService: BLE not supported. Skipping initialization.");
      _state = BluetoothState.unavailable;
      _isInitialized = true;
      notifyListeners(); // Notifier l'état 'unavailable'
      return;
    }
    print("BluetoothService: Initializing...");

    // Écouter les changements d'état de l'adaptateur Bluetooth
    _adapterStateSubscription = blue.FlutterBluePlus.adapterState.listen(
      _onAdapterStateChanged, // Callback défini ci-dessous
      onError: (e) => print("BluetoothService: Error listening to adapter state: $e"),
    );

    // Obtenir l'état initial de manière asynchrone
    await _updateInitialAdapterState();

    _isInitialized = true;
    print("BluetoothService: Initialization complete. Initial State: $_state");
  }

  /// Callback appelé lorsque l'état de l'adaptateur Bluetooth change.
  void _onAdapterStateChanged(BluetoothState newState) {
    print("BluetoothService: Adapter state changed to -> $newState");

    // Gérer le cas où les permissions sont refusées puis le BT éteint
    // On veut conserver l'état 'unauthorized' comme prioritaire.
    if (_state == BluetoothState.unauthorized && newState == BluetoothState.off) {
       print("BluetoothService: Keeping state as 'unauthorized' despite BT turning off.");
       // Ne pas écraser 'unauthorized'
    } else {
      _state = newState;
    }

    // Si le Bluetooth est éteint, arrêter toute opération en cours
    if (newState != BluetoothState.on) {
      if (_isScanning) {
        stopScan(); // Arrête le scan et met _isScanning à false
      }
       // Effacer les résultats précédents si l'adaptateur n'est plus prêt
      _resetScanResultState(notify: false); // Ne pas notifier ici, notifyListeners() est appelé à la fin
    }
    notifyListeners(); // Notifier l'UI du changement d'état
  }

  /// Met à jour l'_state interne avec le premier état stable de l'adaptateur.
  Future<void> _updateInitialAdapterState() async {
    try {
      // Attendre un état stable (ni unknown, ni turningOn/Off)
      _state = await _getFirstStableAdapterState();
      print("BluetoothService: Initial adapter state updated to: $_state");
      notifyListeners();
    } catch (e) {
      print("BluetoothService: Error getting initial adapter state: $e");
      _state = BluetoothState.unknown; // État inconnu en cas d'erreur
      notifyListeners();
    }
  }

  /// Attend et retourne le premier état stable de l'adaptateur Bluetooth.
  /// Évite de démarrer des opérations pendant que le BT s'allume/s'éteint.
  Future<BluetoothState> _getFirstStableAdapterState({Duration timeout = const Duration(seconds: 5)}) async {
    // Si l'état actuel est déjà stable, le retourner immédiatement
     final currentState = blue.FlutterBluePlus.adapterStateNow; // Lire état synchrone
     if (currentState != BluetoothState.unknown &&
         currentState != BluetoothState.turningOn &&
         currentState != BluetoothState.turningOff) {
        return currentState;
     }
     print("BluetoothService: Waiting for stable adapter state...");

    // Attendre la prochaine émission du stream qui correspond à un état stable
    try {
       return await blue.FlutterBluePlus.adapterState
           .where((state) => state != BluetoothState.unknown &&
                             state != BluetoothState.turningOn &&
                             state != BluetoothState.turningOff)
           .first // Prendre le premier état stable reçu
           .timeout(timeout); // Appliquer un timeout
    } on TimeoutException {
       print("BluetoothService: Timeout waiting for stable adapter state. Returning current: $currentState");
       return currentState; // Retourner l'état actuel (potentiellement instable) après timeout
    } catch (e) {
       print("BluetoothService: Error waiting for stable adapter state: $e. Returning current: $currentState");
        return currentState; // Retourner l'état actuel en cas d'erreur
    }
  }


  /// Tente d'activer le Bluetooth via l'API système (Android uniquement).
  /// Retourne `true` si la tentative a été faite, `false` sinon.
  Future<bool> attemptToEnableBluetooth() async {
    if (!isBleSupported || !Platform.isAndroid) {
      print("BluetoothService: attemptToEnableBluetooth only available on Android.");
      return false;
    }
    if (_state == BluetoothState.on) {
      print("BluetoothService: Bluetooth already ON.");
      return true; // Déjà allumé
    }

    print("BluetoothService: Attempting to turn Bluetooth ON via system API...");
    try {
      // Utilise FlutterBluePlus pour demander l'activation à l'OS
      await blue.FlutterBluePlus.turnOn();
      print("BluetoothService: System request to turn ON Bluetooth sent.");
      // Le listener _onAdapterStateChanged mettra à jour l'état réel.
      // On retourne true pour indiquer que la demande a été faite.
      return true;
    } catch (e) {
      // L'utilisateur a peut-être refusé la popup système
      print("BluetoothService: Failed to request Bluetooth turn ON: $e");
      _scanError = "Impossible d'activer le Bluetooth automatiquement."; // Message pour l'UI
      notifyListeners();
      return false;
    }
  }

  // --- Gestion des Permissions ---

  /// Vérifie et demande toutes les permissions requises pour le scan BLE.
  /// Met à jour `_scanError` et `_state` si des permissions sont manquantes.
  /// Retourne `true` si toutes les permissions sont accordées et le service de localisation activé.
  Future<bool> checkAndRequestRequiredPermissions() async {
    if (!isBleSupported) return true; // Pas de permissions si non supporté
    print("BluetoothService: Checking and requesting required permissions...");
    _clearScanError(); // Effacer l'erreur précédente

    // Vérifier les permissions en parallèle pour gagner du temps
    final results = await Future.wait([
      _checkAndRequestLocationPermission(), // Permission et service localisation
      _checkAndRequestBluetoothPermissions(), // Permissions spécifiques BLE (Android)
    ]);

    final bool locationOk = results[0];
    final bool bluetoothOk = results[1];
    final bool overallResult = locationOk && bluetoothOk;

    // Mettre à jour l'état global si les permissions ont échoué
    if (!overallResult) {
       print("BluetoothService: Permissions check failed (Location: $locationOk, Bluetooth: $bluetoothOk). Setting state to unauthorized.");
      _state = BluetoothState.unauthorized; // Marquer comme non autorisé
       // L'erreur spécifique (_scanError) a été définie dans les méthodes check
    } else if (_state == BluetoothState.unauthorized) {
      // Si les permissions sont maintenant OK mais l'état était 'unauthorized',
      // revérifier l'état matériel réel de l'adaptateur.
      print("BluetoothService: Permissions granted, re-checking adapter state...");
      await _updateInitialAdapterState();
    }

    print("BluetoothService: Permissions check complete. Overall result: $overallResult. Final State: $_state");
    notifyListeners(); // Notifier des changements potentiels (_state, _scanError)
    return overallResult;
  }

  /// Vérifie et demande la permission de localisation et l'activation du service.
  /// Utilise le package `location`.
  Future<bool> _checkAndRequestLocationPermission() async {
    // Pas nécessaire sur macOS pour le scan BLE standard a priori
    if (!Platform.isAndroid && !Platform.isIOS) return true;

    print("BluetoothService: Checking Location service & permission...");
    bool serviceEnabled;
    loc.PermissionStatus permissionGranted;

    // 1. Vérifier si le service de localisation est activé
    serviceEnabled = await _location.serviceEnabled();
    if (!serviceEnabled) {
      print("BluetoothService: Location service disabled. Requesting...");
      serviceEnabled = await _location.requestService();
      if (!serviceEnabled) {
        print("BluetoothService: Location service denied by user.");
        _scanError = ErrorMessages.locationServiceDisabled;
        return false; // Service requis
      }
      print("BluetoothService: Location service enabled by user.");
    }

    // 2. Vérifier la permission de localisation
    permissionGranted = await _location.hasPermission();
    print("BluetoothService: Initial location permission status: $permissionGranted");
    if (permissionGranted == loc.PermissionStatus.denied) {
      print("BluetoothService: Location permission denied. Requesting...");
      permissionGranted = await _location.requestPermission();
      print("BluetoothService: Location permission status after request: $permissionGranted");
    }

    // Vérifier si la permission est accordée (granted ou grantedLimited)
    if (permissionGranted == loc.PermissionStatus.granted ||
        permissionGranted == loc.PermissionStatus.grantedLimited) {
      print("BluetoothService: Location permission is sufficient.");
      return true;
    } else {
       print("BluetoothService: Location permission NOT granted.");
       // Gérer le refus définitif si le package le supporte bien (pas standardisé)
       // if (permissionGranted == loc.PermissionStatus.deniedForever) { ... }
       _scanError = ErrorMessages.permissionDeniedLocation;
       return false;
    }
  }

  /// Vérifie et demande les permissions Bluetooth spécifiques (Android 12+).
  /// Utilise le package `permission_handler`.
  Future<bool> _checkAndRequestBluetoothPermissions() async {
    // Nécessaire seulement sur Android
    if (!Platform.isAndroid) return true;

    print("BluetoothService: Checking Android specific BLE permissions...");
    // Permissions requises pour Android 12+ (API 31+)
    final List<ph.Permission> requiredPermissions = [
      ph.Permission.bluetoothScan,
      ph.Permission.bluetoothConnect,
      // ph.Permission.bluetoothAdvertise, // Si l'app doit aussi diffuser
    ];

    // Demander toutes les permissions requises en une fois
    Map<ph.Permission, HandlerPermissionStatus> statuses = await requiredPermissions.request();

    // Vérifier si toutes les permissions sont accordées
    bool allGranted = statuses.values.every((status) => status == HandlerPermissionStatus.granted);

    if (allGranted) {
      print("BluetoothService: All required Android BLE permissions granted.");
      return true;
    } else {
      print("BluetoothService: Not all required Android BLE permissions granted. Statuses: $statuses");
      // Déterminer un message d'erreur plus précis si possible
      if (statuses[ph.Permission.bluetoothScan] != HandlerPermissionStatus.granted) {
         _scanError = "La permission 'Appareils à proximité' (Scan) est requise.";
      } else if (statuses[ph.Permission.bluetoothConnect] != HandlerPermissionStatus.granted) {
          _scanError = "La permission 'Appareils à proximité' (Connexion) est requise.";
      } else {
          _scanError = ErrorMessages.permissionDeniedBluetooth; // Message générique
      }
      // L'état sera mis à 'unauthorized' par checkAndRequestRequiredPermissions si overallResult est false
      return false;
    }
  }

  // --- Logique de Scan ---

  /// Lance un scan BLE pour les appareils Tick, tente d'extraire leur ID, et s'arrête dès le premier trouvé.
  /// Gère les états de l'adaptateur, les permissions, et le timeout.
  /// Met à jour `_foundTickResult`, `_extractedTickId`, `_isScanning`, et `_scanError`.
  /// Retourne `true` si un Tick a été trouvé et son ID extrait, `false` sinon.
  Future<bool> startTickScanAndExtractId(
      {Duration timeout = const Duration(seconds: BluetoothConfig.scanTimeoutSeconds)}) async {
    print("BluetoothService: Attempting to start Tick scan...");
    _resetScanResultState(); // Effacer les résultats précédents
    _clearScanError(); // Effacer l'erreur précédente

    // Vérifier si l'adaptateur est prêt (permissions et état matériel)
    // Cette vérification est cruciale avant de démarrer le scan.
    if (state != BluetoothState.on) {
      _scanError = _scanError ?? "Bluetooth non prêt pour le scan (État: $state)";
      print("BluetoothService: Cannot scan, adapter state is not ON ($state). Error: $_scanError");
      notifyListeners();
      return false;
    }
    if (_isScanning) {
      print("BluetoothService: Scan already in progress.");
      return false; // Ne pas démarrer un nouveau scan si déjà en cours
    }
    if (!isBleSupported) {
      _scanError = ErrorMessages.bleNotSupported;
      print("BluetoothService: Cannot scan, BLE not supported.");
      notifyListeners();
      return false;
    }

    _isScanning = true;
    notifyListeners(); // Notifier l'UI que le scan commence
    print("BluetoothService: Scan started. Timeout: $timeout. Waiting for Tick...");

    final completer = Completer<bool>(); // Pour gérer la fin du scan (trouvé ou timeout)

    // S'abonner aux résultats du scan
    _scanResultsSubscription = blue.FlutterBluePlus.scanResults.listen(
      (results) {
        // Si déjà trouvé ou terminé, ignorer les nouveaux résultats
        if (completer.isCompleted) return;

        // Parcourir les résultats reçus
        for (blue.ScanResult result in results) {
          // Vérifier si l'appareil correspond à un Tick potentiel
          if (_isPotentialTickDevice(result)) {
            final extractedId = getTickIdFromName(result.device.platformName);
            if (extractedId != null) {
              // Tick trouvé et ID extrait !
              print("BluetoothService: >>> Tick Found & ID Extracted: ${result.device.remoteId} / ${result.device.platformName} -> ID: $extractedId <<<");
              _foundTickResult = result;
              _extractedTickId = extractedId;

              // Arrêter le scan immédiatement et compléter le future
              if (!completer.isCompleted) {
                 stopScan(); // Arrête le scan matériel et l'abonnement
                 completer.complete(true); // Indique succès
              }
              notifyListeners(); // Notifier l'UI que le Tick est trouvé
              return; // Sortir de la boucle et du listener
            }
          }
        }
      },
      onError: (error) { // Erreur pendant l'écoute du stream de scan
        print("BluetoothService: Scan Stream Error: $error");
        _scanError = "Erreur pendant le scan: $error";
        if (!completer.isCompleted) {
          stopScan();
          completer.complete(false); // Indique échec
        }
        notifyListeners();
      },
      // onDone n'est pas utilisé car on arrête le scan manuellement ou par timeout
    );

    // Démarrer le scan matériel FlutterBluePlus
    try {
      await blue.FlutterBluePlus.startScan(
        // Filtrer par UUID de service pour optimiser (si possible sur la plateforme)
        withServices: [blue.Guid(BluetoothConfig.tickServiceUuid)],
        timeout: timeout, // Timeout géré par FlutterBluePlus
        androidScanMode: blue.AndroidScanMode.lowLatency, // Mode de scan Android
      );

      // Attendre que le scan se termine (trouvé, timeout, ou erreur)
      final result = await completer.future.timeout(
          timeout + const Duration(seconds: 1), // Ajouter une marge au timeout global
          onTimeout: () {
            print("BluetoothService: Scan future timed out.");
            if (!completer.isCompleted) {
              stopScan(); // Assurer l'arrêt si timeout externe
              return false; // Timeout = non trouvé
            }
            return _extractedTickId != null; // Si trouvé juste avant le timeout
          });

      print("BluetoothService: Scan process finished. Found & ID Extracted: $result");
      // Si non trouvé et pas d'erreur spécifique, définir l'erreur "non trouvé"
      if (!result && _scanError == null) {
        _scanError = ErrorMessages.deviceNotFound;
      }
      // Assurer que _isScanning est false (stopScan devrait le faire)
      if (_isScanning) {
        _isScanning = false;
        notifyListeners();
      }
      return result;

    } catch (e, stacktrace) { // Capturer les erreurs lors du DÉMARRAGE du scan (ex: permissions)
      print("BluetoothService: Exception during startScan execution: $e\n$stacktrace");
      if (e.toString().toLowerCase().contains('permission')) {
        _scanError = ErrorMessages.permissionDeniedBluetooth;
        _state = BluetoothState.unauthorized; // Mettre à jour l'état si erreur de permission
      } else {
        _scanError = "Erreur système lors du démarrage du scan: ${e.toString()}";
      }
      if (!completer.isCompleted) completer.complete(false);
      await stopScan(); // Assurer l'arrêt et la notification
      return false;
    }
  }


  /// Arrête le scan BLE en cours.
  Future<void> stopScan() async {
    // Vérifier si supporté et si un scan est réellement en cours
    if (!isBleSupported || !_isScanning) return;

    print("BluetoothService: Stopping scan...");
    try {
      // Annuler l'abonnement aux résultats d'abord pour éviter de traiter de nouveaux résultats
      await _scanResultsSubscription?.cancel();
      _scanResultsSubscription = null;

      // Arrêter le scan matériel s'il est actif
      // Utiliser la propriété isScanning synchrone pour vérifier
      if (blue.FlutterBluePlus.isScanningNow) {
          await blue.FlutterBluePlus.stopScan();
          print("BluetoothService: Hardware scan stopped via FBP.");
      } else {
          print("BluetoothService: Hardware scan reported as already stopped.");
      }
    } catch (e) {
      // Une erreur ici est moins critique mais doit être loguée
      print('BluetoothService: Error stopping scan: $e');
      // On ne définit pas _scanError ici car ce n'est pas une erreur de scan utilisateur
    } finally {
      // Assurer la mise à jour de l'état interne et notifier l'UI
      // Mettre à jour _isScanning même si stopScan lève une exception
      if (_isScanning) {
         _isScanning = false;
         notifyListeners();
      }
    }
  }

  // --- Helpers & Cleanup ---

  /// Vérifie si un résultat de scan correspond potentiellement à un appareil Tick.
  /// Se base sur le préfixe du nom et l'UUID de service annoncé.
  bool _isPotentialTickDevice(blue.ScanResult result) {
    final deviceName = result.device.platformName;
    // Ignorer les appareils sans nom
    if (deviceName.isEmpty) return false;

    // 1. Vérifier si le nom commence par le préfixe attendu
    if (!deviceName.startsWith(BluetoothConfig.tickNamePrefix)) return false;

    // 2. Vérifier si l'UUID de service attendu est présent dans les données d'annonce
    // Convertir tous les UUID en minuscules pour une comparaison insensible à la casse
    final serviceUuids = result.advertisementData.serviceUuids
        .map((e) => e.toString().toLowerCase())
        .toList();
    final targetUuid = BluetoothConfig.tickServiceUuid.toLowerCase();

    // Debug log (peut être commenté en production)
    // print("Device: $deviceName, Services: $serviceUuids, Target: $targetUuid");
    return serviceUuids.contains(targetUuid);
  }

  /// Extrait l'ID unique du Tick à partir du nom de l'appareil BLE.
  /// Retourne `null` si le nom n'a pas le format attendu ("Tick-ID").
  String? getTickIdFromName(String? deviceName) {
    if (deviceName != null && deviceName.startsWith(BluetoothConfig.tickNamePrefix)) {
      // Retourne la partie après le préfixe
      return deviceName.substring(BluetoothConfig.tickNamePrefix.length);
    }
    return null; // Format invalide ou nom null
  }

  /// Réinitialise les informations liées au dernier scan réussi.
  void _resetScanResultState({bool notify = true}) {
    bool changed = _foundTickResult != null || _extractedTickId != null;
    _foundTickResult = null;
    _extractedTickId = null;
    if (changed && notify) {
      notifyListeners();
    }
  }

   /// Efface le message d'erreur de scan et notifie.
   void _clearScanError() {
      if (_scanError != null) {
         _scanError = null;
         notifyListeners();
      }
   }

  /// Libère les ressources (annule les abonnements aux streams).
  @override
  void dispose() {
    print("BluetoothService: Disposing...");
    _adapterStateSubscription?.cancel();
    stopScan(); // Assure l'arrêt du scan et l'annulation de _scanResultsSubscription
    super.dispose();
    print("BluetoothService: Disposed.");
  }
}

```

--- END OF MODIFIED FILE lib\services\bluetooth_service.dart ---

--- START OF MODIFIED FILE lib\main.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:intl/date_symbol_data_local.dart'; // Pour localisation dates fr_FR

// Amplify imports
import 'package:amplify_flutter/amplify_flutter.dart';
import 'package:amplify_auth_cognito/amplify_auth_cognito.dart';
import 'amplifyconfiguration.dart'; // Fichier généré par Amplify CLI

// Screens
import 'screens/splash_screen.dart';
import 'screens/welcome_page.dart';
import 'screens/auth/login_page.dart';
import 'screens/auth/register_page.dart';
import 'screens/auth/confirmation_page.dart';
import 'screens/auth/reset_password_page.dart';
import 'screens/tick/tick_list_page.dart';
import 'screens/tick/add_tick_page.dart';
import 'screens/tick/tick_settings_page.dart'; // Assurez-vous d'avoir une route ou un moyen d'y accéder
import 'screens/tick/map_page.dart'; // MapPage est généralement poussée sans route nommée
import 'screens/profile_page.dart';
import 'screens/settings_page.dart';

// Services
import 'services/auth_service.dart';
import 'services/tick_service.dart';
import 'services/theme_service.dart';
import 'services/bluetooth_service.dart';
import 'services/api_service.dart';

// Utils
import 'utils/theme.dart';
import 'utils/constants.dart';

// ignore_for_file: avoid_print

/// Point d'entrée principal de l'application.
void main() async {
  // Assurer l'initialisation des bindings Flutter avant toute autre opération
  WidgetsFlutterBinding.ensureInitialized();

  // Initialiser la localisation pour le formatage des dates/heures
  // Charger 'fr_FR' pour utiliser les formats français dans Intl
  await initializeDateFormatting('fr_FR', null);

  // Configurer Amplify (Auth, etc.)
  final bool amplifyConfigured = await _configureAmplify();

  // Lancer l'application Flutter
  if (amplifyConfigured) {
    runApp(const MyApp());
  } else {
    // Gérer l'échec de configuration d'Amplify (afficher un message d'erreur ?)
    runApp(const AmplifyConfigurationErrorApp());
  }
}

/// Configure les plugins Amplify utilisés par l'application.
Future<bool> _configureAmplify() async {
  // Éviter de reconfigurer si déjà fait (utile pour hot reload/restart)
  if (Amplify.isConfigured) {
    print("Amplify is already configured.");
    return true;
  }
  try {
    // Créer et ajouter les plugins nécessaires
    final authPlugin = AmplifyAuthCognito();
    // Ajouter d'autres plugins ici si besoin (Analytics, API Gateway, Storage...)
    await Amplify.addPlugins([authPlugin]);

    // Configurer Amplify avec le fichier amplifyconfiguration.dart
    await Amplify.configure(amplifyconfig);

    print("Amplify configured successfully.");
    return true;
  } on Exception catch (e) {
    // Erreur critique lors de la configuration
    print("CRITICAL: Could not configure Amplify: $e");
    return false;
  }
}

/// Widget racine de l'application. Configure les Providers et MaterialApp.
class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // MultiProvider pour injecter les services dans l'arbre de widgets
    return MultiProvider(
      providers: [
        // --- Services indépendants ---
        ChangeNotifierProvider(create: (_) => ThemeService()),
        ChangeNotifierProvider(create: (_) => BluetoothService()..initialize()), // Initialise BT Service
        Provider(create: (_) => ApiService()), // Service API simple

        // --- Services dépendants (utilisant d'autres services) ---
        // AuthService (utilise Amplify directement maintenant)
        ChangeNotifierProvider(create: (_) => AuthService()),

        // TickService dépend d'AuthService et ApiService
        // Utilise ChangeNotifierProxyProvider pour mettre à jour TickService si AuthService change
        ChangeNotifierProxyProvider<AuthService, TickService>(
          // Crée l'instance initiale de TickService
          create: (context) => TickService(
            Provider.of<ApiService>(context, listen: false), // Fournit ApiService
            Provider.of<AuthService>(context, listen: false), // Fournit AuthService initial
            // mqttService: Provider.of<MQTTService>(context, listen: false), // Optionnel
          ),
          // 'update' est appelé lorsque AuthService notifie un changement
          update: (context, authService, previousTickService) {
            print("ChangeNotifierProxyProvider: AuthService updated, updating TickService.");
            // Met à jour la référence AuthService dans TickService
            previousTickService?.updateAuth(authService);
            // Retourne l'instance existante (ou une nouvelle si null, peu probable)
            return previousTickService ??
                TickService(
                  Provider.of<ApiService>(context, listen: false),
                  authService,
                  // mqttService: ... // Optionnel
                );
          },
        ),
        // Optionnel: Ajouter MQTTService si l'app l'utilise directement
        // Provider(create: (_) => MQTTService(), dispose: (_, service) => service.dispose()),
      ],
      // Consumer<ThemeService> pour reconstruire MaterialApp si le thème change
      child: Consumer<ThemeService>(
        builder: (context, themeService, _) {
          return MaterialApp(
            title: AppTexts.appName,
            theme: AppTheme.getLightTheme(), // Thème clair
            darkTheme: AppTheme.getDarkTheme(), // Thème sombre
            themeMode: themeService.themeMode, // Mode actuel géré par le service
            debugShowCheckedModeBanner: false, // Cacher la bannière Debug

            // Écran de démarrage qui gère la redirection initiale
            initialRoute: Routes.splash, // Utiliser une route nommée pour le splash

            // Définition des routes nommées pour la navigation
            routes: {
              Routes.splash: (context) => const SplashScreen(),
              Routes.welcome: (context) => const WelcomePage(),
              Routes.login: (context) => const LoginPage(),
              Routes.register: (context) => const RegisterPage(),
              Routes.confirmSignUp: (context) => const ConfirmationPage(), // Page de confirmation
              Routes.passwordRecovery: (context) => const ResetPasswordPage(), // Page de réinitialisation mdp
              Routes.tickList: (context) => const TickListPage(), // Écran principal si connecté
              Routes.addTick: (context) => const AddTickPage(),
              Routes.profile: (context) => const ProfilePage(),
              Routes.settings: (context) => const SettingsPage(), // Paramètres généraux
              // Routes pour MapPage et TickSettings ne sont pas définies ici car elles
              // sont généralement poussées avec des arguments (MaterialPageRoute).
              // Si vous préférez des routes nommées, utilisez un package comme GoRouter
              // pour gérer les paramètres (ex: '/ticks/map/:tickId').
            },

            // Gérer les erreurs de navigation (routes inconnues)
            onUnknownRoute: (settings) {
               print("Navigation Error: Unknown route: ${settings.name}");
               return MaterialPageRoute(builder: (_) => const SplashScreen()); // Rediriger vers Splash en cas d'erreur?
            },
          );
        },
      ),
    );
  }
}

/// Widget simple affiché en cas d'échec de configuration d'Amplify.
class AmplifyConfigurationErrorApp extends StatelessWidget {
  const AmplifyConfigurationErrorApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: Scaffold(
        body: Center(
          child: Padding(
            padding: EdgeInsets.all(20.0),
            child: Text(
              'Erreur critique: Impossible de configurer Amplify. Veuillez vérifier la configuration et redémarrer l\'application.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold),
            ),
          ),
        ),
      ),
    );
  }
}

```

--- END OF MODIFIED FILE lib\main.dart ---

--- START OF MODIFIED FILE lib\amplifyconfiguration.dart ---

```dart
/// Configuration Amplify générée par la CLI.
/// Contient les informations nécessaires pour connecter l'application aux services AWS
/// (Cognito User Pool, Identity Pool, etc.).
/// NE PAS MODIFIER MANUELLEMENT CE FICHIER (sauf si vous savez ce que vous faites).
/// Il est regénéré par `amplify pull` ou `amplify push`.

const amplifyconfig = '''{
    "UserAgent": "aws-amplify-cli/2.0",
    "Version": "1.0",
    "auth": {
        "plugins": {
            "awsCognitoAuthPlugin": {
                "UserAgent": "aws-amplify-cli/0.1.0",
                "Version": "0.1.0",
                "IdentityManager": {
                    "Default": {}
                },
                "CredentialsProvider": {
                    "CognitoIdentity": {
                        "Default": {
                            "PoolId": "eu-north-1:1c83f55c-a088-430f-a5b1-82425f1f6667",
                            "Region": "eu-north-1"
                        }
                    }
                },
                "CognitoUserPool": {
                    "Default": {
                        "PoolId": "eu-north-1_EFgbgMufn",
                        "AppClientId": "77jcs4q6oca8fub7sa9ugkkpb3",
                        "Region": "eu-north-1"
                    }
                },
                "Auth": {
                    "Default": {
                        "authenticationFlowType": "USER_SRP_AUTH",
                        "socialProviders": [],
                        "usernameAttributes": [
                            "EMAIL"
                        ],
                        "signupAttributes": [
                            "EMAIL",
                            "NAME"
                        ],
                        "passwordProtectionSettings": {
                            "passwordPolicyMinLength": 8,
                            "passwordPolicyCharacters": []
                        },
                        "mfaConfiguration": "OFF",
                        "mfaTypes": [
                            "SMS"
                        ],
                        "verificationMechanisms": [
                            "EMAIL"
                        ]
                    }
                }
            }
        }
    }
}''';
```

--- END OF MODIFIED FILE lib\amplifyconfiguration.dart ---

--- START OF MODIFIED FILE pubspec.yaml ---

```yaml
name: tick_app
description: Application de suivi IoT pour appareils Tick.
publish_to: 'none' # Empêche la publication accidentelle sur pub.dev

version: 1.0.0+1

environment:
  sdk: '>=3.0.0 <4.0.0' # Utiliser une contrainte SDK plus large mais >= 3.0
  flutter: '>=3.10.0' # Spécifier une version minimale de Flutter si nécessaire

dependencies:
  flutter:
    sdk: flutter
  flutter_localizations: # Ajouté pour la localisation fr_FR
    sdk: flutter

  # --- UI & Utilitaires ---
  cupertino_icons: ^1.0.8 # Icônes style iOS
  provider: ^6.1.2 # Gestion d'état
  intl: ^0.19.0 # Formatage international (dates, nombres) - Version à jour
  url_launcher: ^6.3.0 # Ouvrir des liens externes

  # --- Cartes & Localisation ---
  google_maps_flutter: ^2.6.1 # Cartes Google (vérifier dernière version stable)
  geolocator: ^12.0.0 # Obtenir la position GPS de l'appareil (vérifier dernière version)
  location: ^6.0.2 # Alternative/complément à Geolocator (gestion service/perms)
  permission_handler: ^11.3.1 # Gestion fine des permissions (BLE, Loc)

  # --- Bluetooth Low Energy ---
  flutter_blue_plus: ^1.35.3 # Communication BLE (vérifier dernière version)

  # --- Réseau ---
  http: ^1.2.1 # Requêtes HTTP pour API Lambda

  # --- AWS Amplify ---
  amplify_flutter: ^2.0.1 # Core Amplify Flutter (utiliser la dernière version stable >1.x)
  amplify_auth_cognito: ^2.0.1 # Plugin authentification Cognito

  # --- Stockage Sécurisé (Optionnel si Amplify gère tout) ---
  # flutter_secure_storage: ^9.2.2

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^4.0.0 # Analyse statique du code

flutter:
  uses-material-design: true

  # --- Assets ---
  assets:
    # Répertoire principal des assets (ex: images, logos)
    - assets/
    # Répertoire spécifique pour les certificats MQTT (si utilisés par l'app)
    # S'assurer que ces fichiers existent et sont corrects.
    - assets/certs/AmazonRootCA1.pem
    - assets/certs/DeviceCertificate.crt
    - assets/certs/PrivateKey.key
    # - assets/images/ # Exemple si vous avez des images
```

--- END OF MODIFIED FILE pubspec.yaml ---
