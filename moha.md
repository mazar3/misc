Okay, super ! Merci pour ce résumé détaillé de votre backend. Ça clarifie beaucoup de choses et confirme une intuition : **votre backend gère l'authentification et les opérations sur les Ticks via une API REST personnalisée (avec API Gateway + Lambda), sécurisée par JWT.**

Cela signifie que la refonte précédente de `AuthService` pour utiliser *directement* les fonctions `Amplify.Auth.signIn`, `signUp`, etc., depuis Flutter n'est **pas la bonne approche pour *votre* backend spécifique**. Ces fonctions Amplify sont conçues pour interagir directement avec Cognito en suivant des protocoles standards (comme SRP), ce que vos endpoints `/auth/login` et `/auth/register` personnalisés ne font probablement pas.

**La Bonne Nouvelle :**

1.  Votre backend existe et gère déjà la logique complexe (hashage de mot de passe, interaction avec DynamoDB).
2.  Il utilise la validation JWT, ce qui est standard et sécurisé. On peut supposer que c'est Cognito qui *émet* ces JWTs, même si c'est le backend Lambda qui déclenche leur création lors du login.
3.  Votre `ApiService` est déjà prêt à faire des appels REST.
4.  Votre `TickService` est prêt à utiliser `ApiService`.

**Ce qu'on doit faire maintenant dans Flutter :**

1.  **Revenir en arrière sur `AuthService`** : On va le re-modifier pour qu'il appelle vos endpoints `/auth/login` et `/auth/register` via `ApiService`.
2.  **Gérer le JWT** : `AuthService` devra stocker le JWT reçu lors du login et le fournir à `ApiService`.
3.  **Mettre à jour `ApiService`** : Pour qu'il récupère le JWT depuis `AuthService` (ou depuis le stockage sécurisé) et l'ajoute automatiquement aux headers des requêtes protégées.
4.  **Connecter `TickService`** : S'assurer qu'il utilise `ApiService` (qui inclura maintenant le JWT) pour parler à vos endpoints `/ticks`, `/users/me`, etc.

**Hypothèse Clé sur votre Backend (Important!)**

Je pars du principe que lorsque votre backend Lambda `/auth/login` valide l'email/mot de passe :

*   Il interagit avec **AWS Cognito en utilisant le SDK AWS** (côté serveur, dans la Lambda) pour authentifier l'utilisateur auprès de Cognito (ex: via `AdminInitiateAuth`).
*   Il récupère les **tokens JWT (ID Token, Access Token) générés par Cognito**.
*   Il renvoie (au moins) l'**ID Token JWT** dans la réponse JSON à votre application Flutter.

Si ce n'est pas le cas (si votre backend génère ses propres JWT sans utiliser Cognito), la partie validation JWT et configuration de l'Authorizer API Gateway sera différente, mais le principe dans Flutter reste similaire (stocker et envoyer le token reçu).

Allons-y !

---

**Étape 1 : Modifier `AuthService` pour utiliser `ApiService` et gérer le JWT**

On revient à une version qui utilise `ApiService` pour les appels `/auth/login` et `/auth/register`, et on ajoute la gestion du JWT.

```dart
// lib/services/auth_service.dart

import 'package:flutter/foundation.dart';
// Supprimer les imports Amplify Auth directs si plus utilisés ici
// import 'package:amplify_flutter/amplify_flutter.dart' hide ApiConfig;
// import 'package:amplify_auth_cognito/amplify_auth_cognito.dart';
import '../models/user.dart';
import 'api_service.dart'; // On a besoin d'ApiService
import '../utils/constants.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart'; // Pour stocker le JWT

class AuthService with ChangeNotifier {
  final ApiService _apiService; // Garder ApiService
  User? _currentUser;
  bool _isLoading = false;
  String? _error;
  bool _isInitialized = false;

  // Stockage sécurisé pour le token JWT
  final _storage = const FlutterSecureStorage();
  String? _jwtToken; // Stocker le token en mémoire pour accès rapide

  // Constructeur prend ApiService
  AuthService(this._apiService) {
    // Pas besoin d'écouter les événements Hub Amplify si on n'utilise pas Amplify Auth directement
    // _listenToAuthEvents();
    // L'initialisation va lire le token stocké
    initializeAuth();
  }

  User? get currentUser => _currentUser;
  bool get isLoading => _isLoading;
  // L'authentification est basée sur la présence du token et de l'utilisateur
  bool get isAuthenticated => _jwtToken != null && _currentUser != null;
  String? get error => _error;
  bool get isInitialized => _isInitialized;
  String? get jwtToken => _jwtToken; // Getter pour le token (utilisé par ApiService)

  // Initialisation : vérifier si un token est stocké et valide
  Future<void> initializeAuth() async {
    if (_isInitialized) return;
    _setLoading(true);
    _clearError();

    try {
      // Lire le token depuis le stockage sécurisé
      _jwtToken = await _storage.read(key: StorageKeys.authToken);

      if (_jwtToken != null) {
        print("AuthService: Found stored token. Validating...");
        // Valider le token en récupérant le profil utilisateur
        bool isValid = await _fetchUserProfile();
        if (!isValid) {
           print("AuthService: Stored token is invalid or expired. Logging out.");
          // Token invalide/expiré, nettoyer
          await logout(isTokenExpired: true);
        } else {
           print("AuthService: Token validated successfully. User is authenticated.");
        }
      } else {
        print("AuthService: No stored token found.");
        _currentUser = null; // Assurer qu'il n'y a pas d'utilisateur
      }
    } catch (e) {
      print("AuthService: Error during initialization: $e");
      _setError("Erreur d'initialisation de l'authentification.");
      await logout(isTokenExpired: true); // Forcer le nettoyage en cas d'erreur grave
    } finally {
      _isInitialized = true;
      _setLoading(false);
    }
  }

  // Connexion via l'API REST du backend
  Future<bool> login(String email, String password) async {
    _setLoading(true);
    _clearError();

    try {
      // Appel à l'endpoint /auth/login via ApiService
      final response = await _apiService.post(ApiEndpoints.login, {
        'email': email,
        'password': password, // Le backend gère le hashage/vérification
      });

      if (response['success']) {
        // Assumer que la réponse contient le token et les infos user
        // Adaptez les clés ('token', 'user') selon la réponse réelle de VOTRE API
        final token = response['data']?['token'] as String?;
        final userData = response['data']?['user'] as Map<String, dynamic>?;

        if (token != null && userData != null) {
          // Stocker le token
          await _storage.write(key: StorageKeys.authToken, value: token);
          _jwtToken = token;

          // Mettre à jour l'utilisateur
          _updateCurrentUser(User.fromJson(userData));
           print("AuthService: Login successful for $email. Token stored.");

          _setLoading(false);
          return true;
        } else {
          // Réponse succès mais données manquantes
          _setError("Réponse du serveur invalide après connexion.");
           print("AuthService: Login API response missing token or user data.");
          _setLoading(false);
          return false;
        }
      } else {
        // Erreur renvoyée par l'API backend
        _setError(response['error'] ?? ErrorMessages.invalidCredentials);
         print("AuthService: Login failed. API Error: ${response['error']}");
        _setLoading(false);
        return false;
      }
    } catch (e) {
      print("AuthService: Exception during login: $e");
      _setError(ErrorMessages.connectionFailed);
      _setLoading(false);
      return false;
    }
  }

  // Inscription via l'API REST du backend
  Future<bool> register(String email, String password, String name) async {
    _setLoading(true);
    _clearError();

    try {
       // Appel à l'endpoint /auth/register via ApiService
      final response = await _apiService.post(ApiEndpoints.register, {
        'email': email,
        'password': password,
        'name': name, // Assurez-vous que l'API attend 'name'
      });

      if (response['success']) {
        // L'inscription a réussi côté backend.
        // Le backend a (normalement) créé l'utilisateur dans DynamoDB ET Cognito.
        // L'utilisateur doit maintenant se connecter.
        // Votre API pourrait renvoyer le token directement ici, ou non.
        // Si elle renvoie le token + user : traitez comme dans login().
        // Si elle ne renvoie rien (juste succès):
         print("AuthService: Registration successful for $email. User needs to login.");
        _setLoading(false);
        // Retourner true, mais l'utilisateur n'est PAS connecté automatiquement.
        // L'UI devrait rediriger vers la page de connexion.
        return true;

        // --- OU SI L'API RENVOIE LE TOKEN DIRECTEMENT ---
        /*
        final token = response['data']?['token'] as String?;
        final userData = response['data']?['user'] as Map<String, dynamic>?;
        if (token != null && userData != null) {
          await _storage.write(key: StorageKeys.authToken, value: token);
          _jwtToken = token;
          _updateCurrentUser(User.fromJson(userData));
          print("AuthService: Registration and auto-login successful for $email.");
          _setLoading(false);
          return true;
        } else {
           print("AuthService: Registration successful but no token/user returned. User needs to login.");
           _setLoading(false);
           return true; // Succès de l'inscription, mais pas de connexion auto
        }
        */
        // --- FIN OU ---

      } else {
        // Erreur renvoyée par l'API (ex: email existe déjà)
        _setError(response['error'] ?? ErrorMessages.emailInUse);
         print("AuthService: Registration failed. API Error: ${response['error']}");
        _setLoading(false);
        return false;
      }
    } catch (e) {
      print("AuthService: Exception during registration: $e");
      _setError(ErrorMessages.connectionFailed);
      _setLoading(false);
      return false;
    }
  }

  // Déconnexion : effacer le token et l'utilisateur local
  Future<void> logout({bool isTokenExpired = false}) async {
    if (!isTokenExpired && _jwtToken != null) {
       // Optionnel: Appeler l'endpoint /auth/logout de votre backend s'il existe
       try {
          print("AuthService: Calling backend logout endpoint...");
          await _apiService.post(ApiEndpoints.logout, {}); // Assurez-vous d'avoir le JWT dans les headers
       } catch (e) {
          print("AuthService: Error calling backend logout: $e");
          // Continuer la déconnexion locale même si l'appel API échoue
       }
    }

    _jwtToken = null;
    _currentUser = null;
    // Effacer le token du stockage sécurisé
    await _storage.delete(key: StorageKeys.authToken);
     print("AuthService: User logged out. Token cleared.");
    notifyListeners();
  }

  // Récupérer le profil utilisateur (utilisé pour valider le token)
  Future<bool> _fetchUserProfile() async {
     // Vérifier si un token existe en mémoire
     if (_jwtToken == null) {
       print("AuthService: Cannot fetch profile, no token available.");
       return false;
     }

     try {
       // Appel à /users/me via ApiService (qui ajoutera le token actuel)
       final response = await _apiService.get(ApiEndpoints.getUserProfile);

       if (response['success']) {
          final userData = response['data'] as Map<String, dynamic>?;
          if (userData != null) {
            _updateCurrentUser(User.fromJson(userData));
             print("AuthService: User profile fetched successfully.");
            return true;
          } else {
             _setError("Données utilisateur invalides reçues du serveur.");
              print("AuthService: Fetch profile response missing user data.");
             return false;
          }
       } else {
          // L'erreur peut être une invalidation de token (401/403) gérée par ApiService
          _setError(response['error'] ?? "Impossible de vérifier la session.");
           print("AuthService: Fetch profile failed. API Error: ${response['error']}");
          return false;
       }
     } catch (e) {
       print("AuthService: Exception fetching user profile: $e");
       _setError(ErrorMessages.connectionFailed);
       return false;
     }
  }

  // ---- Méthodes pour mot de passe oublié (si votre API les fournit) ----

  Future<bool> requestPasswordReset(String email) async {
    _setLoading(true);
    _clearError();
    try {
       final response = await _apiService.post(ApiEndpoints.forgotPassword, {'email': email});
       if (response['success']) {
          print("AuthService: Password reset request sent for $email.");
          _setLoading(false);
          return true;
       } else {
          _setError(response['error'] ?? "Erreur lors de la demande.");
           print("AuthService: Password reset request failed. API Error: ${response['error']}");
          _setLoading(false);
          return false;
       }
    } catch (e) {
       print("AuthService: Exception requesting password reset: $e");
       _setError(ErrorMessages.connectionFailed);
       _setLoading(false);
       return false;
    }
  }

  Future<bool> confirmPasswordReset(String email, String newPassword, String confirmationCode) async {
     _setLoading(true);
     _clearError();
     try {
        // Assurez-vous que l'endpoint et le body correspondent à votre API
        final response = await _apiService.post(ApiEndpoints.resetPassword, {
           'email': email,
           'newPassword': newPassword,
           'confirmationCode': confirmationCode,
        });
        if (response['success']) {
           print("AuthService: Password reset confirmed for $email.");
           _setLoading(false);
           // L'utilisateur peut maintenant se connecter avec le nouveau mot de passe
           return true;
        } else {
           _setError(response['error'] ?? "Erreur de confirmation.");
            print("AuthService: Password reset confirmation failed. API Error: ${response['error']}");
           _setLoading(false);
           return false;
        }
     } catch (e) {
        print("AuthService: Exception confirming password reset: $e");
        _setError(ErrorMessages.connectionFailed);
        _setLoading(false);
        return false;
     }
  }

  // ---- Mise à jour du profil (si votre API le fournit) ----
  Future<bool> updateUserName(String newName) async {
    if (!isAuthenticated) {
       _setError("Utilisateur non connecté.");
       return false;
    }
    _setLoading(true);
    _clearError();
    try {
      // Assurez-vous que l'endpoint et le body correspondent à votre API
      // Votre API /users/me pourrait accepter PUT ou PATCH ?
      // Ou un endpoint dédié /users/me/profile ?
      // Ici, on suppose un PUT sur /users/me
      final response = await _apiService.put(ApiEndpoints.getUserProfile, {
        'name': newName,
        // Inclure d'autres champs si l'API les gère (ex: email, mais attention à la vérification)
      });

      if (response['success']) {
        // Mettre à jour l'utilisateur local avec les nouvelles données (si renvoyées)
        // ou juste le nom si l'API ne renvoie pas l'objet complet
        final updatedUserData = response['data'] as Map<String, dynamic>?;
        if (updatedUserData != null) {
           _updateCurrentUser(User.fromJson(updatedUserData));
        } else {
            _updateCurrentUser(currentUser!.copyWith(displayName: newName));
        }
         print("AuthService: User name updated successfully.");
        _setLoading(false);
        return true;
      } else {
        _setError(response['error'] ?? "Erreur mise à jour profil.");
         print("AuthService: Update user name failed. API Error: ${response['error']}");
        _setLoading(false);
        return false;
      }
    } catch (e) {
      print("AuthService: Exception updating user name: $e");
      _setError(ErrorMessages.connectionFailed);
      _setLoading(false);
      return false;
    }
  }


  // --- Méthodes internes de gestion d'état ---
  void _setLoading(bool loading) {
    if (_isLoading == loading) return;
    _isLoading = loading;
    notifyListeners();
  }

  void _setError(String? errorMessage) {
    if (_error == errorMessage && errorMessage != null) return;
    _error = errorMessage;
    notifyListeners(); // Notifier si l'erreur change
  }

  void _clearError() {
    if (_error != null) {
      _error = null;
      notifyListeners(); // Notifier si l'erreur est effacée
    }
  }

  void _updateCurrentUser(User? user) {
    if (_currentUser?.uid == user?.uid && _currentUser?.displayName == user?.displayName) return;
    _currentUser = user;
    notifyListeners(); // Notifier le changement d'utilisateur
  }

  // --- Nettoyage ---
  @override
  void dispose() {
    // Pas de listener Hub Amplify à annuler ici
    super.dispose();
  }
}
```

---

**Étape 2 : Mettre à jour `ApiService` pour utiliser le JWT de `AuthService`**

On modifie `_buildHeaders` pour qu'il récupère le token depuis `AuthService`.

```dart
// lib/services/api_service.dart

import 'package:http/http.dart' as http;
import 'dart:convert';
// Supprimer les imports Amplify si plus utilisés directement ici
// import 'package:amplify_flutter/amplify_flutter.dart' hide ApiConfig;
// import 'package:amplify_auth_cognito/amplify_auth_cognito.dart';
import '../utils/constants.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart'; // Importer storage
// AuthService n'est pas directement utilisé ici, on lit le token du storage

class ApiService {

  // Utiliser FlutterSecureStorage pour récupérer le token
  final _storage = const FlutterSecureStorage();

  // Méthode pour récupérer le token d'autorisation depuis le stockage sécurisé
  Future<String?> getAuthorizationToken() async {
     try {
       // Lire le token depuis le stockage sécurisé
       final token = await _storage.read(key: StorageKeys.authToken);
       // print("ApiService: Retrieved token from storage: ${token != null ? 'found' : 'not found'}"); // Debug
       return token;
     } catch (e) {
       // Gérer les erreurs de lecture du stockage si nécessaire
       print("ApiService: Error reading token from storage: $e");
       return null;
     }
  }

  // Méthode générique pour construire les headers HTTP
  Future<Map<String, String>> _buildHeaders({Map<String, String>? customHeaders}) async {
    // Récupérer le token depuis le stockage
    final token = await getAuthorizationToken();
    final Map<String, String> headers = {
      'Content-Type': 'application/json',
      // Ajouter le token d'autorisation s'il existe
      if (token != null) 'Authorization': 'Bearer $token',
    };
    // Fusionner avec les headers personnalisés éventuels
    if (customHeaders != null) {
      headers.addAll(customHeaders);
    }
    // print("ApiService: Built headers: $headers"); // Debug (Attention en prod avec le token)
    return headers;
  }

  // Méthode générique pour les requêtes GET
  Future<Map<String, dynamic>> get(String endpoint, {Map<String, String>? headers}) async {
    final url = Uri.parse('${ApiConfig.baseUrl}$endpoint');
    final requestHeaders = await _buildHeaders(customHeaders: headers);

    print('API GET Request: $url'); // Log

    try {
      final response = await http.get(
         url,
         headers: requestHeaders,
      ).timeout(AppDurations.apiTimeout); // Ajouter un timeout

       return _handleResponse(response);

    } catch (e) {
       print('API GET Error for $url: $e');
        return {'success': false, 'error': ErrorMessages.connectionFailed};
    }
  }

  // Méthode générique pour les requêtes POST
  Future<Map<String, dynamic>> post(String endpoint, Map<String, dynamic> body, {Map<String, String>? headers}) async {
    final url = Uri.parse('${ApiConfig.baseUrl}$endpoint');
    final requestHeaders = await _buildHeaders(customHeaders: headers);
    final requestBody = jsonEncode(body);

    print('API POST Request: $url'); // Log
    // print('API POST Body: $requestBody'); // Debug

     try {
      final response = await http.post(
         url,
         headers: requestHeaders,
         body: requestBody,
      ).timeout(AppDurations.apiTimeout);

      return _handleResponse(response);

    } catch (e) {
       print('API POST Error for $url: $e');
       return {'success': false, 'error': ErrorMessages.connectionFailed};
    }
  }

   // Méthode générique pour les requêtes PUT
  Future<Map<String, dynamic>> put(String endpoint, Map<String, dynamic> body, {Map<String, String>? headers}) async {
     final url = Uri.parse('${ApiConfig.baseUrl}$endpoint');
     final requestHeaders = await _buildHeaders(customHeaders: headers);
     final requestBody = jsonEncode(body);

     print('API PUT Request: $url'); // Log
     // print('API PUT Body: $requestBody'); // Debug

      try {
        final response = await http.put(
           url,
           headers: requestHeaders,
           body: requestBody,
        ).timeout(AppDurations.apiTimeout);

        return _handleResponse(response);

      } catch (e) {
         print('API PUT Error for $url: $e');
         return {'success': false, 'error': ErrorMessages.connectionFailed};
      }
  }

  // Méthode générique pour les requêtes DELETE
  Future<Map<String, dynamic>> delete(String endpoint, {Map<String, String>? headers}) async {
     final url = Uri.parse('${ApiConfig.baseUrl}$endpoint');
     final requestHeaders = await _buildHeaders(customHeaders: headers);

     print('API DELETE Request: $url'); // Log

      try {
        final response = await http.delete(
           url,
           headers: requestHeaders,
        ).timeout(AppDurations.apiTimeout);

        return _handleResponse(response);

      } catch (e) {
         print('API DELETE Error for $url: $e');
         return {'success': false, 'error': ErrorMessages.connectionFailed};
      }
  }


  // Gestionnaire de réponse HTTP (privé)
  Map<String, dynamic> _handleResponse(http.Response response) {
    final statusCode = response.statusCode;
    final responseBody = response.body;

    print('API Response Status Code: $statusCode'); // Log
    // print('API Response Body: $responseBody'); // Debug

    // Statut de succès (2xx)
    if (statusCode >= 200 && statusCode < 300) {
      try {
        final dynamic decodedBody = responseBody.isNotEmpty ? jsonDecode(responseBody) : {};
        // Adapter la structure si votre API renvoie directement les données sans clé 'data' ou 'success'
        if (decodedBody is Map<String, dynamic> && decodedBody.containsKey('success') && decodedBody.containsKey('data')) {
             // Structure attendue { success: bool, data: ... }
             if (decodedBody['success'] == true) {
                 return {'success': true, 'data': decodedBody['data']};
             } else {
                 // Cas où success: false est dans un 2xx ? Peu probable mais géré.
                 return {'success': false, 'error': decodedBody['error'] ?? decodedBody['message'] ?? 'Erreur serveur inattendue'};
             }
        } else if (decodedBody is Map<String, dynamic> && decodedBody.containsKey('token') && decodedBody.containsKey('user')) {
             // Cas spécifique de la réponse de login réussie (supposée)
             return {'success': true, 'data': decodedBody}; // Retourne tout le corps
        }
        else {
           // Si la réponse est juste les données (ex: GET /users/me retourne juste l'objet user)
           return {'success': true, 'data': decodedBody};
        }
      } catch (e) {
        print("API Response JSON Decode Error (Status: $statusCode): $e");
        return {'success': false, 'error': 'Format de réponse serveur invalide.'};
      }
    }
    // Statut d'erreur (non-2xx)
    else {
      String errorMessage = ErrorMessages.unknownError;
      try {
         final decodedBody = jsonDecode(responseBody);
         if (decodedBody is Map<String, dynamic>) {
            // Votre API semble utiliser 'message' ou 'error'
            errorMessage = decodedBody['message'] ?? decodedBody['error'] ?? errorMessage;
         } else if (decodedBody is String && decodedBody.isNotEmpty) {
             errorMessage = decodedBody;
         }
      } catch (e) {
        print("API Response Error Body Decode Failed (Status: $statusCode): $e");
      }

       // Personnaliser en fonction du code statut
       if (errorMessage == ErrorMessages.unknownError || errorMessage.isEmpty) {
          if (statusCode == 400) errorMessage = ErrorMessages.invalidInput;
          if (statusCode == 401) errorMessage = 'Non autorisé. Veuillez vous reconnecter.'; // Token invalide/expiré
          if (statusCode == 403) errorMessage = 'Accès refusé.'; // Manque de permissions
          if (statusCode == 404) errorMessage = 'Ressource non trouvée.';
          if (statusCode >= 500) errorMessage = ErrorMessages.apiError; // Erreur serveur générique
       }

      // Si c'est une erreur 401 ou 403, on pourrait vouloir déclencher une déconnexion globale
      // Mais c'est mieux géré dans AuthService en vérifiant l'erreur après l'appel
      // if (statusCode == 401 || statusCode == 403) {
      //    // Déclencher un événement global de déconnexion?
      // }

      return {'success': false, 'error': errorMessage};
    }
  }
}
```

---

**Étape 3 : Vérifier `TickService`**

Votre `TickService` utilise déjà `ApiService` pour ses appels (`fetchTicks`, `associateTick`, `updateTickSettings`, `unlinkTick`). Il n'y a **rien à changer dans `TickService`** lui-même, car `ApiService` gère maintenant l'ajout du token JWT automatiquement. Il continuera de fonctionner comme prévu, mais cette fois avec des appels API sécurisés !

---

**Étape 4 : Ajustements UI (Mineurs)**

*   **Pas besoin d'écran de confirmation** : Puisque votre backend gère l'inscription (et potentiellement la vérification), l'écran de confirmation spécifique à Amplify n'est plus nécessaire. La redirection après `register` doit aller vers `LoginPage`.
*   **Pas besoin d'écran de confirmation de reset password** : Votre backend gère le reset via l'API `/auth/reset-password` (qui prend email, code, newPassword). L'UI doit juste collecter ces infos et appeler `authService.confirmPasswordReset`.
*   **Gestion des erreurs 401/403** : Assurez-vous que lorsque `AuthService` ou `TickService` reçoivent une erreur indiquant un token invalide (ex: "Non autorisé. Veuillez vous reconnecter."), l'UI réagit correctement, par exemple en appelant `authService.logout()` et en redirigeant vers la page de connexion. Vous pouvez ajouter cette logique dans les `catch` ou après les appels API qui échouent avec ce type d'erreur.

Exemple dans `TickService` (à ajouter dans `fetchTicks`, `associateTick`, etc.) :

```dart
      // ... dans la gestion d'erreur (après un appel API)
      } else {
        _error = response['error'] ?? ErrorMessages.unknownError;
        print("Error fetching ticks: $_error");
        // Vérifier si l'erreur est due à une session expirée/invalide
        if (response['error']?.contains('Non autorisé') ?? false) { // Adaptez le message d'erreur réel
           // Déclencher la déconnexion via AuthService
           // Utiliser addPostFrameCallback pour éviter les erreurs de build/setState
           WidgetsBinding.instance.addPostFrameCallback((_) {
              // Utiliser listen:false car on est dans un callback asynchrone
              Provider.of<AuthService>(context, listen: false).logout(isTokenExpired: true);
              // La navigation vers login sera gérée par le changement d'état d'AuthService
           });
        }
      }
    } catch (e) {
      // ...
```

---

**Étape 5 : Tester le Nouveau Flux**

C'est l'étape la plus importante !

1.  **Vider le stockage sécurisé** (ou désinstaller/réinstaller l'app) pour être sûr de ne pas avoir d'ancien token Amplify.
2.  **Inscription :**
    *   Créez un compte via l'app.
    *   Vérifiez que l'appel à `/auth/register` est fait.
    *   **Vérifiez dans AWS :** L'utilisateur apparaît-il dans la table `Users` de DynamoDB *ET* dans le User Pool Cognito ? (C'est la responsabilité de votre Lambda backend).
    *   L'app doit vous rediriger vers la page de connexion.
3.  **Connexion :**
    *   Connectez-vous avec le compte créé.
    *   Vérifiez que l'appel à `/auth/login` est fait.
    *   Vérifiez la réponse : recevez-vous bien un `token` et les données `user` ?
    *   Le token est-il stocké (`flutter_secure_storage`) ?
    *   Êtes-vous redirigé vers `TickListPage` ?
4.  **Appels Protégés :**
    *   La `TickListPage` appelle `fetchTicks`. Vérifiez dans les logs de `ApiService` que le header `Authorization: Bearer ...` est bien ajouté à l'appel `/ticks`.
    *   Vérifiez que la liste des ticks (probablement vide au début) s'affiche ou que l'état vide est montré.
    *   Essayez d'ajouter un Tick (si la partie Bluetooth fonctionne). Vérifiez l'appel POST `/ticks` avec le header.
    *   Naviguez vers le profil. Vérifiez l'appel GET `/users/me` avec le header.
5.  **Déconnexion :**
    *   Déconnectez-vous.
    *   Vérifiez que le token est effacé du stockage.
    *   Vérifiez que vous revenez à l'écran de connexion.
6.  **Session Persistante :**
    *   Connectez-vous. Fermez l'app. Rouvrez l'app.
    *   Vérifiez que `initializeAuth` lit le token, appelle `/users/me` (avec le header), valide la session et vous amène directement à `TickListPage`.
7.  **Expiration / Token Invalide (Plus difficile à tester) :**
    *   Si vous pouvez invalider un token côté backend (ou attendre son expiration naturelle), vérifiez qu'un appel API (ex: `fetchTicks`) échoue avec une erreur 401/403 et que l'app vous déconnecte et vous ramène à l'écran de login.

---

Vous avez maintenant un plan clair pour adapter votre Flutter App à votre backend REST spécifique tout en utilisant probablement Cognito pour la génération/validation des JWTs. La clé est de bien comprendre où se fait chaque étape (Flutter, API Gateway, Lambda, Cognito, DynamoDB). Bon courage !
