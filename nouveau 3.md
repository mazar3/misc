Okay, les amis ! On va démystifier ce truc AWS Cognito ensemble. Pas de panique, on va y aller étape par étape, comme si on construisait des LEGOs. Votre code Flutter est déjà bien structuré, c'est une excellente base !

Le fichier `amplifyconfiguration.dart` est LA CLÉ. Il me dit que vous avez **déjà initialisé Amplify dans votre projet et ajouté la fonctionnalité d'authentification (auth)**. C'est une super nouvelle, ça nous évite de devoir tout créer depuis zéro dans la console AWS !

Ce qu'on va faire :

1.  **Comprendre ce que Amplify a créé pour vous** dans AWS Cognito (via la console AWS).
2.  **Refondre votre `AuthService`** dans Flutter pour utiliser les fonctions d'Amplify au lieu de vos appels `http` directs (qui ne marcheront pas avec Cognito sans API Gateway + Lambda, ce qu'on fera plus tard si besoin).
3.  **Adapter (légèrement) vos écrans** pour coller au flux Cognito (notamment la confirmation d'inscription).

Allez, c'est parti !

---

**Étape 1 : Exploration de votre Configuration AWS Cognito (Console AWS)**

Ouvrez votre console AWS (assurez-vous d'être dans la bonne région AWS, indiquée dans `amplifyconfiguration.dart` : **`eu-north-1`**).

1.  **Trouver Cognito :**
    *   Dans la barre de recherche en haut, tapez `Cognito` et cliquez sur le service "Amazon Cognito".

2.  **Trouver votre User Pool (Groupe d'utilisateurs) :**
    *   Sur la page Cognito, vous devriez voir une section "User Pools".
    *   Votre `amplifyconfiguration.dart` indique un `PoolId` : **`eu-north-1_EFgbgMufn`**. Cherchez ce Pool ID dans la liste et cliquez dessus. C'est LÀ que sont gérés vos utilisateurs (inscription, connexion, etc.).

3.  **Explorer les Paramètres du User Pool (Ce qu'il faut regarder/comprendre) :**
    *   **Onglet "General settings" :**
        *   **Pool ARN / Pool ID :** Juste pour confirmer que vous êtes au bon endroit.
        *   **Attributes :** Regardez quels attributs sont requis pour l'inscription. Votre config dit `signupAttributes: ["EMAIL"]`. Amplify demande aussi souvent `name` par défaut lors de la création. Vous pouvez voir ici quels sont les attributs standards (email, name, phone_number...) et si des attributs personnalisés ont été ajoutés (probablement pas pour l'instant). C'est important car votre `User` model Flutter doit correspondre.
    *   **Onglet "Sign-up experience" :**
        *   **Self-service sign-up :** Doit être "Enabled" pour que les utilisateurs puissent s'inscrire eux-mêmes.
        *   **Attribute verification and user account confirmation :** Regardez la section "Verification". Votre config dit `verificationMechanisms: ["EMAIL"]`. Cela signifie que Cognito enverra un code par email pour confirmer l'inscription. **C'est crucial, car votre code Flutter actuel ne gère pas cette étape de confirmation !** On va devoir l'ajouter.
    *   **Onglet "Messaging" :**
        *   **Email :** Ici, vous pouvez (si vous voulez) personnaliser les emails envoyés par Cognito (vérification, mot de passe oublié...). Pour l'instant, laissez les messages par défaut, MAIS assurez-vous que la **"FROM email address"** est configurée (Cognito utilise souvent une adresse par défaut no-reply@verificationemail.com ou vous demande d'en vérifier une via Amazon SES si vous voulez personnaliser).
    *   **Onglet "App integration" :** (TRÈS IMPORTANT)
        *   **App clients and analytics :** Descendez jusqu'à trouver la liste des "App clients". Votre `amplifyconfiguration.dart` a un `AppClientId` : **`77jcs4q6oca8fub7sa9ugkkpb3`**. Cliquez sur cet App Client (il a peut-être un nom du genre `xxxxx_app_clientWeb` ou `xxxxx_app_client`).
        *   **Détails de l'App Client :**
            *   **Client secret :** Assurez-vous que la case **"Generate client secret"** n'est **PAS** cochée. Les applications mobiles utilisent un flux d'authentification (SRP) qui ne nécessite pas de secret client et il est dangereux d'en embarquer un dans une app. Amplify configure ça correctement par défaut.
            *   **Authentication flows :** Vérifiez que **`ALLOW_USER_SRP_AUTH`** et **`ALLOW_REFRESH_TOKEN_AUTH`** sont bien cochés. C'est le flux sécurisé qu'Amplify utilise. `ALLOW_ADMIN_USER_PASSWORD_AUTH` ne devrait PAS être coché pour une app mobile.
            *   **Hosted UI :** Probablement pas configuré, pas besoin pour l'instant.
            *   **Attribute read and write permissions :** Vérifiez que les attributs que vous voulez utiliser (au moins `email` et `name`) sont cochés pour la lecture et l'écriture. Amplify gère ça normalement.

4.  **(Optionnel) Trouver votre Identity Pool (Pool d'identités) :**
    *   Retournez à la page principale de Cognito.
    *   Cliquez sur "Federated Identities" (ou "Identity Pools").
    *   Votre config a un `IdentityManager` -> `PoolId` : **`eu-north-1:1c83f55c-a088-430f-a5b1-82425f1f6667`**. Trouvez-le et cliquez dessus.
    *   **Rôle des Identity Pools :** Ils servent à donner des *permissions AWS temporaires* aux utilisateurs (authentifiés ou non) pour accéder à *d'autres* services AWS (comme API Gateway, S3, etc.). Pour l'instant, on se concentre sur l'authentification du User Pool, mais c'est bon de savoir où il est.
    *   Cliquez sur "Edit identity pool" en haut à droite. Regardez les sections "Authenticated role" et "Unauthenticated role". Ce sont des rôles IAM qui définissent ce que vos utilisateurs peuvent faire dans AWS une fois connectés (ou en tant qu'invités).

**Conclusion Étape 1 :** Vous avez maintenant une idée de ce qu'Amplify a mis en place. Le point clé est la nécessité de gérer la **confirmation par email** lors de l'inscription.

---

**Étape 2 : Refonte de `AuthService` pour utiliser Amplify**

Oubliez les appels `http` pour l'authentification. On va utiliser directement `Amplify.Auth`.

```dart
// lib/services/auth_service.dart

import 'package:flutter/foundation.dart';
import 'package:amplify_flutter/amplify_flutter.dart' hide ApiConfig;
import 'package:amplify_auth_cognito/amplify_auth_cognito.dart'; // Import Cognito specifics
import '../models/user.dart';
// import 'api_service.dart'; // Plus besoin pour les appels auth directs
import '../utils/constants.dart';
// import 'package:http/http.dart' as http; // Supprimer http
// import 'dart:convert'; // Supprimer json convert
// import 'package:flutter_secure_storage/flutter_secure_storage.dart'; // Plus besoin, Amplify gère la session


class AuthService with ChangeNotifier {
  // Plus besoin d'ApiService ici pour l'auth
  // final ApiService _apiService; // Supprimer

  User? _currentUser;
  bool _isLoading = false;
  String? _error;
  bool _isInitialized = false;
  // Track if user needs to confirm their account
  bool _needsConfirmation = false;
  bool get needsConfirmation => _needsConfirmation;

  // Pas besoin de stocker le token manuellement
  // String? _token;
  // final _storage = const FlutterSecureStorage(); // Supprimer

  String? _confirmationUsername; // Store username for confirmation step

  // Le constructeur n'a plus besoin d'ApiService
  AuthService(/*this._apiService*/) { // Modifier le constructeur
    _listenToAuthEvents();
    _initializeAuth();
  }

  User? get currentUser => _currentUser;
  bool get isLoading => _isLoading;
  // L'authentification est gérée par la session Amplify
  bool get isAuthenticated => _currentUser != null;
  String? get error => _error;
  bool get isInitialized => _isInitialized;


  // Listen to Amplify Auth events
  void _listenToAuthEvents() {
    Amplify.Hub.listen(HubChannel.Auth, (AuthHubEvent event) {
      switch (event.type) {
        case AuthHubEventType.signedIn:
          print('Auth Hub: User signed in');
          // Fetch user details after sign in event
          _fetchCurrentUser(setInitialized: true);
          break;
        case AuthHubEventType.signedOut:
          print('Auth Hub: User signed out');
          // Clear user state on sign out event
          _updateCurrentUser(null);
          _isInitialized = true; // Mark as initialized even after logout
           notifyListeners(); // Notify UI about logout
          break;
        case AuthHubEventType.sessionExpired:
          print('Auth Hub: Session expired');
           _updateCurrentUser(null); // Treat as logout
           _isInitialized = true;
           notifyListeners();
          break;
        case AuthHubEventType.userDeleted:
           print('Auth Hub: User deleted');
           _updateCurrentUser(null); // Treat as logout
           _isInitialized = true;
           notifyListeners();
           break;
      }
    });
  }


  Future<void> _initializeAuth() async {
    // Don't re-initialize if already done by Hub event listener or manual call
    if (_isInitialized) return;

    _setLoading(true);
    try {
       // Check if a session already exists (user is logged in)
      final session = await Amplify.Auth.fetchAuthSession();
      if (session.isSignedIn) {
         print("Auth Initialized: User is signed in.");
         await _fetchCurrentUser(setInitialized: false); // Fetch details, don't set initialized yet
      } else {
          print("Auth Initialized: No active session found.");
           _updateCurrentUser(null); // Ensure user is null
      }
    } on Exception catch (e) {
      print("Auth Initialization Error: $e");
       _setError("Erreur lors de la vérification de la session: $e");
       _updateCurrentUser(null); // Ensure user is null on error
    } finally {
       _isInitialized = true; // Mark as initialized regardless of login state
       _setLoading(false); // Ensure loading stops
       // NotifyListeners is handled by _setLoading or _fetchCurrentUser
    }
  }

  // Supprimer initAuth basé sur le token manuel
  /*
  Future<bool> initAuth() async { ... }
  */

  // Helper to fetch and update current user details using Amplify
  Future<void> _fetchCurrentUser({bool setInitialized = true}) async {
     try {
        final cognitoUser = await Amplify.Auth.getCurrentUser();
        final attributes = await Amplify.Auth.fetchUserAttributes();
        String displayName = '';
        String email = '';

        // Itérer sur les attributs récupérés
        for (final attribute in attributes) {
           // Utiliser CognitoUserAttributeKey pour la robustesse
           if (attribute.userAttributeKey == CognitoUserAttributeKey.name) {
              displayName = attribute.value;
           } else if (attribute.userAttributeKey == CognitoUserAttributeKey.email) {
              email = attribute.value;
           }
           // Ajouter d'autres attributs si nécessaire (ex: phone_number)
           // else if (attribute.userAttributeKey == CognitoUserAttributeKey.phoneNumber) { ... }
        }

        // Créer notre User model
        _updateCurrentUser(User(
          uid: cognitoUser.userId, // C'est l'identifiant unique 'sub' de Cognito
          email: email,
          displayName: displayName,
        ));

        // Si on réussit à récupérer l'utilisateur, c'est qu'il est confirmé
        _needsConfirmation = false;
        _confirmationUsername = null;

        print("Current user fetched: ID=${cognitoUser.userId}, Email=$email, Name=$displayName");

     } on AuthException catch (e) {
        // Gérer spécifiquement les erreurs d'authentification Amplify
        print("Error fetching current user (AuthException): ${e.message}");
        if (e is SignedOutException) {
          print("User is signed out.");
           _updateCurrentUser(null); // Assurer que l'utilisateur est null
        } else {
          _setError("Impossible de récupérer les informations utilisateur: ${e.message}");
          // Faut-il déconnecter l'utilisateur si on ne peut pas récupérer ses infos ?
          // Probablement oui, pour éviter un état incohérent.
           _updateCurrentUser(null);
        }
     } on Exception catch (e) {
        // Gérer les autres erreurs
        print("Error fetching current user (Exception): $e");
        _setError(ErrorMessages.unknownError);
        _updateCurrentUser(null);
     } finally {
         if (setInitialized) _isInitialized = true; // Marquer comme initialisé si demandé
          // Notifier les listeners des changements (utilisateur, erreur...)
          notifyListeners();
     }
  }


  // Connexion avec Amplify
  Future<bool> login(String email, String password) async {
    _setLoading(true);
    _clearError(); // Utiliser _clearError() qui gère notifyListeners

    try {
      // Utiliser Amplify.Auth.signIn
      final result = await Amplify.Auth.signIn(
        username: email,
        password: password,
      );

      if (result.isSignedIn) {
        print("Login successful for $email");
        // Le listener _listenToAuthEvents va appeler _fetchCurrentUser
        // _setLoading(false); // _fetchCurrentUser s'en chargera
        return true;
      } else {
        // Gérer les étapes suivantes si nécessaire (ex: MFA, custom challenge)
        // Pour une config simple, isSignedIn devrait être true si succès
        print("Login status unexpected: ${result.nextStep.signInStep}");
        _setError("Statut de connexion inattendu.");
        _setLoading(false);
        return false;
      }
    } on AuthException catch (e) {
      print("Login AuthException: ${e.message}");
       // Gérer les erreurs spécifiques de Cognito
       if (e is UserNotFoundException) {
          _setError(ErrorMessages.invalidCredentials); // Ou "Utilisateur non trouvé."
       } else if (e is NotAuthorizedException) {
          _setError(ErrorMessages.invalidCredentials);
       } else if (e is UserNotConfirmedException) {
          print("User $email is not confirmed.");
          _setError("Veuillez confirmer votre compte avant de vous connecter.");
          _needsConfirmation = true; // Marquer pour UI
          _confirmationUsername = email; // Stocker pour renvoi/confirmation
       } else {
          _setError(e.message); // Message d'erreur Cognito
       }
       _setLoading(false);
       return false;
    } on Exception catch (e) {
       print("Login Generic Exception: $e");
       _setError(ErrorMessages.connectionFailed);
       _setLoading(false);
       return false;
    }
  }


  // Inscription avec Amplify
  Future<bool> register(String email, String password, String name) async {
    _setLoading(true);
    _clearError();

    try {
      // Définir les attributs utilisateur (doivent correspondre à ceux configurés dans Cognito)
      final userAttributes = {
        CognitoUserAttributeKey.email: email,
        CognitoUserAttributeKey.name: name,
        // Ajoutez d'autres attributs si nécessaire ici
        // CognitoUserAttributeKey.phoneNumber: '+15551234567',
      };

      // Utiliser Amplify.Auth.signUp
      final result = await Amplify.Auth.signUp(
        username: email, // L'email est utilisé comme username ici
        password: password,
        options: SignUpOptions(userAttributes: userAttributes),
      );

      // Vérifier si l'inscription nécessite une confirmation
      if (result.isSignUpComplete) {
        print("Sign up complete for $email (auto-verified or no verification needed)");
        _needsConfirmation = false;
        // L'utilisateur peut maintenant se connecter directement (si auto-vérifié)
        _setLoading(false);
        return true; // Indiquer succès, mais pas connecté encore
      } else if (result.nextStep.signUpStep == AuthSignUpStep.confirmSignUp) {
        print("Sign up requires confirmation for $email. Code delivery: ${result.nextStep.codeDeliveryDetails}");
        _needsConfirmation = true; // Indiquer qu'il faut confirmer
        _confirmationUsername = email; // Stocker l'email pour l'étape de confirmation
        _setLoading(false);
        return true; // Indiquer succès de l'étape d'inscription, mais attente confirmation
      } else {
         // Cas inattendu
          print("Sign up status unexpected: ${result.nextStep.signUpStep}");
         _setError("Statut d'inscription inattendu.");
          _setLoading(false);
          return false;
      }

    } on AuthException catch (e) {
       print("Register AuthException: ${e.message}");
       // Gérer les erreurs (UsernameExistsException, InvalidPasswordException, etc.)
       if (e is UsernameExistsException) {
         _setError(ErrorMessages.emailInUse);
       } else if (e is InvalidPasswordException) {
          _setError("Le mot de passe ne respecte pas les critères requis."); // Ou un message plus précis basé sur la politique Cognito
       } else {
          _setError(e.message);
       }
       _setLoading(false);
       return false;
    } on Exception catch (e) {
      print("Register Generic Exception: $e");
      _setError(ErrorMessages.connectionFailed);
      _setLoading(false);
      return false;
    }
  }

  // Confirmer l'inscription avec le code reçu (par email)
  Future<bool> confirmSignUp(String confirmationCode) async {
     if (_confirmationUsername == null) {
        _setError("Aucun utilisateur à confirmer.");
        return false;
     }
     _setLoading(true);
     _clearError();

     try {
        final result = await Amplify.Auth.confirmSignUp(
           username: _confirmationUsername!, // Utiliser l'email stocké
           confirmationCode: confirmationCode,
        );

        if (result.isSignUpComplete) {
           print("Sign up confirmed successfully for $_confirmationUsername!");
           _needsConfirmation = false;
           _confirmationUsername = null;
           _setLoading(false);
           // L'utilisateur peut maintenant se connecter (login page)
           return true;
        } else {
            // Ne devrait pas arriver si isSignUpComplete est le critère de succès
             print("Confirmation status unexpected: ${result.nextStep.signUpStep}");
             _setError("Statut de confirmation inattendu.");
             _setLoading(false);
             return false;
        }
     } on AuthException catch (e) {
         print("Confirm SignUp AuthException: ${e.message}");
          if (e is CodeMismatchException) {
             _setError("Code de confirmation invalide.");
          } else if (e is ExpiredCodeException) {
             _setError("Le code de confirmation a expiré. Veuillez en demander un nouveau.");
          } else if (e is UserNotFoundException) {
              _setError("Utilisateur non trouvé pour la confirmation."); // Ne devrait pas arriver
          } else if (e is AliasExistsException) {
               _setError("Cet email est déjà associé à un autre compte confirmé.");
          }
          else {
              _setError(e.message);
          }
          _setLoading(false);
          return false;
     } on Exception catch (e) {
         print("Confirm SignUp Generic Exception: $e");
         _setError(ErrorMessages.unknownError);
         _setLoading(false);
         return false;
     }
  }

   // Renvoyer le code de confirmation
  Future<bool> resendConfirmationCode() async {
     if (_confirmationUsername == null) {
        _setError("Aucun utilisateur trouvé pour renvoyer le code.");
        return false;
     }
     _setLoading(true); // Peut-être utiliser un indicateur spécifique ?
     _clearError();

     try {
        await Amplify.Auth.resendSignUpCode(username: _confirmationUsername!);
        print("Confirmation code resent successfully for $_confirmationUsername");
        _setLoading(false);
        // Afficher un message de succès dans l'UI (ex: SnackBar)
        return true;
     } on AuthException catch (e) {
        print("Resend Code AuthException: ${e.message}");
         // Gérer LimitExceededException etc.
         _setError("Erreur lors du renvoi du code: ${e.message}");
         _setLoading(false);
         return false;
     } on Exception catch (e) {
        print("Resend Code Generic Exception: $e");
        _setError(ErrorMessages.unknownError);
        _setLoading(false);
         return false;
     }
  }


  // Déconnexion avec Amplify
  Future<void> logout() async {
    _setLoading(true); // Indiquer chargement pendant la déconnexion
    _clearError();
    try {
      await Amplify.Auth.signOut();
      // Le listener _listenToAuthEvents va mettre _currentUser à null
      print("Logout successful via Amplify.");
    } on AuthException catch (e) {
      print('Error during Amplify logout: ${e.message}');
      _setError("Erreur lors de la déconnexion: ${e.message}");
      // Même en cas d'erreur, on force l'état local à déconnecté
      _updateCurrentUser(null);
      notifyListeners();
    } finally {
       _setLoading(false); // Arrêter le chargement
    }
  }


  // Demander réinitialisation de mot de passe avec Amplify
  Future<bool> requestPasswordReset(String email) async {
    _setLoading(true);
    _clearError();

    try {
      final result = await Amplify.Auth.resetPassword(username: email);
      // Vérifier l'étape suivante pour confirmer que le code a été envoyé
      if (result.nextStep.updateStep == AuthResetPasswordStep.confirmResetPasswordWithCode) {
         print("Password reset code sent to: ${result.nextStep.codeDeliveryDetails?.destination}");
          _setLoading(false);
         return true; // Succès, l'utilisateur doit entrer le code
      } else {
          print("Password reset status unexpected: ${result.nextStep.updateStep}");
          _setError("Statut de réinitialisation inattendu.");
           _setLoading(false);
          return false;
      }
    } on AuthException catch (e) {
      print("Reset Password AuthException: ${e.message}");
      // Gérer UserNotFoundException, LimitExceededException etc.
      if (e is UserNotFoundException) {
         _setError("Aucun compte trouvé pour cet email.");
      } else {
         _setError("Erreur lors de la demande de réinitialisation: ${e.message}");
      }
       _setLoading(false);
      return false;
    } on Exception catch (e) {
      print("Reset Password Generic Exception: $e");
      _setError(ErrorMessages.connectionFailed);
      _setLoading(false);
      return false;
    }
  }

   // Confirmer la réinitialisation avec Amplify
   Future<bool> confirmPasswordReset(String email, String newPassword, String confirmationCode) async {
      _setLoading(true);
      _clearError();

      try {
         await Amplify.Auth.confirmResetPassword(
            username: email, // Important: utiliser l'email/username fourni
            newPassword: newPassword,
            confirmationCode: confirmationCode,
         );
         print("Password reset confirmed successfully for $email");
          _setLoading(false);
          // L'utilisateur peut maintenant se connecter avec le nouveau mot de passe
          return true;
      } on AuthException catch (e) {
          print("Confirm Password Reset AuthException: ${e.message}");
          // Gérer CodeMismatchException, ExpiredCodeException, InvalidPasswordException etc.
           _setError("Erreur confirmation réinitialisation: ${e.message}");
           _setLoading(false);
           return false;
      } on Exception catch (e) {
         print("Confirm Password Reset Generic Exception: $e");
         _setError(ErrorMessages.unknownError);
         _setLoading(false);
         return false;
      }
   }

  // Mettre à jour le nom d'utilisateur (attribut 'name') avec Amplify
  Future<bool> updateUserName(String newName) async {
    if (_currentUser == null) {
      _setError("Utilisateur non connecté.");
      return false;
    }
    _setLoading(true);
    _clearError();

    try {
      // Créer l'attribut à mettre à jour
      final attributes = [
        AuthUserAttribute(
          userAttributeKey: CognitoUserAttributeKey.name,
          value: newName,
        ),
        // Ajoutez d'autres attributs à mettre à jour ici si nécessaire
      ];

      // Appeler Amplify.Auth.updateUserAttributes
      await Amplify.Auth.updateUserAttributes(attributes: attributes);
      print("User attribute 'name' update requested successfully.");

      // Mettre à jour l'objet utilisateur local IMMÉDIATEMENT pour l'UI
      // (Même si la mise à jour réelle peut prendre un instant côté backend)
      _updateCurrentUser(_currentUser!.copyWith(displayName: newName));

      _setLoading(false);
      // Pas besoin de notifyListeners() ici car _updateCurrentUser le fait déjà
      return true;

    } on AuthException catch (e) {
      print("Update User Name AuthException: ${e.message}");
      // Gérer les erreurs potentielles (ex: validation, alias exists si on change email/phone)
      _setError("Erreur mise à jour du nom: ${e.message}");
       _setLoading(false);
      return false;
    } on Exception catch (e) {
      print("Update User Name Generic Exception: $e");
      _setError(ErrorMessages.unknownError);
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
      // Nettoyer le message d'erreur Amplify pour être plus user-friendly si besoin
      String? finalMessage = errorMessage;
      if (errorMessage != null && errorMessage.contains("]")) {
          finalMessage = errorMessage.substring(errorMessage.indexOf("]") + 1).trim();
      }

      if (_error == finalMessage && finalMessage != null) return; // Éviter notif si même erreur

      _error = finalMessage;
      // Notifier SEULEMENT si l'erreur change OU si loading change aussi
      // Le notifyListeners() est appelé à la fin des méthodes publiques
      // ou dans _setLoading / _updateCurrentUser
   }

  // Appelé avant chaque opération pour effacer l'erreur précédente
  void _clearError() {
     if (_error != null) {
        _error = null;
        notifyListeners(); // Notifier si l'erreur est effacée
     }
   }

   // Met à jour l'utilisateur et notifie
   void _updateCurrentUser(User? user) {
      // Vérifier si l'utilisateur a réellement changé
      if (_currentUser?.uid == user?.uid &&
          _currentUser?.displayName == user?.displayName &&
          _currentUser?.email == user?.email) return;

       _currentUser = user;
       notifyListeners(); // Notifier le changement d'utilisateur
   }

   // --- Nettoyage ---
   @override
   void dispose() {
      // Annuler l'écouteur Amplify Hub (important !)
      // Malheureusement, Amplify ne fournit pas de méthode simple pour annuler
      // un listener spécifique. La meilleure approche est de s'assurer que
      // AuthService est un singleton ou fourni au plus haut niveau et ne
      // se 'dispose' que lorsque l'application se ferme réellement.
      print("AuthService dispose called - Hub listener cancellation might be complex.");
      super.dispose();
   }

  // Supprimer _validateToken car Amplify gère la session
  /*
  Future<bool> _validateToken(String token) async { ... }
  */
}
```

---

**Étape 3 : Modifications dans l'Interface Utilisateur (UI)**

1.  **`main.dart` :**
    *   Votre `_configureAmplify()` est parfait. Assurez-vous qu'il est bien appelé *avant* `runApp`.
    *   La structure de `MultiProvider` et `ChangeNotifierProxyProvider` pour `AuthService` et `TickService` semble correcte.

2.  **`SplashScreen` / Écran de Démarrage :**
    *   Actuellement, votre `SplashScreen` navigue directement vers `LoginPage` ou `RegisterPage`. Il faudrait plutôt vérifier l'état d'authentification *réel* au démarrage.
    *   Transformez `SplashScreen` en `StatefulWidget`.
    *   Dans `initState`, appelez une méthode (ex: `_checkAuthState`) qui utilise `Provider.of<AuthService>(context, listen: false)` pour vérifier `isInitialized` et `isAuthenticated`.
        ```dart
        class _SplashScreenState extends State<SplashScreen> {
          @override
          void initState() {
            super.initState();
            _checkAuthState();
          }

          Future<void> _checkAuthState() async {
            // Attendre que AuthService soit initialisé
            final authService = Provider.of<AuthService>(context, listen: false);
            // Boucle simple pour attendre l'initialisation (ou utiliser un FutureBuilder/listener)
            while (!authService.isInitialized) {
                await Future.delayed(const Duration(milliseconds: 100));
                // Vérifier si le widget est toujours monté
                if (!mounted) return;
            }

            // Une fois initialisé, vérifier l'authentification
            if (authService.isAuthenticated) {
              // Utilisateur connecté -> Aller à la liste des Ticks
              Navigator.pushReplacementNamed(context, Routes.tickList);
            } else {
              // Utilisateur non connecté -> Aller à la page de bienvenue/login
              Navigator.pushReplacementNamed(context, Routes.welcome); // Ou Routes.login directement
            }
          }

          @override
          Widget build(BuildContext context) {
            // Afficher un indicateur de chargement pendant la vérification
            return const Scaffold(
              body: Center(child: LoadingIndicator()), // Votre widget LoadingIndicator
            );
          }
        }
        ```
    *   Assurez-vous que `SplashScreen` est bien la `home` de votre `MaterialApp` ou la première route chargée.

3.  **`LoginPage` :**
    *   Dans `_submitLogin`, remplacez l'appel `http` par :
        ```dart
        // final success = await authService.login(email, password); // LIGNE ACTUELLE
        final loginSuccess = await authService.login(email, password);

        if (!mounted) return; // Vérifier si le widget est toujours là après l'await

        if (loginSuccess) {
            if (authService.needsConfirmation) {
               // Si la connexion échoue car l'utilisateur n'est pas confirmé
               CustomSnackBar.showError(context, authService.error ?? "Confirmation requise");
               // Optionnel: Naviguer vers l'écran de confirmation
               Navigator.push(context, MaterialPageRoute(builder: (_) => ConfirmationPage())); // Écran à créer
            } else {
               // Connexion réussie et confirmée
               Navigator.pushNamedAndRemoveUntil(context, Routes.tickList, (route) => false);
            }
        } else {
            // Afficher l'erreur renvoyée par AuthService (déjà géré)
            CustomSnackBar.showError(context, authService.error ?? ErrorMessages.unknownError);
        }
        ```
    *   Dans `_handleForgotPassword`, remplacez l'appel `http` par `authService.requestPasswordReset(result)`. Vous aurez besoin d'un écran/dialogue pour entrer le code et le nouveau mot de passe, qui appellera `authService.confirmPasswordReset`.

4.  **`RegisterPage` :**
    *   Dans `_submitRegistration`, remplacez l'appel `http` par :
        ```dart
        // final success = await authService.register(email, password, name); // LIGNE ACTUELLE
        final registerSuccess = await authService.register(email, password, name);

        if (!mounted) return;

        if (registerSuccess) {
            if (authService.needsConfirmation) {
              // Inscription réussie, MAIS nécessite confirmation
              // Naviguer vers un écran de confirmation
              Navigator.pushReplacement( // Remplacer pour ne pas pouvoir revenir à l'inscription
                  context,
                  MaterialPageRoute(builder: (_) => ConfirmationPage()), // Écran à créer
              );
              // Afficher un message indiquant de vérifier l'email
               CustomSnackBar.show(
                  context,
                  message: "Inscription réussie ! Veuillez vérifier votre email pour le code de confirmation.",
                  type: AlertType.info,
                  duration: const Duration(seconds: 6), // Plus long
               );
            } else {
               // Inscription réussie ET auto-confirmée (rare, dépend config Cognito)
                // Naviguer vers la page de connexion pour que l'utilisateur se connecte
               Navigator.pushNamedAndRemoveUntil(context, Routes.login, (route) => false);
                CustomSnackBar.showSuccess(context, "Inscription réussie ! Vous pouvez maintenant vous connecter.");
            }
        } else {
            // Afficher l'erreur (ex: email déjà utilisé) (déjà géré)
            CustomSnackBar.showError(context, authService.error ?? ErrorMessages.unknownError);
        }
        ```

5.  **Nouvel Écran : `ConfirmationPage` (à créer)**
    *   Ce nouvel écran (`lib/screens/auth/confirmation_page.dart`) est nécessaire si votre Cognito demande une confirmation.
    *   Il doit contenir :
        *   Un champ de texte (`TextFormField`) pour entrer le code de confirmation.
        *   Un bouton "Confirmer" qui appelle `authService.confirmSignUp(code)`. En cas de succès, naviguez vers `LoginPage`.
        *   Un bouton/lien "Renvoyer le code" qui appelle `authService.resendConfirmationCode()`.
        *   Afficher les messages d'erreur/succès de `AuthService`.
        *   Utiliser `Consumer<AuthService>` pour afficher `isLoading` et `error`.

6.  **Nouvel Écran/Dialogue : `ResetPasswordPage` (à créer)**
    *   Il faut un moyen pour l'utilisateur d'entrer le code reçu par email ET le nouveau mot de passe.
    *   Cela peut être une nouvelle page ou un dialogue affiché après `requestPasswordReset`.
    *   Contient :
        *   Champ pour le code.
        *   Champ pour le nouveau mot de passe.
        *   Champ pour confirmer le nouveau mot de passe.
        *   Bouton "Confirmer" qui appelle `authService.confirmPasswordReset(email, newPassword, code)`. En cas de succès, naviguez vers `LoginPage`.

7.  **`ProfilePage`:**
    *   La logique pour `updateUserName` appelle maintenant la bonne méthode `AuthService`. C'est bon.
    *   Le bouton "Changer le mot de passe" devrait naviguer vers un écran dédié qui prend l'ancien et le nouveau mot de passe (non couvert ici, mais utilise `Amplify.Auth.updatePassword`).

8.  **`ApiService` :**
    *   La méthode `getAuthorizationToken` utilise déjà Amplify, c'est parfait ! Elle sera utilisée pour sécuriser vos *autres* appels API (ceux pour les Ticks, etc.) une fois que vous aurez mis en place API Gateway.
    *   **Important :** Le `ApiConfig.baseUrl` devra pointer vers votre **API Gateway**, pas vers des endpoints d'authentification inexistants.

---

**Étape 4 : Tester !**

1.  Lancez l'application.
2.  **Inscription :**
    *   Créez un nouveau compte.
    *   Vérifiez si vous êtes redirigé vers la page de confirmation (c'est le comportement attendu).
    *   Vérifiez votre boîte mail pour le code Cognito.
    *   Entrez le code sur la page de confirmation.
    *   Vérifiez que la confirmation réussit et que vous êtes redirigé vers la page de connexion.
3.  **Connexion :**
    *   Connectez-vous avec l'email et le mot de passe que vous venez de créer.
    *   Vérifiez que vous arrivez sur la `TickListPage`.
4.  **Déconnexion :**
    *   Allez dans le profil (ou via le menu de `TickListPage`) et déconnectez-vous.
    *   Vérifiez que vous revenez à l'écran de bienvenue/login.
5.  **Mot de passe oublié :**
    *   Sur la page de login, cliquez sur "Mot de passe oublié".
    *   Entrez votre email.
    *   Vérifiez votre boîte mail pour le code de réinitialisation.
    *   Entrez le code et votre nouveau mot de passe sur l'écran/dialogue correspondant.
    *   Essayez de vous reconnecter avec le nouveau mot de passe.
6.  **Session persistante :**
    *   Fermez complètement l'application après vous être connecté.
    *   Relancez l'application.
    *   Vérifiez que le `SplashScreen` vous redirige directement vers `TickListPage` (grâce à `_initializeAuth` et `fetchAuthSession`).

---

**Prochaines étapes (après avoir fait fonctionner l'authentification) :**

1.  **API Gateway & Lambdas :** Pour que vos appels API dans `TickService` (`fetchTicks`, `associateTick`, etc.) fonctionnent, vous devrez :
    *   Créer une API Gateway dans AWS.
    *   Créer des fonctions Lambda (en Node.js, Python...) qui contiendront la logique métier (interagir avec une base de données comme DynamoDB pour stocker les Ticks, etc.).
    *   Connecter les routes API Gateway (ex: `/ticks` en GET) aux fonctions Lambda correspondantes.
    *   **Sécuriser l'API Gateway** en utilisant votre **Cognito User Pool comme Authorizer**. Ainsi, seuls les utilisateurs connectés via votre app pourront appeler ces endpoints, et l'API Gateway vérifiera automatiquement le token JWT (`Authorization: Bearer ...`) que votre `ApiService` envoie.
    *   Mettre à jour `ApiConfig.baseUrl` dans votre code Flutter avec l'URL de votre API Gateway déployée.
2.  **Base de données (DynamoDB ?) :** Vos Lambdas auront besoin d'une base de données pour stocker les informations sur les Ticks, leur association avec les utilisateurs, leur historique, etc. DynamoDB est un choix courant et s'intègre bien avec Lambda et API Gateway.

C'est un gros morceau, mais en suivant ces étapes, vous devriez pouvoir remplacer votre système d'authentification actuel par AWS Cognito en utilisant Amplify. N'hésitez pas si certains points ne sont pas clairs ! Bon courage !
C'est un gros morceau, mais en suivant ces étapes, vous devriez pouvoir remplacer votre système d'authentification actuel par AWS Cognito en utilisant Amplify. N'hésitez pas si certains points ne sont pas clairs ! Bon courage !