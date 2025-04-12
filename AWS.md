Okay, let's integrate AWS Cognito for real user authentication (Sign Up & Sign In) into your Flutter app. This replaces the simulated logic in `AuthService`.

We'll use **Amazon Cognito User Pools**, which acts as your user directory in the cloud. Users will register and sign in against this pool.

There are two main ways to interact with Cognito from Flutter:

1.  **AWS Amplify Flutter:** A framework provided by AWS that simplifies integration with various AWS services, including Cognito. It uses a CLI to configure the backend and provides Flutter libraries. **(Recommended for beginners)**
2.  **AWS SDK for Flutter (e.g., `aws_cognito_identity_provider`):** Lower-level SDKs give more granular control but require more manual setup and handling of authentication flows.

We'll proceed with **Amplify Flutter** as it significantly reduces boilerplate and complexity.

**Overall Steps:**

1.  **Set up AWS Amplify CLI:** Install and configure the command-line tool.
2.  **Initialize Amplify in your Flutter project:** Connect your project to an AWS backend environment.
3.  **Add the "Auth" category:** Configure Cognito via the Amplify CLI.
4.  **Push changes to AWS:** Create the Cognito User Pool in your AWS account.
5.  **Configure Amplify in Flutter:** Initialize Amplify within your `main.dart`.
6.  **Modify `AuthService`:** Replace simulation logic with Amplify Auth calls.
7.  **Modify `ApiService`:** Fetch the real authentication token (JWT) from Amplify to use in API calls.
8.  **(Future Step) Protect API Gateway:** Configure your API Gateway endpoints (like `/ticks`) to require a valid Cognito token.

---

**Step 1: Set up AWS Amplify CLI**

If you haven't already, you need Node.js and npm installed.

```bash
npm install -g @aws-amplify/cli
amplify configure
```

The `amplify configure` command will walk you through:
*   Signing into your AWS account via the browser.
*   Creating an IAM (Identity and Access Management) user specifically for Amplify to manage resources on your behalf. **Follow the prompts carefully**, ensuring you give the IAM user the necessary permissions (usually administrator access during setup).
*   Saving the Access Key ID and Secret Access Key for this IAM user locally, so the CLI can interact with your AWS account.

---

**Step 2: Initialize Amplify in your Flutter Project**

Navigate to the root directory of your Flutter project in your terminal and run:

```bash
amplify init
```

Follow the prompts:
*   **Enter a name for the project:** (e.g., `ticktracker`)
*   **Enter a name for the environment:** (e.g., `dev`)
*   **Choose your default editor:** (Select your preferred editor)
*   **Choose the type of app you're building:** `flutter`
*   **Choose the AWS profile you want to use:** Select the profile you configured with `amplify configure`.

This will create an `amplify` folder in your project and initial backend configuration files.

---

**Step 3: Add the "Auth" Category (Configure Cognito)**

Run the following command:

```bash
amplify add auth
```

Follow the prompts. For a basic email/password setup matching your current flow:
*   **Do you want to use the default authentication and security configuration?** `Default configuration` (Easiest start)
*   **How do you want users to be able to sign in?** `Email`
*   **Do you want to configure advanced settings?** `No, I am done.` (You can revisit this later for MFA, etc.)

This configures how your Cognito User Pool will behave.

---

**Step 4: Push Changes to AWS**

This command actually creates the Cognito User Pool and associated resources in your AWS account based on the configuration from the previous step.

```bash
amplify push --y
```

*   This process can take a few minutes.
*   It will show you the resources being created/updated.
*   Crucially, it will generate a file named `amplifyconfiguration.dart` inside your `lib/` directory. **Do NOT commit this file to public git repositories if it contains sensitive info**, although for basic auth it usually doesn't. The standard `.gitignore` provided by Flutter often ignores it.

**After this step, log in to your AWS Management Console:**
*   Go to the **Amazon Cognito** service.
*   Select **User Pools**.
*   You should see a new User Pool created by Amplify (e.g., `ticktrackerxxxxxxx_userpool_xxxxxxx`).
*   Explore its settings (especially under "App integration" -> "App client list") to see the configuration done by Amplify. Note the **User pool ID** and the **Client ID**.

---

**Step 5: Configure Amplify in Flutter (`main.dart`)**

1.  **Add Dependencies:** Add the necessary Amplify packages to your `pubspec.yaml`:

    ```yaml
    dependencies:
      flutter:
        sdk: flutter
      # ... other dependencies
      amplify_flutter: ^1.0.0 # Use the latest compatible version
      amplify_auth_cognito: ^1.0.0 # Use the latest compatible version
      # ... other dependencies
    ```
    Run `flutter pub get`.

2.  **Initialize Amplify in `main.dart`:** You need to initialize Amplify before `runApp`.

    ```dart
    // lib/main.dart
    import 'package:flutter/material.dart';
    import 'package:provider/provider.dart';

    // Import Amplify packages
    import 'package:amplify_flutter/amplify_flutter.dart';
    import 'package:amplify_auth_cognito/amplify_auth_cognito.dart';

    // Import your generated configuration file
    import 'amplifyconfiguration.dart';

    // ... (other imports: services, screens, utils)

    Future<void> main() async { // Make main async
      WidgetsFlutterBinding.ensureInitialized(); // Ensure bindings are ready

      try {
        // Configure Amplify ONLY ONCE at startup
        await _configureAmplify();
        print("Amplify configured successfully");
      } catch (e) {
        print("Error configuring Amplify: $e");
        // Handle configuration error (e.g., show an error message or exit)
        // It's critical Amplify configures correctly.
      }

      // Run the app after configuration
      runApp(const MyApp());
    }

    // Function to configure Amplify
    Future<void> _configureAmplify() async {
       // Create and add the Auth plugin
       final authPlugin = AmplifyAuthCognito();
       await Amplify.addPlugin(authPlugin);

       // Configure Amplify with the generated file
       // Note: Amplify.configure should only be called once per app launch.
       // Use Amplify.isConfigured to check if already configured if necessary.
       if (!Amplify.isConfigured) {
          await Amplify.configure(amplifyconfig);
       }
    }


    class MyApp extends StatelessWidget {
      const MyApp({super.key});

      @override
      Widget build(BuildContext context) {
        return MultiProvider(
           // ... (Your existing providers: Theme, Bluetooth, Api, Auth, Tick) ...
           providers: [
             ChangeNotifierProvider(create: (_) => ThemeService()),
             ChangeNotifierProvider(create: (_) => BluetoothService()..initialize()),
             Provider(create: (_) => ApiService()),
             ChangeNotifierProvider(
               create: (context) => AuthService(
                 Provider.of<ApiService>(context, listen: false),
                 // Pass Amplify instance or just use Amplify directly inside AuthService
               ),
             ),
             ChangeNotifierProxyProvider<AuthService, TickService>(
                // ... (TickService provider setup)
             ),
           ],
           child: Consumer<ThemeService>(
             builder: (context, themeService, _) {
               return MaterialApp(
                 // ... (Your MaterialApp setup) ...
                  home: const SplashScreen(), // Start with SplashScreen
               );
             },
           ),
        );
      }
    }
    ```

---

**Step 6: Modify `AuthService`**

Replace the simulation logic with calls to `Amplify.Auth`.

```dart
// lib/services/auth_service.dart
import 'package:flutter/foundation.dart';
import 'package:amplify_flutter/amplify_flutter.dart';
import 'package:amplify_auth_cognito/amplify_auth_cognito.dart'; // Import Cognito specifics
import '../models/user.dart';
import 'api_service.dart';
import '../utils/constants.dart';

class AuthService with ChangeNotifier {
  final ApiService _apiService;
  User? _currentUser;
  bool _isLoading = false;
  String? _error;
  bool _isInitialized = false;
  // Track if user needs to confirm their account
  bool _ GENDER_needsConfirmation = false;
  bool get needsConfirmation => _needsConfirmation;
  String? _confirmationUsername; // Store username for confirmation step

  AuthService(this._apiService) {
    // Listen to Hub events for auth changes (optional but useful)
    _listenToAuthEvents();
    _initializeAuth();
  }

  User? get currentUser => _currentUser;
  bool get isLoading => _isLoading;
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
    // Don't re-initialize if already done by Hub event listener
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

  // Helper to fetch and update current user details
  Future<void> _fetchCurrentUser({bool setInitialized = true}) async {
     try {
        final cognitoUser = await Amplify.Auth.getCurrentUser();
        final attributes = await Amplify.Auth.fetchUserAttributes();
        String displayName = '';
        String email = '';
        for (final attribute in attributes) {
           if (attribute.userAttributeKey == CognitoUserAttributeKey.name) {
              displayName = attribute.value;
           } else if (attribute.userAttributeKey == CognitoUserAttributeKey.email) {
              email = attribute.value;
           }
        }
        _updateCurrentUser(User(
          uid: cognitoUser.userId, // Cognito 'sub' attribute
          email: email,
          displayName: displayName,
        ));
        // Clear confirmation flag on successful fetch (implies logged in and possibly confirmed)
        _needsConfirmation = false;
        _confirmationUsername = null;
     } on Exception catch (e) {
        print("Error fetching current user: $e");
        _setError("Impossible de récupérer les informations utilisateur.");
        _updateCurrentUser(null); // Log out user if fetch fails? Or keep previous state?
     } finally {
         if (setInitialized) _isInitialized = true; // Mark as init if requested
          notifyListeners(); // Always notify after attempting fetch
     }
  }


  Future<bool> login(String email, String password) async {
    _setLoading(true);
    _clearError();
    _needsConfirmation = false; // Reset confirmation flag

    try {
      final result = await Amplify.Auth.signIn(
        username: email,
        password: password,
      );

      if (result.isSignedIn) {
         print("Login Successful");
         await _fetchCurrentUser(); // Fetch details after successful sign-in
         // _setLoading(false) and notifyListeners() are handled by _fetchCurrentUser
         return true;
      } else {
          // This part might not be reached if isSignedIn is the only success criteria
          // Handle other potential SignInResult states if necessary
          print("Login Status: ${result.nextStep?.signInStep}");
           _setError("Statut de connexion inattendu: ${result.nextStep?.signInStep}");
          _setLoading(false);
           return false;
      }

    } on AuthException catch (e) { // Catch specific Amplify Auth exceptions
      print("Login AuthException: ${e.message}");
      // Handle specific errors (user not found, incorrect password, user not confirmed)
      if (e is UserNotFoundException) {
         _setError(ErrorMessages.invalidCredentials);
      } else if (e is NotAuthorizedException) {
         _setError(ErrorMessages.invalidCredentials);
      } else if (e is UserNotConfirmedException) {
         print("User needs confirmation");
          _needsConfirmation = true;
          _confirmationUsername = email; // Store username for confirmation screen
          _setError("Veuillez confirmer votre compte. Vérifiez vos emails.");
          // Optionally, resend confirmation code here:
          // await resendConfirmationCode(email);
      } else {
          _setError(e.message); // Use the message from the exception
      }
       _setLoading(false);
       return false;
    } on Exception catch (e) { // Catch generic exceptions
      print("Login Generic Exception: $e");
      _setError(ErrorMessages.unknownError);
       _setLoading(false);
       return false;
    }
  }

  Future<bool> register(String email, String password, String name) async {
    _setLoading(true);
    _clearError();
    _needsConfirmation = false;

    try {
      // Prepare attributes for Cognito
      final userAttributes = {
        CognitoUserAttributeKey.email: email,
        CognitoUserAttributeKey.name: name,
        // Add other attributes if needed (e.g., given_name, family_name)
      };

      final result = await Amplify.Auth.signUp(
        username: email, // Use email as username
        password: password,
        options: SignUpOptions(userAttributes: userAttributes),
      );

      if (result.isSignUpComplete) {
        print("Registration Successful and Complete (Auto-confirmed or no confirmation needed)");
         // If auto-confirmed, maybe sign them in? Or let them login separately.
         // For simplicity, let's require login after registration.
        _setLoading(false);
         return true; // Indicate success, user needs to login
      } else {
        // Registration requires confirmation step
         print("Registration requires confirmation. Next step: ${result.nextStep.signUpStep}");
         _needsConfirmation = true;
         _confirmationUsername = email; // Store username for confirmation screen
         _setLoading(false);
         // Don't set error here, just indicate confirmation needed
         return true; // Indicate success (registration initiated), but needs confirmation
      }

    } on AuthException catch (e) {
      print("Register AuthException: ${e.message}");
      if (e is UsernameExistsException) {
         _setError(ErrorMessages.emailInUse);
      } else if (e is InvalidPasswordException) {
          _setError("Mot de passe invalide: ${e.message}"); // Provide more details if available
      } else {
          _setError(e.message);
      }
       _setLoading(false);
       return false;
    } on Exception catch (e) {
      print("Register Generic Exception: $e");
      _setError(ErrorMessages.unknownError);
      _setLoading(false);
       return false;
    }
  }

  // **** NOUVELLE MÉTHODE pour confirmer l'inscription ****
  Future<bool> confirmSignUp(String confirmationCode) async {
     if (_confirmationUsername == null) {
        _setError("Aucun utilisateur à confirmer.");
        return false;
     }
     _setLoading(true);
     _clearError();

     try {
        final result = await Amplify.Auth.confirmSignUp(
           username: _confirmationUsername!,
           confirmationCode: confirmationCode,
        );

        if (result.isSignUpComplete) {
           print("Sign up confirmed successfully!");
           _needsConfirmation = false;
           _confirmationUsername = null;
           _setLoading(false);
           // User can now login
           return true;
        } else {
            // Should not happen if isSignUpComplete is the success criteria
             print("Confirmation status: ${result.nextStep.signUpStep}");
             _setError("Statut de confirmation inattendu.");
             _setLoading(false);
             return false;
        }
     } on AuthException catch (e) {
         print("Confirm SignUp AuthException: ${e.message}");
          if (e is CodeMismatchException) {
             _setError("Code de confirmation invalide.");
          } else if (e is ExpiredCodeException) {
             _setError("Le code de confirmation a expiré.");
          } else if (e is UserNotFoundException) {
              _setError("Utilisateur non trouvé pour la confirmation."); // Should not happen if _confirmationUsername is set
          } else {
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

   // **** NOUVELLE MÉTHODE pour renvoyer le code de confirmation ****
  Future<bool> resendConfirmationCode() async {
     if (_confirmationUsername == null) {
        _setError("Aucun utilisateur trouvé pour renvoyer le code.");
        return false;
     }
     _setLoading(true); // Maybe use a different loading flag?
     _clearError();

     try {
        await Amplify.Auth.resendSignUpCode(username: _confirmationUsername!);
        print("Confirmation code resent successfully for $_confirmationUsername");
        _setLoading(false);
        // Afficher un message de succès dans l'UI
        return true;
     } on AuthException catch (e) {
        print("Resend Code AuthException: ${e.message}");
         // Handle errors like LimitExceededException
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


  Future<void> logout() async {
    _setLoading(true);
    _clearError();

    try {
      await Amplify.Auth.signOut();
      // User state is cleared via Hub event listener (_listenToAuthEvents)
       print("Logout successful via Amplify.Auth.signOut");
    } on AuthException catch (e) {
      print("Logout AuthException: $e");
       _setError("Erreur lors de la déconnexion: ${e.message}");
       // Even on error, try to clear local state?
        _updateCurrentUser(null);
        notifyListeners();
    } finally {
       // Hub listener handles setting loading to false and clearing user state
       // _setLoading(false); // Avoid double notification if Hub listener works
       // Ensure initialized is true after attempting logout
       _isInitialized = true;
    }
  }

   // **** MODIFIÉ pour utiliser Amplify ****
  Future<bool> requestPasswordReset(String email) async {
     _setLoading(true);
     _clearError();

     try {
       final result = await Amplify.Auth.resetPassword(username: email);
       print("Password reset initiated. Next step: ${result.nextStep.updateStep}");
        // Le code est envoyé par email par Cognito
        _setLoading(false);
        return true; // Indique que la demande a été envoyée
     } on AuthException catch (e) {
       print("Password Reset AuthException: ${e.message}");
        // Handle UserNotFoundException, LimitExceededException etc.
        _setError("Erreur de réinitialisation: ${e.message}");
        _setLoading(false);
        return false;
     } on Exception catch (e) {
        print("Password Reset Generic Exception: $e");
        _setError(ErrorMessages.unknownError);
        _setLoading(false);
        return false;
     }
  }

   // **** NOUVELLE MÉTHODE pour confirmer la réinitialisation ****
   Future<bool> confirmPasswordReset(String email, String newPassword, String confirmationCode) async {
      _setLoading(true);
      _clearError();

      try {
         await Amplify.Auth.confirmResetPassword(
            username: email,
            newPassword: newPassword,
            confirmationCode: confirmationCode,
         );
         print("Password reset confirmed successfully for $email");
          _setLoading(false);
          // User can now login with the new password
          return true;
      } on AuthException catch (e) {
          print("Confirm Password Reset AuthException: ${e.message}");
          // Handle CodeMismatchException, ExpiredCodeException, etc.
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

  // Update user name (garder la structure, l'implémentation est #TODO API)
  Future<bool> updateUserName(String newName) async {
    if (_currentUser == null) return false;
    _setLoading(true);
    _clearError();
    try {
      // #TODO: Appel API pour mettre à jour le nom (pas une fonction Cognito directe standard)
      // Vous devrez créer un endpoint API Gateway -> Lambda qui utilise l'Admin SDK Cognito
      // ou met à jour votre propre table de profils utilisateur.
      // Pour l'instant, on simule et met à jour localement.

      print("AuthService: Simulating name update API call to '$newName'");
      await Future.delayed(const Duration(seconds: 1));
      bool success = newName.isNotEmpty && !newName.toLowerCase().contains("error_api");

      if (success) {
         // Mettre à jour l'attribut 'name' dans Cognito (via API/Lambda) est complexe.
         // Pour la SIMULATION, on met juste à jour l'objet local.
         _updateCurrentUser(_currentUser!.copyWith(displayName: newName));
         _setLoading(false);
         notifyListeners(); // Notifier le changement de nom
         return true;
      } else {
         _setError("Erreur lors de la mise à jour du nom (simulée)");
          _setLoading(false);
         return false;
      }
    } catch (e) {
      print("Update User Name Error: $e");
       _setError(e is Exception ? e.toString() : ErrorMessages.connectionFailed);
       _setLoading(false);
       return false;
    }
  }

  // Méthodes internes (inchangées)
  void _setLoading(bool loading) {
    if (_isLoading == loading) return;
    _isLoading = loading;
    notifyListeners();
  }
   void _setError(String? errorMessage) {
      if (_error == errorMessage && errorMessage != null) return; // Avoid notifying if same error
      _error = errorMessage;
      // Notify ONLY if error changes OR loading is also changing
      // notifyListeners(); // Déplacé vers la fin des méthodes publiques pour regrouper les notifs
   }
  void _clearError() {
     _setError(null);
  }
   void _updateCurrentUser(User? user) {
      if (_currentUser?.uid == user?.uid && _currentUser?.displayName == user?.displayName) return; // Compare basic fields
       _currentUser = user;
       // Notify listeners sera appelé par la méthode appelante (login, fetch, logout...)
   }

   // Ne pas oublier d'annuler le listener Hub dans dispose
   @override
   void dispose() {
      // TODO: Vérifier comment annuler correctement le listener Amplify Hub si nécessaire
      super.dispose();
   }
}
```

**7. Modify `ApiService`**

Fetch the real JWT from Amplify instead of using the simulation.

```dart
// lib/services/api_service.dart
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:amplify_flutter/amplify_flutter.dart'; // Importer Amplify
import 'package:amplify_auth_cognito/amplify_auth_cognito.dart'; // Importer Cognito
import '../utils/constants.dart';
// flutter_secure_storage n'est plus nécessaire ici car Amplify gère le stockage

class ApiService {

  // ... (get, post, put, delete - garder la logique d'appel http) ...

  // MODIFIER: Renommer et implémenter la récupération du token depuis Amplify
  Future<String?> getAuthorizationToken() async {
     try {
       final session = await Amplify.Auth.fetchAuthSession();
       if (session.isSignedIn) {
          // Utiliser l'ID Token car il contient les infos utilisateur (sub, email, etc.)
          // L'Access Token est plutôt pour autoriser des appels API spécifiques.
           final idToken = session.userPoolTokensResult.value?.idToken.raw;
           // print("Fetched ID Token: ${idToken?.substring(0, 10)}..."); // Log tronqué
           return idToken;
       } else {
         return null; // Pas connecté
       }
     } on AuthException catch (e) {
        print("Error fetching auth session for token: ${e.message}");
        return null;
     }
  }

  // **** SUPPRIMER les méthodes de simulation ****
  // Future<String?> getToken() async { ... } // Supprimer l'ancienne
  // Future<void> saveToken(String token) async { ... } // Supprimer
  // Future<void> deleteToken() async { ... } // Supprimer
  // String? _fakeToken; // Supprimer

  // ... (Reste de la classe : _handleResponse, _fakeTicksData pour simulation API) ...

   // Modifier les méthodes get/post/put/delete pour utiliser getAuthorizationToken()
   Future<Map<String, dynamic>> get(String endpoint, {Map<String, String>? headers}) async {
    // **** MODIFIER: Utiliser getAuthorizationToken ****
    final token = await getAuthorizationToken();
    final defaultHeaders = {
      'Content-Type': 'application/json',
      if (token != null) 'Authorization': 'Bearer $token', // Utiliser le token récupéré
      ...?headers
    };

    print('API GET Request: ${ApiConfig.baseUrl}$endpoint');
    // #TODO: Implémenter l'appel API GET réel avec http
    // final response = await http.get(...)
    // return _handleResponse(response);

    // --- Simulation --- (Garder la simulation pour les endpoints non-auth pour l'instant)
     await Future.delayed(const Duration(milliseconds: 500));
     if (endpoint == ApiEndpoints.fetchTicks) {
       // ... simulation fetchTicks ...
       return {'success': true, 'data': _fakeTicksData};
     } else if (endpoint == ApiEndpoints.getUserProfile) {
        // ... simulation getUserProfile ...
         return {'success': true, 'data': {'userId': 'user_sim_123', 'email': 'sim@example.com', 'name': 'Simulated User'}};
     }
     return {'success': false, 'error': 'Endpoint not implemented in simulation'};
     // --- Fin Simulation ---
  }

   // Faire la même modification pour post, put, delete (récupérer token via getAuthorizationToken)
   Future<Map<String, dynamic>> post(String endpoint, Map<String, dynamic> body, {Map<String, String>? headers}) async {
     final token = await getAuthorizationToken(); // **** MODIFIER ****
     final defaultHeaders = { /* ... */ if (token != null) 'Authorization': 'Bearer $token', /* ... */ };
     // ... (reste de la méthode POST) ...
   }

   Future<Map<String, dynamic>> put(String endpoint, Map<String, dynamic> body, {Map<String, String>? headers}) async {
      final token = await getAuthorizationToken(); // **** MODIFIER ****
      final defaultHeaders = { /* ... */ if (token != null) 'Authorization': 'Bearer $token', /* ... */ };
      // ... (reste de la méthode PUT) ...
   }

    Future<Map<String, dynamic>> delete(String endpoint, {Map<String, String>? headers}) async {
      final token = await getAuthorizationToken(); // **** MODIFIER ****
      final defaultHeaders = { /* ... */ if (token != null) 'Authorization': 'Bearer $token', /* ... */ };
      // ... (reste de la méthode DELETE) ...
   }

   // ... (Reste de ApiService: _handleResponse, _fakeTicksData) ...
   // --- Données pour la simulation API ---
   final List<Map<String, dynamic>> _fakeTicksData = [
       // ... (données factices) ...
   ];

}
```

**8. (Future Step) Protect API Gateway**

This is crucial but done on the AWS side:

*   Go to your API Gateway in the AWS Console.
*   Select your API.
*   Go to "Authorizers".
*   Create a new Authorizer:
    *   **Type:** Cognito
    *   **Cognito User Pool:** Select the User Pool created by Amplify.
    *   **Token Source:** `Authorization` (This is the header where the app sends the `Bearer <ID_Token>`).
*   Go to "Resources".
*   For each method (GET, POST, etc.) on your protected endpoints (like `/ticks`, `/users/me`), edit the "Method Request".
*   Under "Authorization", select the Cognito Authorizer you just created.
*   **Deploy your API** for the changes to take effect.

Now, API Gateway will automatically validate the `Authorization` header token against your Cognito User Pool before forwarding the request to your Lambda functions. Invalid or missing tokens will be rejected with a 401 or 403 error.

**UI Changes Needed:**

*   **Confirmation Screen:** You need a new screen or dialog where the user can enter the confirmation code sent to their email after registration. This screen should call `authService.confirmSignUp()` and `authService.resendConfirmationCode()`. Modify the navigation flow in `RegisterPage` to go to this confirmation screen when `authService.needsConfirmation` is true after `register()` returns successfully.
*   **Password Reset Screens:**
    *   Modify the "Forgot Password" flow in `LoginPage` to navigate to a new screen after `requestPasswordReset()` is successful.
    *   This new screen needs fields for the code (sent via email), the new password, and confirmation of the new password. It should call `authService.confirmPasswordReset()`.

This is a significant update. Test each part carefully: registration flow (including confirmation), login flow, logout, password reset request, and confirm password reset. Then verify that authenticated API calls (like fetching ticks) work correctly once you protect API Gateway.

Okay, great! You've successfully integrated the Amplify Auth foundation into your Flutter app. Now, let's connect the dots and make the authentication flow fully operational by handling the UI aspects and securing your backend API.

Here are the next detailed steps:

**Step 8: Protect API Gateway Endpoints**

This is a *critical security step* performed in the **AWS Management Console**. Right now, your `ApiService` is sending the authentication token, but your API Gateway isn't configured to *validate* it.

1.  **Navigate to API Gateway:** Log in to your AWS Console and go to the API Gateway service. Find the API that Amplify created for your backend (or the one you created manually if you didn't use `amplify add api`).
2.  **Create a Cognito Authorizer:**
    *   In your API's navigation pane, click on **Authorizers**.
    *   Click **Create Authorizer**.
    *   **Authorizer name:** Give it a descriptive name (e.g., `TickTrackerCognitoAuthorizer`).
    *   **Type:** Select `Cognito`.
    *   **Cognito User Pool:** Choose the User Pool created by Amplify in Step 4 (its name usually includes your project name and environment).
    *   **Token Source:** Enter `Authorization`. This tells API Gateway to look for the token in the `Authorization` HTTP header (where your app sends `Bearer <token>`).
    *   **Token Validation:** Leave empty (API Gateway handles validation against the pool).
    *   Click **Create Authorizer**.
3.  **Apply the Authorizer to Your Methods:**
    *   In your API's navigation pane, click on **Resources**.
    *   Navigate to each resource (path) that requires authentication (e.g., `/ticks`, `/users/me`, `/ticks/{tickId}/locate`).
    *   For *each HTTP method* on that resource (e.g., GET, POST, PUT, DELETE under `/ticks`), click on the method name.
    *   Click on **Method Request**.
    *   Under **Settings**:
        *   Click the pencil icon next to **Authorization**.
        *   Select the Cognito Authorizer you just created (e.g., `TickTrackerCognitoAuthorizer`).
        *   Click the checkmark icon to save.
    *   Repeat this for *all methods* on *all resources* that should be protected. Authentication endpoints like `/auth/login` or `/auth/register` should *not* have an authorizer applied.
4.  **Deploy Your API:** After applying the authorizer to all necessary methods, you **must deploy your API** for the changes to take effect.
    *   Click on **Actions** (usually near the top).
    *   Select **Deploy API**.
    *   Choose a **Deployment stage** (e.g., `dev`, `prod` - the one Amplify uses or one you created).
    *   Click **Deploy**.

**Result:** Now, API Gateway will automatically check for a valid Cognito ID token in the `Authorization` header for the protected routes. If the token is missing, invalid, or expired, API Gateway will reject the request with a `401 Unauthorized` or `403 Forbidden` error *before* it even reaches your Lambda function. Your Lambda functions can now trust that if a request arrives, it came from an authenticated user.

---

**Step 9: Implement UI for Confirmation & Password Reset**

Your `AuthService` now handles the *logic* for confirming sign-up and resetting passwords using Amplify, but your UI needs screens/dialogs for the user interaction.

1.  **Sign-Up Confirmation Screen:**
    *   **Create a new screen:** `lib/screens/auth/confirm_signup_page.dart`.
    *   **Navigation:** Modify `RegisterPage`'s `_submitRegistration` function. If `authService.register()` returns `true` AND `authService.needsConfirmation` is `true`, navigate to `ConfirmSignUpPage` instead of `TickListPage`. Pass the `email` (which is `_confirmationUsername` in `AuthService`) as an argument to this new page.
    *   **UI Elements:**
        *   Display a message explaining that a code was sent to the user's email.
        *   A `TextFormField` for the user to enter the confirmation code.
        *   A "Confirm" button.
        *   A "Resend Code" button/link.
    *   **Logic:**
        *   The "Confirm" button calls `authService.confirmSignUp(code)`. On success, navigate to the `LoginPage` (or maybe `TickListPage` if you want to auto-login, though explicit login is often clearer). On failure, display the error from `authService.error`.
        *   The "Resend Code" button calls `authService.resendConfirmationCode()`. Show a success/error message using `CustomSnackBar`.

2.  **Password Reset Confirmation Screen:**
    *   **Create a new screen:** `lib/screens/auth/confirm_reset_password_page.dart`.
    *   **Navigation:** Modify `LoginPage`'s `_handleForgotPassword` (or the dialog it shows). If `authService.requestPasswordReset()` returns `true`, navigate to `ConfirmResetPasswordPage`. Pass the `email` as an argument.
    *   **UI Elements:**
        *   Display the email address for clarity.
        *   A `TextFormField` for the confirmation code (sent via email).
        *   A `TextFormField` for the new password (use `Validators.validatePassword`).
        *   A `TextFormField` to confirm the new password (use `Validators.validateConfirmPassword`).
        *   A "Set New Password" button.
    *   **Logic:**
        *   The button calls `authService.confirmPasswordReset(email, newPassword, code)`. On success, navigate to the `LoginPage` with a success message. On failure, display the error from `authService.error`.

---

**Step 10: Implement Backend APIs (Lambda & API Gateway)**

This is where you replace the `#TODO`s for actual API logic in `ApiService` by building the corresponding backend. This involves creating Lambda functions and connecting them to API Gateway endpoints (which you just protected in Step 8).

**General Workflow for Each Protected Endpoint (e.g., `GET /ticks`):**

1.  **Create Lambda Function (AWS Console or Amplify CLI):**
    *   Go to the Lambda service in the AWS Console.
    *   Click "Create function".
    *   Choose "Author from scratch".
    *   **Function name:** e.g., `tickTrackerGetTicksFunction`.
    *   **Runtime:** Choose Node.js or Python (popular choices).
    *   **Permissions:** Create or use an execution role that has permissions to access **DynamoDB** (read permissions for this specific function, like `dynamodb:Query`) and potentially **CloudWatch Logs** for logging.
2.  **Write Lambda Code:**
    *   Use the AWS SDK for your chosen runtime (e.g., `@aws-sdk/client-dynamodb` for Node.js, `boto3` for Python).
    *   **Get User ID:** When API Gateway invokes the Lambda via the Cognito Authorizer, the user's information (including the `sub` - unique user ID) is available in the `event.requestContext.authorizer.claims` object. Extract the `sub`.
    *   **Interact with DynamoDB:**
        *   For `GET /ticks`: Perform a DynamoDB `Query` operation on your table using the `USER#{userId}` as the Partition Key (PK) and a condition where the Sort Key (SK) `begins_with('TICK#')`.
        *   For `POST /ticks` (Associate): Get `userId` from claims, `tickName` & `macAddress` from `event.body`. Check if `TICK#{macAddress}` PK exists. If not, create the `USER#{userId}/TICK#{tickId}` and `TICK#{tickId}/METADATA` items using `PutItem`. **Crucially:** Generate a unique `tickId` within the Lambda (e.g., using `uuid`). After successful DynamoDB write, use the AWS IoT SDK (`@aws-sdk/client-iot-data-plane` or `boto3.client('iot-data')`) to publish the `link_device` MQTT message to the specific Tick via AWS IoT Core.
        *   For `PUT /users/me`: Get `userId` from claims, `name` from `event.body`. Update the corresponding user item in DynamoDB. (Updating Cognito attributes directly from Lambda requires Admin SDK calls and more complex permissions).
        *   For `DELETE /ticks/{tickId}`: Get `userId` from claims, `tickId` from `event.pathParameters`. Delete the `USER#{userId}/TICK#{tickId}` item. *Also* delete the `TICK#{tickId}/METADATA` item and potentially the history items `USER#{userId}/HISTORY#...` associated with that tick. Send the `unlink_device` MQTT command via IoT Core.
    *   **Return Response:** Format the response according to API Gateway Lambda proxy integration requirements (statusCode, headers, body as a JSON string).
3.  **Connect API Gateway Method to Lambda:**
    *   Go back to API Gateway -> Resources.
    *   Select the method (e.g., `GET` under `/ticks`).
    *   Click **Integration Request**.
    *   **Integration type:** `Lambda Function`.
    *   **Lambda Region:** Your AWS region.
    *   **Lambda Function:** Select the function you created (e.g., `tickTrackerGetTicksFunction`).
    *   Save. API Gateway might ask for permission to invoke your Lambda function – grant it.
4.  **Deploy API Gateway:** Remember to deploy the API again after linking the Lambda function.

**Replace `#TODO`s in `ApiService`:**

*   Now that the backend endpoints exist, go back to `lib\services\api_service.dart`.
*   Replace the simulation logic within the `get`, `post`, `put`, `delete` methods with actual calls using the `http` package, pointing to your deployed API Gateway endpoints. Make sure you are using the correct HTTP method and sending the body/parameters as expected by your Lambda functions.
*   Use the real response handling in `_handleResponse`.

---

**Step 11: Implement Real-time Updates (Optional but Recommended)**

Polling the API repeatedly for Tick updates isn't efficient. Use Push Notifications.

1.  **Backend Setup (SNS + Push Services):**
    *   **IoT Rule:** Create a rule in AWS IoT Core that triggers when a Tick sends specific MQTT messages (e.g., `periodic_update`, `movement_alert`, `low_battery`).
    *   **Lambda Target:** Configure the IoT Rule to send the MQTT message payload to a new Lambda function (e.g., `tickTrackerNotificationHandler`).
    *   **Notification Lambda:** This Lambda function:
        *   Receives the Tick data (ID, lat, lng, bat, eventType).
        *   Looks up the owner (`userId`) associated with the `tickId` (query DynamoDB `TICK#{tickId}/METADATA`).
        *   Finds the Push Notification token(s) (e.g., FCM token for Android, APNs token for iOS) registered for that `userId` (you'll need to store these tokens, perhaps in another DynamoDB table: `USER#{userId}/NOTIFICATION_TOKENS`).
        *   Uses **AWS SNS (Simple Notification Service)** to send a push notification payload specifically formatted for FCM (Android) and/or APNs (iOS) to the user's registered device(s). The payload should contain the updated Tick data or at least the `tickId` and `eventType` so the app knows what happened.
2.  **Flutter Setup (Receive Push Notifications):**
    *   **Add Packages:** `firebase_core`, `firebase_messaging` (for Android/iOS via FCM). You'll need to set up a Firebase project and configure your Flutter app for FCM. For direct APNs on iOS, the setup is different.
    *   **Initialization:** Initialize Firebase in `main.dart`.
    *   **Request Permissions:** Request notification permissions from the user.
    *   **Get Token:** Get the FCM device token using `FirebaseMessaging.instance.getToken()`.
    *   **Register Token:** Send this token to your backend (e.g., via a `POST /users/me/register-token` API endpoint) so the Notification Lambda can find it later. Store it associated with the `userId`.
    *   **Handle Messages:** Set up listeners using `FirebaseMessaging.onMessage.listen` (foreground), `FirebaseMessaging.onMessageOpenedApp.listen` (background tap), and `FirebaseMessaging.instance.getInitialMessage()` (terminated tap) to receive notifications.
    *   **Update UI:** When a notification related to a Tick update is received, parse the payload, get the updated `Tick` data, and call `Provider.of<TickService>(context, listen: false).updateTickDataLocally(updatedTick)` to refresh the UI without needing a full API poll.

---

**Step 12: Refinement and Testing**

*   **Thorough Testing:** Test every authentication flow (sign-up, confirm, sign-in, forgot password, confirm reset, sign-out). Test API calls with and without valid tokens. Test Tick association and commands.
*   **Loading Indicators:** Make loading indicators more granular where appropriate (e.g., indicating *which* specific action is loading).
*   **Error Messages:** Provide user-friendly error messages based on the specific exceptions caught (from Amplify or your API).
*   **User Feedback:** Add success messages (e.g., using `CustomSnackBar`) for actions like password reset confirmation, profile updates, etc.
*   **Edge Cases:** Consider what happens if the network connection is lost during multi-step processes.

This is a comprehensive roadmap. Tackle it step-by-step, starting with API protection and the UI for the remaining auth flows. Good luck!
