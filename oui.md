Okay, I've reviewed and refactored the provided Flutter code snippets according to your requirements.

Here's a summary of the main changes applied across the files:

1.  **Amplify Integration:** Refactored `AuthService` entirely to use the `amplify_flutter` and `amplify_auth_cognito` packages for all authentication operations (login, register, confirm, reset password, fetch user, update attributes, logout) instead of direct API calls. Removed `ApiService` dependency from `AuthService`. Added Amplify initialization in `main.dart`.
2.  **Bluetooth Logic:** Refactored `BluetoothService` and `AddTickPage`. The flow no longer requires connecting via BLE. It now scans for devices advertising the specific service UUID, extracts a unique ID from the device name (`Tick-XXXX`), and then calls the backend API (`TickService.associateTick`) with this extracted ID and the user-provided nickname. `permission_handler` and `location` packages are used for robust permission checks.
3.  **API Service:** Modified `ApiService` methods (`get`, `post`, `put`, `delete`) to accept the *full* Lambda Function URL directly, as provided in `constants.dart`, instead of constructing URLs from a base URL and endpoint. This aligns with using Lambda Function URLs for backend operations.
4.  **Tick Service:** Updated `TickService` to use the modified `ApiService` and the correct Lambda Function URLs from `constants.dart`. Added `getTickHistory`. Updated `associateTick` to match the new Bluetooth flow. Added placeholders for other API calls (`ring`, `update`, `unlink`). Integrated tightly with `AuthService` changes using `ChangeNotifierProxyProvider`.
5.  **Constants & Utils:** Centralized more strings, durations, routes, and configurations in `constants.dart` and `theme.dart`. Reviewed and standardized validators. Added helper methods/getters in models (`Tick`, `HistoryEvent`).
6.  **UI & Widgets:**
    *   Extracted reusable UI components (e.g., `_buildTickInfoCard`, `_buildInfoRow`, `_buildSectionTitle`).
    *   Used custom widgets (`LoadingIndicator`, `AlertCard`, `CustomSnackBar`, `ActionButton`, `StepIndicator`, `ThemeToggleButton`, `BluetoothStatusWidget`) consistently.
    *   Improved loading states across screens using `Consumer` and `Stack`/`AbsorbPointer`.
    *   Implemented `RefreshIndicator` in `TickListPage` and `HistoryPage`.
    *   Used `FutureBuilder` correctly in `HistoryPage`.
    *   Improved inline editing UI in `TickSettingsPage` and `ProfilePage`.
    *   Added platform checks where necessary (e.g., FAB in `TickListPage`, Bluetooth functionality).
7.  **State Management:** Reviewed Provider usage (`Consumer`, `context.watch`, `context.read`, `listen: false`). Ensured `ChangeNotifierProxyProvider` is used correctly for `TickService`'s dependency on `AuthService`.
8.  **Error Handling & Feedback:** Used `CustomSnackBar` for user feedback. Improved error messages using constants and parsing API/Amplify errors better. Ensured `mounted` checks after `await` calls in `StatefulWidget`s.
9.  **Null Safety & Types:** Ensured proper null safety handling and used specific types where possible.
10. **Code Style & Comments:** Added `///` documentation comments, clarified implementation comments (kept in French where existing), removed redundant comments, organized imports, and ensured consistent formatting. Renamed some variables/methods for clarity.
11. **Configuration Files:** Added comments to `Info.plist`. Updated `Podfile` platform target to a more common default (`13.0`). Added necessary Amplify dependencies and other required packages (`location`, `permission_handler`) to `pubspec.yaml`. Ensured assets were correctly listed.

Here is the cleaned-up code for each modified file:

--- START OF MODIFIED FILE ios\Runner\Info.plist ---

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CADisableMinimumFrameDurationOnPhone</key>
	<true/>
	<key>CFBundleDevelopmentRegion</key>
	<string>$(DEVELOPMENT_LANGUAGE)</string>
	<key>CFBundleDisplayName</key>
	<string>Tick App</string> <!-- Nom affiché de l'app -->
	<key>CFBundleExecutable</key>
	<string>$(EXECUTABLE_NAME)</string>
	<key>CFBundleIdentifier</key>
	<string>$(PRODUCT_BUNDLE_IDENTIFIER)</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>tick_app</string> <!-- Nom technique -->
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>$(FLUTTER_BUILD_NAME)</string>
	<key>CFBundleSignature</key>
	<string>????</string>
	<key>CFBundleVersion</key>
	<string>$(FLUTTER_BUILD_NUMBER)</string>
	<key>LSRequiresIPhoneOS</key>
	<true/>

	<!-- Permissions Bluetooth -->
	<key>NSBluetoothAlwaysUsageDescription</key>
	<string>Le Bluetooth est nécessaire pour rechercher et communiquer avec vos appareils Tick à proximité lors de l'association.</string>
	<key>NSBluetoothPeripheralUsageDescription</key>
	<string>Le Bluetooth est nécessaire pour rechercher et communiquer avec vos appareils Tick à proximité lors de l'association.</string>

	<!-- Permissions Localisation (Nécessaires pour le scan Bluetooth et la carte) -->
	<key>NSLocationWhenInUseUsageDescription</key>
	<string>Nous avons besoin de votre position pour localiser vos Ticks, afficher la carte, et scanner les appareils Bluetooth à proximité.</string>
	<key>NSLocationAlwaysAndWhenInUseUsageDescription</key>
    <string>Nous avons besoin de votre position pour localiser vos Ticks, afficher la carte, et scanner les appareils Bluetooth à proximité.</string>
    <key>NSLocationAlwaysUsageDescription</key>
    <string>Nous avons besoin de votre position pour localiser vos Ticks, afficher la carte, et scanner les appareils Bluetooth à proximité.</string>

	<!-- Autres clés standard Flutter -->
	<key>LSApplicationQueriesSchemes</key>
	<array>
		<string>https</string>
		<string>http</string>
	</array>
	<key>UIApplicationSupportsIndirectInputEvents</key>
	<true/>
	<key>UILaunchStoryboardName</key>
	<string>LaunchScreen</string>
	<key>UIMainStoryboardFile</key>
	<string>Main</string>
	<key>UISupportedInterfaceOrientations</key>
	<array>
		<string>UIInterfaceOrientationPortrait</string>
		<!-- Retirer Landscape si l'app est uniquement Portrait -->
		<!-- <string>UIInterfaceOrientationLandscapeLeft</string> -->
		<!-- <string>UIInterfaceOrientationLandscapeRight</string> -->
	</array>
	<key>UISupportedInterfaceOrientations~ipad</key>
	<array>
		<string>UIInterfaceOrientationPortrait</string>
		<string>UIInterfaceOrientationPortraitUpsideDown</string>
		<string>UIInterfaceOrientationLandscapeLeft</string>
		<string>UIInterfaceOrientationLandscapeRight</string>
	</array>
</dict>
</plist>
```

--- END OF MODIFIED FILE ios\Runner\Info.plist ---

--- START OF MODIFIED FILE ios\Podfile ---

```ruby
# Uncomment this line to define a global platform for your project
# platform :ios, '18.0' # 18.0 est très récent, préférer une version plus stable compatible avec vos plugins
platform :ios, '13.0' # Version minimum iOS raisonnable

# CocoaPods analytics sends network stats synchronously affecting flutter build latency.
ENV['COCOAPODS_DISABLE_STATS'] = 'true'

project 'Runner', {
  'Debug' => :debug,
  'Profile' => :release,
  'Release' => :release,
}

def flutter_root
  generated_xcode_build_settings_path = File.expand_path(File.join('..', 'Flutter', 'Generated.xcconfig'), __FILE__)
  unless File.exist?(generated_xcode_build_settings_path)
    raise "#{generated_xcode_build_settings_path} must exist. If you're running pod install manually, make sure flutter pub get is executed first"
  end

  File.foreach(generated_xcode_build_settings_path) do |line|
    matches = line.match(/FLUTTER_ROOT\=(.*)/)
    return matches[1].strip if matches
  end
  raise "FLUTTER_ROOT not found in #{generated_xcode_build_settings_path}. Try deleting Generated.xcconfig, then run flutter pub get"
end

require File.expand_path(File.join('packages', 'flutter_tools', 'bin', 'podhelper'), flutter_root)

flutter_ios_podfile_setup

target 'Runner' do
  # Comment this line if you don't want Flutter and native related pods to be built as dynamic frameworks
  # Requis par certains plugins (ex: Firebase, Google Maps)
  use_frameworks!
  use_modular_headers! # Souvent recommandé avec use_frameworks!

  flutter_install_all_ios_pods File.dirname(File.realpath(__FILE__))

  # Décommenter si vous avez une target de tests
  # target 'RunnerTests' do
  #   inherit! :search_paths
  # end
end

post_install do |installer|
  installer.pods_project.targets.each do |target|
    flutter_additional_ios_build_settings(target)
     # Configuration spécifique si nécessaire, par exemple pour certains plugins:
     # target.build_configurations.each do |config|
     #    config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '13.0' # Assurer cohérence
     # end
  end
end
```

--- END OF MODIFIED FILE ios\Podfile ---

--- START OF MODIFIED FILE lib\utils\theme.dart ---

```dart
import 'package:flutter/material.dart';

/// Enumération des types d'alertes utilisées dans l'application.
enum AlertType { info, success, warning, error }

/// Contient les configurations de thèmes Light et Dark pour l'application.
class AppTheme {
  // --- Couleurs Primaires et d'Accentuation ---
  static const Color primaryColor = Color(0xFF2196F3); // Bleu principal
  static const Color secondaryColor = Color(0xFF03A9F4); // Bleu clair
  static const Color accentColor = Color(0xFF4CAF50); // Vert pour accents/succès
  static const Color errorColor = Color(0xFFF44336); // Rouge pour erreurs
  static const Color warningColor = Color(0xFFFF9800); // Orange pour avertissements

  // Alias sémantiques (peuvent pointer vers les couleurs ci-dessus)
  static const Color successColor = accentColor;
  static const Color infoColor = primaryColor;

  // --- Couleurs Neutres (Mode Clair) ---
  static const Color backgroundColorLight = Color(0xFFF5F5F5); // Gris très clair pour fond
  static const Color surfaceColorLight = Color(0xFFFFFFFF); // Blanc pour surfaces (cartes, appbar)
  static const Color textPrimaryColorLight = Color(0xFF212121); // Noir/Gris foncé pour texte principal
  static const Color textSecondaryColorLight = Color(0xFF757575); // Gris moyen pour texte secondaire
  static const Color dividerColorLight = Color(0xFFBDBDBD); // Gris clair pour séparateurs

  // --- Couleurs Neutres (Mode Sombre) ---
  static const Color backgroundColorDark = Color(0xFF121212); // Noir/Gris très foncé pour fond
  static const Color surfaceColorDark = Color(0xFF1E1E1E); // Gris foncé pour surfaces (cartes, appbar)
  static const Color textPrimaryColorDark = Color(0xFFFFFFFF); // Blanc pour texte principal
  static const Color textSecondaryColorDark = Color(0xFFB0B0B0); // Gris clair pour texte secondaire
  static const Color dividerColorDark = Color(0xFF3C3C3C); // Gris foncé pour séparateurs

  /// Retourne la configuration ThemeData pour le mode clair.
  static ThemeData getLightTheme() {
    final baseTheme = ThemeData.light(useMaterial3: true);
    return baseTheme.copyWith(
      // Schéma de couleurs principal
      colorScheme: const ColorScheme.light(
        primary: primaryColor,
        secondary: secondaryColor,
        surface: surfaceColorLight,
        background: backgroundColorLight,
        error: errorColor,
        onPrimary: Colors.white, // Texte sur couleur primaire
        onSecondary: Colors.white, // Texte sur couleur secondaire
        onSurface: textPrimaryColorLight, // Texte sur surfaces claires
        onBackground: textPrimaryColorLight, // Texte sur fond clair
        onError: Colors.white, // Texte sur couleur d'erreur
        brightness: Brightness.light,
      ),
      // Fond général des écrans
      scaffoldBackgroundColor: backgroundColorLight,
      // Style de l'AppBar
      appBarTheme: AppBarTheme(
        centerTitle: true,
        elevation: 0, // Pas d'ombre par défaut
        backgroundColor: surfaceColorLight, // Fond blanc par défaut
        foregroundColor: textPrimaryColorLight, // Couleur par défaut pour titre/icônes (peut être surchargée)
        titleTextStyle: TextStyle(
          color: textPrimaryColorLight,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
        iconTheme: IconThemeData(
          color: primaryColor, // Icônes AppBar en bleu par défaut
        ),
      ),
      // Style des boutons
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryColor,
          foregroundColor: Colors.white, // Texte/icône sur le bouton
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 24),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          elevation: 1,
          textStyle: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: primaryColor, // Texte/icône et bordure par défaut
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 24),
          side: const BorderSide(color: primaryColor, width: 1.5),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          textStyle: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: primaryColor, // Texte du bouton
          padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
          textStyle: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      // Style des cartes
      cardTheme: CardTheme(
        elevation: 1, // Ombre subtile
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        color: surfaceColorLight, // Fond de la carte
        margin: const EdgeInsets.symmetric(vertical: 6.0, horizontal: 8.0), // Marge par défaut
      ),
      // Style des champs de texte
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.grey[100], // Fond légèrement grisé
        contentPadding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
        border: OutlineInputBorder( // Bordure par défaut (utilisée si enabled/focused/error non définies)
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: dividerColorLight),
        ),
        enabledBorder: OutlineInputBorder( // Bordure quand le champ est activé mais pas en focus
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: dividerColorLight),
        ),
        focusedBorder: OutlineInputBorder( // Bordure quand le champ est en focus
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: primaryColor, width: 2),
        ),
        errorBorder: OutlineInputBorder( // Bordure en cas d'erreur
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: errorColor, width: 1.5),
        ),
        focusedErrorBorder: OutlineInputBorder( // Bordure en cas d'erreur ET focus
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: errorColor, width: 2),
        ),
        labelStyle: TextStyle(color: textSecondaryColorLight), // Style du label flottant
        hintStyle: TextStyle(color: Colors.grey[500]), // Style du placeholder
        // helperStyle: TextStyle(color: textSecondaryColorLight), // Style texte d'aide
        // errorStyle: TextStyle(color: errorColor), // Style texte d'erreur
      ),
      // Style des séparateurs
      dividerTheme: const DividerThemeData(
        color: dividerColorLight,
        space: 1,
        thickness: 1,
      ),
      // Style des SnackBars
      snackBarTheme: SnackBarThemeData(
        behavior: SnackBarBehavior.floating, // Flottant en bas
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        // backgroundColor: ..., // Défini dynamiquement par CustomSnackBar
        // contentTextStyle: ..., // Défini dynamiquement par CustomSnackBar
        // actionTextColor: accentColor, // Couleur pour SnackBarAction standard
      ),
      // Style des tooltips (info-bulles)
      tooltipTheme: TooltipThemeData(
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.8),
          borderRadius: BorderRadius.circular(4),
        ),
        textStyle: const TextStyle(color: Colors.white, fontSize: 12),
      ),
      // Définir les styles de texte par défaut si nécessaire
      // textTheme: baseTheme.textTheme.copyWith( ... )
    );
  }

  /// Retourne la configuration ThemeData pour le mode sombre.
  static ThemeData getDarkTheme() {
    final baseTheme = ThemeData.dark(useMaterial3: true);
    return baseTheme.copyWith(
      // Schéma de couleurs principal
      colorScheme: const ColorScheme.dark(
        primary: primaryColor, // Garder le même bleu primaire pour la cohérence?
        secondary: secondaryColor, // Garder le même bleu secondaire?
        surface: surfaceColorDark, // Fond des composants (cartes, appbar)
        background: backgroundColorDark, // Fond général de l'écran
        error: errorColor, // Garder le même rouge
        onPrimary: Colors.white, // Texte sur bleu primaire
        onSecondary: Colors.white, // Texte sur bleu secondaire
        onSurface: textPrimaryColorDark, // Texte sur surfaces sombres (blanc)
        onBackground: textPrimaryColorDark, // Texte sur fond sombre (blanc)
        onError: Colors.black, // Texte sur couleur d'erreur (noir pour contraste)
        brightness: Brightness.dark,
      ),
      // Fond général des écrans
      scaffoldBackgroundColor: backgroundColorDark,
      // Style de l'AppBar
      appBarTheme: AppBarTheme(
        centerTitle: true,
        elevation: 0,
        backgroundColor: surfaceColorDark, // App bar se fond avec la surface
        foregroundColor: textPrimaryColorDark, // Couleur par défaut titre/icônes (blanc)
        titleTextStyle: TextStyle(
          color: textPrimaryColorDark,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
        iconTheme: IconThemeData(
          color: primaryColor, // Garder icônes bleues pour accent? Ou blanches (textPrimaryColorDark)?
        ),
      ),
      // Style des boutons
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryColor,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 24),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          elevation: 1,
          textStyle: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: primaryColor, // Garder le texte/bordure bleu? Ou blanc (textPrimaryColorDark)?
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 24),
          side: const BorderSide(color: primaryColor, width: 1.5), // Garder bordure bleue?
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          textStyle: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: primaryColor, // Texte bleu
          padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
          textStyle: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      // Style des cartes
      cardTheme: CardTheme(
        elevation: 1,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        color: surfaceColorDark, // Couleur de la carte
        margin: const EdgeInsets.symmetric(vertical: 6.0, horizontal: 8.0),
      ),
      // Style des champs de texte
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.grey[850], // Fond des champs texte (légèrement différent de surface)
        contentPadding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: dividerColorDark), // Bordure par défaut
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: dividerColorDark), // Bordure quand activé
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: primaryColor, width: 2), // Bordure focus
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: errorColor, width: 1.5), // Bordure erreur
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: errorColor, width: 2), // Bordure erreur focus
        ),
        labelStyle: TextStyle(color: textSecondaryColorDark), // Label flottant
        hintStyle: TextStyle(color: Colors.grey[600]), // Placeholder
        // helperStyle: TextStyle(color: textSecondaryColorDark),
        // errorStyle: TextStyle(color: errorColor),
      ),
      // Style des séparateurs
      dividerTheme: const DividerThemeData(
        color: dividerColorDark,
        space: 1,
        thickness: 1,
      ),
      // Style des SnackBars
      snackBarTheme: SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        // backgroundColor: const Color(0xFFE0E0E0), // Fond clair pour contraste sur fond sombre
        // contentTextStyle: const TextStyle(color: Colors.black), // Texte sombre
        // actionTextColor: primaryColor, // Action en bleu
      ),
      // Style des tooltips
      tooltipTheme: TooltipThemeData(
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.9),
          borderRadius: BorderRadius.circular(4),
        ),
        textStyle: const TextStyle(color: Colors.black, fontSize: 12),
      ),
      // Adapter les styles de texte si nécessaire pour le thème sombre
      textTheme: baseTheme.textTheme.apply(
        bodyColor: textPrimaryColorDark, // Couleur par défaut du texte
        displayColor: textPrimaryColorDark,
      ).copyWith(
          // Ex: Rendre les titres un peu plus clairs si besoin
          // titleLarge: baseTheme.textTheme.titleLarge?.copyWith(color: Colors.white.withOpacity(0.9)),
          ),
    );
  }

  /// Retourne la couleur associée à un type d'alerte.
  ///
  /// Peut être ajustée pour le mode sombre si nécessaire via `isDark`.
  static Color getAlertColor(AlertType type, {bool isDark = false}) {
    // Actuellement, les couleurs sont les mêmes en light/dark, mais on pourrait les différencier ici.
    switch (type) {
      case AlertType.error:
        return errorColor;
      case AlertType.warning:
        return warningColor;
      case AlertType.success:
        return successColor;
      case AlertType.info:
      default:
        return primaryColor; // Ou `infoColor`
    }
  }
}
```

--- END OF MODIFIED FILE lib\utils\theme.dart ---

--- START OF MODIFIED FILE lib\utils\constants.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:tick_app/utils/theme.dart'; // Pour AppTheme.errorColor, etc.
import '../models/tick_model.dart'; // Pour TickStatus

// ignore_for_file: constant_identifier_names

/// Contient les configurations liées à l'API backend.
class ApiConfig {
  /// URL de base de l'API Gateway (si utilisée).
  /// Mettre en commentaire ou supprimer si seules les Function URLs sont utilisées.
  // static const String apiGatewayBaseUrl = 'https://votre-api-id.execute-api.votre-region.amazonaws.com/prod'; // Exemple

  /// URL de la fonction Lambda pour récupérer les Ticks de l'utilisateur.
  /// Doit retourner un JSON: { "success": true, "data": [...] } ou { "success": false, "error": "..." }
  static const String getMyTicksFunctionUrl = 'https://g2wgiv7n5vy4b4xdjbisbvgsny0asnki.lambda-url.eu-north-1.on.aws/';

  /// URL de la fonction Lambda pour demander la localisation d'un Tick.
  /// Attend un JSON POST: { "tickId": "..." }
  /// Doit retourner un JSON: { "success": true, "data": { ... } } ou { "success": false, "error": "..." }
  static const String requestLocationFunctionUrl = 'https://khrjd2jrqsyubgq3vad3o7z4li0eezcb.lambda-url.eu-north-1.on.aws/';

  /// URL de la fonction Lambda pour associer un nouveau Tick.
  /// Attend un JSON POST: { "tickName": "...", "tickId": "..." }
  /// Doit retourner un JSON: { "success": true, "data": { ... } } ou { "success": false, "error": "..." }
  static const String associateTickFunctionUrl = 'https://sxnfyrupf2navy3327vv7uvjvy0phdas.lambda-url.eu-north-1.on.aws/';

  /// URL de la fonction Lambda pour récupérer l'historique d'un Tick.
  /// Attend un GET avec query parameter: ?tickId=...
  /// Doit retourner un JSON: { "success": true, "data": [...] } ou { "success": false, "error": "..." }
  static const String getTickHistoryFunctionUrl = 'https://svhswmtmjywgi35alhomkjwhwm0yhngo.lambda-url.eu-north-1.on.aws/';

  /// URL de la fonction Lambda pour désassocier un Tick.
  /// Attend probablement un POST ou DELETE avec { "tickId": "..." } dans le body ou query parameter.
  /// #TODO: Définir l'URL et la méthode (POST/DELETE?) réelles.
  static const String removeTickFunctionUrl = ''; // Exemple: 'https://xxxx.lambda-url.region.on.aws/'

  /// URL de la fonction Lambda pour mettre à jour les paramètres d'un Tick (nom, etc.).
  /// Attend probablement un PUT ou POST avec { "tickId": "...", "name": "..." }
  /// #TODO: Définir l'URL réelle.
  static const String updateTickSettingsFunctionUrl = '';

  /// URL de la fonction Lambda pour faire sonner un Tick.
  /// Attend probablement un POST avec { "tickId": "...", "soundType": "..." }
  /// #TODO: Définir l'URL réelle.
  static const String ringTickFunctionUrl = '';

  /// URL de la fonction Lambda pour désactiver temporairement un Tick.
  /// Attend probablement un POST avec { "tickId": "...", "duration": 3600 }
  /// #TODO: Définir l'URL réelle.
  static const String temporaryDisableFunctionUrl = '';

  /// URL de la fonction Lambda pour mettre à jour le profil utilisateur (nom, etc.).
  /// Attend probablement un PUT ou POST avec { "userId": "...", "name": "..." }
  /// #TODO: Définir l'URL réelle. (Alternative: utiliser Amplify Auth pour mettre à jour les attributs Cognito)
  static const String updateUserProfileFunctionUrl = '';
}


/// Endpoints relatifs si une API Gateway est utilisée (ApiConfig.apiGatewayBaseUrl).
/// Mis en commentaire car les Function URLs semblent être utilisées directement.
class ApiEndpoints {
  /*
  // Authentification (gérée par Amplify Auth maintenant)
  // static const String login = '/auth/login';
  // static const String register = '/auth/register';
  // static const String forgotPassword = '/auth/forgot-password';
  // static const String resetPassword = '/auth/reset-password';
  // static const String getUserProfile = '/users/me';

  // Ticks (endpoints relatifs si API Gateway)
  // static const String fetchTicks = '/ticks';             // GET
  // static const String associateTick = '/ticks';          // POST
  // static const String getTickDetails = '/ticks/{tickId}';  // GET
  // static const String unlinkTick = '/ticks/{tickId}';      // DELETE
  // static const String updateTickSettings = '/ticks/{tickId}/settings'; // PUT/PATCH
  // static const String getTickHistory = '/ticks/{tickId}/history';    // GET

  // Commandes Tick (endpoints relatifs si API Gateway)
  // static const String requestLocation = '/ticks/{tickId}/locate';    // POST
  // static const String ringTick = '/ticks/{tickId}/ring';             // POST
  // static const String temporaryDisable = '/ticks/{tickId}/disable';  // POST
  */
}


/// Clés utilisées pour le stockage local sécurisé (si nécessaire en dehors d'Amplify).
/// Amplify gère la session et les tokens Cognito automatiquement.
class StorageKeys {
  // static const String authToken = 'auth_token'; // Géré par Amplify
  // static const String userId = 'user_id'; // Peut être obtenu depuis AuthService.currentUser.uid
  // static const String userEmail = 'user_email'; // Peut être obtenu depuis AuthService.currentUser.email
  // static const String themeMode = 'theme_mode'; // Pour persister le thème
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
  static const String bleNotSupported = 'Le Bluetooth Low Energy n\'est pas supporté sur cet appareil.'; // Ajouté

  // Erreurs de permissions
  static const String permissionDenied = 'Permission nécessaire refusée.';
  static const String permissionDeniedLocation = 'La permission de localisation est requise.'; // Plus spécifique
  static const String permissionDeniedBluetooth = 'Les permissions Bluetooth sont requises.'; // Plus spécifique
  static const String permissionDeniedForever = 'Permission refusée définitivement.'; // Non utilisé directement
  static const String permissionDeniedLocationExplain = 'Permission de localisation refusée définitivement. Veuillez l\'activer manuellement dans les paramètres de l\'application pour utiliser cette fonctionnalité.';
  static const String locationServiceDisabled = 'Le service de localisation doit être activé.'; // Ajouté

  // Erreurs de formulaire
  static const String invalidInput = 'Veuillez vérifier les informations saisies.';
}


/// Textes utilisés dans l'interface utilisateur.
class AppTexts {
  static const String appName = 'Tick Tracker'; // Peut être mis à jour ici
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
  static const String done = 'Terminé'; // Ajouté

  // --- Authentification ---
  static const String forgotPassword = 'Mot de passe oublié ?';
  static const String noAccount = 'Pas encore de compte ?';
  static const String alreadyAccount = 'Déjà un compte ?';
  static const String passwordRecovery = 'Récupération de mot de passe';
  static const String passwordRecoveryInstructions = 'Entrez votre email pour recevoir les instructions de réinitialisation.';
  static const String sendRecoveryLink = 'Envoyer le lien';
  static const String recoveryLinkSent = 'Email de réinitialisation envoyé.';
  static const String confirmAccount = "Confirmer l'inscription"; // Ajouté
  static const String confirmationCode = "Code de confirmation"; // Ajouté
  static const String resendCode = "Renvoyer le code"; // Ajouté
  static const String codeSentTo = "Un code de confirmation a été envoyé à :"; // Ajouté
  static const String enterConfirmationCode = "Entrez le code à 6 chiffres"; // Ajouté
  static const String checkEmailForCode = "Vérifiez votre email pour le code"; // Ajouté
  static const String resetPassword = "Réinitialiser Mot de Passe"; // Ajouté
  static const String newPassword = "Nouveau mot de passe"; // Ajouté
  static const String enterResetCode = "Code de réinitialisation"; // Ajouté

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
  static const String invalidCode = 'Code invalide'; // Ajouté

  // --- Écrans & Sections ---
  static const String myTicks = 'Mes Ticks';
  static const String addTick = 'Ajouter un Tick';
  static const String tickDetails = 'Détails du Tick';
  static const String noTicksAvailable = 'Aucun Tick associé';
  static const String addFirstTick = 'Ajoutez votre premier Tick pour commencer'; // Reformulé
  static const String map = 'Carte';
  static const String settings = 'Paramètres';
  static const String profile = 'Profil';
  static const String history = 'Historique';
  static const String alerts = 'Alertes récentes'; // Ou 'Événements récents'
  static const String information = 'Informations';
  static const String general = 'Général'; // Ajouté pour settings
  static const String features = 'Fonctionnalités'; // Ajouté pour settings
  static const String dangerZone = 'Zone de Danger'; // Ajouté pour settings
  static const String appearance = 'Apparence'; // Ajouté pour settings
  static const String notifications = 'Notifications'; // Ajouté pour settings
  static const String security = 'Sécurité'; // Ajouté pour settings

  // --- Association Tick (Refactorisée) ---
  static const String associateNewTick = 'Associer un nouveau Tick';
  static const String associationSteps = 'Suivez les étapes pour connecter votre appareil.';
  static const String step1_Naming = '1. Nommez votre Tick';
  static const String step2_Scanning = '2. Recherche du Tick'; // Étape simplifiée
  static const String step3_Sending = '3. Association'; // Étape simplifiée
  static const String step4_Done = '4. Terminé'; // Étape simplifiée
  static const String enableBluetoothPrompt = 'Le Bluetooth est nécessaire pour trouver votre Tick.';
  static const String enableBluetoothButton = 'Activer le Bluetooth';
  static const String bluetoothEnabled = 'Bluetooth activé';
  static const String activateTickPrompt = 'Assurez-vous que votre Tick est allumé et à proximité.'; // Simplifié
  static const String searchTickButton = 'Rechercher mon Tick';
  static const String searchingTick = 'Recherche du Tick en cours...';
  static const String connectingTick = 'Connexion au Tick...'; // Plus utilisé dans le flux simplifié
  static const String tickFound = 'Tick trouvé !';
  static const String tickMacAddress = 'Adresse MAC'; // Affiché dans les détails/settings
  static const String associatingTick = 'Association en cours...'; // Texte UI pendant l'appel API
  static const String associateTickButton = 'Associer ce Tick'; // Pas directement utilisé
  static const String tickAssociatedSuccess = 'Tick associé avec succès !';
  static const String tickIdExtracted = 'ID du Tick détecté'; // Ajouté

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
  static const String ringingTick = 'Sonnerie en cours...'; // Texte UI
  static const String updating = 'Mise à jour...';
  static const String errorFetchingLocation = 'Erreur de localisation';
  static const String locationUpdated = 'Position mise à jour';
  static const String locationRequestSent = 'Demande de localisation envoyée...'; // Ajouté
  static const String noLocationAvailable = 'Position non disponible'; // Ajouté
  static const String tryToLocate = 'Tenter de localiser'; // Ajouté

  // --- Paramètres Tick ---
  static const String tickSettings = 'Paramètres du Tick';
  static const String changeName = 'Changer le nom';
  static const String soundSettings = 'Paramètres de sonnerie';
  static const String temporaryDisable = 'Désactivation temporaire';
  static const String disableDuration = 'Durée de désactivation';
  static const String unlinkDevice = 'Désassocier cet appareil';
  static const String unlinkDeviceConfirm = 'Êtes-vous sûr de vouloir désassocier ce Tick ? Cette action est irréversible.'; // Précisé
  static const String unlinkSuccess = 'Tick désassocié avec succès.';
  static const String featureComingSoon = 'Fonctionnalité bientôt disponible'; // Ajouté

  // --- Autres ---
  static const String loading = 'Chargement...'; // Ajouté
  static const String noHistoryAvailable = 'Aucun historique disponible'; // Ajouté
  static const String loadingHistoryError = 'Erreur de chargement de l\'historique'; // Ajouté
  static const String featureNotAvailableOnPlatform = 'Fonctionnalité non disponible sur cette plateforme.'; // Ajouté
  static const String openSettings = 'Ouvrir les paramètres'; // Ajouté
  static const String unknownUser = 'Utilisateur inconnu'; // Ajouté
  static const String notConnected = 'Non connecté'; // Ajouté
  static const String updateSuccess = 'Mise à jour réussie.'; // Ajouté
  static const String updateError = 'Erreur lors de la mise à jour.'; // Ajouté
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
  static const String tickSettings = '/ticks/settings'; // Devrait utiliser un paramètre ex: /ticks/settings/{tickId}
  static const String tickHistory = '/ticks/history'; // Devrait utiliser un paramètre ex: /ticks/history/{tickId}
  static const String profile = '/profile'; // Page profil utilisateur
  static const String settings = '/settings'; // Page paramètres généraux de l'app
}


/// Configuration spécifique au Bluetooth.
class BluetoothConfig {
  /// UUID du service principal exposé par les Ticks ESP32 en BLE.
  /// Doit correspondre exactement à celui défini dans le firmware ESP32.
  static const String tickServiceUuid = "7a8274fc-0723-44da-ad07-2293d5a5212a"; // Votre UUID

  /// Préfixe attendu pour le nom des appareils Tick diffusé en BLE.
  /// Le format attendu est "Tick-ID_UNIQUE_DU_TICK".
  static const String tickNamePrefix = "Tick-"; // Doit correspondre au firmware

  /// Durée maximale (en secondes) pour un scan Bluetooth.
  static const int scanTimeoutSeconds = 15;

  // Caractéristique pour lire/écrire des commandes (optionnel, non utilisé dans le flux actuel)
  // static const String commandCharacteristicUuid = " VOTRE_UUID_CARACTERISTIQUE_COMMANDE";
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
  static const Duration apiTimeout = Duration(seconds: 20); // Augmenté légèrement
  static const Duration shortDelay = Duration(milliseconds: 500);
  static const Duration mediumDelay = Duration(seconds: 1);
  static const Duration longDelay = Duration(seconds: 3);
}
```

--- END OF MODIFIED FILE lib\utils\constants.dart ---

--- START OF MODIFIED FILE lib\utils\map_styles.dart ---

```dart
/// Contient les styles JSON pour personnaliser l'apparence de Google Maps.
/// Générés via https://mapstyle.withgoogle.com/
class MapStyles {

  /// Style JSON pour le mode sombre.
  /// Vise à réduire la luminosité et masquer les éléments non essentiels.
  static const String darkStyle = '''
[
  { "elementType": "geometry", "stylers": [ { "color": "#212121" } ] }, // Fond général
  { "elementType": "labels.icon", "stylers": [ { "visibility": "off" } ] }, // Masquer icônes POI
  { "elementType": "labels.text.fill", "stylers": [ { "color": "#757575" } ] }, // Texte labels gris
  { "elementType": "labels.text.stroke", "stylers": [ { "color": "#212121" } ] }, // Contour texte labels
  { "featureType": "administrative", "elementType": "geometry", "stylers": [ { "color": "#757575" }, { "visibility": "off" } ] }, // Masquer lignes admin
  { "featureType": "administrative.country", "elementType": "labels.text.fill", "stylers": [ { "color": "#9e9e9e" } ] },
  { "featureType": "administrative.land_parcel", "stylers": [ { "visibility": "off" } ] },
  { "featureType": "administrative.locality", "elementType": "labels.text.fill", "stylers": [ { "color": "#bdbdbd" } ] },
  { "featureType": "poi", "stylers": [ { "visibility": "off" } ] }, // Masquer tous les POI
  { "featureType": "poi", "elementType": "labels.text.fill", "stylers": [ { "color": "#757575" } ] },
  { "featureType": "poi.park", "elementType": "geometry", "stylers": [ { "color": "#181818" }, { "visibility": "on" } ] }, // Afficher parcs (discret)
  { "featureType": "poi.park", "elementType": "labels.text.fill", "stylers": [ { "color": "#616161" } ] },
  { "featureType": "poi.park", "elementType": "labels.text.stroke", "stylers": [ { "color": "#1b1b1b" } ] },
  { "featureType": "road", "elementType": "geometry.fill", "stylers": [ { "color": "#2c2c2c" } ] }, // Routes sombres
  { "featureType": "road", "elementType": "labels.text.fill", "stylers": [ { "color": "#8a8a8a" } ] }, // Texte routes gris clair
  { "featureType": "road", "elementType": "labels.icon", "stylers": [ { "visibility": "off" } ] }, // Masquer icônes routes (N°, etc.)
  { "featureType": "road.arterial", "elementType": "geometry", "stylers": [ { "color": "#373737" } ] },
  { "featureType": "road.highway", "elementType": "geometry", "stylers": [ { "color": "#3c3c3c" } ] },
  { "featureType": "road.highway.controlled_access", "elementType": "geometry", "stylers": [ { "color": "#4e4e4e" } ] },
  { "featureType": "road.local", "elementType": "labels.text.fill", "stylers": [ { "color": "#616161" } ] },
  { "featureType": "transit", "stylers": [ { "visibility": "off" } ] }, // Masquer transports en commun
  { "featureType": "transit", "elementType": "labels.text.fill", "stylers": [ { "color": "#757575" } ] },
  { "featureType": "water", "elementType": "geometry", "stylers": [ { "color": "#000000" } ] }, // Eau en noir
  { "featureType": "water", "elementType": "labels.text.fill", "stylers": [ { "color": "#3d3d3d" } ] }
]
''';

  /// Style JSON pour le mode clair.
  /// Exemple simple masquant POI/transit et ajustant la couleur de l'eau.
  static const String lightStyle = '''
[
  { "featureType": "poi", "stylers": [ { "visibility": "off" } ] },
  { "featureType": "transit", "stylers": [ { "visibility": "off" } ] },
  { "featureType": "water", "elementType": "geometry", "stylers": [ { "color": "#aadaff" } ] }, // Eau bleu clair
  { "featureType": "road", "elementType": "labels.icon", "stylers": [ { "visibility": "off" } ] }
]
''';
}
```

--- END OF MODIFIED FILE lib\utils\map_styles.dart ---

--- START OF MODIFIED FILE lib\utils\validators.dart ---

```dart
// lib/utils/validators.dart
import 'constants.dart'; // Pour les messages d'erreur constants

/// Fournit des méthodes de validation statiques pour les formulaires.
class Validators {

  /// Valide qu'une chaîne de caractères n'est pas nulle ou vide (après trim).
  ///
  /// Retourne un message d'erreur si invalide, sinon `null`.
  /// [message] optionnel pour personnaliser le message d'erreur.
  static String? validateNotEmpty(String? value, [String? message]) {
    if (value == null || value.trim().isEmpty) {
      return message ?? AppTexts.requiredField;
    }
    return null;
  }

  /// Valide un format d'email simple.
  ///
  /// Vérifie d'abord si le champ est vide, puis utilise une regex.
  /// Retourne un message d'erreur si invalide, sinon `null`.
  static String? validateEmail(String? value) {
    final notEmptyValidation = validateNotEmpty(value, AppTexts.invalidEmail);
    if (notEmptyValidation != null) {
      return notEmptyValidation; // Retourne l'erreur "champ requis" ou "email invalide" si vide
    }
    // Regex simple pour vérifier la structure de base d'un email
    // Source: https://emailregex.com/ (compromis entre simplicité et couverture)
    final emailRegex = RegExp(
        r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$");
    if (!emailRegex.hasMatch(value!)) {
      return AppTexts.invalidEmail;
    }
    return null;
  }

  /// Valide un mot de passe basé sur une longueur minimale.
  ///
  /// La longueur minimale est définie dans `AppTexts.passwordTooShort`.
  /// Retourne un message d'erreur si invalide, sinon `null`.
  /// Note: La politique de complexité réelle est définie dans AWS Cognito.
  static String? validatePassword(String? value) {
    final notEmptyValidation = validateNotEmpty(value, AppTexts.passwordTooShort);
    if (notEmptyValidation != null) {
      return notEmptyValidation;
    }
    // Utiliser la longueur définie dans les constantes pour le message d'erreur
    // La vraie validation de complexité est faite par Cognito.
    // On vérifie juste la longueur minimale affichée à l'utilisateur.
    if (value!.length < 8) { // Correspond à la politique Cognito par défaut (8 caractères)
      return AppTexts.passwordTooShort;
    }
    return null;
  }

  /// Valide que le mot de passe de confirmation correspond à l'original.
  ///
  /// Retourne un message d'erreur si invalide, sinon `null`.
  static String? validateConfirmPassword(String? confirmValue, String originalValue) {
    final notEmptyValidation = validateNotEmpty(confirmValue, AppTexts.passwordsDoNotMatch);
     if (notEmptyValidation != null) {
      return notEmptyValidation; // Retourne "champ requis" si vide, ou "ne correspond pas" si différent
    }
    if (confirmValue != originalValue) {
      return AppTexts.passwordsDoNotMatch;
    }
    return null;
  }

  /// Valide un code de confirmation (doit être non vide et d'une certaine longueur).
  ///
  /// Retourne un message d'erreur si invalide, sinon `null`.
  static String? validateConfirmationCode(String? value, {int length = 6}) {
     final notEmptyValidation = validateNotEmpty(value, AppTexts.invalidCode);
     if (notEmptyValidation != null) {
       return notEmptyValidation;
     }
     if (value!.length != length) {
        return AppTexts.invalidCode; // Ou message plus précis: "Le code doit faire $length chiffres"
     }
     // Pourrait ajouter une regex pour vérifier que ce sont des chiffres: RegExp(r'^[0-9]+$')
     return null;
  }

}
```

--- END OF MODIFIED FILE lib\utils\validators.dart ---

--- START OF MODIFIED FILE lib\models\user.dart ---

```dart
import 'package:flutter/foundation.dart' show immutable;

/// Représente un utilisateur de l'application.
/// Les données proviennent généralement des attributs Cognito.
@immutable // Marque la classe comme immuable
class User {
  /// Identifiant unique de l'utilisateur (Cognito Sub).
  final String uid;

  /// Adresse email de l'utilisateur (utilisée pour la connexion).
  final String email;

  /// Nom d'affichage de l'utilisateur (peut être le nom complet).
  final String displayName;

  /// Constructeur principal.
  const User({
    required this.uid,
    required this.email,
    required this.displayName,
  });

  /// Crée une instance `User` à partir d'une Map JSON (ex: réponse API ou attributs Cognito).
  factory User.fromJson(Map<String, dynamic> json) {
    // Tente de trouver l'ID utilisateur sous différentes clés communes
    final String userId = json['userId'] ?? json['uid'] ?? json['sub'] ?? '';
    final String userEmail = json['email'] ?? '';
    // Combine nom/prénom si présents séparément, sinon utilise 'name' ou 'displayName'
    final String name = json['name'] ?? json['displayName'] ?? '';
    final String firstName = json['firstName'] ?? json['prénom'] ?? '';
    final String lastName = json['lastName'] ?? json['nom'] ?? '';

    String finalDisplayName = name;
    if (finalDisplayName.isEmpty && (firstName.isNotEmpty || lastName.isNotEmpty)) {
      finalDisplayName = '$firstName $lastName'.trim();
    }

    return User(
      uid: userId,
      email: userEmail,
      displayName: finalDisplayName.isNotEmpty ? finalDisplayName : 'Utilisateur', // Nom par défaut
    );
  }

  /// Convertit l'objet `User` en une Map JSON.
  /// Utile si l'objet doit être envoyé à une API (moins courant avec Amplify Auth).
  Map<String, dynamic> toJson() {
    return {
      'uid': uid,
      'email': email,
      'displayName': displayName,
    };
  }

  /// Crée une copie de l'objet `User` avec certaines valeurs modifiées.
  /// Utile pour la gestion d'état immuable.
  User copyWith({
    String? uid,
    String? email,
    String? displayName,
  }) {
    return User(
      uid: uid ?? this.uid,
      email: email ?? this.email,
      displayName: displayName ?? this.displayName,
    );
  }

  // Pour faciliter la comparaison et l'utilisation dans les Sets/Maps
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is User &&
          runtimeType == other.runtimeType &&
          uid == other.uid &&
          email == other.email &&
          displayName == other.displayName;

  @override
  int get hashCode => uid.hashCode ^ email.hashCode ^ displayName.hashCode;

  @override
  String toString() {
    return 'User(uid: $uid, email: $email, displayName: $displayName)';
  }
}
```

--- END OF MODIFIED FILE lib\models\user.dart ---

--- START OF MODIFIED FILE lib\models\tick_model.dart ---

```dart
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
  });

  /// Crée une instance `Tick` à partir d'une Map JSON (venant de l'API/DB).
  factory Tick.fromJson(Map<String, dynamic> json) {
    // Extraction prudente des données, gérant plusieurs noms de clés possibles
    final String tickId = json['id'] ?? json['tickId'] ?? '';
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
    );
  }

  /// Retourne une chaîne de caractères formatée représentant la dernière mise à jour.
  /// Utilise `intl` pour une meilleure localisation et formatage relatif.
  String get formattedLastUpdate {
    if (lastUpdate == null) return 'jamais';

    final now = DateTime.now();
    final difference = now.difference(lastUpdate!);

    // Initialiser DateFormat pour la localisation française
    final timeFormat = DateFormat('HH:mm', 'fr_FR');
    final dateFormat = DateFormat('le dd/MM/yy', 'fr_FR');
    final dateTimeFormat = DateFormat('le dd/MM/yy à HH:mm', 'fr_FR');
    final weekdayFormat = DateFormat('EEEE', 'fr_FR'); // Jour de la semaine complet

    if (difference.inSeconds < 60) {
      return "à l'instant";
    } else if (difference.inMinutes < 60) {
      return "il y a ${difference.inMinutes} min";
    } else if (difference.inHours < now.hour) { // Moins de X heures, mais aujourd'hui
      return "auj. à ${timeFormat.format(lastUpdate!)}";
    } else if (difference.inHours < 24 + now.hour) { // Hier
      return "hier à ${timeFormat.format(lastUpdate!)}";
    } else if (difference.inDays < 7) { // Dans la semaine
      return "${weekdayFormat.format(lastUpdate!)} à ${timeFormat.format(lastUpdate!)}";
    } else { // Plus ancien
      return dateTimeFormat.format(lastUpdate!);
    }
  }

  /// Retourne une description textuelle du statut actuel.
  String get statusDescription {
    switch (status) {
      case TickStatus.active: return 'Actif';
      case TickStatus.inactive: return 'Hors ligne'; // Plus clair qu'inactif?
      case TickStatus.lowBattery: return 'Batterie faible';
      case TickStatus.moving: return 'En mouvement';
      case TickStatus.theftAlert: return 'Alerte Vol';
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
          macAddress == other.macAddress;

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
      macAddress.hashCode;

   @override
   String toString() {
      return 'Tick(id: $id, name: "$name", status: $status, lat: $latitude, lng: $longitude, bat: $batteryLevel, lastUpdate: $lastUpdate, ownerId: $ownerId)';
   }
}

/// Valeur sentinelle interne utilisée par `copyWith` pour différencier
/// une valeur non fournie (garder l'ancienne) d'une valeur explicitement `null`.
const Object _sentinel = Object();
```

--- END OF MODIFIED FILE lib\models\tick_model.dart ---

--- START OF MODIFIED FILE lib\screens\auth\login_page.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../services/auth_service.dart';
import '../../utils/constants.dart';
import '../../utils/validators.dart';
import '../../widgets/alert_card.dart'; // Pour CustomSnackBar
import '../../widgets/theme_toggle_button.dart';
import '../../widgets/loading_indicator.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({Key? key}) : super(key: key);

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _obscurePassword = true;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  /// Soumet le formulaire de connexion.
  Future<void> _submitLogin() async {
    // Masquer le clavier
    FocusScope.of(context).unfocus();

    // Valider le formulaire
    if (!(_formKey.currentState?.validate() ?? false)) {
      return;
    }

    final authService = Provider.of<AuthService>(context, listen: false);
    final email = _emailController.text.trim();
    final password = _passwordController.text;

    // Appeler le service d'authentification
    final loginSuccess = await authService.login(email, password);

    if (!mounted) return; // Vérifier après l'appel asynchrone

    if (loginSuccess) {
      // Connexion réussie, AuthService a mis à jour l'état et notifié.
      // Le widget parent (AuthWrapper ou similaire) devrait gérer la navigation.
      // Alternativement, si on est sûr que l'état est à jour:
      Navigator.pushNamedAndRemoveUntil(context, Routes.tickList, (route) => false);
    } else {
      // Gérer les erreurs spécifiques renvoyées par AuthService
      if (authService.needsConfirmation) {
        // L'utilisateur doit confirmer son compte
        CustomSnackBar.showError(context, authService.error ?? ErrorMessages.userNotConfirmed);
        // Naviguer vers la page de confirmation, en passant l'email
        Navigator.pushNamed(context, Routes.confirmSignUp, arguments: email);
      } else {
        // Autre erreur de connexion (mot de passe incorrect, etc.)
        CustomSnackBar.showError(context, authService.error ?? ErrorMessages.unknownError);
      }
    }
  }

  /// Gère la demande de réinitialisation de mot de passe.
  Future<void> _handleForgotPassword() async {
    FocusScope.of(context).unfocus();
    final authService = Provider.of<AuthService>(context, listen: false);

    // Pré-remplir l'email si possible
    String? initialEmail = _emailController.text.trim();
    if (!Validators.validateEmail(initialEmail) == null) { // Vérifier si c'est un email valide
       initialEmail = null; // Ne pas pré-remplir si invalide
    }

    // Afficher la boîte de dialogue pour entrer l'email
    final resultEmail = await showDialog<String>(
      context: context,
      builder: (context) => _buildForgotPasswordDialog(context, initialEmail: initialEmail),
    );

    // Si l'utilisateur a confirmé avec un email valide
    if (resultEmail != null && resultEmail.isNotEmpty) {
      final success = await authService.requestPasswordReset(resultEmail);
      if (!mounted) return;

      if (success) {
        CustomSnackBar.showSuccess(context, "Email de réinitialisation envoyé à $resultEmail. Vérifiez votre boîte mail.");
        // Naviguer vers la page de saisie du code et du nouveau mot de passe
        Navigator.pushNamed(context, Routes.passwordRecovery, arguments: resultEmail);
      } else {
        // Afficher l'erreur renvoyée par AuthService
        CustomSnackBar.showError(context, authService.error ?? ErrorMessages.unknownError);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(AppTexts.login),
        actions: const [ThemeToggleButton()],
      ),
      body: Consumer<AuthService>(
        builder: (context, authService, child) {
          // Utiliser un Stack pour superposer l'indicateur de chargement
          return Stack(
            children: [
              // Contenu principal (formulaire), désactivé pendant le chargement
              AbsorbPointer(
                absorbing: authService.isLoading,
                child: Center( // Centrer le formulaire verticalement
                  child: SingleChildScrollView( // Permet le défilement sur petits écrans
                    padding: const EdgeInsets.all(24.0),
                    child: Form(
                      key: _formKey,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch, // Étire les boutons
                        mainAxisAlignment: MainAxisAlignment.center, // Centre verticalement
                        children: [
                          // Logo avec animation Hero
                          Padding(
                            padding: const EdgeInsets.only(bottom: 48.0), // Espace sous le logo
                            child: Hero(
                              tag: 'logo', // Doit correspondre au tag dans WelcomePage/SplashScreen
                              child: Image.asset('assets/logo.png', height: 120),
                            ),
                          ),

                          // Champ Email
                          TextFormField(
                            controller: _emailController,
                            decoration: const InputDecoration(
                              labelText: AppTexts.email,
                              prefixIcon: Icon(Icons.email_outlined),
                            ),
                            keyboardType: TextInputType.emailAddress,
                            validator: Validators.validateEmail,
                            textInputAction: TextInputAction.next, // Aller au champ suivant
                            enabled: !authService.isLoading,
                          ),
                          const SizedBox(height: 16),

                          // Champ Mot de passe
                          TextFormField(
                            controller: _passwordController,
                            decoration: InputDecoration(
                              labelText: AppTexts.password,
                              prefixIcon: const Icon(Icons.lock_outline),
                              suffixIcon: IconButton(
                                icon: Icon(
                                  _obscurePassword
                                      ? Icons.visibility_outlined
                                      : Icons.visibility_off_outlined,
                                ),
                                onPressed: () {
                                  setState(() {
                                    _obscurePassword = !_obscurePassword;
                                  });
                                },
                              ),
                            ),
                            obscureText: _obscurePassword,
                            validator: Validators.validatePassword,
                            textInputAction: TextInputAction.done, // Action 'terminé' sur le clavier
                            onFieldSubmitted: (_) => _submitLogin(), // Permet de soumettre avec Entrée
                            enabled: !authService.isLoading,
                          ),
                          const SizedBox(height: 8),

                          // Lien Mot de passe oublié
                          Align(
                            alignment: Alignment.centerRight,
                            child: TextButton(
                              onPressed: authService.isLoading ? null : _handleForgotPassword,
                              child: const Text(AppTexts.forgotPassword),
                            ),
                          ),
                          const SizedBox(height: 24),

                          // Bouton de Connexion
                          ElevatedButton(
                            onPressed: authService.isLoading ? null : _submitLogin,
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 16.0),
                            ),
                            child: authService.isLoading
                                ? const LoadingIndicator(size: 20, color: Colors.white)
                                : const Text(
                                    AppTexts.login,
                                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                                  ),
                          ),
                          const SizedBox(height: 32),

                          // Lien vers l'Inscription
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const Text(AppTexts.noAccount),
                              TextButton(
                                onPressed: authService.isLoading
                                    ? null
                                    : () => Navigator.pushNamed(context, Routes.register),
                                child: const Text(AppTexts.register),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),

              // Indicateur de chargement global superposé
              if (authService.isLoading)
                Container(
                  color: Colors.black.withOpacity(0.3), // Fond semi-transparent
                  child: const Center(child: LoadingIndicator(size: 40)),
                ),
            ],
          );
        },
      ),
    );
  }

  /// Construit la boîte de dialogue pour la récupération de mot de passe.
  Widget _buildForgotPasswordDialog(BuildContext context, {String? initialEmail}) {
    final emailController = TextEditingController(text: initialEmail ?? '');
    final dialogFormKey = GlobalKey<FormState>(); // Clé spécifique pour ce formulaire

    return AlertDialog(
      title: const Text(AppTexts.passwordRecovery),
      content: Form(
        key: dialogFormKey,
        child: Column(
          mainAxisSize: MainAxisSize.min, // Ajuste la hauteur au contenu
          children: [
            const Text(AppTexts.passwordRecoveryInstructions),
            const SizedBox(height: 16),
            TextFormField(
              controller: emailController,
              decoration: const InputDecoration(
                labelText: AppTexts.email,
                prefixIcon: Icon(Icons.email_outlined),
              ),
              keyboardType: TextInputType.emailAddress,
              validator: Validators.validateEmail, // Utiliser le validateur
              autofocus: true,
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context), // Ferme sans renvoyer de valeur
          child: const Text(AppTexts.cancel),
        ),
        ElevatedButton(
          onPressed: () {
            // Valider le formulaire de la boîte de dialogue avant de fermer
            if (dialogFormKey.currentState?.validate() ?? false) {
              Navigator.pop(context, emailController.text.trim()); // Renvoie l'email
            }
          },
          child: const Text(AppTexts.sendRecoveryLink),
        ),
      ],
    );
  }
}
```

--- END OF MODIFIED FILE lib\screens\auth\login_page.dart ---

--- START OF MODIFIED FILE lib\screens\auth\register_page.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../services/auth_service.dart';
import '../../utils/constants.dart';
import '../../utils/validators.dart';
import '../../widgets/alert_card.dart'; // Pour CustomSnackBar
import '../../widgets/theme_toggle_button.dart';
import '../../widgets/loading_indicator.dart';

class RegisterPage extends StatefulWidget {
  const RegisterPage({Key? key}) : super(key: key);

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  final _nameController = TextEditingController(); // Pour le nom complet

  bool _obscurePassword = true;
  bool _obscureConfirmPassword = true;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    _nameController.dispose();
    super.dispose();
  }

  /// Soumet le formulaire d'inscription.
  Future<void> _submitRegistration() async {
    FocusScope.of(context).unfocus();

    if (!(_formKey.currentState?.validate() ?? false)) {
      return;
    }

    final authService = Provider.of<AuthService>(context, listen: false);
    final email = _emailController.text.trim();
    final password = _passwordController.text;
    final name = _nameController.text.trim();

    // Appeler le service d'inscription
    final registerSuccess = await authService.register(email, password, name);

    if (!mounted) return;

    if (registerSuccess) {
      if (authService.needsConfirmation) {
        // Inscription réussie, mais nécessite confirmation
        CustomSnackBar.show(
          context,
          message: "${AppTexts.checkEmailForCode} ($email).",
          type: AlertType.info,
          duration: AppDurations.longDelay * 2, // Durée plus longue
        );
        // Naviguer vers la page de confirmation, en remplaçant la page actuelle
        Navigator.pushReplacementNamed(context, Routes.confirmSignUp, arguments: email);
      } else {
        // Inscription réussie ET auto-confirmée (moins courant)
        CustomSnackBar.showSuccess(context, "Inscription réussie ! Vous pouvez maintenant vous connecter.");
        // Rediriger vers la page de connexion, en retirant les pages précédentes
        Navigator.pushNamedAndRemoveUntil(context, Routes.login, (route) => false);
      }
    } else {
      // Afficher l'erreur renvoyée par AuthService (ex: email déjà utilisé)
      CustomSnackBar.showError(context, authService.error ?? ErrorMessages.unknownError);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(AppTexts.register),
        actions: const [ThemeToggleButton()],
      ),
      body: Consumer<AuthService>(
        builder: (context, authService, child) {
          return Stack(
            children: [
              // Formulaire, désactivé pendant le chargement
              AbsorbPointer(
                absorbing: authService.isLoading,
                child: Center( // Centrer le formulaire
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(24.0),
                    child: Form(
                      key: _formKey,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          // Logo (optionnel, plus petit que sur Login)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 32.0),
                            child: Hero(
                              tag: 'logo', // Match tag
                              child: Image.asset('assets/logo.png', height: 100),
                            ),
                          ),

                          // Champ Nom Complet
                          TextFormField(
                            controller: _nameController,
                            decoration: const InputDecoration(
                              labelText: AppTexts.name,
                              prefixIcon: Icon(Icons.person_outline),
                            ),
                            validator: (value) => Validators.validateNotEmpty(value, "Veuillez entrer votre nom"),
                            textInputAction: TextInputAction.next,
                            enabled: !authService.isLoading,
                            textCapitalization: TextCapitalization.words, // Majuscule au début des mots
                          ),
                          const SizedBox(height: 16),

                          // Champ Email
                          TextFormField(
                            controller: _emailController,
                            decoration: const InputDecoration(
                              labelText: AppTexts.email,
                              prefixIcon: Icon(Icons.email_outlined),
                            ),
                            keyboardType: TextInputType.emailAddress,
                            validator: Validators.validateEmail,
                            textInputAction: TextInputAction.next,
                            enabled: !authService.isLoading,
                          ),
                          const SizedBox(height: 16),

                          // Champ Mot de passe
                          TextFormField(
                            controller: _passwordController,
                            decoration: InputDecoration(
                              labelText: AppTexts.password,
                              prefixIcon: const Icon(Icons.lock_outline),
                              suffixIcon: IconButton(
                                icon: Icon(
                                  _obscurePassword ? Icons.visibility_outlined : Icons.visibility_off_outlined,
                                ),
                                onPressed: () => setState(() => _obscurePassword = !_obscurePassword),
                              ),
                            ),
                            obscureText: _obscurePassword,
                            validator: Validators.validatePassword,
                            textInputAction: TextInputAction.next,
                            enabled: !authService.isLoading,
                          ),
                          const SizedBox(height: 16),

                          // Champ Confirmation Mot de passe
                          TextFormField(
                            controller: _confirmPasswordController,
                            decoration: InputDecoration(
                              labelText: AppTexts.confirmPassword,
                              prefixIcon: const Icon(Icons.lock_outline),
                              suffixIcon: IconButton(
                                icon: Icon(
                                  _obscureConfirmPassword ? Icons.visibility_outlined : Icons.visibility_off_outlined,
                                ),
                                onPressed: () => setState(() => _obscureConfirmPassword = !_obscureConfirmPassword),
                              ),
                            ),
                            obscureText: _obscureConfirmPassword,
                            // Validation comparant avec le champ mot de passe
                            validator: (value) => Validators.validateConfirmPassword(value, _passwordController.text),
                            textInputAction: TextInputAction.done,
                            onFieldSubmitted: (_) => _submitRegistration(), // Soumettre avec Entrée
                            enabled: !authService.isLoading,
                          ),
                          const SizedBox(height: 32),

                          // Bouton d'Inscription
                          ElevatedButton(
                            onPressed: authService.isLoading ? null : _submitRegistration,
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 16.0),
                            ),
                            child: authService.isLoading
                                ? const LoadingIndicator(size: 20, color: Colors.white)
                                : const Text(
                                    AppTexts.register,
                                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                                  ),
                          ),
                          const SizedBox(height: 32),

                          // Lien vers la Connexion
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const Text(AppTexts.alreadyAccount),
                              TextButton(
                                onPressed: authService.isLoading
                                    ? null
                                    : () => Navigator.pop(context), // Retourne à la page précédente (Login)
                                child: const Text(AppTexts.login),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),

              // Indicateur de chargement global
              if (authService.isLoading)
                Container(
                  color: Colors.black.withOpacity(0.3),
                  child: const Center(child: LoadingIndicator(size: 40)),
                ),
            ],
          );
        },
      ),
    );
  }
}
```

--- END OF MODIFIED FILE lib\screens\auth\register_page.dart ---

--- START OF MODIFIED FILE lib\screens\auth\confirmation_page.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../services/auth_service.dart';
import '../../utils/constants.dart';
import '../../utils/validators.dart';
import '../../widgets/alert_card.dart'; // Pour CustomSnackBar
import '../../widgets/loading_indicator.dart';
import '../../widgets/theme_toggle_button.dart';

class ConfirmationPage extends StatefulWidget {
  const ConfirmationPage({Key? key}) : super(key: key);

  @override
  State<ConfirmationPage> createState() => _ConfirmationPageState();
}

class _ConfirmationPageState extends State<ConfirmationPage> {
  final _formKey = GlobalKey<FormState>();
  final _codeController = TextEditingController();
  String? _username; // Pour stocker l'email passé en argument

  bool _isResendingCode = false; // État spécifique pour le bouton de renvoi

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Tenter de récupérer l'email passé en argument lors de la navigation
    final arguments = ModalRoute.of(context)?.settings.arguments;
    if (arguments is String && arguments.isNotEmpty) {
      _username = arguments;
    } else {
      // Sinon, essayer de le récupérer depuis l'état du AuthService
      _username = Provider.of<AuthService>(context, listen: false).pendingUsername;
    }

    // Gérer le cas où l'email n'est pas disponible (sécurité)
    if (_username == null || _username!.isEmpty) {
      print("ERREUR: Aucun username trouvé pour la page de confirmation.");
      // Option: rediriger immédiatement vers la page de connexion
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) {
          CustomSnackBar.showError(context, ErrorMessages.unknownError);
          Navigator.pushNamedAndRemoveUntil(context, Routes.login, (route) => false);
        }
      });
    }
  }

  @override
  void dispose() {
    _codeController.dispose();
    super.dispose();
  }

  /// Soumet le code de confirmation.
  Future<void> _submitConfirmation() async {
    FocusScope.of(context).unfocus();

    if (_username == null) {
      CustomSnackBar.showError(context, "Impossible de confirmer sans email.");
      return;
    }
    if (!(_formKey.currentState?.validate() ?? false)) {
      return;
    }

    final authService = Provider.of<AuthService>(context, listen: false);
    final code = _codeController.text.trim();

    final success = await authService.confirmSignUp(_username!, code);

    if (!mounted) return;

    if (success) {
      CustomSnackBar.showSuccess(context, "Compte confirmé avec succès ! Vous pouvez maintenant vous connecter.");
      // Rediriger vers la page de connexion
      Navigator.pushNamedAndRemoveUntil(context, Routes.login, (route) => false);
    } else {
      // Afficher l'erreur renvoyée par AuthService (ex: code invalide)
      CustomSnackBar.showError(context, authService.error ?? "Erreur de confirmation.");
    }
  }

  /// Demande le renvoi du code de confirmation.
  Future<void> _resendCode() async {
    if (_username == null) {
      CustomSnackBar.showError(context, "Impossible de renvoyer le code sans email.");
      return;
    }
    setState(() => _isResendingCode = true); // Afficher indicateur sur le bouton
    final authService = Provider.of<AuthService>(context, listen: false);

    final success = await authService.resendConfirmationCode(_username!);

    if (!mounted) return;
    // Arrêter l'indicateur du bouton après l'appel
    setState(() => _isResendingCode = false);

    if (success) {
      CustomSnackBar.showSuccess(context, "${AppTexts.resendCode} envoyé à $_username.");
    } else {
      // Afficher l'erreur du service (ex: limite dépassée, utilisateur déjà confirmé)
      CustomSnackBar.showError(context, authService.error ?? "Erreur lors du renvoi du code.");
      // Cas spécifique: si l'utilisateur est déjà confirmé, le rediriger
      if (authService.error?.contains("déjà confirmé") ?? false) {
        await Future.delayed(AppDurations.shortDelay); // Petit délai pour lire le message
        if (mounted) {
          Navigator.pushNamedAndRemoveUntil(context, Routes.login, (route) => false);
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // Récupérer l'état de chargement global d'AuthService
    final authService = context.watch<AuthService>();
    final isLoading = authService.isLoading;

    return Scaffold(
      appBar: AppBar(
        title: const Text(AppTexts.confirmAccount),
        actions: const [ThemeToggleButton()],
      ),
      body: Stack(
        children: [
          // Formulaire, désactivé si chargement global ou renvoi en cours
          AbsorbPointer(
            absorbing: isLoading || _isResendingCode,
            child: Center( // Centrer
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(24.0),
                child: Form(
                  key: _formKey,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Icône
                      const Icon(Icons.mark_email_read_outlined, size: 60, color: AppTheme.primaryColor),
                      const SizedBox(height: 24),
                      // Instructions
                      Text(
                        AppTexts.codeSentTo,
                        style: Theme.of(context).textTheme.titleMedium,
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        _username ?? AppTexts.unknownUser, // Afficher l'email cible
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 32),
                      // Champ Code
                      TextFormField(
                        controller: _codeController,
                        decoration: const InputDecoration(
                          labelText: AppTexts.confirmationCode,
                          prefixIcon: Icon(Icons.pin_outlined),
                          hintText: AppTexts.enterConfirmationCode,
                          counterText: "", // Masquer le compteur par défaut
                        ),
                        keyboardType: TextInputType.number,
                        validator: Validators.validateConfirmationCode, // Validateur spécifique
                        maxLength: 6, // Longueur standard Cognito
                        enabled: !(isLoading || _isResendingCode),
                        autofocus: true,
                        textAlign: TextAlign.center, // Centrer le texte du code
                        style: const TextStyle(fontSize: 18, letterSpacing: 4), // Style pour code
                        textInputAction: TextInputAction.done,
                        onFieldSubmitted: (_) => _submitConfirmation(),
                      ),
                      const SizedBox(height: 32),
                      // Bouton Confirmer
                      ElevatedButton(
                        onPressed: (isLoading || _isResendingCode) ? null : _submitConfirmation,
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16.0),
                        ),
                        // Afficher indicateur seulement si c'est la confirmation qui charge
                        child: (isLoading && !_isResendingCode)
                            ? const LoadingIndicator(size: 20, color: Colors.white)
                            : const Text(AppTexts.confirm, style: TextStyle(fontSize: 16)),
                      ),
                      const SizedBox(height: 24),
                      // Bouton Renvoyer le code
                      TextButton(
                        onPressed: (isLoading || _isResendingCode) ? null : _resendCode,
                        child: _isResendingCode
                            ? const Row( // Afficher texte + indicateur pendant le renvoi
                                mainAxisSize: MainAxisSize.min,
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  LoadingIndicator(size: 16),
                                  SizedBox(width: 8),
                                  Text("Renvoi en cours...")
                                ],
                              )
                            : const Text(AppTexts.resendCode),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          // Indicateur de chargement global (pour la soumission du code)
          if (isLoading && !_isResendingCode)
            Container(
              color: Colors.black.withOpacity(0.3),
              child: const Center(child: LoadingIndicator(size: 40)),
            ),
        ],
      ),
    );
  }
}
```

--- END OF MODIFIED FILE lib\screens\auth\confirmation_page.dart ---

--- START OF MODIFIED FILE lib\screens\auth\reset_password_page.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../services/auth_service.dart';
import '../../utils/constants.dart';
import '../../utils/validators.dart';
import '../../widgets/loading_indicator.dart';
import '../../widgets/theme_toggle_button.dart';
import '../../widgets/alert_card.dart'; // Pour CustomSnackBar

class ResetPasswordPage extends StatefulWidget {
  const ResetPasswordPage({Key? key}) : super(key: key);

  @override
  State<ResetPasswordPage> createState() => _ResetPasswordPageState();
}

class _ResetPasswordPageState extends State<ResetPasswordPage> {
  final _formKey = GlobalKey<FormState>();
  final _codeController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();

  String? _username; // Email pour lequel réinitialiser

  bool _obscurePassword = true;
  bool _obscureConfirmPassword = true;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Récupérer l'email passé en argument depuis la page précédente (Login)
    final arguments = ModalRoute.of(context)?.settings.arguments;
    if (arguments is String && arguments.isNotEmpty) {
      _username = arguments;
    } else {
      // Fallback: essayer de le récupérer depuis l'état du AuthService
      _username = Provider.of<AuthService>(context, listen: false).pendingUsername;
    }

    if (_username == null || _username!.isEmpty) {
      print("ERREUR: Aucun username trouvé pour la page de réinitialisation.");
      // Option: revenir en arrière ou vers login
      WidgetsBinding.instance.addPostFrameCallback((_) {
         if (mounted) {
           CustomSnackBar.showError(context, ErrorMessages.unknownError);
           Navigator.pop(context); // Revenir à la page précédente (Login)
         }
      });
    }
  }

  @override
  void dispose() {
    _codeController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  /// Soumet la demande de confirmation de réinitialisation.
  Future<void> _submitResetPassword() async {
    FocusScope.of(context).unfocus();
    if (_username == null) {
      CustomSnackBar.showError(context, "Impossible de réinitialiser sans email.");
      return;
    }

    if (!(_formKey.currentState?.validate() ?? false)) {
      return;
    }

    final authService = Provider.of<AuthService>(context, listen: false);
    final code = _codeController.text.trim();
    final newPassword = _passwordController.text;

    final success = await authService.confirmPasswordReset(_username!, newPassword, code);

    if (!mounted) return;

    if (success) {
      CustomSnackBar.showSuccess(context, "Mot de passe réinitialisé avec succès. Vous pouvez maintenant vous connecter.");
      // Rediriger vers la page de connexion
      Navigator.pushNamedAndRemoveUntil(context, Routes.login, (route) => false);
    } else {
      // Afficher l'erreur renvoyée par AuthService (ex: code invalide, politique mdp)
      CustomSnackBar.showError(context, authService.error ?? "Erreur de réinitialisation.");
    }
  }

  // #TODO: Ajouter un bouton "Renvoyer le code" similaire à ConfirmationPage si nécessaire
  // Future<void> _resendResetCode() async { ... }

  @override
  Widget build(BuildContext context) {
    final authService = context.watch<AuthService>();
    final isLoading = authService.isLoading;

    return Scaffold(
      appBar: AppBar(
        title: const Text(AppTexts.resetPassword),
        actions: const [ThemeToggleButton()],
      ),
      body: Stack(
        children: [
          // Formulaire, désactivé si chargement
          AbsorbPointer(
            absorbing: isLoading,
            child: Center( // Centrer
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(24.0),
                child: Form(
                  key: _formKey,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Icône
                      const Icon(Icons.lock_reset_outlined, size: 60, color: AppTheme.primaryColor),
                      const SizedBox(height: 24),
                      // Instructions
                      Text(
                        "${AppTexts.checkEmailForCode} (${_username ?? AppTexts.unknownUser}).",
                        style: Theme.of(context).textTheme.titleMedium,
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 32),
                      // Champ Code
                      TextFormField(
                        controller: _codeController,
                        decoration: const InputDecoration(
                          labelText: AppTexts.enterResetCode,
                          prefixIcon: Icon(Icons.pin_outlined),
                          counterText: "",
                        ),
                        keyboardType: TextInputType.number,
                        validator: Validators.validateConfirmationCode, // Même validation que code confirm
                        maxLength: 6,
                        enabled: !isLoading,
                        textInputAction: TextInputAction.next,
                        autofocus: true,
                        textAlign: TextAlign.center,
                        style: const TextStyle(fontSize: 18, letterSpacing: 4),
                      ),
                      const SizedBox(height: 16),
                      // Champ Nouveau Mot de Passe
                      TextFormField(
                        controller: _passwordController,
                        decoration: InputDecoration(
                          labelText: AppTexts.newPassword,
                          prefixIcon: const Icon(Icons.lock_outline),
                          suffixIcon: IconButton(
                            icon: Icon(_obscurePassword ? Icons.visibility_outlined : Icons.visibility_off_outlined),
                            onPressed: () => setState(() => _obscurePassword = !_obscurePassword),
                          ),
                        ),
                        obscureText: _obscurePassword,
                        validator: Validators.validatePassword, // Vérifie longueur min
                        textInputAction: TextInputAction.next,
                        enabled: !isLoading,
                      ),
                      const SizedBox(height: 16),
                      // Champ Confirmer Mot de Passe
                      TextFormField(
                        controller: _confirmPasswordController,
                        decoration: InputDecoration(
                          labelText: AppTexts.confirmPassword,
                          prefixIcon: const Icon(Icons.lock_outline),
                          suffixIcon: IconButton(
                            icon: Icon(_obscureConfirmPassword ? Icons.visibility_outlined : Icons.visibility_off_outlined),
                            onPressed: () => setState(() => _obscureConfirmPassword = !_obscureConfirmPassword),
                          ),
                        ),
                        obscureText: _obscureConfirmPassword,
                        validator: (value) => Validators.validateConfirmPassword(value, _passwordController.text),
                        textInputAction: TextInputAction.done,
                        onFieldSubmitted: (_) => _submitResetPassword(),
                        enabled: !isLoading,
                      ),
                      const SizedBox(height: 32),
                      // Bouton Confirmer
                      ElevatedButton(
                        onPressed: isLoading ? null : _submitResetPassword,
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16.0),
                        ),
                        child: isLoading
                            ? const LoadingIndicator(size: 20, color: Colors.white)
                            : const Text(AppTexts.confirm, style: TextStyle(fontSize: 16)),
                      ),
                      // Optionnel: Ajouter bouton Renvoyer le code ici
                      // const SizedBox(height: 16),
                      // TextButton(onPressed: isLoading ? null : _resendResetCode, child: Text(AppTexts.resendCode)),
                    ],
                  ),
                ),
              ),
            ),
          ),
          // Indicateur de chargement global
          if (isLoading)
            Container(
              color: Colors.black.withOpacity(0.3),
              child: const Center(child: LoadingIndicator(size: 40)),
            ),
        ],
      ),
    );
  }
}
```

--- END OF MODIFIED FILE lib\screens\auth\reset_password_page.dart ---

--- START OF MODIFIED FILE lib\screens\tick\map_page.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:provider/provider.dart';
import 'package:geolocator/geolocator.dart'; // Pour la localisation utilisateur
import 'package:permission_handler/permission_handler.dart' as ph; // Pour ouvrir les paramètres
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
  bool _isTickActionLoading = false; // Pour les actions 'Localiser', 'Sonner'
  bool _isUserLocationLoading = false; // Pour la récupération de la position utilisateur

  // La position de l'utilisateur (peut être null)
  LatLng? _userPosition;

  // Les marqueurs à afficher sur la carte (contiendra le Tick et potentiellement l'utilisateur)
  final Set<Marker> _markers = {};

  // Référence locale au Tick, mise à jour par Provider
  late Tick _currentTickData;

  @override
  void initState() {
    super.initState();
    // Initialiser avec les données passées via le constructeur
    _currentTickData = widget.tick;
    // Mettre à jour le marqueur initial du Tick
    _updateTickMarker();
    // Essayer de récupérer la position initiale de l'utilisateur (sans bloquer)
    _getCurrentUserLocation(centerMap: false);
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Écouter les changements du Tick spécifique via le Provider
    // Si TickService notifie un changement, on récupère la nouvelle version du Tick
    final updatedTick = context.watch<TickService>().getTickById(widget.tick.id);
    if (updatedTick != null && updatedTick != _currentTickData) {
       print("MapPage: Received update for Tick ${_currentTickData.id}");
       setState(() {
          _currentTickData = updatedTick;
          _updateTickMarker(); // Mettre à jour le marqueur avec les nouvelles données
       });
    } else if (updatedTick == null) {
        // Le Tick a été supprimé (désassocié?), revenir en arrière
        print("MapPage: Tick ${widget.tick.id} no longer found in service. Popping.");
         WidgetsBinding.instance.addPostFrameCallback((_) {
            if (mounted) {
               CustomSnackBar.showError(context, "Ce Tick n'est plus disponible.");
               Navigator.of(context).pop();
            }
         });
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
      case TickStatus.inactive:
      case TickStatus.unknown:
      default: hue = BitmapDescriptor.hueViolet; break; // Violet/Pourpre pour inactif/inconnu
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
        17.0, // Zoom plus proche sur le Tick
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
    if (_isTickActionLoading) return; // Éviter les appels multiples
    setState(() => _isTickActionLoading = true);

    final tickService = Provider.of<TickService>(context, listen: false);
    final success = await tickService.requestTickLocation(_currentTickData.id);

    if (!mounted) return;
    setState(() => _isTickActionLoading = false);

    if (success) {
      CustomSnackBar.show(context, message: AppTexts.locationRequestSent, type: AlertType.info);
      // La mise à jour de la position arrivera via le listener du Provider ou une notification push
    } else {
      CustomSnackBar.showError(context, tickService.error ?? ErrorMessages.apiError);
    }
  }

  /// Demande à faire sonner le Tick via le service.
  Future<void> _ringTick() async {
    if (_isTickActionLoading) return;
    setState(() => _isTickActionLoading = true);

    final tickService = Provider.of<TickService>(context, listen: false);
    // #TODO: Appeler la méthode `ringTick` de TickService (à implémenter dans le service)
    // final success = await tickService.ringTick(_currentTickData.id);
    await Future.delayed(const Duration(seconds: 1)); // Simuler l'appel API
    final success = true; // Placeholder

    if (!mounted) return;
    setState(() => _isTickActionLoading = false);

    if (success) {
      CustomSnackBar.showSuccess(context, AppTexts.ringingTick);
    } else {
      CustomSnackBar.showError(context, tickService.error ?? ErrorMessages.apiError);
    }
  }

  // --- Navigation ---

  /// Navigue vers la page des paramètres du Tick.
  void _navigateToTickSettings() {
    // Utilise une route nommée et passe l'ID du Tick en argument
    Navigator.pushNamed(context, Routes.tickSettings, arguments: _currentTickData.id);
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
            icon: _isTickActionLoading ? const LoadingIndicator(size: 18) : const Icon(Icons.refresh),
            tooltip: AppTexts.locate,
            onPressed: _isTickActionLoading ? null : _requestLocationUpdate,
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
                    zoomControlsEnabled: true, // Contrôles +/-
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
                            icon: _isTickActionLoading ? const LoadingIndicator(size: 18, color: Colors.white) : const Icon(Icons.refresh, size: 18),
                            label: const Text(AppTexts.tryToLocate),
                            onPressed: _isTickActionLoading ? null : _requestLocationUpdate,
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
                    child: const Icon(Icons.gps_fixed),
                  ),
                ),
                Positioned(
                  right: 16,
                  bottom: 20,
                  child: FloatingActionButton.small(
                    heroTag: "centerUserBtn",
                    onPressed: _centerOnUser,
                    tooltip: AppTexts.centerOnMe,
                    child: _isUserLocationLoading
                      ? const LoadingIndicator(size: 18) // Indicateur pendant la recherche utilisateur
                      : const Icon(Icons.my_location),
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
                        isLoading: _isTickActionLoading,
                        isDisabled: _isTickActionLoading,
                        color: AppTheme.accentColor, // Vert pour localiser
                      ),
                      ActionButton(
                        icon: Icons.volume_up_outlined,
                        label: AppTexts.ring,
                        onPressed: _ringTick,
                        isLoading: _isTickActionLoading,
                        isDisabled: _isTickActionLoading,
                        color: AppTheme.warningColor, // Orange pour sonner
                      ),
                       ActionButton(
                        icon: Icons.history_outlined,
                        label: AppTexts.history,
                        onPressed: _navigateToHistory,
                        // isDisabled: _isTickActionLoading, // Accès historique pendant action?
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
              value: (_currentTickData.latitude != null && _currentTickData.longitude != null)
                  ? '${_currentTickData.latitude!.toStringAsFixed(5)}, ${_currentTickData.longitude!.toStringAsFixed(5)}'
                  : AppTexts.noLocationAvailable,
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
  }) {
    final textTheme = Theme.of(context).textTheme;
    final secondaryColor = textTheme.bodySmall?.color;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        children: [
          Icon(icon, size: 18, color: secondaryColor),
          const SizedBox(width: 12),
          Text('$label:', style: textTheme.bodyMedium?.copyWith(color: secondaryColor)),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              value,
              textAlign: TextAlign.right,
              style: textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w500,
                color: valueColor ?? textTheme.bodyMedium?.color, // Applique couleur si fournie
              ),
              overflow: TextOverflow.ellipsis,
              maxLines: 1,
            ),
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

--- END OF MODIFIED FILE lib\screens\tick\map_page.dart ---

--- START OF MODIFIED FILE lib\screens\tick\history_page.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart'; // Pour formater dates/heures

import '../../models/tick_model.dart'; // Pour TickStatus (si utilisé indirectement)
import '../../services/tick_service.dart';
import '../../utils/constants.dart'; // Pour textes et couleurs
import '../../utils/theme.dart'; // Pour AppTheme et AlertType
import '../../widgets/loading_indicator.dart';

/// Modèle simple pour représenter un événement d'historique.
/// Les champs doivent correspondre à ce que renvoie la fonction Lambda `getTickHistoryFunctionUrl`.
class HistoryEvent {
  final String id; // ID unique de l'événement (peut être généré ou basé sur la clé DB)
  final DateTime timestamp;
  final String eventType; // Ex: 'periodic_update', 'movement_alert', etc. (doit correspondre aux types MQTT)
  final double? latitude;
  final double? longitude;
  final int? batteryLevel;
  final String description; // Description formatée de l'événement

  HistoryEvent({
    required this.id,
    required this.timestamp,
    required this.eventType,
    required this.description,
    this.latitude,
    this.longitude,
    this.batteryLevel,
  });

  /// Factory pour créer une instance `HistoryEvent` depuis une Map JSON renvoyée par l'API Lambda.
  factory HistoryEvent.fromJson(Map<String, dynamic> json) {
    // Parsing prudent des données, gérant différents noms de clés possibles
    double? lat;
    if (json['position']?['lat'] is num) lat = (json['position']['lat'] as num).toDouble();
    if (lat == null && json['lat'] is num) lat = (json['lat'] as num).toDouble();
    if (lat == null && json['latitude'] is num) lat = (json['latitude'] as num).toDouble();

    double? lng;
    if (json['position']?['lng'] is num) lng = (json['position']['lng'] as num).toDouble();
    if (lng == null && json['lng'] is num) lng = (json['lng'] as num).toDouble();
    if (lng == null && json['longitude'] is num) lng = (json['longitude'] as num).toDouble();

    int? bat;
    if (json['batteryLevel'] is num) bat = (json['batteryLevel'] as num).toInt();
    if (bat == null && json['bat'] is num) bat = (json['bat'] as num).toInt();
    if (bat == null && json['battery'] is num) bat = (json['battery'] as num).toInt();

    DateTime ts = DateTime.now(); // Valeur par défaut
    // Essayer de parser depuis une chaîne ISO 8601
    if (json['timestamp'] is String) {
      ts = DateTime.tryParse(json['timestamp']) ?? DateTime.now();
    }
    // Essayer de parser depuis la clé de tri DynamoDB (si utilisée comme timestamp)
    else if (json['SK'] is String && json['SK'].startsWith('HISTORY#')) {
      final String tsString = json['SK'].substring(8); // Enlève 'HISTORY#'
      ts = DateTime.tryParse(tsString) ?? DateTime.now();
    }
     // Essayer de parser depuis un timestamp epoch (millisecondes ou secondes)
     else if (json['timestamp'] is num) {
        final tsNum = json['timestamp'] as num;
        if (tsNum > 1000000000000) { // Millisecondes
           ts = DateTime.fromMillisecondsSinceEpoch(tsNum.toInt());
        } else { // Secondes
           ts = DateTime.fromMillisecondsSinceEpoch(tsNum.toInt() * 1000);
        }
     }

    String type = json['eventType'] ?? 'unknown';
    // Utiliser la description fournie par l'API si elle existe, sinon générer une description par défaut
    String desc = json['description'] ?? _generateDefaultDescription(type, lat, lng, bat, ts);

    // Générer un ID unique si non fourni par l'API
    String eventId = json['id'] ?? json['eventId'] ?? json['PK'] ?? UniqueKey().toString();

    return HistoryEvent(
      id: eventId,
      timestamp: ts,
      eventType: type,
      latitude: lat,
      longitude: lng,
      batteryLevel: bat,
      description: desc,
    );
  }

  /// Génère une description par défaut basée sur le type d'événement et les données.
  static String _generateDefaultDescription(String type, double? lat, double? lng, int? bat, DateTime ts) {
    String timeStr = DateFormat('HH:mm', 'fr_FR').format(ts); // Heure formatée
    switch (type) {
      case 'periodic_update': return "Mise à jour périodique reçue à $timeStr.";
      case 'movement_alert': return "Mouvement détecté à $timeStr.";
      case 'theft_alert': return "⚠️ ALERTE DÉPLACEMENT à $timeStr !";
      case 'low_battery': return "Batterie faible ($bat%) détectée à $timeStr.";
      case 'location_response': return "Position reçue à $timeStr.";
      case 'link_device': return "Appareil associé à $timeStr.";
      case 'unlink_device': return "Appareil désassocié à $timeStr.";
      case 'sound_alert': return "Sonnerie activée à $timeStr.";
      case 'temporary_disable': return "Surveillance désactivée à $timeStr.";
      case 'reactivate': return "Surveillance réactivée à $timeStr.";
      default: return "Événement système ($type) à $timeStr.";
    }
  }

  /// Retourne une icône basée sur le type d'événement.
  IconData get eventIcon {
    switch (eventType) {
      case 'periodic_update': return Icons.update_outlined;
      case 'movement_alert': return Icons.directions_walk; // Icône de marche
      case 'theft_alert': return Icons.warning_amber_rounded; // Icône d'alerte
      case 'low_battery': return Icons.battery_alert_outlined;
      case 'location_response': return Icons.gps_fixed;
      case 'link_device': return Icons.link;
      case 'unlink_device': return Icons.link_off;
      case 'sound_alert': return Icons.volume_up_outlined;
      case 'temporary_disable': return Icons.pause_circle_outline;
      case 'reactivate': return Icons.play_circle_outline;
      default: return Icons.history_edu_outlined; // Icône générique pour inconnu
    }
  }

  /// Retourne une couleur basée sur le type d'événement pour l'affichage (ex: icône).
  Color getEventColor(BuildContext context) {
    switch (eventType) {
      case 'theft_alert': return AppTheme.errorColor;
      case 'low_battery': return AppTheme.warningColor;
      case 'link_device':
      case 'unlink_device':
      case 'reactivate':
          return AppTheme.accentColor; // Vert pour actions système?
      case 'movement_alert':
      case 'location_response':
      case 'sound_alert':
      case 'temporary_disable':
          return AppTheme.primaryColor; // Bleu pour infos/actions standard
      case 'periodic_update': // Gris pour événements fréquents/moins importants
          return Theme.of(context).colorScheme.onSurface.withOpacity(0.5);
      default: // Gris clair pour inconnu
          return Theme.of(context).colorScheme.onSurface.withOpacity(0.7);
    }
  }
}

/// Page affichant l'historique des événements pour un Tick spécifique.
class HistoryPage extends StatefulWidget {
  final String tickId;
  final String tickName; // Passer aussi le nom pour l'AppBar

  const HistoryPage({
    Key? key,
    required this.tickId,
    required this.tickName,
  }) : super(key: key);

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  // Future pour stocker le résultat de l'appel API
  late Future<List<HistoryEvent>> _historyFuture;

  @override
  void initState() {
    super.initState();
    // Lancer la récupération de l'historique lors de l'initialisation
    _historyFuture = _fetchHistory();
  }

  /// Fonction interne pour appeler le [TickService] et parser la réponse.
  Future<List<HistoryEvent>> _fetchHistory() async {
    final tickService = Provider.of<TickService>(context, listen: false);

    try {
      // Appeler la méthode du service qui retourne la Map brute de l'API
      final response = await tickService.getTickHistory(widget.tickId);

      // Vérifier si l'appel API a réussi et si les données sont une liste
      if (response['success'] == true && response['data'] is List) {
        final List<dynamic> rawDataList = response['data'];

        // Mapper les données JSON en objets HistoryEvent
        // Le parsing est géré dans HistoryEvent.fromJson
        // La Lambda devrait trier par timestamp descendant
        List<HistoryEvent> events = rawDataList.map((data) {
          try {
            return HistoryEvent.fromJson(data as Map<String, dynamic>);
          } catch (e) {
            print("HistoryPage: Error parsing HistoryEvent JSON item: $e \nData: $data");
            return null; // Ignorer l'élément mal formé
          }
        }).whereType<HistoryEvent>().toList(); // Filtrer les éléments nulls

        print("HistoryPage: History processed successfully: ${events.length} events for ${widget.tickId}.");
        return events;

      } else {
        // Si l'appel API a échoué ou les données sont mal formées
        String errorMessage = response['error']?.toString() ?? ErrorMessages.apiError;
        print("HistoryPage: Error response received from getTickHistory: $errorMessage");
        // Lancer une exception pour que le FutureBuilder l'attrape
        throw Exception(errorMessage);
      }
    } catch (e) {
       // Capturer les exceptions de connexion ou autres erreurs inattendues
       print("HistoryPage: Exception fetching history for ${widget.tickId}: $e");
       throw Exception(ErrorMessages.connectionFailed); // Remonter une erreur générique
    }
  }

  /// Rafraîchit la liste de l'historique en relançant l'appel API.
  Future<void> _refreshHistory() async {
    // Le RefreshIndicator affiche déjà un indicateur de chargement.
    // On met simplement à jour le Future pour que le FutureBuilder se reconstruise.
    setState(() {
      _historyFuture = _fetchHistory();
    });
    // Attendre la fin du nouveau future pour que le RefreshIndicator disparaisse
    await _historyFuture;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("${AppTexts.history} - ${widget.tickName}"),
      ),
      body: RefreshIndicator(
        onRefresh: _refreshHistory, // Action pour le pull-to-refresh
        child: FutureBuilder<List<HistoryEvent>>(
          future: _historyFuture,
          builder: (context, snapshot) {
            // --- Cas 1: Chargement en cours ---
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const Center(child: LoadingIndicator());
            }

            // --- Cas 2: Erreur lors de la récupération ---
            if (snapshot.hasError) {
              return _buildErrorState(snapshot.error);
            }

            // --- Cas 3: Données récupérées avec succès ---
            if (!snapshot.hasData || snapshot.data!.isEmpty) {
              // Afficher un état vide si la liste est vide
              return _buildEmptyState();
            }

            // Afficher la liste des événements
            final historyEvents = snapshot.data!;
            return ListView.separated(
              padding: const EdgeInsets.symmetric(vertical: 8.0),
              itemCount: historyEvents.length,
              itemBuilder: (context, index) {
                final event = historyEvents[index];
                // Afficher un en-tête de date si le jour change
                 bool showDateHeader = index == 0 ||
                     !_isSameDay(historyEvents[index-1].timestamp, event.timestamp);

                return Column(
                   children: [
                      if (showDateHeader) _buildDateHeader(context, event.timestamp),
                      _buildHistoryListItem(event),
                   ],
                );
              },
              separatorBuilder: (context, index) => const Divider(height: 0, indent: 72, thickness: 0.5), // Séparateur fin
            );
          },
        ),
      ),
    );
  }

   /// Vérifie si deux DateTime correspondent au même jour (ignore l'heure).
   bool _isSameDay(DateTime dt1, DateTime dt2) {
      return dt1.year == dt2.year && dt1.month == dt2.month && dt1.day == dt2.day;
   }

   /// Construit un en-tête de date pour séparer les jours dans la liste.
   Widget _buildDateHeader(BuildContext context, DateTime timestamp) {
      final now = DateTime.now();
      String dateText;
      if (_isSameDay(timestamp, now)) {
         dateText = 'Aujourd\'hui';
      } else if (_isSameDay(timestamp, now.subtract(const Duration(days: 1)))) {
         dateText = 'Hier';
      } else {
         dateText = DateFormat('EEEE d MMMM yyyy', 'fr_FR').format(timestamp); // Format complet
      }

      return Container(
         padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
         // color: Theme.of(context).colorScheme.surfaceVariant.withOpacity(0.5), // Fond léger
         alignment: Alignment.centerLeft,
         child: Text(
            dateText,
            style: Theme.of(context).textTheme.titleSmall?.copyWith(
               fontWeight: FontWeight.bold,
               color: Theme.of(context).colorScheme.primary,
            ),
         ),
      );
   }

  /// Construit le widget pour afficher un seul événement de l'historique.
  Widget _buildHistoryListItem(HistoryEvent event) {
    final eventColor = event.getEventColor(context);
    final timeFormat = DateFormat('HH:mm:ss'); // Format heure précise
    final theme = Theme.of(context);

    return ListTile(
      leading: CircleAvatar(
        backgroundColor: eventColor.withOpacity(0.15),
        child: Icon(event.eventIcon, color: eventColor, size: 20),
      ),
      title: Text(
        event.description, // Utilise la description déjà formatée
        style: const TextStyle(fontWeight: FontWeight.w500, fontSize: 14),
      ),
      subtitle: Row(
        children: [
          Text(
            timeFormat.format(event.timestamp),
            style: theme.textTheme.bodySmall,
          ),
          // Afficher position si disponible et pertinente
          if (event.latitude != null && event.longitude != null &&
              (event.eventType == 'theft_alert' || event.eventType == 'location_response'))
            Padding(
              padding: const EdgeInsets.only(left: 8.0),
              child: Icon(Icons.location_on_outlined, size: 12, color: theme.disabledColor),
            ),
          // Afficher batterie si disponible et pertinente
          if (event.batteryLevel != null &&
              (event.eventType == 'low_battery' || event.eventType == 'periodic_update')) ...[
            Padding(
              padding: const EdgeInsets.only(left: 8.0),
              child: Icon(Icons.battery_std_outlined, size: 12, color: AppColors.getBatteryColor(event.batteryLevel)),
            ),
            const SizedBox(width: 2),
            Text('${event.batteryLevel}%', style: TextStyle(fontSize: 11, color: AppColors.getBatteryColor(event.batteryLevel))),
          ]
        ],
      ),
      // Trailing peut afficher la date si pas groupé, ou être vide
      // trailing: Text(DateFormat('dd/MM', 'fr_FR').format(event.timestamp), style: theme.textTheme.bodySmall),
      dense: true, // Rendre plus compact
      onTap: () {
        // Optionnel: Afficher plus de détails ou centrer sur la carte
        if (event.latitude != null && event.longitude != null) {
          // #TODO: Naviguer vers MapPage en centrant sur cette position?
          CustomSnackBar.show(context,
              message: 'Pos: ${event.latitude?.toStringAsFixed(5)}, ${event.longitude?.toStringAsFixed(5)} @ ${timeFormat.format(event.timestamp)}',
              type: AlertType.info);
        }
      },
    );
  }

   /// Construit le widget affiché quand la liste d'historique est vide.
   Widget _buildEmptyState() {
     return Center(
       child: Padding(
         padding: const EdgeInsets.all(32.0),
         child: Column(
           mainAxisAlignment: MainAxisAlignment.center,
           children: [
             Icon(Icons.history_toggle_off_outlined, size: 80, color: Theme.of(context).disabledColor),
             const SizedBox(height: 24),
             Text(
                AppTexts.noHistoryAvailable,
                style: Theme.of(context).textTheme.headlineSmall,
                textAlign: TextAlign.center,
             ),
             const SizedBox(height: 12),
             Text(
               "Les événements de votre Tick apparaîtront ici.",
               textAlign: TextAlign.center,
               style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                 color: Theme.of(context).textTheme.bodyMedium?.color?.withOpacity(0.7),
               ),
             ),
              const SizedBox(height: 32),
              // Bouton pour rafraîchir manuellement
              ElevatedButton.icon(
                 icon: const Icon(Icons.refresh),
                 label: const Text(AppTexts.retry),
                 onPressed: _refreshHistory,
              )
           ],
         ),
       ),
     );
   }

   /// Construit le widget affiché en cas d'erreur de chargement.
   Widget _buildErrorState(Object? error) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.error_outline, color: AppTheme.errorColor, size: 60),
              const SizedBox(height: 16),
              Text(
                AppTexts.loadingHistoryError,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 8),
              Text(
                // Afficher le message d'erreur spécifique si possible
                error.toString().replaceFirst("Exception: ", ""),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                icon: const Icon(Icons.refresh),
                label: const Text(AppTexts.retry),
                onPressed: _refreshHistory,
              )
            ],
          ),
        ),
      );
   }
}
```

--- END OF MODIFIED FILE lib\screens\tick\history_page.dart ---

--- START OF MODIFIED FILE lib\screens\tick\add_tick_page.dart ---

```dart
import 'dart:async';
import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:permission_handler/permission_handler.dart' as ph; // Alias pour permission_handler

import '../../services/bluetooth_service.dart';
import '../../services/tick_service.dart';
import '../../utils/constants.dart';
import '../../utils/theme.dart';
import '../../utils/validators.dart';
import '../../widgets/alert_card.dart';
import '../../widgets/bluetooth_status_widget.dart';
import '../../widgets/loading_indicator.dart';
import '../../widgets/step_indicator.dart';
import '../../widgets/theme_toggle_button.dart';

/// Étapes du processus d'association d'un nouveau Tick.
enum AssociationStep {
  /// 1. Entrer le nom du Tick.
  naming,
  /// 2. Vérifier les permissions et l'état du Bluetooth.
  bluetoothCheck,
  /// 3. Scanner les appareils Tick à proximité.
  scanning,
  /// 4. Envoyer les informations au backend pour l'association.
  sending,
  /// 5. Association réussie.
  done,
  /// Une erreur est survenue.
  error
}

class AddTickPage extends StatefulWidget {
  const AddTickPage({Key? key}) : super(key: key);

  @override
  State<AddTickPage> createState() => _AddTickPageState();
}

class _AddTickPageState extends State<AddTickPage> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();

  AssociationStep _currentStep = AssociationStep.naming; // Étape initiale
  String? _errorMessage; // Message d'erreur à afficher
  String? _extractedTickId; // ID unique du Tick extrait du nom BLE
  bool _isProcessing = false; // Indicateur de traitement global (scan, API)
  bool _isPlatformSupported = false; // Flag pour Web/Desktop

  // Garder une référence au service pour éviter les lookups répétitifs
  late BluetoothService _bluetoothService;

  @override
  void initState() {
    super.initState();
    _isPlatformSupported = !kIsWeb && (Platform.isAndroid || Platform.isIOS || Platform.isMacOS);
    if (_isPlatformSupported) {
      _bluetoothService = Provider.of<BluetoothService>(context, listen: false);
      // On pourrait vérifier l'état initial BT ici, mais on le fera au clic sur "Suivant"
    }
  }

  /// Change l'étape actuelle du processus et met à jour l'UI.
  void _setStep(AssociationStep step) {
    if (!mounted || _currentStep == step) return;

    // Éviter de revenir en arrière depuis les états finaux (sauf pour retry depuis error)
    if ((_currentStep == AssociationStep.done || _currentStep == AssociationStep.error) &&
        step.index < _currentStep.index && step != AssociationStep.naming) {
      print("AddTickPage: Cannot go back from step: $_currentStep to $step");
      return;
    }

    setState(() {
      _currentStep = step;
      _isProcessing = false; // Réinitialiser le flag de traitement
      // Effacer le message d'erreur quand on quitte l'état d'erreur
      if (step != AssociationStep.error) _errorMessage = null;
      // Si on revient au nommage, effacer l'ID potentiellement extrait
      if (step == AssociationStep.naming) _extractedTickId = null;
    });
    print("AddTickPage: Current Step set to: $_currentStep");
  }

  /// Met l'UI en état d'erreur avec un message spécifique.
  void _setError(String message) {
    if (!mounted) return;
    print("AddTickPage: Setting Error: $message");
    setState(() {
      _currentStep = AssociationStep.error;
      _errorMessage = message;
      _isProcessing = false; // Arrêter le traitement en cas d'erreur
    });
  }

  // --- Logique Principale du Processus ---

  /// Étape 1 -> 2: Vérifie les permissions et l'état Bluetooth avant de lancer le scan.
  /// Appelé par le bouton "Rechercher le Tick" après avoir entré le nom.
  Future<void> _checkPermissionsAndStartScan() async {
    if (!mounted || _isProcessing) return;

    // Valider d'abord le nom
    if (!(_formKey.currentState?.validate() ?? false)) {
      return;
    }
    FocusScope.of(context).unfocus(); // Masquer clavier
    setState(() => _isProcessing = true); // Indiquer un traitement

    print("AddTickPage: Checking permissions and Bluetooth state...");
    // Demander au service de vérifier/demander les permissions nécessaires
    bool permissionsOk = await _bluetoothService.checkAndRequestRequiredPermissions();
    if (!mounted) return;

    if (!permissionsOk) {
      print("AddTickPage: Permissions check failed. State: ${_bluetoothService.state}, Error: ${_bluetoothService.scanError}");
      // Afficher l'étape de vérification Bluetooth qui montrera l'erreur de permission via BluetoothStatusWidget
      _setError(_bluetoothService.scanError ?? ErrorMessages.permissionDenied); // S'assurer qu'il y a un message
      _setStep(AssociationStep.bluetoothCheck); // Passer à l'étape qui montre l'erreur
      return; // Arrêter ici
    }

    print("AddTickPage: Permissions OK. Adapter State: ${_bluetoothService.state}");
    // Vérifier si le Bluetooth est allumé
    if (_bluetoothService.state != BluetoothState.on) {
      print("AddTickPage: Bluetooth is not ON.");
      _setError(_bluetoothService.scanError ?? ErrorMessages.bluetoothNotEnabled);
      _setStep(AssociationStep.bluetoothCheck); // Afficher l'étape pour l'activer
      return; // Arrêter ici
    }

    // Permissions et état OK -> Lancer le scan
    print("AddTickPage: Permissions and State OK. Proceeding to scan...");
    await _startScanAndProcess(); // Lance le scan et gère la suite

    // L'état _isProcessing est géré dans _startScanAndProcess ou _setError
     if (mounted && _currentStep != AssociationStep.scanning && _currentStep != AssociationStep.sending) {
        // Assurer que le processing s'arrête si on n'est pas dans une étape d'attente
        setState(() { _isProcessing = false; });
     }
  }

  /// Étape 2 -> 3: Lance le scan Bluetooth, extrait l'ID, et déclenche l'appel API.
  Future<void> _startScanAndProcess() async {
    if (!mounted) return;
    // On suppose que les permissions/état sont OK car vérifiés juste avant

    _setStep(AssociationStep.scanning); // Afficher UI "Recherche..."
    setState(() => _isProcessing = true); // Maintient l'état de traitement

    try {
      // Lancer le scan via le service et attendre le résultat (trouvé ou timeout/erreur)
      final bool found = await _bluetoothService.startTickScanAndExtractId();
      if (!mounted) return; // Vérifier après l'attente du scan

      if (found && _bluetoothService.extractedTickId != null) {
        // Tick trouvé et ID extrait avec succès
        _extractedTickId = _bluetoothService.extractedTickId;
        print("AddTickPage: Scan successful, Extracted ID: $_extractedTickId");
        // Passer à l'étape d'envoi API
        _setStep(AssociationStep.sending);
        await _triggerAssociationApi(); // Déclencher l'appel API
      } else {
        // Scan échoué (timeout) ou ID non extrait
        print("AddTickPage: Scan failed or ID not extracted. Error: ${_bluetoothService.scanError}");
        _setError(_bluetoothService.scanError ?? ErrorMessages.deviceNotFound);
        // L'état reviendra à 'error' via _setError
      }
    } catch (e) {
      print("AddTickPage: Exception during scan process: $e");
      _setError("Erreur inattendue pendant le scan: ${e.toString()}");
      // L'état reviendra à 'error' via _setError
    }
    // _isProcessing est géré dans _triggerAssociationApi ou _setError
  }

  /// Étape 3 -> 4: Appelle l'API backend pour associer le Tick trouvé.
  Future<void> _triggerAssociationApi() async {
    if (!mounted) return;
    // _isProcessing doit déjà être true et l'étape est 'sending'

    if (_extractedTickId == null) {
      _setError("Erreur interne: ID du Tick non trouvé après le scan.");
      _setStep(AssociationStep.scanning); // Revenir à l'étape du scan
      return;
    }

    final tickService = Provider.of<TickService>(context, listen: false);
    final tickNickname = _nameController.text.trim();

    print("AddTickPage: Triggering association API call with Name: $tickNickname, Extracted ID: $_extractedTickId");

    try {
      // Appeler le service Tick pour associer (qui appelle l'API Lambda)
      final success = await tickService.associateTick(tickNickname, _extractedTickId!);

      // Pas de déconnexion Bluetooth nécessaire car on ne s'est pas connecté.

      if (!mounted) return;

      if (success) {
        // Association réussie !
        print("AddTickPage: Association API successful!");
        _setStep(AssociationStep.done); // Passer à l'étape finale
        // Attendre un peu pour afficher le message de succès avant de fermer
        await Future.delayed(AppDurations.longDelay);
        if (mounted) {
          Navigator.pop(context); // Fermer la page d'ajout
        }
      } else {
        // L'API a renvoyé une erreur (ex: Tick déjà associé, erreur serveur...)
        print("AddTickPage: Association API failed. Error: ${tickService.error}");
        _setError(tickService.error ?? ErrorMessages.associationFailed);
      }
    } catch (e) {
      // Erreur de communication réseau ou autre exception
      print("AddTickPage: Exception during association API call: $e");
      _setError(ErrorMessages.connectionFailed); // Erreur générique
    } finally {
      // Gérer _isProcessing seulement si on n'est pas à l'étape 'done'
      if (mounted && _currentStep != AssociationStep.done) {
        setState(() { _isProcessing = false; });
      }
    }
  }

  @override
  void dispose() {
    _nameController.dispose();
    // Arrêter le scan s'il est en cours lorsque la page est quittée prématurément
    if (_currentStep == AssociationStep.scanning && _isProcessing) {
      _bluetoothService.stopScan();
    }
    super.dispose();
  }

  // --- Construction de l'UI ---

  @override
  Widget build(BuildContext context) {
    // Gérer les plateformes non supportées (Web, Desktop)
    if (!_isPlatformSupported) {
      return _buildUnsupportedPlatformWidget();
    }

    // Écouter les changements d'état Bluetooth pour potentiellement revenir en arrière
    context.watch<BluetoothService>(); // Assure la reconstruction si l'état BT change

     // Si BT se désactive pendant le processus (après l'étape de check initiale)
     if (_currentStep.index > AssociationStep.bluetoothCheck.index &&
         _currentStep != AssociationStep.done &&
         _currentStep != AssociationStep.error &&
         _bluetoothService.state != BluetoothState.on) {
       WidgetsBinding.instance.addPostFrameCallback((_) {
         // Ne pas appeler _setError si on est déjà en erreur pour éviter boucle
          if(_currentStep != AssociationStep.error && mounted) {
             _setError(ErrorMessages.bluetoothNotEnabled);
