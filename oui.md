Okay, here's the rest of the code for `lib\screens\tick\add_tick_page.dart`, completing the `build` method and the helper widgets for each step.

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
      // On vérifie l'état BT/perms au clic sur "Suivant"
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
    if (!mounted) { setState(() => _isProcessing = false); return; } // Vérif après await

    if (!permissionsOk) {
      print("AddTickPage: Permissions check failed. State: ${_bluetoothService.state}, Error: ${_bluetoothService.scanError}");
      _setError(_bluetoothService.scanError ?? ErrorMessages.permissionDenied);
      _setStep(AssociationStep.bluetoothCheck);
      return;
    }

    print("AddTickPage: Permissions OK. Adapter State: ${_bluetoothService.state}");
    if (_bluetoothService.state != BluetoothState.on) {
      print("AddTickPage: Bluetooth is not ON.");
      _setError(_bluetoothService.scanError ?? ErrorMessages.bluetoothNotEnabled);
      _setStep(AssociationStep.bluetoothCheck);
      return;
    }

    print("AddTickPage: Permissions and State OK. Proceeding to scan...");
    await _startScanAndProcess();

     if (mounted && _currentStep != AssociationStep.scanning && _currentStep != AssociationStep.sending) {
        setState(() { _isProcessing = false; });
     }
  }

  /// Étape 2 -> 3: Lance le scan Bluetooth, extrait l'ID, et déclenche l'appel API.
  Future<void> _startScanAndProcess() async {
    if (!mounted) return;

    _setStep(AssociationStep.scanning);
    setState(() => _isProcessing = true);

    try {
      final bool found = await _bluetoothService.startTickScanAndExtractId();
      if (!mounted) return;

      if (found && _bluetoothService.extractedTickId != null) {
        _extractedTickId = _bluetoothService.extractedTickId;
        print("AddTickPage: Scan successful, Extracted ID: $_extractedTickId");
        _setStep(AssociationStep.sending);
        await _triggerAssociationApi();
      } else {
        print("AddTickPage: Scan failed or ID not extracted. Error: ${_bluetoothService.scanError}");
        _setError(_bluetoothService.scanError ?? ErrorMessages.deviceNotFound);
      }
    } catch (e) {
      print("AddTickPage: Exception during scan process: $e");
      _setError("Erreur inattendue pendant le scan: ${e.toString()}");
    }
  }

  /// Étape 3 -> 4: Appelle l'API backend pour associer le Tick trouvé.
  Future<void> _triggerAssociationApi() async {
    if (!mounted) return;

    if (_extractedTickId == null) {
      _setError("Erreur interne: ID du Tick non trouvé après le scan.");
      _setStep(AssociationStep.scanning);
      return;
    }

    final tickService = Provider.of<TickService>(context, listen: false);
    final tickNickname = _nameController.text.trim();

    print("AddTickPage: Triggering association API call with Name: $tickNickname, Extracted ID: $_extractedTickId");

    try {
      final success = await tickService.associateTick(tickNickname, _extractedTickId!);
      if (!mounted) return;

      if (success) {
        print("AddTickPage: Association API successful!");
        _setStep(AssociationStep.done);
        await Future.delayed(AppDurations.longDelay);
        if (mounted) {
          Navigator.pop(context);
        }
      } else {
        print("AddTickPage: Association API failed. Error: ${tickService.error}");
        _setError(tickService.error ?? ErrorMessages.associationFailed);
      }
    } catch (e) {
      print("AddTickPage: Exception during association API call: $e");
      _setError(ErrorMessages.connectionFailed);
    } finally {
      if (mounted && _currentStep != AssociationStep.done) {
        setState(() { _isProcessing = false; });
      }
    }
  }

  @override
  void dispose() {
    _nameController.dispose();
    if (_isPlatformSupported && _currentStep == AssociationStep.scanning && _isProcessing) {
      _bluetoothService.stopScan();
    }
    super.dispose();
  }

  // --- Construction de l'UI ---

  @override
  Widget build(BuildContext context) {
    if (!_isPlatformSupported) {
      return _buildUnsupportedPlatformWidget();
    }

    // Écouter les changements d'état Bluetooth pour potentiellement revenir en arrière
    final btState = context.watch<BluetoothService>().state;

    // Si BT se désactive pendant le processus (après l'étape de check initiale)
     if (_currentStep.index >= AssociationStep.scanning.index &&
         _currentStep != AssociationStep.done &&
         _currentStep != AssociationStep.error &&
         btState != BluetoothState.on) {
       WidgetsBinding.instance.addPostFrameCallback((_) {
         // Ne pas afficher l'erreur si on est déjà en état d'erreur
          if(_currentStep != AssociationStep.error && mounted) {
             print("AddTickPage: Bluetooth turned off during association process. Returning to check step.");
             _setError(ErrorMessages.bluetoothNotEnabled);
             _setStep(AssociationStep.bluetoothCheck); // Revenir à l'étape BT
          }
       });
     }

    return Scaffold(
      appBar: AppBar(
        title: const Text(AppTexts.addTick),
        actions: const [ThemeToggleButton()],
      ),
      body: SingleChildScrollView( // Permettre le défilement si contenu dépasse
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
             // Titre et description
             Text(AppTexts.associateNewTick, style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
             const SizedBox(height: 8),
             Text(AppTexts.associationSteps, style: Theme.of(context).textTheme.bodyMedium),
             const SizedBox(height: 24),

             // Indicateur d'étapes
             StepIndicator(
                stepCount: 4, // 1:Nom, 2:Scan, 3:API, 4:Fini
                currentStep: _getCurrentStepIndex(),
                activeColor: AppTheme.primaryColor,
                inactiveColor: Theme.of(context).dividerColor,
                errorStep: _currentStep == AssociationStep.error ? _getPreviousStepIndex() : null,
                doneStep: _currentStep == AssociationStep.done ? 3 : null, // Index 3 pour la 4ème étape
             ),
             const SizedBox(height: 32),

             // Contenu spécifique à l'étape actuelle
             AnimatedSwitcher(
                duration: AppDurations.shortFade,
                child: Container(
                   // Clé unique pour forcer l'animation au changement d'étape
                   key: ValueKey<AssociationStep>(_currentStep),
                   child: _buildStepContent(),
                ),
             ),
             const SizedBox(height: 32),

             // Bouton d'action principal (Suivant, Rechercher, Annuler, Réessayer, Terminé)
             _buildActionButton(),
          ],
        ),
      ),
    );
  }

  /// Retourne l'index (0-based) de l'étape actuelle pour le StepIndicator.
  int _getCurrentStepIndex() {
     switch (_currentStep) {
       case AssociationStep.naming: return 0;
       case AssociationStep.bluetoothCheck: return 1; // Considérer check comme faisant partie de l'étape 2
       case AssociationStep.scanning: return 1; // Scan = étape 2
       case AssociationStep.sending: return 2; // Envoi API = étape 3
       case AssociationStep.done: return 3; // Terminé = étape 4
       case AssociationStep.error: return _getPreviousStepIndex(); // Se positionne sur l'étape qui a échoué
     }
  }

  /// Retourne l'index (0-based) de l'étape où l'erreur s'est probablement produite.
  int _getPreviousStepIndex() {
     // Si on a un ID extrait, l'erreur vient de l'envoi API (étape 3 -> index 2)
     if (_extractedTickId != null) return 2;
     // Sinon, l'erreur vient du scan ou du check BT (étape 2 -> index 1)
     return 1;
     // Si on est à l'étape naming et qu'il y a une erreur (peu probable sans action), on reste à 0
     // if (_currentStep == AssociationStep.naming) return 0;
  }

  /// Construit le contenu principal de l'UI en fonction de l'étape actuelle.
  Widget _buildStepContent() {
    // Utilise une clé pour l'AnimatedSwitcher
    switch (_currentStep) {
      case AssociationStep.naming: return _buildNamingStep();
      case AssociationStep.bluetoothCheck: return _buildBluetoothCheckStep();
      case AssociationStep.scanning: return _buildScanningStep();
      case AssociationStep.sending: return _buildSendingStep();
      case AssociationStep.done: return _buildDoneStep();
      case AssociationStep.error: return _buildErrorStep();
    }
  }

  /// Widget pour l'étape 1: Saisie du nom.
  Widget _buildNamingStep() {
    return Form(
      key: _formKey,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(AppTexts.step1_Naming, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          TextFormField(
            controller: _nameController,
            decoration: const InputDecoration(
              labelText: AppTexts.tickName,
              hintText: AppTexts.tickNameHint,
              prefixIcon: Icon(Icons.label_outline),
            ),
            validator: (value) => Validators.validateNotEmpty(value, "Veuillez nommer votre Tick"),
            textInputAction: TextInputAction.done,
            onFieldSubmitted: (_) => _checkPermissionsAndStartScan(), // Peut lancer la recherche directement
            enabled: !_isProcessing,
            textCapitalization: TextCapitalization.sentences,
          ),
        ],
      ),
    );
  }

  /// Widget pour l'étape 2 (intermédiaire): Vérification Bluetooth/Permissions.
  /// Affiche le statut et les erreurs potentielles via BluetoothStatusWidget.
  Widget _buildBluetoothCheckStep() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(AppTexts.step2_Scanning, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
        const SizedBox(height: 16),
        Text(AppTexts.enableBluetoothPrompt),
        const SizedBox(height: 16),
        // Le widget BluetoothStatusWidget affiche les erreurs de permission ou l'état Off
        BluetoothStatusWidget(showOnlyWhenOff: false),
      ],
    );
  }

  /// Widget pour l'étape 2 (active): Recherche du Tick en cours.
  Widget _buildScanningStep() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        const LoadingIndicator(size: 40),
        const SizedBox(height: 24),
        Text(AppTexts.searchingTick, style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        Text(
          AppTexts.activateTickPrompt, // Texte d'instruction
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.bodyMedium,
        ),
      ],
    );
  }

  /// Widget pour l'étape 3: Association en cours (appel API).
  Widget _buildSendingStep() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        const LoadingIndicator(size: 40),
        const SizedBox(height: 24),
        Text(AppTexts.associatingTick, style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        Text(
          // Afficher le nom et l'ID en cours d'association
          "Enregistrement de '${_nameController.text}' (ID: ${_extractedTickId ?? '...'})...",
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.bodyMedium,
        ),
      ],
    );
  }

  /// Widget pour l'étape 4: Association terminée avec succès.
  Widget _buildDoneStep() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Icon(Icons.check_circle_outline, color: AppTheme.successColor, size: 60),
        const SizedBox(height: 24),
        Text(AppTexts.tickAssociatedSuccess, style: Theme.of(context).textTheme.titleLarge?.copyWith(color: AppTheme.successColor)),
        const SizedBox(height: 8),
        Text(
          "'${_nameController.text}' est maintenant visible dans votre liste.",
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.bodyMedium,
        ),
      ],
    );
  }

  /// Widget pour l'état d'erreur. Affiche le message d'erreur.
  Widget _buildErrorStep() {
    return AlertCard(
      title: AppTexts.error,
      message: _errorMessage ?? ErrorMessages.unknownError,
      type: AlertType.error,
    );
  }

  /// Construit le bouton d'action principal en bas de page.
  /// Le texte et l'action dépendent de l'étape actuelle.
  Widget _buildActionButton() {
    String buttonText = AppTexts.next;
    VoidCallback? onPressedAction;
    bool isEnabled = !_isProcessing; // Désactivé par défaut si une opération est en cours

    switch (_currentStep) {
      case AssociationStep.naming:
        buttonText = AppTexts.searchTickButton;
        onPressedAction = _checkPermissionsAndStartScan;
        break;

      case AssociationStep.bluetoothCheck:
        // Le bouton "Réessayer" dépend de la cause de l'échec affichée par BluetoothStatusWidget
        final state = _bluetoothService.state;
        if (state == BluetoothState.unauthorized || (_bluetoothService.scanError?.contains("Permission") ?? false)) {
          // Si erreur de permission, proposer d'ouvrir les paramètres
          buttonText = AppTexts.openSettings;
          onPressedAction = () async => await ph.openAppSettings();
        } else if (state == BluetoothState.off) {
          // Si Bluetooth est éteint, proposer de l'activer
          buttonText = AppTexts.enableBluetoothButton;
          onPressedAction = () async {
             await _bluetoothService.attemptToEnableBluetooth();
             // L'état sera mis à jour par le listener, on n'attend pas ici
             // On pourrait forcer un retour à Naming pour relancer le check complet
             await Future.delayed(AppDurations.shortDelay); // Laisser le temps à l'état de changer
             if (mounted && _bluetoothService.state == BluetoothState.on) {
                _setStep(AssociationStep.naming); // Retourner au nommage si activation réussie
             }
          };
        } else {
          // Autre erreur (unavailable, unknown), proposer de réessayer
          buttonText = AppTexts.retry;
          onPressedAction = () => _setStep(AssociationStep.naming); // Recommencer depuis le début
        }
        break; // isEnabled est déjà !isProcessing

      case AssociationStep.scanning:
        buttonText = AppTexts.cancel;
        // Activer le bouton Annuler seulement si le scan est en cours
        isEnabled = isEnabled && _bluetoothService.isScanning;
        onPressedAction = isEnabled ? () async {
            await _bluetoothService.stopScan();
            _setStep(AssociationStep.naming); // Revenir au début après annulation
        } : null;
        break;

      case AssociationStep.sending:
        buttonText = AppTexts.associatingTick; // Texte indicatif pendant l'envoi
        onPressedAction = null; // Pas d'action possible pendant l'envoi API
        isEnabled = false; // Bouton désactivé
        break;

      case AssociationStep.done:
        buttonText = AppTexts.done;
        onPressedAction = () => Navigator.pop(context); // Fermer la page
        isEnabled = true;
        break;

      case AssociationStep.error:
        buttonText = AppTexts.retry;
        // Retourner à l'étape Naming pour relancer tout le processus
        onPressedAction = () => _setStep(AssociationStep.naming);
        isEnabled = true; // Permettre de réessayer même si isProcessing était true (erreur)
        break;
    }

    return ElevatedButton(
      onPressed: isEnabled ? onPressedAction : null,
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 16),
        minimumSize: const Size(double.infinity, 50), // Bouton pleine largeur
      ),
      // Afficher le loader sur le bouton seulement pendant le scan ou l'envoi API
      child: (_isProcessing && (_currentStep == AssociationStep.scanning || _currentStep == AssociationStep.sending))
          ? const LoadingIndicator(size: 20, color: Colors.white)
          : Text(buttonText, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
    );
  }

   /// Construit le widget affiché si la plateforme n'est pas supportée.
   Widget _buildUnsupportedPlatformWidget() {
     return Scaffold(
       appBar: AppBar(title: const Text(AppTexts.addTick)),
       body: Center(
         child: Padding(
           padding: const EdgeInsets.all(24.0),
           child: Column(
             mainAxisAlignment: MainAxisAlignment.center,
             children: [
               Icon(Icons.bluetooth_disabled, size: 60, color: Theme.of(context).disabledColor),
               const SizedBox(height: 16),
               Text(
                 'Fonctionnalité non supportée',
                 style: Theme.of(context).textTheme.headlineSmall,
                 textAlign: TextAlign.center,
               ),
               const SizedBox(height: 8),
               Text(
                 AppTexts.featureNotAvailableOnPlatform,
                 textAlign: TextAlign.center,
                 style: Theme.of(context).textTheme.bodyMedium,
               ),
               const SizedBox(height: 24),
               ElevatedButton(
                 onPressed: () => Navigator.pop(context),
                 child: const Text(AppTexts.back),
               )
             ],
           ),
         ),
       ),
     );
   }
}
```

--- END OF MODIFIED FILE lib\screens\tick\add_tick_page.dart ---

--- START OF MODIFIED FILE lib\screens\tick\tick_list_page.dart ---

```dart
import 'dart:io' show Platform; // Pour vérifier la plateforme
import 'package:flutter/foundation.dart' show kIsWeb; // Pour vérifier si Web
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/tick_model.dart';
import '../../services/auth_service.dart';
import '../../services/tick_service.dart';
import '../../utils/constants.dart';
import '../../utils/theme.dart'; // Pour AppTheme, AlertType
import '../../widgets/alert_card.dart'; // Pour CustomSnackBar
import '../../widgets/theme_toggle_button.dart';
import '../../widgets/loading_indicator.dart';
import 'map_page.dart'; // Pour la navigation vers MapPage

class TickListPage extends StatefulWidget {
  const TickListPage({Key? key}) : super(key: key);

  @override
  State<TickListPage> createState() => _TickListPageState();
}

class _TickListPageState extends State<TickListPage> {

  // La récupération initiale des ticks est gérée par TickService
  // via le listener sur AuthService (_handleAuthChange)

  /// Navigue vers la page de carte pour un Tick spécifique.
  void _navigateToMapPage(Tick tick) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => MapPage(tick: tick), // Passer l'objet Tick complet
      ),
    );
  }

  /// Navigue vers la page d'ajout de Tick.
  void _navigateToAddTickPage() {
    Navigator.pushNamed(context, Routes.addTick);
  }

  /// Navigue vers la page de profil utilisateur.
  void _navigateToProfilePage() {
    Navigator.pushNamed(context, Routes.profile);
  }

  /// Navigue vers la page des paramètres généraux.
  void _navigateToSettingsPage() {
     Navigator.pushNamed(context, Routes.settings);
  }

  /// Gère la déconnexion de l'utilisateur.
  Future<void> _logout() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text(AppTexts.logout),
        content: const Text('Êtes-vous sûr de vouloir vous déconnecter ?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text(AppTexts.cancel),
          ),
          // Bouton de confirmation en rouge pour attirer l'attention
          TextButton(
            style: TextButton.styleFrom(foregroundColor: AppTheme.errorColor),
            onPressed: () => Navigator.pop(context, true),
            child: const Text(AppTexts.logout),
          ),
        ],
      ),
    );

    if (confirm == true) {
      final authService = Provider.of<AuthService>(context, listen: false);
      await authService.logout();
      // La redirection est gérée par le SplashScreen/AuthWrapper qui écoute AuthService
      // Si la navigation n'est pas gérée ailleurs, faire ici:
      if (mounted) {
         Navigator.pushNamedAndRemoveUntil(context, Routes.welcome, (route) => false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(AppTexts.myTicks),
        actions: [
          const ThemeToggleButton(),
          // Menu pour Profil, Paramètres et Déconnexion
          PopupMenuButton<String>(
            tooltip: "Menu", // Ajouter un tooltip
            onSelected: (value) {
              if (value == 'profile') _navigateToProfilePage();
              if (value == 'settings') _navigateToSettingsPage();
              if (value == 'logout') _logout();
            },
            itemBuilder: (BuildContext context) => <PopupMenuEntry<String>>[
              const PopupMenuItem<String>(
                value: 'profile',
                child: ListTile(
                  leading: Icon(Icons.person_outline),
                  title: Text(AppTexts.profile),
                ),
              ),
               const PopupMenuItem<String>(
                 value: 'settings',
                 child: ListTile(
                    leading: Icon(Icons.settings_outlined),
                    title: Text(AppTexts.settings),
                 ),
              ),
              const PopupMenuDivider(), // Séparateur
              const PopupMenuItem<String>(
                value: 'logout',
                child: ListTile(
                  leading: Icon(Icons.logout, color: AppTheme.errorColor), // Icône rouge
                  title: Text(AppTexts.logout, style: TextStyle(color: AppTheme.errorColor)), // Texte rouge
                ),
              ),
            ],
            icon: const Icon(Icons.account_circle_outlined),
          ),
        ],
      ),
      body: Consumer<TickService>(
        builder: (context, tickService, child) {
          // --- Cas 1: Chargement initial ---
          if (tickService.isLoading && tickService.ticks.isEmpty) {
            return const Center(child: LoadingIndicator());
          }

          // --- Cas 2: Erreur de chargement initial ---
          if (tickService.error != null && tickService.ticks.isEmpty) {
            return _buildErrorState(tickService.error!, () => tickService.fetchTicks());
          }

          // --- Cas 3: Aucun Tick associé ---
          if (tickService.ticks.isEmpty) {
            return _buildEmptyState();
          }

          // --- Cas 4: Afficher la liste des Ticks ---
          // Utiliser RefreshIndicator pour permettre le pull-to-refresh
          return RefreshIndicator(
            onRefresh: () => tickService.fetchTicks(),
            child: ListView.builder(
              padding: const EdgeInsets.all(8.0), // Padding autour de la liste
              itemCount: tickService.ticks.length,
              itemBuilder: (context, index) {
                final tick = tickService.ticks[index];
                return _buildTickListItem(tick);
              },
            ),
          );
        },
      ),
      // Bouton flottant pour ajouter un Tick (uniquement sur mobile)
      floatingActionButton: (kIsWeb || !(Platform.isAndroid || Platform.isIOS))
          ? null // Pas de FAB sur Web/Desktop
          : FloatingActionButton(
              onPressed: _navigateToAddTickPage,
              tooltip: AppTexts.addTick,
              child: const Icon(Icons.add),
            ),
    );
  }

  /// Construit le widget pour afficher un élément Tick dans la liste.
  Widget _buildTickListItem(Tick tick) {
    final batteryLevel = tick.batteryLevel;
    final batteryColor = AppColors.getBatteryColor(batteryLevel);
    final statusColor = AppColors.getStatusColor(tick.status, context);
    final theme = Theme.of(context);

    return Card(
      // Utilise la CardTheme globale (margin, elevation, shape, color)
      clipBehavior: Clip.antiAlias, // Pour que l'effet InkWell reste dans les coins arrondis
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: statusColor.withOpacity(0.15), // Fond léger basé sur statut
          child: Icon(_getTickIcon(tick.status), color: statusColor, size: 24), // Icône basée sur statut
        ),
        title: Text(
          tick.name,
          style: const TextStyle(fontWeight: FontWeight.bold),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 2), // Petit espace
            // Afficher le statut textuellement
            Text(
              tick.statusDescription,
              style: TextStyle(fontSize: 12, color: statusColor, fontWeight: FontWeight.w500),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
            const SizedBox(height: 4),
            // Ligne pour dernière MàJ et Batterie
            Row(
              children: [
                Icon(Icons.access_time, size: 14, color: theme.textTheme.bodySmall?.color),
                const SizedBox(width: 4),
                // Utiliser le getter formaté pour la date/heure
                Text(tick.formattedLastUpdate, style: theme.textTheme.bodySmall),
                const Spacer(), // Pousse la batterie à droite
                // Afficher l'icône et le niveau de batterie si disponible
                if (batteryLevel != null) ...[
                  Icon(_getBatteryIcon(batteryLevel), size: 14, color: batteryColor),
                  const SizedBox(width: 4),
                  Text(
                    '$batteryLevel%',
                    style: TextStyle(fontSize: 12, color: batteryColor, fontWeight: FontWeight.w500),
                  ),
                ]
              ],
            ),
          ],
        ),
        trailing: Icon(Icons.arrow_forward_ios, size: 16, color: theme.textTheme.bodySmall?.color),
        onTap: () => _navigateToMapPage(tick), // Action au clic
      ),
    );
  }

  /// Construit le widget affiché lorsqu'aucun Tick n'est associé.
  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.explore_off_outlined, size: 80, color: Theme.of(context).disabledColor),
            const SizedBox(height: 24),
            Text(
              AppTexts.noTicksAvailable,
              style: Theme.of(context).textTheme.headlineSmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            Text(
              AppTexts.addFirstTick,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: Theme.of(context).textTheme.bodyMedium?.color?.withOpacity(0.7),
                  ),
            ),
            const SizedBox(height: 32),
             // Bouton pour ajouter (si plateforme supportée)
            if (!kIsWeb && (Platform.isAndroid || Platform.isIOS))
              ElevatedButton.icon(
                onPressed: _navigateToAddTickPage,
                icon: const Icon(Icons.add_circle_outline),
                label: const Text(AppTexts.addTick),
                style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12)),
              ),
          ],
        ),
      ),
    );
  }

  /// Construit le widget affiché en cas d'erreur de chargement.
  Widget _buildErrorState(String errorMessage, VoidCallback onRetry) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.error_outline, color: AppTheme.errorColor, size: 60),
            const SizedBox(height: 16),
            Text(
              'Erreur de chargement',
              style: Theme.of(context).textTheme.headlineSmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              errorMessage,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              icon: const Icon(Icons.refresh),
              label: const Text(AppTexts.retry),
              onPressed: onRetry, // Appeler la fonction de retry fournie
            )
          ],
        ),
      ),
    );
  }

  /// Retourne l'icône principale pour le Tick en fonction de son statut.
  IconData _getTickIcon(TickStatus status) {
     switch (status) {
       case TickStatus.theftAlert: return Icons.warning_amber_rounded;
       case TickStatus.lowBattery: return Icons.battery_alert_outlined;
       case TickStatus.moving: return Icons.directions_walk; // Ou animation?
       case TickStatus.inactive: return Icons.cloud_off_outlined;
       case TickStatus.active:
       case TickStatus.unknown:
       default: return Icons.location_pin; // Icône par défaut
     }
   }

  /// Retourne l'icône de batterie appropriée en fonction du niveau.
  IconData _getBatteryIcon(int level) {
    // Utilise les icônes outlined pour cohérence
    if (level > 95) return Icons.battery_full_outlined;
    if (level > 80) return Icons.battery_6_bar_outlined;
    if (level > 60) return Icons.battery_5_bar_outlined;
    if (level > 40) return Icons.battery_3_bar_outlined;
    if (level > 20) return Icons.battery_1_bar_outlined;
    return Icons.battery_alert_outlined; // <= 20%
  }
}
```

--- END OF MODIFIED FILE lib\screens\tick\tick_list_page.dart ---

--- START OF MODIFIED FILE lib\screens\tick\tick_settings_page.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../models/tick_model.dart';
import '../../services/tick_service.dart';
import '../../utils/constants.dart';
import '../../utils/theme.dart';
import '../../utils/validators.dart';
import '../../widgets/alert_card.dart'; // Pour CustomSnackBar
import '../../widgets/loading_indicator.dart';
// import '../../widgets/theme_toggle_button.dart'; // Pas forcément utile ici

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
  bool _isRinging = false; // Sonnerie en cours ?
  bool _isDisabling = false; // Désactivation temporaire en cours ?

  // Stocke les données actuelles du Tick (mis à jour via Provider)
  late Tick? _tick;

  @override
  void initState() {
    super.initState();
    // Récupérer les données initiales du Tick depuis le service
    _tick = Provider.of<TickService>(context, listen: false).getTickById(widget.tickId);
    // Initialiser le contrôleur de nom avec la valeur actuelle
    _nameController.text = _tick?.name ?? '';
  }

  @override
  void didChangeDependencies() {
     super.didChangeDependencies();
     // S'assurer que _tick est à jour si le Provider notifie un changement
     final updatedTick = context.watch<TickService>().getTickById(widget.tickId);
     if (updatedTick == null) {
         // Le Tick a été supprimé (désassocié?), fermer la page
         print("TickSettingsPage: Tick ${widget.tickId} no longer found. Popping.");
          WidgetsBinding.instance.addPostFrameCallback((_) {
             if (mounted) {
                // Eviter d'afficher un SnackBar si la page est en cours de fermeture
                // CustomSnackBar.showError(context, "Ce Tick n'est plus disponible.");
                Navigator.of(context).pop();
             }
          });
     } else if (_tick != updatedTick) {
          _tick = updatedTick;
          // Mettre à jour le contrôleur seulement si on n'est pas en mode édition
          if (!_isEditingName) {
             _nameController.text = _tick!.name;
          }
     }
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  // --- Actions ---

  /// Sauvegarde le nouveau nom du Tick via le service.
  Future<void> _saveName() async {
    if (!(_nameFormKey.currentState?.validate() ?? false)) {
      return; // Ne pas sauvegarder si le formulaire est invalide
    }
    FocusScope.of(context).unfocus(); // Masquer le clavier
    setState(() => _isSavingName = true);
    final newName = _nameController.text.trim();
    final tickService = Provider.of<TickService>(context, listen: false);

    // #TODO: Implémenter `updateTickSettings` dans TickService et ApiService
    // final success = await tickService.updateTickSettings(widget.tickId, name: newName);
    await Future.delayed(AppDurations.mediumDelay); // Simuler appel API
    final success = true; // Placeholder

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

   /// Active la sonnerie du Tick.
  Future<void> _ringTick() async {
    if (_isRinging) return;
    setState(() => _isRinging = true);
    final tickService = Provider.of<TickService>(context, listen: false);

    // #TODO: Implémenter `ringTick` dans TickService et ApiService
    // final success = await tickService.ringTick(widget.tickId);
    await Future.delayed(AppDurations.mediumDelay); // Simuler appel API
    final success = true; // Placeholder

     if (mounted) {
        setState(() => _isRinging = false);
        if (success) {
           CustomSnackBar.showSuccess(context, AppTexts.ringingTick);
        } else {
           CustomSnackBar.showError(context, tickService.error ?? "Erreur lors de la sonnerie.");
        }
     }
  }

   /// Désactive temporairement la surveillance du Tick.
  Future<void> _temporaryDisable() async {
    if (_isDisabling) return;

     // #TODO: Afficher un dialogue pour choisir la durée
     final Duration? selectedDuration = await showDialog<Duration>(
        context: context,
        builder: (context) => _buildDurationPickerDialog() // À implémenter
     );

     if (selectedDuration == null) return; // L'utilisateur a annulé

    setState(() => _isDisabling = true);
    final tickService = Provider.of<TickService>(context, listen: false);

    // #TODO: Implémenter `temporaryDisable` dans TickService et ApiService
    // final success = await tickService.temporaryDisable(widget.tickId, duration: selectedDuration.inSeconds);
    await Future.delayed(AppDurations.mediumDelay); // Simuler appel API
    final success = true; // Placeholder

     if (mounted) {
        setState(() => _isDisabling = false);
        if (success) {
           CustomSnackBar.showSuccess(context, "Surveillance désactivée pour ${selectedDuration.inMinutes} minutes.");
        } else {
           CustomSnackBar.showError(context, tickService.error ?? "Erreur de désactivation.");
        }
     }
  }

  /// Lance le processus de désassociation du Tick.
  Future<void> _unlinkDevice() async {
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

      // #TODO: Implémenter `unlinkTick` dans TickService et ApiService
      // final success = await tickService.unlinkTick(widget.tickId);
      await Future.delayed(AppDurations.mediumDelay); // Simuler appel API
      final success = true; // Placeholder

      // Ne pas vérifier `mounted` ici car on va pop la page de toute façon si succès
      if (success) {
        // Afficher le message de succès sur la page précédente (TickList)
        Navigator.pop(context); // Ferme la page des paramètres
        // Le SnackBar doit être affiché APRÈS la navigation retour
        WidgetsBinding.instance.addPostFrameCallback((_) {
           ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                 content: Text(AppTexts.unlinkSuccess),
                 backgroundColor: AppTheme.successColor,
              )
           );
        });
      } else {
        // Gérer l'erreur de désassociation (reste sur la page)
         if (mounted) {
            setState(() => _isUnlinking = false);
            CustomSnackBar.showError(context, tickService.error ?? "Erreur de désassociation.");
         }
      }
    }
  }

  // --- Construction de l'UI ---

  @override
  Widget build(BuildContext context) {
    // Si _tick devient null (suite à une mise à jour du Provider pendant le build),
    // on affiche un état d'erreur/chargement simple pour éviter un crash.
    if (_tick == null) {
      return Scaffold(
        appBar: AppBar(title: const Text(AppTexts.error)),
        body: const Center(child: LoadingIndicator()), // Ou un message d'erreur
      );
    }

    return Scaffold(
      appBar: AppBar(
        // Afficher le nom actuel du Tick dans l'AppBar
        title: Text("Paramètres - ${_tick!.name}"),
      ),
      // Utiliser AbsorbPointer pour désactiver toute la page pendant la désassociation
      body: AbsorbPointer(
        absorbing: _isUnlinking,
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
                _buildFeatureTile(
                   icon: Icons.music_note_outlined,
                   title: AppTexts.ring, // Action directe: Faire sonner
                   subtitle: 'Faire sonner le Tick pour le retrouver',
                   onTap: _isRinging ? null : _ringTick, // Désactiver pendant sonnerie
                   trailing: _isRinging ? const LoadingIndicator(size: 18) : null,
                ),
                _buildFeatureTile(
                  icon: Icons.notifications_active_outlined, // Modifier icône pour sonnerie
                  title: AppTexts.soundSettings,
                  subtitle: 'Configurer le type de sonnerie', // Placeholder
                  onTap: () {
                    CustomSnackBar.show(context, message: AppTexts.featureComingSoon, type: AlertType.info);
                    // #TODO: Naviguer vers une page/dialogue de configuration sonnerie
                  },
                ),
                _buildFeatureTile(
                  icon: Icons.pause_circle_outline,
                  title: AppTexts.temporaryDisable,
                  subtitle: 'Désactiver la surveillance pour une durée définie', // Placeholder
                  onTap: _isDisabling ? null : _temporaryDisable,
                   trailing: _isDisabling ? const LoadingIndicator(size: 18) : null,
                ),

                const Divider(height: 32),

                // --- Section Danger Zone ---
                _buildSectionTitle(context, AppTexts.dangerZone, color: AppTheme.errorColor),
                _buildFeatureTile(
                  icon: Icons.link_off,
                  title: AppTexts.unlinkDevice,
                  subtitle: 'Supprimer ce Tick de votre compte (irréversible)',
                  color: AppTheme.errorColor, // Couleur rouge pour le titre/icône
                  onTap: _isUnlinking ? null : _unlinkDevice, // Désactiver pendant désassociation
                  trailing: _isUnlinking
                      ? const LoadingIndicator(size: 18)
                      : Icon(Icons.delete_forever_outlined, color: AppTheme.errorColor),
                ),
              ],
            ),
            // Loader global superposé pendant la désassociation
            if (_isUnlinking)
               Container(
                 color: Colors.black.withOpacity(0.3),
                 child: const Center(child: LoadingIndicator(size: 40)),
               ),
          ],
        ),
      ),
    );
  }

  /// Construit le ListTile pour afficher/éditer le nom.
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
                     _nameController.text = _tick!.name; // Rétablir la valeur initiale
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
        subtitle: Text(_tick!.name),
        trailing: IconButton(
          icon: const Icon(Icons.edit_outlined, size: 20),
          tooltip: AppTexts.edit,
          onPressed: () => setState(() => _isEditingName = true),
        ),
        onTap: () => setState(() => _isEditingName = true), // Permet de cliquer sur toute la ligne
      );
    }
  }

  /// Construit un ListTile simple pour afficher une information (non éditable).
  Widget _buildInfoTile({required IconData icon, required String title, required String subtitle}) {
     return ListTile(
       leading: Icon(icon),
       title: Text(title),
       subtitle: Text(subtitle),
       dense: true,
     );
  }

  /// Construit un ListTile pour une fonctionnalité ou une action.
  Widget _buildFeatureTile({
     required IconData icon,
     required String title,
     required String subtitle,
     VoidCallback? onTap,
     Color? color, // Pour le titre/icône
     Widget? trailing,
  }) {
     return ListTile(
       leading: Icon(icon, color: color),
       title: Text(title, style: TextStyle(color: color)),
       subtitle: Text(subtitle),
       onTap: onTap,
       trailing: trailing ?? (onTap != null ? const Icon(Icons.arrow_forward_ios, size: 16) : null),
     );
  }

  /// Construit le titre d'une section dans la liste.
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

  /// Widget pour choisir la durée de désactivation (placeholder).
  Widget _buildDurationPickerDialog() {
      // #TODO: Implémenter un dialogue plus élaboré (ex: NumberPicker, liste de choix)
      return SimpleDialog(
         title: const Text(AppTexts.disableDuration),
         children: [
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
             SimpleDialogOption(
               onPressed: () => Navigator.pop(context), // Annuler
               child: const Text(AppTexts.cancel, style: TextStyle(color: AppTheme.errorColor)),
            ),
         ],
      );
  }
}
```

--- END OF MODIFIED FILE lib\screens\tick\tick_settings_page.dart ---

--- START OF MODIFIED FILE lib\screens\profile_page.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../services/auth_service.dart';
// import '../../services/api_service.dart'; // Pas nécessaire si on utilise Amplify Auth pour update
import '../../models/user.dart';
import '../../utils/constants.dart';
import '../../utils/validators.dart';
import '../../utils/theme.dart'; // Pour les couleurs
import '../../widgets/theme_toggle_button.dart';
import '../../widgets/loading_indicator.dart';
import '../../widgets/alert_card.dart'; // Pour CustomSnackBar

class ProfilePage extends StatefulWidget {
  const ProfilePage({Key? key}) : super(key: key);

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  // États pour l'édition du nom
  bool _isEditingName = false;
  final _nameController = TextEditingController();
  final _nameFormKey = GlobalKey<FormState>();
  bool _isSavingName = false;

  // Référence au service d'authentification
  late AuthService _authService;

  @override
  void initState() {
    super.initState();
    // Obtenir la référence au service (sans écouter ici)
    _authService = Provider.of<AuthService>(context, listen: false);
    // Initialiser le contrôleur avec le nom actuel de l'utilisateur connecté
    _nameController.text = _authService.currentUser?.displayName ?? '';
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  /// Gère la déconnexion de l'utilisateur.
  Future<void> _logout() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text(AppTexts.logout),
        content: const Text('Êtes-vous sûr de vouloir vous déconnecter ?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text(AppTexts.cancel),
          ),
          TextButton(
            style: TextButton.styleFrom(foregroundColor: AppTheme.errorColor),
            onPressed: () => Navigator.pop(context, true),
            child: const Text(AppTexts.logout),
          ),
        ],
      ),
    );

    if (confirm == true) {
      await _authService.logout();
      // La redirection est gérée par l'AuthWrapper/SplashScreen qui écoute AuthService
      // Si ce n'est pas le cas, ajouter la navigation ici:
      // if (mounted) {
      //   Navigator.pushNamedAndRemoveUntil(context, Routes.welcome, (route) => false);
      // }
    }
  }

  /// Sauvegarde le nouveau nom d'utilisateur via AuthService.
  Future<void> _saveName() async {
    if (!(_nameFormKey.currentState?.validate() ?? false)) {
      return;
    }
    FocusScope.of(context).unfocus();
    setState(() => _isSavingName = true);
    final newName = _nameController.text.trim();

    // Appeler la méthode d'update d'AuthService (qui utilise Amplify)
    bool success = await _authService.updateUserName(newName);

    if (mounted) {
      setState(() {
        _isSavingName = false;
        if (success) {
          _isEditingName = false; // Fermer le mode édition
          CustomSnackBar.showSuccess(context, "Nom mis à jour.");
          // L'utilisateur est déjà mis à jour dans AuthService et notifié
        } else {
          // Afficher l'erreur renvoyée par AuthService
          CustomSnackBar.showError(context, _authService.error ?? AppTexts.updateError);
        }
      });
    }
  }

  /// Navigue vers la page de changement de mot de passe (à implémenter).
  void _navigateToChangePassword() {
     // #TODO: Créer une page/dialogue pour le changement de mot de passe sécurisé
     // Cette page demanderait l'ancien mot de passe et le nouveau (x2).
     // Elle appellerait ensuite une méthode comme `_authService.changePassword(...)`.
     CustomSnackBar.show(context, message: AppTexts.featureComingSoon, type: AlertType.info);
     // Navigator.pushNamed(context, '/change-password'); // Exemple de route
  }

  @override
  Widget build(BuildContext context) {
    // Utiliser Consumer pour écouter les changements de l'utilisateur
    // et reconstruire l'UI si le nom change après une sauvegarde.
    return Scaffold(
      appBar: AppBar(
        title: const Text(AppTexts.profile),
        actions: const [ThemeToggleButton()],
      ),
      body: Consumer<AuthService>(
        builder: (context, authService, child) {
          final user = authService.currentUser; // Obtenir l'utilisateur actuel

          // Gérer le cas (peu probable ici) où l'utilisateur n'est plus connecté
          if (user == null) {
            return const Center(child: Text(AppTexts.notConnected));
          }

          // Mettre à jour le contrôleur si l'utilisateur change et qu'on n'est pas en édition
          if (!_isEditingName && _nameController.text != user.displayName) {
             WidgetsBinding.instance.addPostFrameCallback((_) {
                if (mounted) _nameController.text = user.displayName;
             });
          }

          return ListView(
            padding: const EdgeInsets.all(16.0),
            children: [
              // --- Section Informations Utilisateur ---
              _buildProfileHeader(context, user),
              const SizedBox(height: 32),

              // Affichage/Édition du nom
              _buildNameTile(user, authService.isLoading), // Passer isLoading pour désactiver
              const SizedBox(height: 8),

              // Affichage de l'email (non modifiable)
              ListTile(
                leading: const Icon(Icons.email_outlined),
                title: const Text(AppTexts.email),
                subtitle: Text(user.email),
                dense: true,
              ),
              const Divider(height: 32),

              // --- Section Sécurité ---
              _buildSectionTitle(context, AppTexts.security),
              ListTile(
                leading: const Icon(Icons.lock_outline),
                title: const Text('Changer le mot de passe'),
                trailing: const Icon(Icons.arrow_forward_ios, size: 16),
                onTap: _navigateToChangePassword,
              ),
              const Divider(height: 32),

              // --- Bouton Déconnexion ---
              Center(
                child: TextButton.icon(
                  icon: const Icon(Icons.logout),
                  label: const Text(AppTexts.logout),
                  style: TextButton.styleFrom(foregroundColor: AppTheme.errorColor),
                  onPressed: authService.isLoading ? null : _logout, // Désactiver si une op est en cours
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  /// Construit l'en-tête du profil avec l'avatar.
  Widget _buildProfileHeader(BuildContext context, User user) {
    final theme = Theme.of(context);
    // Utiliser les initiales ou une icône par défaut
    String initials = user.displayName.isNotEmpty
        ? user.displayName.trim().split(' ').map((part) => part.isNotEmpty ? part[0] : '').join().toUpperCase()
        : '?';
    if (initials.length > 2) initials = initials.substring(0, 2); // Limiter à 2 initiales

    return Center(
      child: CircleAvatar(
        radius: 50,
        backgroundColor: theme.colorScheme.primaryContainer,
        child: Text(
          initials,
          style: TextStyle(fontSize: 40, color: theme.colorScheme.onPrimaryContainer),
        ),
      ),
    );
  }

  /// Construit le ListTile pour afficher/éditer le nom.
  Widget _buildNameTile(User user, bool isAuthLoading) {
    if (_isEditingName) {
      return Form(
        key: _nameFormKey,
        child: ListTile(
          contentPadding: EdgeInsets.zero,
          leading: const Icon(Icons.person_outline),
          title: TextFormField(
            controller: _nameController,
            decoration: const InputDecoration(labelText: AppTexts.name, isDense: true),
            validator: (value) => Validators.validateNotEmpty(value, "Le nom ne peut pas être vide"),
            textInputAction: TextInputAction.done,
            onFieldSubmitted: (_) => _saveName(),
            autofocus: true,
            enabled: !_isSavingName && !isAuthLoading, // Désactivé si sauvegarde ou auth en cours
            textCapitalization: TextCapitalization.words,
          ),
          trailing: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              IconButton(
                icon: _isSavingName ? const LoadingIndicator(size: 18) : const Icon(Icons.check, color: AppTheme.successColor),
                tooltip: AppTexts.save,
                onPressed: (_isSavingName || isAuthLoading) ? null : _saveName,
              ),
              IconButton(
                icon: const Icon(Icons.close),
                tooltip: AppTexts.cancel,
                onPressed: (_isSavingName || isAuthLoading) ? null : () => setState(() {
                  _isEditingName = false;
                  _nameController.text = user.displayName;
                  _nameFormKey.currentState?.reset();
                }),
              ),
            ],
          ),
        ),
      );
    } else {
      return ListTile(
        leading: const Icon(Icons.person_outline),
        title: const Text(AppTexts.name),
        subtitle: Text(user.displayName.isNotEmpty ? user.displayName : 'Non défini'),
        trailing: IconButton(
          icon: const Icon(Icons.edit_outlined, size: 20),
          tooltip: AppTexts.edit,
          onPressed: isAuthLoading ? null : () => setState(() => _isEditingName = true),
        ),
        onTap: isAuthLoading ? null : () => setState(() => _isEditingName = true),
        dense: true,
      );
    }
  }

  /// Construit un titre de section.
  Widget _buildSectionTitle(BuildContext context, String title) {
    return Padding(
      padding: const EdgeInsets.only(top: 16.0, bottom: 8.0),
      child: Text(
        title.toUpperCase(),
        style: TextStyle(
          color: Theme.of(context).colorScheme.primary,
          fontWeight: FontWeight.bold,
          fontSize: 12,
          letterSpacing: 0.8,
        ),
      ),
    );
  }
}
```

--- END OF MODIFIED FILE lib\screens\profile_page.dart ---

--- START OF MODIFIED FILE lib\screens\welcome_page.dart ---

```dart
import 'package:flutter/material.dart';
import '../utils/constants.dart';
import '../widgets/theme_toggle_button.dart';

/// Page d'accueil affichée aux utilisateurs non connectés.
/// Propose la connexion ou l'inscription.
class WelcomePage extends StatelessWidget {
  const WelcomePage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context); // Accès au thème actuel

    return Scaffold(
      // appBar: AppBar( // AppBar redondante si le contenu est centré et clair
      //   // title: Text(AppTexts.welcome),
      //   backgroundColor: Colors.transparent, // Fond transparent
      //   elevation: 0, // Pas d'ombre
      //   actions: const [ThemeToggleButton()],
      // ),
      body: SafeArea( // Assure que le contenu ne déborde pas sur les zones système
        child: Container(
          width: double.infinity, // Prend toute la largeur
          padding: const EdgeInsets.all(24.0),
          child: Column(
            // crossAxisAlignment: CrossAxisAlignment.center, // Column est centré par défaut
            mainAxisAlignment: MainAxisAlignment.center, // Centre verticalement les éléments principaux
            children: <Widget>[
              const Spacer(flex: 2), // Pousse le contenu vers le centre

              // Logo avec animation Hero
              Hero(
                tag: 'logo', // Doit correspondre au tag dans SplashScreen
                child: Image.asset('assets/logo.png', height: 150), // Taille ajustée
              ),
              const SizedBox(height: 40),

              // Texte d'introduction
              Text(
                AppTexts.tagline,
                style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              Text(
                AppTexts.description,
                style: theme.textTheme.bodyLarge?.copyWith(color: theme.textTheme.bodyMedium?.color?.withOpacity(0.7)),
                textAlign: TextAlign.center,
              ),

              const Spacer(flex: 3), // Plus d'espace avant les boutons

              // Boutons d'action
              ElevatedButton(
                onPressed: () => Navigator.pushNamed(context, Routes.login),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16.0),
                  minimumSize: const Size(double.infinity, 50), // Prend toute la largeur
                ),
                child: const Text(
                  AppTexts.login,
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ),
              const SizedBox(height: 16),
              OutlinedButton(
                onPressed: () => Navigator.pushNamed(context, Routes.register),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16.0),
                  minimumSize: const Size(double.infinity, 50), // Prend toute la largeur
                  // side: BorderSide(width: 1.5, color: theme.colorScheme.primary), // Utilise la couleur du thème
                ),
                child: const Text(
                  AppTexts.register,
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ),

              // Optionnel: Bouton Continuer sans compte (si applicable)
              // if (allowGuestMode)
              //   Padding(
              //     padding: const EdgeInsets.only(top: 16.0),
              //     child: TextButton(
              //       onPressed: () {
              //         // #TODO: Gérer la navigation en mode invité
              //         Navigator.pushNamedAndRemoveUntil(context, Routes.tickList, (route) => false);
              //       },
              //       child: Text(AppTexts.continueWithoutAccount),
              //     ),
              //   ),

              const SizedBox(height: 24), // Espace en bas
            ],
          ),
        ),
      ),
    );
  }
}
```

--- END OF MODIFIED FILE lib\screens\welcome_page.dart ---

--- START OF MODIFIED FILE lib\screens\settings_page.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart'; // Pour ouvrir les liens externes

import '../../services/theme_service.dart';
import '../../utils/constants.dart';
import '../../utils/theme.dart'; // Pour AppTheme.errorColor
import '../../widgets/theme_toggle_button.dart';

/// Page pour les paramètres généraux de l'application.
class SettingsPage extends StatefulWidget {
  const SettingsPage({Key? key}) : super(key: key);

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  // État local (placeholder) pour les notifications
  // Dans une vraie app, cet état serait synchronisé avec les préférences utilisateur/serveur
  bool _notificationsEnabled = true;

  /// Ouvre une URL externe dans le navigateur par défaut.
  /// Gère les erreurs si l'URL ne peut pas être lancée.
  Future<void> _launchURL(String urlString) async {
    final Uri url = Uri.parse(urlString);
    try {
      bool launched = await launchUrl(url, mode: LaunchMode.externalApplication);
      if (!launched && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Impossible d\'ouvrir le lien: $urlString')),
        );
      }
    } catch (e) {
      print("Error launching URL $urlString: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Erreur lors de l\'ouverture du lien.')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // Écouter ThemeService pour afficher le nom du thème actuel
    final themeService = context.watch<ThemeService>();

    return Scaffold(
      appBar: AppBar(
        title: const Text(AppTexts.settings),
      ),
      body: ListView(
        padding: const EdgeInsets.symmetric(vertical: 8.0),
        children: [
          // --- Section Apparence ---
          _buildSectionTitle(context, AppTexts.appearance),
          ListTile(
            leading: const Icon(Icons.brightness_6_outlined),
            title: const Text('Thème de l\'application'),
            subtitle: Text(themeService.getThemeModeName()), // Nom du thème actuel
            // Utiliser le widget dédié pour changer le thème
            trailing: const ThemeToggleButton(),
            // Permettre de cliquer sur toute la ligne pour changer le thème
            onTap: () => themeService.toggleThemeMode(context),
          ),
          const Divider(height: 16),

          // --- Section Notifications ---
          _buildSectionTitle(context, AppTexts.notifications),
          SwitchListTile(
            secondary: const Icon(Icons.notifications_outlined), // Icône à gauche
            title: const Text('Activer les notifications'),
            subtitle: const Text('Alertes mouvement, batterie faible...'),
            value: _notificationsEnabled,
            onChanged: (bool value) {
              setState(() => _notificationsEnabled = value);
              // #TODO: Implémenter la logique réelle pour (dés)activer les notifications Push
              // Cela implique généralement:
              // 1. Obtenir le token FCM/APNS de l'appareil.
              // 2. Envoyer ce token (ou une info de désactivation) au backend.
              // 3. Stocker la préférence utilisateur localement (SharedPreferences).
              print("Notifications switched: $value");
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('Notifications ${value ? "activées" : "désactivées"} (placeholder)')),
              );
            },
            activeColor: AppTheme.accentColor, // Couleur verte pour le switch activé
          ),
          const Divider(height: 16),

          // --- Section Informations & Support ---
          _buildSectionTitle(context, AppTexts.information),
          _buildInfoTile(
            icon: Icons.help_outline,
            title: 'Aide & Support',
            onTap: () => _launchURL('https://github.com/votre-repo/issues'), // #TODO: Mettre URL réelle
          ),
          _buildInfoTile(
            icon: Icons.info_outline,
            title: 'À propos de ${AppTexts.appName}',
            onTap: () => _showAboutDialog(context),
          ),
          _buildInfoTile(
            icon: Icons.privacy_tip_outlined,
            title: 'Politique de confidentialité',
            onTap: () => _launchURL('https://example.com/privacy'), // #TODO: Mettre URL réelle
          ),
          _buildInfoTile(
            icon: Icons.description_outlined,
            title: 'Conditions d\'utilisation',
            onTap: () => _launchURL('https://example.com/terms'), // #TODO: Mettre URL réelle
          ),
        ],
      ),
    );
  }

  /// Helper pour construire un ListTile cliquable pour les liens d'info.
  Widget _buildInfoTile({required IconData icon, required String title, required VoidCallback onTap}) {
    return ListTile(
      leading: Icon(icon),
      title: Text(title),
      trailing: const Icon(Icons.arrow_forward_ios, size: 16),
      onTap: onTap,
    );
  }

  /// Helper pour créer les titres de section.
  Widget _buildSectionTitle(BuildContext context, String title) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16.0, 16.0, 16.0, 8.0), // Espacement standard
      child: Text(
        title.toUpperCase(),
        style: TextStyle(
          color: Theme.of(context).colorScheme.primary,
          fontWeight: FontWeight.bold,
          fontSize: 12,
          letterSpacing: 0.8,
        ),
      ),
    );
  }

  /// Affiche la boîte de dialogue standard "À propos".
  void _showAboutDialog(BuildContext context) {
     // #TODO: Obtenir la version dynamiquement (ex: package_info_plus)
     const String appVersion = '1.0.0+1';

     showAboutDialog(
       context: context,
       applicationName: AppTexts.appName,
       applicationVersion: appVersion,
       applicationIcon: Padding(
         padding: const EdgeInsets.all(8.0),
         // Utiliser une version plus petite du logo si disponible
         child: Image.asset('assets/logo.png', width: 40),
       ),
       // #TODO: Mettre le nom de votre équipe/entreprise
       applicationLegalese: '© ${DateTime.now().year} HEPIA - Groupe X',
       children: <Widget>[
         const SizedBox(height: 16),
         const Text('Application de suivi IoT pour vos objets de valeur.'),
         // Ajouter d'autres informations si nécessaire
       ],
     );
  }
}

```

--- END OF MODIFIED FILE lib\screens\settings_page.dart ---

--- START OF MODIFIED FILE lib\screens\splash_screen.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/auth_service.dart';
import '../utils/constants.dart'; // Pour Routes
import '../widgets/loading_indicator.dart';

/// Écran de démarrage affiché au lancement de l'application.
/// Vérifie l'état d'authentification et redirige l'utilisateur.
class SplashScreen extends StatefulWidget {
  const SplashScreen({Key? key}) : super(key: key);

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    // Lancer la vérification de l'état d'authentification après un court délai
    // pour laisser le temps aux Providers et à Amplify de s'initialiser.
    WidgetsBinding.instance.addPostFrameCallback((_) => _checkAuthenticationState());
  }

  /// Vérifie si l'utilisateur est connecté via [AuthService] et navigue.
  Future<void> _checkAuthenticationState() async {
    // Attendre que le service d'authentification soit prêt
    final authService = Provider.of<AuthService>(context, listen: false);

    // Boucle d'attente simple (alternative: utiliser un Completer dans AuthService)
    while (!authService.isInitialized) {
      print("SplashScreen: Waiting for AuthService initialization...");
      await Future.delayed(AppDurations.shortDelay); // Attendre 500ms
      if (!mounted) return; // Quitter si le widget est démonté pendant l'attente
    }

    print("SplashScreen: AuthService initialized. Auth State: ${authService.isAuthenticated}");

    // Naviguer vers la page appropriée en fonction de l'état d'authentification
    if (authService.isAuthenticated) {
      // Utilisateur connecté -> Aller à la liste des Ticks
      Navigator.pushReplacementNamed(context, Routes.tickList);
    } else {
      // Utilisateur non connecté -> Aller à la page de bienvenue/login
      Navigator.pushReplacementNamed(context, Routes.welcome);
    }
  }

  @override
  Widget build(BuildContext context) {
    // Afficher un logo et un indicateur de chargement pendant la vérification.
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Utiliser Hero pour une transition douce depuis l'icône de l'app (si configuré)
            const Hero(
              tag: 'logo',
              child: Image(image: AssetImage('assets/logo.png'), width: 150, height: 150),
            ),
            const SizedBox(height: 40),
            const LoadingIndicator(size: 30),
            const SizedBox(height: 16),
            Text(
              AppTexts.loading, // "Chargement..."
              style: Theme.of(context).textTheme.bodyLarge,
            ),
          ],
        ),
      ),
    );
  }
}
```

--- END OF MODIFIED FILE lib\screens\splash_screen.dart ---

--- START OF MODIFIED FILE lib\widgets\alert_card.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:intl/intl.dart'; // Pour le formatage de date relatif
import '../utils/theme.dart'; // Contient AlertType et AppTheme.getAlertColor
import '../utils/constants.dart'; // Pour AppTexts, AppDurations

/// Un widget [Card] stylisé pour afficher des alertes ou informations.
/// Peut être utilisé dans des listes (ex: historique) ou seul.
class AlertCard extends StatelessWidget {
  final String title;
  final String message;
  final AlertType type;
  final DateTime? time; // Pour afficher le temps écoulé
  final VoidCallback? onTap; // Action au clic sur la carte
  final VoidCallback? onDismiss; // Action si la carte est dismissible

  const AlertCard({
    Key? key,
    required this.title,
    required this.message,
    this.type = AlertType.info,
    this.time,
    this.onTap,
    this.onDismiss,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final bool isDark = theme.brightness == Brightness.dark;
    // Obtenir la couleur et l'icône basées sur le type d'alerte
    final Color color = AppTheme.getAlertColor(type, isDark: isDark);
    final IconData iconData = _getIconForType(type);

    // Contenu interne de la carte (ListTile pour une structure standard)
    final cardContent = ListTile(
      leading: Icon(iconData, color: color, size: 28), // Icône à gauche
      title: Text(
        title,
        style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
      ),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(height: 4),
          Text(message, style: theme.textTheme.bodyMedium),
          // Afficher le temps écoulé si fourni
          if (time != null)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                _formatTimeAgo(time!), // Utilise le helper de formatage
                style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurface.withOpacity(0.6),
                    ),
              ),
            ),
        ],
      ),
      onTap: onTap, // Action au clic
      // Bouton 'Fermer' si une action onDismiss est fournie
      trailing: onDismiss != null
          ? IconButton(
              icon: const Icon(Icons.close, size: 20),
              tooltip: AppTexts.close,
              onPressed: onDismiss,
              color: theme.colorScheme.onSurface.withOpacity(0.6),
              padding: EdgeInsets.zero, // Réduire padding du IconButton
              constraints: const BoxConstraints(), // Réduire contraintes taille
            )
          : null,
      contentPadding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
    );

    // La carte elle-même, utilisant la CardTheme globale
    final cardWidget = Card(
      clipBehavior: Clip.antiAlias, // Pour l'effet InkWell
      child: InkWell( // Effet visuel au clic
        onTap: onTap,
        child: cardContent,
      ),
    );

    // Si onDismiss est fourni, rendre la carte dismissible
    if (onDismiss != null) {
      return Dismissible(
        // Clé unique basée sur le titre et le temps (ou ID si disponible)
        key: Key(title + (time?.toIso8601String() ?? UniqueKey().toString())),
        // Apparence lors du glissement
        background: Container(
          alignment: Alignment.centerRight,
          padding: const EdgeInsets.only(right: 20.0),
          decoration: BoxDecoration(
            color: AppTheme.errorColor.withOpacity(0.8), // Fond rouge pour suppression
            borderRadius: BorderRadius.circular(12), // Doit correspondre à CardTheme.shape
          ),
          child: const Icon(Icons.delete_outline, color: Colors.white),
        ),
        direction: DismissDirection.endToStart, // Glisser vers la gauche pour supprimer
        onDismissed: (_) => onDismiss!(), // Appeler la callback
        child: cardWidget,
      );
    }

    // Sinon, retourner simplement la carte
    return cardWidget;
  }

  /// Retourne l'icône appropriée pour un [AlertType].
  IconData _getIconForType(AlertType type) {
    switch (type) {
      case AlertType.success: return Icons.check_circle_outline;
      case AlertType.warning: return Icons.warning_amber_rounded;
      case AlertType.error: return Icons.error_outline;
      case AlertType.info:
      default: return Icons.info_outline;
    }
  }

  /// Formatte un [DateTime] en une chaîne de caractères relative ("il y a 5 min", "hier", etc.).
  String _formatTimeAgo(DateTime time) {
    final now = DateTime.now();
    final difference = now.difference(time);

    // Utiliser la localisation française pour les formats
    final timeFormat = DateFormat('HH:mm', 'fr_FR');
    final dateFormat = DateFormat('dd/MM/yy', 'fr_FR');
    final dateTimeFormat = DateFormat('dd MMM à HH:mm', 'fr_FR'); // Format plus lisible
    final weekdayFormat = DateFormat('EEE', 'fr_FR'); // Jour abrégé

    if (difference.inSeconds < 60) {
      return "à l'instant";
    } else if (difference.inMinutes < 60) {
      return "il y a ${difference.inMinutes} min";
    } else if (difference.inHours < now.hour) {
      return "auj. à ${timeFormat.format(time)}";
    } else if (difference.inHours < 24 + now.hour) {
      return "hier à ${timeFormat.format(time)}";
    } else if (difference.inDays < 7) {
      return "${weekdayFormat.format(time)} à ${timeFormat.format(time)}";
    } else {
      return dateTimeFormat.format(time); // Format date et heure pour plus ancien
    }
  }
}


/// Helper pour afficher des messages SnackBar personnalisés et stylisés.
class CustomSnackBar {

  /// Affiche un SnackBar avec un style basé sur [AlertType].
  static void show(
    BuildContext context, {
    required String message,
    AlertType type = AlertType.info,
    Duration duration = AppDurations.snackbarDuration,
    SnackBarAction? action,
  }) {
    // Cacher le SnackBar précédent s'il y en a un
    ScaffoldMessenger.of(context).hideCurrentSnackBar();

    final bool isDark = Theme.of(context).brightness == Brightness.dark;
    // Obtenir la couleur de fond et l'icône basées sur le type
    final Color backgroundColor = AppTheme.getAlertColor(type, isDark: isDark);
    final IconData iconData = _getIconForType(type); // Utilise le helper interne
    // Choisir la couleur du texte pour un bon contraste
    final Color textColor = ThemeData.estimateBrightnessForColor(backgroundColor) == Brightness.dark
                           ? Colors.white
                           : Colors.black;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            Icon(iconData, color: textColor.withOpacity(0.8)), // Icône légèrement transparente
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                message,
                style: TextStyle(color: textColor),
              ),
            ),
          ],
        ),
        backgroundColor: backgroundColor,
        duration: duration,
        action: action, // Utilise SnackBarAction standard
        behavior: SnackBarBehavior.floating, // Style flottant
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)), // Coins arrondis
        margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 10), // Marges pour flottant
      ),
    );
  }

  /// Raccourci pour afficher un message d'erreur standard.
  static void showError(BuildContext context, String? errorMessage, {SnackBarAction? action}) {
    show(
      context,
      message: errorMessage ?? ErrorMessages.unknownError,
      type: AlertType.error,
      action: action,
       duration: const Duration(seconds: 6), // Erreurs affichées plus longtemps
    );
  }

  /// Raccourci pour afficher un message de succès standard.
  static void showSuccess(BuildContext context, String message, {SnackBarAction? action}) {
    show(
      context,
      message: message,
      type: AlertType.success,
      action: action,
    );
  }

  /// Retourne l'icône appropriée pour un [AlertType].
  /// (Dupliqué de AlertCard pour indépendance, pourrait être dans un utilitaire commun).
  static IconData _getIconForType(AlertType type) {
    switch (type) {
      case AlertType.success: return Icons.check_circle_outline;
      case AlertType.warning: return Icons.warning_amber_rounded;
      case AlertType.error: return Icons.error_outline;
      case AlertType.info:
      default: return Icons.info_outline;
    }
  }
}
```

--- END OF MODIFIED FILE lib\widgets\alert_card.dart ---

--- START OF MODIFIED FILE lib\widgets\action_button.dart ---

```dart
import 'package:flutter/material.dart';
import 'loading_indicator.dart'; // Assurez-vous que LoadingIndicator est importé

/// Un bouton d'action vertical simple avec une icône et un label.
/// Utilise InkWell pour un effet visuel léger au clic.
class ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onPressed; // Rendre optionnel pour gérer l'état désactivé
  final Color? color; // Couleur pour l'icône et potentiellement le texte
  final bool isLoading; // Affiche un indicateur de chargement à la place de l'icône
  final bool isDisabled; // Désactive le bouton visuellement et fonctionnellement
  final double size; // Taille de l'icône

  const ActionButton({
    Key? key,
    required this.icon,
    required this.label,
    this.onPressed, // Null si désactivé
    this.color,
    this.isLoading = false,
    this.isDisabled = false,
    this.size = 28.0, // Taille par défaut
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    // Déterminer si le bouton est effectivement inactif
    final bool isInactive = isDisabled || isLoading || onPressed == null;

    // Couleur effective pour l'icône et le texte
    final Color effectiveColor = isInactive
        ? theme.disabledColor // Couleur désactivée standard
        // Utilise la couleur fournie ou la couleur primaire du thème
        : color ?? theme.colorScheme.primary;

    return InkWell(
      // Désactiver l'effet InkWell si inactif
      onTap: isInactive ? null : onPressed,
      borderRadius: BorderRadius.circular(8), // Rayon pour l'effet d'encre
      // Ajouter un peu de transparence si désactivé
      child: Opacity(
        opacity: isInactive ? 0.5 : 1.0,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8), // Padding autour
          child: Column(
            mainAxisSize: MainAxisSize.min,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Affiche soit l'indicateur de chargement, soit l'icône
              if (isLoading)
                SizedBox(
                  width: size,
                  height: size,
                  // Utiliser LoadingIndicator pour la cohérence
                  child: LoadingIndicator(size: size * 0.8, color: effectiveColor, strokeWidth: 2.5),
                )
              else
                Icon(
                  icon,
                  size: size,
                  color: effectiveColor,
                ),
              const SizedBox(height: 6), // Espace entre icône et texte
              Text(
                label,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 12,
                  color: effectiveColor, // Utilise la même couleur que l'icône
                  fontWeight: FontWeight.w500,
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis, // Gère les textes longs
              ),
            ],
          ),
        ),
      ),
    );
  }
}


// --- Variante ActionIconButton (moins utilisée dans ce projet mais gardée pour référence) ---

/// Un bouton standard (Elevated ou Outlined) avec une icône et un label.
/// Gère les états `isLoading` et `isDisabled`.
class ActionIconButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onPressed;
  final Color? backgroundColor; // Pour ElevatedButton
  final Color? foregroundColor; // Pour texte/icône des deux types
  final Color? borderColor; // Pour OutlinedButton
  final bool isLoading;
  final bool isOutlined;
  final bool isDisabled;

  const ActionIconButton({
    Key? key,
    required this.icon,
    required this.label,
    this.onPressed,
    this.backgroundColor,
    this.foregroundColor,
    this.borderColor,
    this.isLoading = false,
    this.isOutlined = false,
    this.isDisabled = false,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final bool isInactive = isDisabled || isLoading || onPressed == null;

    // Déterminer les couleurs effectives en fonction de l'état et du type
    final Color effectiveForegroundColor = isInactive
        ? theme.disabledColor
        : foregroundColor ?? (isOutlined ? theme.colorScheme.primary : Colors.white);

    final Color effectiveBackgroundColor = isInactive
        ? theme.disabledColor.withOpacity(0.12) // Fond désactivé standard
        : backgroundColor ?? theme.colorScheme.primary;

    final Color effectiveBorderColor = isInactive
        ? theme.disabledColor.withOpacity(0.12)
        : borderColor ?? theme.colorScheme.primary;

    // Widget pour l'icône ou l'indicateur de chargement
    final iconWidget = isLoading
        ? SizedBox(
            width: 18, // Taille cohérente avec la police du bouton
            height: 18,
            child: LoadingIndicator(size: 18, color: effectiveForegroundColor, strokeWidth: 2),
          )
        : Icon(icon, size: 18, color: effectiveForegroundColor);

    // Construire le bouton Outlined ou Elevated
    if (isOutlined) {
      return OutlinedButton.icon(
        onPressed: isInactive ? null : onPressed,
        icon: iconWidget,
        label: Text(label),
        style: OutlinedButton.styleFrom(
          foregroundColor: effectiveForegroundColor,
          // Gérer la couleur de la bordure pour l'état désactivé
          side: BorderSide(color: effectiveBorderColor),
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
        ).copyWith(
          // Utiliser MaterialStateProperty pour gérer la couleur de bordure désactivée
          side: MaterialStateProperty.resolveWith<BorderSide?>(
            (Set<MaterialState> states) {
              if (states.contains(MaterialState.disabled)) {
                return BorderSide(color: theme.disabledColor.withOpacity(0.12));
              }
              return BorderSide(color: effectiveBorderColor); // Bordure normale
            },
          ),
        ),
      );
    } else {
      return ElevatedButton.icon(
        onPressed: isInactive ? null : onPressed,
        icon: iconWidget,
        label: Text(label),
        style: ElevatedButton.styleFrom(
          backgroundColor: effectiveBackgroundColor,
          foregroundColor: effectiveForegroundColor,
          disabledBackgroundColor: theme.disabledColor.withOpacity(0.12), // Fond désactivé
          disabledForegroundColor: theme.disabledColor, // Texte/icône désactivé
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
        ),
      );
    }
  }
}

```

--- END OF MODIFIED FILE lib\widgets\action_button.dart ---

--- START OF MODIFIED FILE lib\widgets\step_indicator.dart ---

```dart
import 'package:flutter/material.dart';

/// Un widget indicateur d'étapes horizontal simple.
/// Affiche des cercles connectés par des lignes, avec des états visuels pour
/// les étapes actives, complétées, inactives et en erreur.
class StepIndicator extends StatelessWidget {
  final int stepCount; // Nombre total d'étapes
  final int currentStep; // Index (0-based) de l'étape actuelle
  final Color activeColor; // Couleur pour l'étape active et les lignes complétées
  final Color inactiveColor; // Couleur pour les étapes/lignes inactives
  final Color? errorColor; // Couleur pour une étape en erreur (utilise theme.error si null)
  final int? errorStep; // Index (0-based) de l'étape en erreur, si applicable
  final int? doneStep; // Index (0-based) de la dernière étape explicitement marquée comme terminée

  const StepIndicator({
    Key? key,
    required this.stepCount,
    required this.currentStep,
    required this.activeColor,
    required this.inactiveColor,
    this.errorColor,
    this.errorStep,
    this.doneStep,
  }) : assert(stepCount > 0),
       assert(currentStep >= 0 && currentStep < stepCount),
       assert(errorStep == null || (errorStep >= 0 && errorStep < stepCount)),
       assert(doneStep == null || (doneStep >= 0 && doneStep < stepCount)),
       super(key: key);

  @override
  Widget build(BuildContext context) {
    // Génère une liste alternant cercles et lignes connectrices
    // Ex: pour 3 étapes: [Cercle0, Ligne0, Cercle1, Ligne1, Cercle2] -> 2*3 - 1 = 5 éléments
    return Row(
      mainAxisAlignment: MainAxisAlignment.center, // Centre l'indicateur
      children: List.generate(stepCount * 2 - 1, (index) {
        // Index pair: C'est un cercle représentant une étape
        if (index.isEven) {
          final stepIndex = index ~/ 2; // Index 0-based de l'étape
          return _buildStepCircle(context, stepIndex);
        }
        // Index impair: C'est une ligne connectant deux étapes
        else {
          final stepIndex = index ~/ 2; // Index de l'étape *précédant* la ligne
          return _buildConnectorLine(context, stepIndex);
        }
      }),
    );
  }

  /// Construit un cercle représentant une étape.
  Widget _buildStepCircle(BuildContext context, int index) {
    Color circleColor = inactiveColor; // Couleur par défaut
    Widget child = const SizedBox.shrink(); // Pas d'icône par défaut

    // Déterminer les états de l'étape
    final bool isError = errorStep == index;
    // Une étape est considérée comme 'done' si elle est avant ou égale à doneStep
    final bool isDone = doneStep != null && index <= doneStep!;
    // Une étape est active si c'est l'étape courante ET qu'elle n'est pas en erreur
    final bool isActive = index == currentStep && !isError;
    // Une étape est complétée si elle est avant l'étape actuelle (et pas en erreur)
    // OU si elle est explicitement marquée comme done (et pas en erreur)
    final bool isCompleted = (index < currentStep || isDone) && !isError;

    // Définir la couleur d'erreur effective
    final Color effectiveErrorColor = errorColor ?? Theme.of(context).colorScheme.error;

    // Appliquer les styles en fonction de l'état
    if (isError) {
      circleColor = effectiveErrorColor;
      // Icône X pour erreur
      child = Icon(Icons.close, color: Theme.of(context).colorScheme.onError, size: 14);
    } else if (isCompleted) {
      circleColor = activeColor;
      // Icône Coche pour étape complétée
      child = Icon(Icons.check, color: Theme.of(context).colorScheme.onPrimary, size: 14);
    } else if (isActive) {
      circleColor = activeColor;
      // L'étape active n'a pas d'icône par défaut, elle est juste pleine
      // Option: afficher le numéro de l'étape:
      // child = Text('${index + 1}', style: TextStyle(color: Theme.of(context).colorScheme.onPrimary, fontSize: 12, fontWeight: FontWeight.bold));
    }
    // Si inactive (ni error, ni completed, ni active), garde la couleur et l'enfant par défaut (vide).

    // Construction du cercle avec décoration
    return Container(
      width: 24,
      height: 24,
      decoration: BoxDecoration(
        color: circleColor,
        shape: BoxShape.circle,
        // Ajouter une bordure pour mieux délimiter, surtout pour les étapes inactives
        border: Border.all(
          color: isCompleted || isActive ? activeColor : (isError ? effectiveErrorColor : inactiveColor.withOpacity(0.5)),
          width: 1.5,
        ),
      ),
      child: Center(child: child), // Centrer l'icône/texte à l'intérieur
    );
  }

  /// Construit une ligne connectant deux étapes.
  Widget _buildConnectorLine(BuildContext context, int precedingStepIndex) {
    // La ligne est active si l'étape *précédente* est complétée (ou active)
    // ET si aucune erreur n'est survenue à ou avant cette étape précédente.
    final bool isPreviousStepCompleted =
        (precedingStepIndex < currentStep || (doneStep != null && precedingStepIndex <= doneStep!)) &&
        (errorStep == null || precedingStepIndex < errorStep!);

    // Si une erreur est survenue à l'étape précédente ou avant, la ligne reste inactive.
    final bool isAfterError = errorStep != null && precedingStepIndex >= errorStep!;

    return Expanded( // Prend l'espace disponible entre les cercles
      child: Container(
        height: 2, // Épaisseur de la ligne
        // La ligne est active si l'étape précédente est complétée, sinon inactive (sauf si après une erreur)
        color: isAfterError ? inactiveColor : (isPreviousStepCompleted ? activeColor : inactiveColor),
        margin: const EdgeInsets.symmetric(horizontal: 4), // Petit espace autour de la ligne
      ),
    );
  }
}
```

--- END OF MODIFIED FILE lib\widgets\step_indicator.dart ---

--- START OF MODIFIED FILE lib\widgets\loading_indicator.dart ---

```dart
import 'package:flutter/material.dart';

/// Un indicateur de chargement circulaire simple et centré.
class LoadingIndicator extends StatelessWidget {
  final double size; // Diamètre du cercle
  final Color? color; // Couleur de l'indicateur (utilise la couleur primaire si null)
  final double strokeWidth; // Épaisseur du trait

  const LoadingIndicator({
    Key? key,
    this.size = 24.0, // Taille par défaut raisonnable
    this.color,
    this.strokeWidth = 3.0, // Épaisseur par défaut
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Center est souvent redondant car le widget parent (ex: ElevatedButton, Center)
    // gère généralement le centrage. Mais il ne nuit pas.
    return Center(
      child: SizedBox(
        width: size,
        height: size,
        child: CircularProgressIndicator(
          strokeWidth: strokeWidth,
          // Utilise la couleur fournie, ou la couleur primaire du thème actuel
          valueColor: AlwaysStoppedAnimation<Color>(
              color ?? Theme.of(context).colorScheme.primary),
        ),
      ),
    );
  }
}
```

--- END OF MODIFIED FILE lib\widgets\loading_indicator.dart ---

--- START OF MODIFIED FILE lib\widgets\theme_toggle_button.dart ---

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/theme_service.dart';
// import '../utils/constants.dart'; // Pas besoin ici a priori

/// Un [IconButton] qui permet de basculer entre les modes de thème (Clair/Sombre/Système).
class ThemeToggleButton extends StatelessWidget {
  /// Callback optionnel exécuté après le changement de thème.
  final VoidCallback? onToggle;

  const ThemeToggleButton({
    Key? key,
    this.onToggle,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Utiliser 'watch' pour que le bouton se reconstruise si le thème change
    // (pour mettre à jour l'icône et le tooltip).
    final themeService = context.watch<ThemeService>();
    final bool isCurrentlyDark = themeService.isDarkMode(context);
    final String tooltipMessage;
    final IconData iconData;

    // Déterminer l'icône et le tooltip en fonction du mode actuel effectif
    if (isCurrentlyDark) {
      iconData = Icons.light_mode_outlined; // Icône pour passer en mode clair
      tooltipMessage = 'Passer au thème clair';
    } else {
      iconData = Icons.dark_mode_outlined; // Icône pour passer en mode sombre
      tooltipMessage = 'Passer au thème sombre';
    }
    // Alternative si on veut un cycle Light -> Dark -> System:
    // switch (themeService.themeMode) {
    //   case ThemeMode.light:
    //     iconData = Icons.dark_mode_outlined;
    //     tooltipMessage = 'Passer au thème sombre';
    //     break;
    //   case ThemeMode.dark:
    //     iconData = Icons.brightness_auto_outlined; // Icône pour système
    //     tooltipMessage = 'Utiliser le thème système';
    //     break;
    //   case ThemeMode.system:
    //   default:
    //     iconData = isCurrentlyDark ? Icons.light_mode_outlined : Icons.dark_mode_outlined;
    //     tooltipMessage = isCurrentlyDark ? 'Passer au thème clair' : 'Passer au thème sombre';
    //     break;
    // }

    return IconButton(
      icon: Icon(iconData),
      tooltip: tooltipMessage,
      onPressed: () {
        // Appeler la méthode pour basculer le thème dans le service
        themeService.toggleThemeMode(context);
        // Exécuter le callback si fourni
        onToggle?.call();
      },
    );
  }
}
```

--- END OF MODIFIED FILE lib\widgets\theme_toggle_button.dart ---

--- START OF MODIFIED FILE lib\widgets\bluetooth_status_widget.dart ---

```dart
import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart' as blue; // Alias pour FlutterBluePlus
import 'package:permission_handler/permission_handler.dart' as ph; // Alias pour permission_handler

import '../services/bluetooth_service.dart';
import '../utils/constants.dart'; // Pour ErrorMessages, AppTexts
import '../utils/theme.dart'; // Pour AppTheme, AlertType
import 'alert_card.dart'; // Pour CustomSnackBar

/// Un widget qui affiche l'état actuel du Bluetooth et des permissions associées.
/// Peut proposer des actions pour résoudre les problèmes (activer BT, ouvrir paramètres).
class BluetoothStatusWidget extends StatelessWidget {
  /// Si `true`, n'affiche rien si le Bluetooth est activé et autorisé.
  final bool showOnlyWhenOffOrUnauthorized;

  const BluetoothStatusWidget({
    Key? key,
    this.showOnlyWhenOffOrUnauthorized = false,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Vérifier si le BLE est pertinent sur la plateforme actuelle
    final bool isBleRelevant = !kIsWeb && (Platform.isAndroid || Platform.isIOS || Platform.isMacOS);
    if (!isBleRelevant) {
      return const SizedBox.shrink(); // Ne rien afficher si non pertinent
    }

    // Écouter les changements du BluetoothService
    return Consumer<BluetoothService>(
      builder: (context, bluetoothService, child) {
        final state = bluetoothService.state;
        final scanError = bluetoothService.scanError; // Lire l'erreur potentielle

        // Si on ne doit afficher que les problèmes et que tout est OK
        if (showOnlyWhenOffOrUnauthorized && state == blue.BluetoothAdapterState.on && state != blue.BluetoothAdapterState.unauthorized) {
          return const SizedBox.shrink();
        }

        // Gérer les états transitoires (activation/désactivation)
        if (state == blue.BluetoothAdapterState.turningOn || state == blue.BluetoothAdapterState.turningOff) {
          return _buildTransitionStateCard(context, state);
        }

        // Gérer les états problématiques (Off, Non autorisé, Indisponible)
        if (state == blue.BluetoothAdapterState.off ||
            state == blue.BluetoothAdapterState.unavailable ||
            state == blue.BluetoothAdapterState.unauthorized) {

          String title;
          String message;
          AlertType type;
          IconData icon;
          VoidCallback? buttonAction;
          String? buttonLabel;
          IconData? buttonIcon;

          switch (state) {
            case blue.BluetoothAdapterState.off:
              title = 'Bluetooth désactivé';
              message = AppTexts.enableBluetoothPrompt;
              type = AlertType.warning;
              icon = Icons.bluetooth_disabled;
              buttonLabel = AppTexts.enableBluetoothButton;
              buttonIcon = Icons.bluetooth_audio;
              buttonAction = () async {
                // Tenter d'activer via le service
                final success = await bluetoothService.attemptToEnableBluetooth();
                // Sur iOS, guider l'utilisateur car l'activation programmatique n'est pas possible
                if (!success && Platform.isIOS && context.mounted) {
                  CustomSnackBar.show(
                    context,
                    message: 'Veuillez activer le Bluetooth dans le Centre de contrôle ou les Réglages.',
                    type: AlertType.info,
                  );
                }
              };
              break;

            case blue.BluetoothAdapterState.unauthorized:
              title = 'Permissions requises';
              // Utiliser l'erreur spécifique du service si disponible
              message = scanError ?? 'L\'application nécessite les permissions Bluetooth et/ou Localisation pour scanner les appareils.';
              type = AlertType.error;
              icon = Icons.lock_outline; // Icône de permission
              buttonLabel = AppTexts.openSettings; // Bouton pour ouvrir les paramètres
              buttonIcon = Icons.settings;
              buttonAction = () async => await ph.openAppSettings(); // Utilise permission_handler
              break;

            case blue.BluetoothAdapterState.unavailable:
              title = 'Bluetooth non disponible';
              message = scanError ?? AppTexts.featureNotAvailableOnPlatform; // Message plus générique
              type = AlertType.error;
              icon = Icons.bluetooth_disabled;
              buttonLabel = null; // Pas d'action possible
              buttonAction = null;
              break;

            default: // Ne devrait pas être atteint
              return const SizedBox.shrink();
          }

          // Construire la carte d'alerte avec les informations déterminées
          return _buildProblemStateCard(
              context, type, icon, title, message, buttonLabel, buttonIcon, buttonAction);
        }

        // Si l'état est 'on' et qu'on n'a pas filtré, ou tout autre état non géré
        return const SizedBox.shrink(); // Ne rien afficher dans les autres cas
      },
    );
  }

  /// Construit une carte simple pour les états transitoires (turningOn/turningOff).
  Widget _buildTransitionStateCard(BuildContext context, blue.BluetoothAdapterState state) {
     return Card(
       margin: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 16.0),
       color: AppTheme.infoColor.withOpacity(0.1),
       child: Padding(
         padding: const EdgeInsets.all(16.0),
         child: Row(
           children: [
             const SizedBox(
               width: 20,
               height: 20,
               child: LoadingIndicator(size: 20, strokeWidth: 2, color: AppTheme.infoColor),
             ),
             const SizedBox(width: 16),
             Expanded(
               child: Text(
                 state == blue.BluetoothAdapterState.turningOn
                     ? 'Activation du Bluetooth...'
                     : 'Désactivation du Bluetooth...',
                 style: const TextStyle(color: AppTheme.infoColor),
               ),
             ),
           ],
         ),
       ),
     );
  }

  /// Construit une carte d'alerte pour les états problématiques (Off, Unauthorized, Unavailable).
  Widget _buildProblemStateCard(
      BuildContext context,
      AlertType type,
      IconData icon,
      String title,
      String message,
      String? buttonLabel,
      IconData? buttonIcon,
      VoidCallback? buttonAction) {
    final alertColor = AppTheme.getAlertColor(type);
    // Choisir une couleur de texte contrastante pour le bouton
    final buttonTextColor = ThemeData.estimateBrightnessForColor(alertColor) == Brightness.dark
                           ? Colors.white : Colors.black;

     return Card(
       margin: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 16.0),
       // Utiliser une couleur de fond légère basée sur le type d'alerte
       color: alertColor.withOpacity(0.1),
       child: Padding(
         padding: const EdgeInsets.all(16.0),
         child: Column(
           crossAxisAlignment: CrossAxisAlignment.start,
           children: [
             Row(
               children: [
                 Icon(icon, color: alertColor),
                 const SizedBox(width: 16),
                 Expanded(
                   child: Text(
                     title,
                     style: TextStyle(fontWeight: FontWeight.bold, color: alertColor),
                   ),
                 ),
               ],
             ),
             const SizedBox(height: 8),
             Text(message),
             // Afficher le bouton d'action si défini
             if (buttonAction != null && buttonLabel != null) ...[
               const SizedBox(height: 16),
               ElevatedButton.icon(
                 icon: buttonIcon != null ? Icon(buttonIcon, size: 18) : const SizedBox.shrink(),
                 label: Text(buttonLabel),
                 onPressed: buttonAction,
                 // Style pour correspondre à la couleur de l'alerte
                 style: ElevatedButton.styleFrom(
                   backgroundColor: alertColor,
                   foregroundColor: buttonTextColor,
                   visualDensity: VisualDensity.compact, // Bouton plus petit
                 ),
               ),
             ]
           ],
         ),
       ),
     );
  }
}
```

--- END OF MODIFIED FILE lib\widgets\bluetooth_status_widget.dart ---

--- START OF MODIFIED FILE lib\services\api_service.dart ---

```dart
import 'dart:async';
import 'dart:convert'; // Pour jsonEncode/Decode
import 'package:http/http.dart' as http;
import 'package:amplify_flutter/amplify_flutter.dart'; // Pour Auth et erreurs
import 'package:amplify_auth_cognito/amplify_auth_cognito.dart'; // Pour CognitoAuthSession
import '../utils/constants.dart'; // Pour ErrorMessages et AppDurations

/// Service responsable des interactions avec les API backend (Lambda Function URLs).
/// Gère l'ajout automatique du token d'authentification Cognito et le parsing des réponses.
class ApiService {

  /// Récupère le token d'identification (ID Token) de l'utilisateur Cognito connecté.
  /// Retourne `null` si l'utilisateur n'est pas connecté ou en cas d'erreur.
  Future<String?> getAuthorizationToken() async {
    try {
      final session = await Amplify.Auth.fetchAuthSession();
      if (session.isSignedIn) {
        // S'assurer que c'est une session Cognito pour accéder aux tokens
        final cognitoSession = session as CognitoAuthSession;
        // Utiliser l'ID Token pour l'authentification avec API Gateway (Lambda Authorizer) ou Function URLs
        final idToken = cognitoSession.userPoolTokensResult.valueOrNull?.idToken.raw;
        if (idToken == null) {
           print("ApiService: ID Token not found in Cognito session.");
        }
        return idToken;
      } else {
        print("ApiService: User not signed in, cannot get token.");
        return null;
      }
    } on AuthException catch (e) {
      // Gérer les erreurs spécifiques d'Amplify Auth
      print("ApiService: Error fetching auth session for token: ${e.message}");
      // Si l'erreur est SignedOutException, c'est normal si l'utilisateur est déconnecté
      if (e is SignedOutException) {
         return null;
      }
      // Pour d'autres erreurs, on pourrait vouloir logger différemment
      return null;
    } catch (e) {
      // Gérer les erreurs inattendues
      print("ApiService: Unexpected error fetching auth session: $e");
      return null;
    }
  }


  /// Construit les en-têtes HTTP standards pour les requêtes API.
  /// Inclut 'Content-Type' et le token d'autorisation 'Bearer' si disponible.
  /// Permet d'ajouter des en-têtes personnalisés.
  Future<Map<String, String>> _buildHeaders({Map<String, String>? customHeaders}) async {
    final token = await getAuthorizationToken();
    final Map<String, String> headers = {
      'Content-Type': 'application/json; charset=UTF-8', // Spécifier UTF-8
      // Ajouter l'en-tête Authorization seulement si un token est obtenu
      if (token != null) 'Authorization': 'Bearer $token',
    };
    if (customHeaders != null) {
      headers.addAll(customHeaders);
    }
    return headers;
  }


  /// Effectue une requête GET vers une URL de fonction Lambda spécifiée.
  ///
  /// [functionUrl]: L'URL complète de la fonction Lambda.
  /// [headers]: En-têtes HTTP personnalisés optionnels.
  /// Retourne une Map: `{'success': true, 'data': <données>}` ou `{'success': false, 'error': <message>}`.
  Future<Map<String, dynamic>> get(String functionUrl, {Map<String, String>? headers}) async {
    // Vérifier si l'URL est vide (sécurité)
    if (functionUrl.isEmpty) {
       print('ApiService GET Error: Function URL is empty.');
       return {'success': false, 'error': ErrorMessages.unknownError};
    }

    final url = Uri.parse(functionUrl);
    final requestHeaders = await _buildHeaders(customHeaders: headers);

    safePrint('API GET Request URL: $url'); // Utilise safePrint pour éviter log excessif en release
    // safePrint('API GET Headers: $requestHeaders'); // Token sensible, éviter en prod

    try {
      final response = await http.get(
        url,
        headers: requestHeaders,
      ).timeout(AppDurations.apiTimeout); // Utiliser timeout constant

      return _handleResponse(response, url.toString()); // Passer l'URL pour le log d'erreur

    } on TimeoutException catch (_) {
        print('API GET Timeout for $url');
        return {'success': false, 'error': ErrorMessages.connectionFailed};
    } on http.ClientException catch (e) { // Erreur de connexion/socket
        print('API GET ClientException for $url: $e');
        return {'success': false, 'error': ErrorMessages.connectionFailed};
    } catch (e) { // Autres erreurs (parsing URL, etc.)
      print('API GET Error for $url: $e');
      return {'success': false, 'error': ErrorMessages.unknownError};
    }
  }

  /// Effectue une requête POST vers une URL de fonction Lambda spécifiée.
  ///
  /// [functionUrl]: L'URL complète de la fonction Lambda.
  /// [body]: Le corps de la requête (sera encodé en JSON).
  /// [headers]: En-têtes HTTP personnalisés optionnels.
  /// Retourne une Map: `{'success': true, 'data': <données>}` ou `{'success': false, 'error': <message>}`.
  Future<Map<String, dynamic>> post(String functionUrl, Map<String, dynamic> body, {Map<String, String>? headers}) async {
     if (functionUrl.isEmpty) {
       print('ApiService POST Error: Function URL is empty.');
       return {'success': false, 'error': ErrorMessages.unknownError};
     }

     final url = Uri.parse(functionUrl);
     final requestHeaders = await _buildHeaders(customHeaders: headers);
     final requestBody = jsonEncode(body); // Encoder le corps en JSON

     safePrint('API POST Request URL: $url');
     // safePrint('API POST Headers: $requestHeaders');
     safePrint('API POST Body: $requestBody'); // Log du corps (attention si données sensibles)

     try {
       final response = await http.post(
         url,
         headers: requestHeaders,
         body: requestBody,
       ).timeout(AppDurations.apiTimeout);

       return _handleResponse(response, url.toString());

     } on TimeoutException catch (_) {
         print('API POST Timeout for $url');
         return {'success': false, 'error': ErrorMessages.connectionFailed};
     } on http.ClientException catch (e) {
         print('API POST ClientException for $url: $e');
         return {'success': false, 'error': ErrorMessages.connectionFailed};
     } catch (e) {
       print('API POST Error for $url: $e');
       return {'success': false, 'error': ErrorMessages.unknownError};
     }
  }

  /// Effectue une requête PUT vers une URL de fonction Lambda spécifiée.
  ///
  /// [functionUrl]: L'URL complète de la fonction Lambda.
  /// [body]: Le corps de la requête (sera encodé en JSON).
  /// [headers]: En-têtes HTTP personnalisés optionnels.
  /// Retourne une Map: `{'success': true, 'data': <données>}` ou `{'success': false, 'error': <message>}`.
  Future<Map<String, dynamic>> put(String functionUrl, Map<String, dynamic> body, {Map<String, String>? headers}) async {
    if (functionUrl.isEmpty) {
      print('ApiService PUT Error: Function URL is empty.');
      return {'success': false, 'error': ErrorMessages.unknownError};
    }

    final url = Uri.parse(functionUrl);
    final requestHeaders = await _buildHeaders(customHeaders: headers);
    final requestBody = jsonEncode(body);

    safePrint('API PUT Request URL: $url');
    // safePrint('API PUT Headers: $requestHeaders');
    safePrint('API PUT Body: $requestBody');

    try {
      final response = await http.put(
        url,
        headers: requestHeaders,
        body: requestBody,
      ).timeout(AppDurations.apiTimeout);

      return _handleResponse(response, url.toString());

    } on TimeoutException catch (_) {
        print('API PUT Timeout for $url');
        return {'success': false, 'error': ErrorMessages.connectionFailed};
    } on http.ClientException catch (e) {
        print('API PUT ClientException for $url: $e');
        return {'success': false, 'error': ErrorMessages.connectionFailed};
    } catch (e) {
      print('API PUT Error for $url: $e');
      return {'success': false, 'error': ErrorMessages.unknownError};
    }
  }

  /// Effectue une requête DELETE vers une URL de fonction Lambda spécifiée.
  /// Note: Le corps est généralement ignoré pour DELETE. Passer les ID via l'URL (query params).
  ///
  /// [functionUrl]: L'URL complète de la fonction Lambda (peut inclure des query params).
  /// [headers]: En-têtes HTTP personnalisés optionnels.
  /// Retourne une Map: `{'success': true, 'data': <données>}` ou `{'success': false, 'error': <message>}`.
  Future<Map<String, dynamic>> delete(String functionUrl, {Map<String, String>? headers}) async {
     if (functionUrl.isEmpty) {
       print('ApiService DELETE Error: Function URL is empty.');
       return {'success': false, 'error': ErrorMessages.unknownError};
     }

     final url = Uri.parse(functionUrl);
     final requestHeaders = await _buildHeaders(customHeaders: headers);

     safePrint('API DELETE Request URL: $url');
     // safePrint('API DELETE Headers: $requestHeaders');

     try {
       final response = await http.delete(
         url,
         headers: requestHeaders,
       ).timeout(AppDurations.apiTimeout);

       return _handleResponse(response, url.toString());

     } on TimeoutException catch (_) {
         print('API DELETE Timeout for $url');
         return {'success': false, 'error': ErrorMessages.connectionFailed};
     } on http.ClientException catch (e) {
         print('API DELETE ClientException for $url: $e');
         return {'success': false, 'error': ErrorMessages.connectionFailed};
     } catch (e) {
       print('API DELETE Error for $url: $e');
       return {'success': false, 'error': ErrorMessages.unknownError};
     }
  }

  /// Gère la réponse HTTP, parse le JSON et retourne une Map standardisée.
  Map<String, dynamic> _handleResponse(http.Response response, String requestUrl) {
    final statusCode = response.statusCode;
    final responseBody = response.body; // Corps de la réponse brute

    safePrint('API Response Status Code: $statusCode for $requestUrl');
    // Log du corps complet seulement en debug, peut contenir des infos sensibles
    // safePrint('API Response Body: $responseBody');

    if (statusCode >= 200 && statusCode < 300) { // Succès HTTP
      try {
        // Tenter de décoder le JSON. Gérer le cas où le corps est vide.
        final dynamic decodedBody = responseBody.isNotEmpty ? jsonDecode(responseBody) : {};

        // La Lambda devrait idéalement retourner un objet avec une clé 'data'
        // On gère le cas où 'data' manque ou si la réponse est directement les données.
        final dynamic data = (decodedBody is Map && decodedBody.containsKey('data'))
                             ? decodedBody['data']
                             : decodedBody;

        // Retourner succès avec les données extraites
        return {'success': true, 'data': data};

      } catch (e) { // Erreur de parsing JSON
        print("API Response JSON Decode Error (Status: $statusCode) for $requestUrl: $e \nBody: $responseBody");
        return {'success': false, 'error': 'Format de réponse serveur invalide.'};
      }
    } else { // Erreur HTTP (4xx, 5xx)
      String errorMessage = ErrorMessages.unknownError; // Message par défaut
      try {
        // Tenter de lire un message d'erreur depuis le corps JSON
        final decodedBody = jsonDecode(responseBody);
        if (decodedBody is Map<String, dynamic>) {
          // Chercher des clés communes pour les messages d'erreur
          errorMessage = decodedBody['error'] ?? decodedBody['message'] ?? decodedBody['detail'] ?? errorMessage;
        } else if (decodedBody is String && decodedBody.isNotEmpty) {
          errorMessage = decodedBody; // Si le corps est juste une chaîne d'erreur
        }
      } catch (e) {
        // Si le corps n'est pas du JSON valide, garder le message par défaut ou utiliser le corps brut (si court)
        print("API Response Error Body Decode Failed (Status: $statusCode) for $requestUrl: $e");
        if (responseBody.isNotEmpty && responseBody.length < 100) {
           errorMessage = responseBody; // Utiliser le corps brut si court et non JSON
        }
      }

      // Raffiner le message d'erreur basé sur le code de statut HTTP si pas déjà spécifique
      if (errorMessage == ErrorMessages.unknownError || errorMessage.isEmpty) {
        switch (statusCode) {
           case 400: errorMessage = 'Requête invalide.'; break;
           case 401: errorMessage = 'Authentification requise ou invalide.'; break; // Non autorisé (token)
           case 403: errorMessage = 'Accès refusé.'; break; // Autorisé mais interdit (permissions IAM?)
           case 404: errorMessage = 'Ressource non trouvée.'; break; // Fonction Lambda introuvable?
           case 500:
           case 502:
           case 503:
           case 504: errorMessage = ErrorMessages.connectionFailed; break; // Erreurs serveur/Lambda
           default: errorMessage = 'Erreur serveur ($statusCode)';
        }
      }
      print("API Error Response (Status $statusCode) for $requestUrl: $errorMessage");
      return {'success': false, 'error': errorMessage};
    }
  }
}
```

--- END OF MODIFIED FILE lib\services\api_service.dart ---

--- START OF MODIFIED FILE lib\services\auth_service.dart ---

```dart
import 'dart:async'; // Pour StreamSubscription
import 'package:flutter/foundation.dart'; // Pour ChangeNotifier
import 'package:amplify_flutter/amplify_flutter.dart';
import 'package:amplify_auth_cognito/amplify_auth_cognito.dart'; // Pour types spécifiques Cognito

import '../models/user.dart'; // Notre modèle User
import '../utils/constants.dart'; // Pour ErrorMessages

/// Service gérant l'authentification utilisateur via AWS Cognito avec Amplify.
class AuthService with ChangeNotifier {
  User? _currentUser;
  bool _isLoading = false;
  String? _error;
  bool _isInitialized = false;
  bool _needsConfirmation = false; // Flag pour l'étape de confirmation
  String? _pendingUsername; // Stocke l'username pour confirm/reset

  // Stream pour écouter les événements d'authentification Amplify
  StreamSubscription<AuthHubEvent>? _hubSubscription;

  // Getters publics
  User? get currentUser => _currentUser;
  bool get isLoading => _isLoading;
  bool get isAuthenticated => _currentUser != null;
  String? get error => _error;
  bool get isInitialized => _isInitialized;
  bool get needsConfirmation => _needsConfirmation;
  String? get pendingUsername => _pendingUsername;

  /// Constructeur: lance l'initialisation et écoute les événements Hub.
  AuthService() {
    print("AuthService: Initializing...");
    _listenToAuthEvents(); // Démarrer l'écoute des événements
    _initializeAuthStatus(); // Vérifier l'état initial
  }

  /// Écoute les événements d'authentification d'Amplify Hub.
  void _listenToAuthEvents() {
    _hubSubscription = Amplify.Hub.listen(HubChannel.Auth, (AuthHubEvent event) {
      safePrint('Auth Hub Event: ${event.type}'); // Utiliser safePrint
      switch (event.type) {
        case AuthHubEventType.signedIn:
          // L'utilisateur s'est connecté avec succès
          _fetchCurrentUser(setInitialized: true); // Récupérer détails utilisateur
          break;
        case AuthHubEventType.signedOut:
        case AuthHubEventType.sessionExpired:
        case AuthHubEventType.userDeleted:
          // L'utilisateur s'est déconnecté, la session a expiré ou compte supprimé
          _clearUserState(); // Effacer les données utilisateur
          _isInitialized = true; // Marquer comme initialisé même après déconnexion
          notifyListeners(); // Notifier l'UI du changement d'état
          break;
      }
    });
  }

  /// Vérifie l'état d'authentification initial au démarrage.
  Future<void> _initializeAuthStatus() async {
    // Éviter réinitialisation si déjà fait (ex: par événement Hub)
    if (_isInitialized) return;
    _setLoading(true);

    try {
      // Tenter de récupérer la session actuelle
      final session = await Amplify.Auth.fetchAuthSession();
      if (session.isSignedIn) {
        print("AuthService Initial Check: User is signed in.");
        // Si connecté, récupérer les détails de l'utilisateur
        await _fetchCurrentUser(setInitialized: false); // Ne pas marquer initialisé ici
      } else {
        print("AuthService Initial Check: No active session.");
        _clearUserState(); // Assurer que l'état est propre
      }
    } on Exception catch (e) {
      print("AuthService Initial Check Error: $e");
      _setError("Erreur lors de la vérification de la session.");
      _clearUserState();
    } finally {
      // Marquer comme initialisé à la fin de la vérification initiale
      _isInitialized = true;
      _setLoading(false);
      // notifyListeners() est appelé par _fetchCurrentUser ou _clearUserState si nécessaire
    }
  }

  /// Récupère les détails de l'utilisateur connecté et met à jour l'état.
  Future<void> _fetchCurrentUser({bool setInitialized = true}) async {
    _setLoading(true); // Indiquer le chargement pendant la récupération
    _clearError(); // Effacer les erreurs précédentes

    try {
      final cognitoUser = await Amplify.Auth.getCurrentUser();
      final attributes = await Amplify.Auth.fetchUserAttributes();

      // Construire notre objet User à partir des attributs Cognito
      String displayName = '';
      String email = '';
      for (final attribute in attributes) {
        if (attribute.userAttributeKey == CognitoUserAttributeKey.name) {
          displayName = attribute.value;
        } else if (attribute.userAttributeKey == CognitoUserAttributeKey.email) {
          email = attribute.value;
        }
        // Ajouter d'autres attributs si nécessaire
      }

      _updateCurrentUser(User(
        uid: cognitoUser.userId, // Utiliser userId comme 'sub'
        email: email,
        displayName: displayName.isNotEmpty ? displayName : 'Utilisateur', // Nom par défaut
      ));

      // Si on réussit à récupérer l'utilisateur, il est confirmé
      _needsConfirmation = false;
      _pendingUsername = null; // Plus besoin de l'username en attente

      print("AuthService: Current user details fetched: ID=${cognitoUser.userId}");

    } on AuthException catch (e) {
      print("AuthService: Error fetching current user (Amplify): ${e.message}");
      if (e is SignedOutException) {
        _clearUserState(); // L'utilisateur s'est déconnecté entre-temps
      } else {
        _setError("Impossible de récupérer les informations utilisateur.");
        _clearUserState(); // Effacer l'état si erreur critique
      }
    } catch (e) {
      print("AuthService: Error fetching current user (Other): $e");
      _setError(ErrorMessages.unknownError);
      _clearUserState();
    } finally {
      if (setInitialized) _isInitialized = true;
      _setLoading(false); // Arrêter le chargement
    }
  }

  /// Réinitialise l'état utilisateur (déconnexion locale).
  void _clearUserState() {
    if (_currentUser != null || _needsConfirmation || _pendingUsername != null) {
        _currentUser = null;
        _needsConfirmation = false;
        _pendingUsername = null;
        print("AuthService: User state cleared.");
        notifyListeners(); // Notifier seulement si quelque chose a changé
    }
  }

  // --- Méthodes d'Authentification Publiques ---

  /// Tente de connecter l'utilisateur avec email et mot de passe.
  Future<bool> login(String email, String password) async {
    _setLoading(true);
    _clearError();
    _needsConfirmation = false; // Réinitialiser le flag de confirmation

    try {
      final result = await Amplify.Auth.signIn(
        username: email,
        password: password,
      );

      if (result.isSignedIn) {
        print("AuthService: Login successful for $email.");
        // Le listener Hub appellera _fetchCurrentUser pour mettre à jour _currentUser
        _pendingUsername = null; // Effacer username en attente
        // _setLoading(false) sera appelé par _fetchCurrentUser
        return true;
      } else {
        // Gérer les étapes intermédiaires si nécessaire (MFA, etc.) - non configuré ici
        print("AuthService: Login status unexpected: ${result.nextStep.signInStep}");
        _setError("Statut de connexion inattendu.");
        _setLoading(false);
        return false;
      }
    } on AuthException catch (e) {
      print("AuthService: Login AuthException: ${e.message}");
      _handleAuthError(e, contextUsername: email); // Gérer l'erreur spécifique
      _setLoading(false);
      return false;
    } catch (e) {
      print("AuthService: Login Generic Exception: $e");
      _setError(ErrorMessages.connectionFailed);
      _setLoading(false);
      return false;
    }
  }

  /// Tente d'inscrire un nouvel utilisateur.
  Future<bool> register(String email, String password, String name) async {
    _setLoading(true);
    _clearError();
    _needsConfirmation = false;

    try {
      // Attributs à envoyer à Cognito lors de l'inscription
      final userAttributes = {
        CognitoUserAttributeKey.email: email,
        CognitoUserAttributeKey.name: name, // S'assurer que 'name' est activé dans Cognito
      };

      final result = await Amplify.Auth.signUp(
        username: email, // Utiliser l'email comme username Cognito
        password: password,
        options: SignUpOptions(userAttributes: userAttributes),
      );

      if (result.isSignUpComplete) {
        // Inscription terminée et auto-vérifiée (ou pas de vérification requise)
        print("AuthService: Sign up complete for $email (auto-verified).");
        _pendingUsername = null;
        _setLoading(false);
        return true; // Succès, mais l'utilisateur doit se connecter
      } else if (result.nextStep.signUpStep == AuthSignUpStep.confirmSignUp) {
        // Inscription nécessite confirmation par code
        print("AuthService: Sign up requires confirmation for $email. Delivery: ${result.nextStep.codeDeliveryDetails}");
        _needsConfirmation = true; // Marquer pour l'UI
        _pendingUsername = email; // Stocker pour l'étape de confirmation
        _setLoading(false);
        return true; // Succès de l'étape, attente confirmation
      } else {
        print("AuthService: Sign up status unexpected: ${result.nextStep.signUpStep}");
        _setError("Statut d'inscription inattendu.");
        _setLoading(false);
        return false;
      }

    } on AuthException catch (e) {
      print("AuthService: Register AuthException: ${e.message}");
      _handleAuthError(e);
      _setLoading(false);
      return false;
    } catch (e) {
      print("AuthService: Register Generic Exception: $e");
      _setError(ErrorMessages.connectionFailed);
      _setLoading(false);
      return false;
    }
  }

  /// Confirme l'inscription avec le code reçu.
  Future<bool> confirmSignUp(String username, String confirmationCode) async {
    _setLoading(true);
    _clearError();

    try {
      final result = await Amplify.Auth.confirmSignUp(
        username: username,
        confirmationCode: confirmationCode,
      );

      if (result.isSignUpComplete) {
        print("AuthService: Sign up confirmed successfully for $username.");
        _needsConfirmation = false; // Confirmation terminée
        _pendingUsername = null;
        _setLoading(false);
        // L'utilisateur peut maintenant se connecter via la page de Login
        return true;
      } else {
        // Ne devrait pas arriver si isSignUpComplete est le critère
        print("AuthService: Confirmation status unexpected: ${result.nextStep.signUpStep}");
        _setError("Statut de confirmation inattendu.");
        _setLoading(false);
        return false;
      }
    } on AuthException catch (e) {
      print("AuthService: Confirm SignUp AuthException: ${e.message}");
      _handleAuthError(e);
      _setLoading(false);
      return false;
    } catch (e) {
      print("AuthService: Confirm SignUp Generic Exception: $e");
      _setError(ErrorMessages.unknownError);
      _setLoading(false);
      return false;
    }
  }

  /// Renvoie le code de confirmation pour un utilisateur non confirmé.
  Future<bool> resendConfirmationCode(String username) async {
    _setLoading(true); // Pourrait utiliser un flag spécifique si besoin
    _clearError();

    try {
      await Amplify.Auth.resendSignUpCode(username: username);
      print("AuthService: Confirmation code resent successfully for $username.");
      _setLoading(false);
      return true; // Indique que la demande a été envoyée
    } on AuthException catch (e) {
      print("AuthService: Resend Code AuthException: ${e.message}");
      _handleAuthError(e);
       // Si déjà confirmé, mettre à jour l'état local
       if (e is InvalidParameterException && e.message.contains('confirmed user')) {
           _needsConfirmation = false;
           _pendingUsername = null;
       }
      _setLoading(false);
      return false;
    } catch (e) {
      print("AuthService: Resend Code Generic Exception: $e");
      _setError(ErrorMessages.unknownError);
      _setLoading(false);
      return false;
    }
  }

  /// Déconnecte l'utilisateur actuel.
  Future<void> logout() async {
    _setLoading(true);
    _clearError();
    try {
      await Amplify.Auth.signOut();
      // Le listener Hub mettra _currentUser à null et notifiera
      print("AuthService: Logout successful via Amplify.");
    } on AuthException catch (e) {
      print('AuthService: Error during Amplify logout: ${e.message}');
      _setError("Erreur lors de la déconnexion.");
      // Forcer l'état déconnecté même si l'API échoue
      _clearUserState();
    } finally {
      _setLoading(false);
    }
  }

  /// Lance le processus de réinitialisation de mot de passe pour un email donné.
  Future<bool> requestPasswordReset(String email) async {
    _setLoading(true);
    _clearError();

    try {
      final result = await Amplify.Auth.resetPassword(username: email);
      // Vérifier si l'étape suivante est bien l'attente du code
      if (result.nextStep.updateStep == AuthResetPasswordStep.confirmResetPasswordWithCode) {
        print("AuthService: Password reset code sent to: ${result.nextStep.codeDeliveryDetails?.destination}");
        _pendingUsername = email; // Stocker pour l'étape de confirmation
        _setLoading(false);
        return true; // Succès, l'UI doit passer à l'étape suivante
      } else {
        print("AuthService: Password reset status unexpected: ${result.nextStep.updateStep}");
        _setError("Statut de réinitialisation inattendu.");
        _setLoading(false);
        return false;
      }
    } on AuthException catch (e) {
      print("AuthService: Reset Password AuthException: ${e.message}");
      _handleAuthError(e);
      _setLoading(false);
      return false;
    } catch (e) {
      print("AuthService: Reset Password Generic Exception: $e");
      _setError(ErrorMessages.connectionFailed);
      _setLoading(false);
      return false;
    }
  }

  /// Confirme la réinitialisation du mot de passe avec le code et le nouveau mot de passe.
  Future<bool> confirmPasswordReset(String username, String newPassword, String confirmationCode) async {
    _setLoading(true);
    _clearError();

    try {
      await Amplify.Auth.confirmResetPassword(
        username: username,
        newPassword: newPassword,
        confirmationCode: confirmationCode,
      );
      print("AuthService: Password reset confirmed successfully for $username.");
      _pendingUsername = null; // Réinitialisation terminée
      _setLoading(false);
      // L'utilisateur peut maintenant se connecter avec le nouveau mot de passe
      return true;
    } on AuthException catch (e) {
      print("AuthService: Confirm Password Reset AuthException: ${e.message}");
      _handleAuthError(e);
      _setLoading(false);
      return false;
    } catch (e) {
      print("AuthService: Confirm Password Reset Generic Exception: $e");
      _setError(ErrorMessages.unknownError);
      _setLoading(false);
      return false;
    }
  }

  /// Met à jour l'attribut 'name' de l'utilisateur Cognito.
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
      ];
      await Amplify.Auth.updateUserAttributes(attributes: attributes);
      print("AuthService: User attribute 'name' update requested successfully.");

      // Mettre à jour immédiatement l'état local pour réactivité UI
      _updateCurrentUser(_currentUser!.copyWith(displayName: newName));
      _setLoading(false); // notifyListeners est appelé par _updateCurrentUser
      return true;

    } on AuthException catch (e) {
      print("AuthService: Update User Name AuthException: ${e.message}");
      _handleAuthError(e);
      _setLoading(false);
      return false;
    } catch (e) {
      print("AuthService: Update User Name Generic Exception: $e");
      _setError(ErrorMessages.unknownError);
      _setLoading(false);
      return false;
    }
  }

  // --- Méthodes internes de gestion d'état ---

  /// Met à jour l'état de chargement et notifie les listeners.
  void _setLoading(bool loading) {
    if (_isLoading == loading) return; // Éviter notifications inutiles
    _isLoading = loading;
    notifyListeners();
  }

  /// Met à jour le message d'erreur et notifie les listeners.
  void _setError(String? errorMessage) {
    // Nettoyer le message d'erreur Amplify pour être plus lisible
    String? finalMessage = errorMessage;
    if (errorMessage != null) {
        // Tenter d'extraire le message principal après 'message:' ou après le premier crochet ']'
        final messageIndex = errorMessage.indexOf('message:');
        final bracketIndex = errorMessage.indexOf(']');
        if (messageIndex != -1) {
           finalMessage = errorMessage.substring(messageIndex + 8).trim();
        } else if (bracketIndex != -1) {
           finalMessage = errorMessage.substring(bracketIndex + 1).trim();
        }
        // Si commence par une accolade (souvent structure JSON), prendre le message court
        if (finalMessage.startsWith('{')) finalMessage = "Erreur de communication serveur.";
    }

    // Éviter notification si l'erreur n'a pas changé
    if (_error == finalMessage) return;

    _error = finalMessage;
    notifyListeners();
  }

  /// Efface le message d'erreur actuel et notifie si nécessaire.
  void _clearError() {
    if (_error != null) {
      _error = null;
      notifyListeners();
    }
  }

  /// Met à jour l'objet `_currentUser` et notifie les listeners si l'utilisateur a changé.
  void _updateCurrentUser(User? user) {
    // Comparer l'ancien et le nouvel utilisateur pour éviter notifications inutiles
    if (_currentUser == user) return;
    _currentUser = user;
    notifyListeners();
  }

  /// Gère les exceptions Amplify Auth et définit le message d'erreur approprié.
  void _handleAuthError(AuthException e, {String? contextUsername}) {
     if (e is UserNotFoundException) {
        _setError(ErrorMessages.invalidCredentials);
     } else if (e is NotAuthorizedServiceException || e is NotAuthorizedException) {
        _setError(ErrorMessages.invalidCredentials);
     } else if (e is UserNotConfirmedException) {
        _setError(ErrorMessages.userNotConfirmed);
        _needsConfirmation = true; // Marquer pour l'UI
        _pendingUsername = contextUsername; // Stocker l'email pour la confirmation
     } else if (e is UsernameExistsException) {
        _setError(ErrorMessages.emailInUse);
     } else if (e is InvalidPasswordException) {
        _setError("Le mot de passe ne respecte pas les critères requis."); // Améliorer si possible
     } else if (e is CodeMismatchException) {
        _setError("Code de confirmation invalide.");
     } else if (e is ExpiredCodeException) {
        _setError("Le code de confirmation a expiré. Veuillez en demander un nouveau.");
     } else if (e is AliasExistsException) {
        _setError("Cet email est déjà associé à un autre compte confirmé.");
        _needsConfirmation = false; // Considérer que la confirmation n'est plus nécessaire
        _pendingUsername = null;
     } else if (e is LimitExceededException) {
         _setError("Trop de tentatives. Veuillez réessayer plus tard.");
     } else if (e is InvalidParameterException && e.message.contains('confirmed user')) {
         _setError("Cet utilisateur est déjà confirmé.");
     } else if (e is SignedOutException) {
         _setError("Vous avez été déconnecté."); // Informer l'utilisateur
         _clearUserState(); // Effacer l'état local
     }
      else {
        // Erreur générique Amplify
        _setError(e.message); // Utiliser le message brut d'Amplify
     }
  }

  /// Nettoie les ressources (listener Hub) lors de la destruction du service.
  @override
  void dispose() {
    print("AuthService: Disposing...");
    _hubSubscription?.cancel(); // Annuler l'abonnement aux événements Hub
    super.dispose();
  }
}
```

--- END OF MODIFIED FILE lib\services\auth_service.dart ---

--- START OF MODIFIED FILE lib\services\mqtt_service.dart ---

```dart
import 'dart:async';
import 'dart:convert';
import 'dart:io'; // Pour Platform specific checks si nécessaire pour certificats
import 'package:flutter/services.dart' show rootBundle; // Pour charger les certificats depuis assets
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';
import 'package:flutter/foundation.dart'; // Pour kIsWeb

// ignore_for_file: avoid_print

/// Service pour gérer la connexion et la communication MQTT avec AWS IoT Core.
/// NOTE: Ce service est une ébauche et nécessite une configuration de sécurité
///       appropriée (certificats) pour se connecter à AWS IoT Core.
///       L'application mobile n'interagit généralement PAS directement avec MQTT.
///       C'est plutôt le backend (Lambda) qui communique avec les Ticks via MQTT.
///       Ce fichier est donc probablement plus pertinent pour le firmware ESP32
///       ou pour un outil de débogage/test côté serveur/admin.
class MQTTService {
  // --- Configuration (Doit correspondre à votre endpoint AWS IoT Core) ---
  final String _broker = 'a3iuhmxzhgk88s-ats.iot.eu-central-1.amazonaws.com'; // Votre endpoint ATS
  final int _port = 8883; // Port standard MQTT sécurisé (TLS)
  // Client ID doit être unique par connexion
  final String _clientId = 'flutter_app_${DateTime.now().millisecondsSinceEpoch}';
  final int _keepAlivePeriod = 60; // Secondes

  // --- Client MQTT ---
  MqttServerClient? _client;
  bool _connected = false;
  bool _isConnecting = false; // Flag pour éviter connexions multiples

  // --- Stream pour les messages reçus (si l'app doit écouter) ---
  final StreamController<Map<String, dynamic>> _messageStreamController =
      StreamController<Map<String, dynamic>>.broadcast();
  Stream<Map<String, dynamic>> get messageStream => _messageStreamController.stream;

  // --- Statut de connexion ---
  bool get isConnected => _connected;

  // --- Initialisation et Connexion (Sécurisée) ---
  /// Tente de se connecter au broker MQTT AWS IoT Core.
  /// Nécessite une configuration de sécurité (certificats).
  Future<bool> connect() async {
    if (_connected || _isConnecting) {
      print('MQTTService: Already connected or connecting.');
      return _connected;
    }
    if (kIsWeb) {
       print('MQTTService: MQTT over TLS not directly supported on Web without specific libraries/proxies.');
       return false; // MQTT direct sur Web est complexe
    }

    _isConnecting = true;
    print('MQTTService: Attempting to connect to $_broker...');

    // Création du client MQTT
    _client = MqttServerClient.withPort(_broker, _clientId, _port);

    // Configuration de base
    _client!.logging(on: true); // Activer logs pour debug
    _client!.keepAlivePeriod = _keepAlivePeriod;
    _client!.secure = true; // Connexion TLS requise pour AWS IoT
    _client!.securityContext = SecurityContext(withTrustedRoots: true); // Utiliser CAs système

    // Configuration des callbacks
    _client!.onConnected = _onConnected;
    _client!.onDisconnected = _onDisconnected;
    _client!.onSubscribed = _onSubscribed;
    _client!.onSubscribeFail = _onSubscribeFail;
    _client!.onUnsubscribed = _onUnsubscribed;
    _client!.pongCallback = _pong; // Réponse au ping keep-alive

    // Écoute des messages publiés sur les sujets auxquels on est abonné
    _client!.updates?.listen(_onMessageReceived);

    // --- Configuration de la sécurité (TLS avec certificats) ---
    // Ceci est la partie cruciale et complexe pour AWS IoT.
    // Vous devez fournir les certificats client et la clé privée, ainsi que la CA racine AWS.
    // Ces fichiers doivent être inclus dans les assets de l'app (voir pubspec.yaml).

    try {
       // Charger la CA Racine Amazon depuis les assets
      // TODO: Vérifier le chemin et nom exacts du fichier CA
      final String caCert = await rootBundle.loadString('assets/certs/AmazonRootCA1.pem');
      // Charger le certificat de l'appareil (Thing Certificate)
      // TODO: Vérifier le chemin et nom exacts du fichier certificat appareil
      final String deviceCert = await rootBundle.loadString('assets/certs/DeviceCertificate.crt');
       // Charger la clé privée de l'appareil
      // TODO: Vérifier le chemin et nom exacts du fichier clé privée
      final String privateKey = await rootBundle.loadString('assets/certs/PrivateKey.key');

       // Configurer le contexte de sécurité du client MQTT
      // NOTE: L'implémentation exacte peut varier légèrement selon les mises à jour du package mqtt_client.
      // Il faut s'assurer que le SecurityContext est correctement configuré.
      final context = SecurityContext.defaultContext;
      context.setTrustedCertificatesBytes(utf8.encode(caCert)); // CA Racine
      context.useCertificateChainBytes(utf8.encode(deviceCert)); // Certificat appareil
      context.usePrivateKeyBytes(utf8.encode(privateKey)); // Clé privée appareil

      _client!.securityContext = context;

      // Message de connexion (Will Message optionnel non défini ici)
      final connMessage = MqttConnectMessage()
          .withClientIdentifier(_clientId)
          .startClean() // Session propre (pas de messages persistants)
          .withWillQos(MqttQos.atLeastOnce);
      _client!.connectionMessage = connMessage;

      print('MQTTService: Attempting secure connection...');
      // Tentative de connexion réelle
      await _client!.connect();

    } catch (e) {
      print('MQTTService: CONNECTION EXCEPTION: $e');
      _client?.disconnect(); // Assurer la déconnexion en cas d'erreur
      _connected = false;
      _isConnecting = false;
      return false;
    }

    _isConnecting = false; // Connexion terminée (succès ou échec géré par callbacks)
    return _connected; // Retourne l'état actuel après tentative
  }

  /// Déconnecte proprement du broker MQTT.
  void disconnect() {
    if (_client != null && _connected) {
      print('MQTTService: Disconnecting...');
      _client!.disconnect();
    } else {
       print('MQTTService: Already disconnected.');
    }
    // _onDisconnected sera appelé pour mettre à jour _connected = false
  }

  /// S'abonne à un sujet MQTT.
  Future<bool> subscribe(String topic, {MqttQos qos = MqttQos.atLeastOnce}) async {
    if (!_connected) {
      print('MQTTService: Cannot subscribe, not connected. Trying to connect...');
      bool connected = await connect();
      if (!connected) return false; // Échec de connexion
    }

    try {
      print('MQTTService: Subscribing to topic: $topic with QoS: $qos');
      _client?.subscribe(topic, qos);
      // La confirmation vient via le callback _onSubscribed
      return true; // Indique que la demande a été envoyée
    } catch (e) {
      print('MQTTService: Error subscribing to topic $topic: $e');
      return false;
    }
  }

   /// Se désabonne d'un sujet MQTT.
   void unsubscribe(String topic) {
      if (_connected) {
         print('MQTTService: Unsubscribing from topic: $topic');
         _client?.unsubscribe(topic);
         // Confirmation via _onUnsubscribed
      } else {
          print('MQTTService: Cannot unsubscribe, not connected.');
      }
   }


  /// Publie un message sur un sujet MQTT.
  Future<bool> publish(String topic, String message, {MqttQos qos = MqttQos.atLeastOnce, bool retain = false}) async {
    if (!_connected) {
      print('MQTTService: Cannot publish, not connected. Trying to connect...');
      bool connected = await connect();
      if (!connected) return false;
    }

    try {
      final builder = MqttClientPayloadBuilder();
      builder.addString(message);

      print('MQTTService: Publishing to $topic (QoS: $qos, Retain: $retain): $message');
      _client?.publishMessage(topic, qos, builder.payload!, retain: retain);
      return true; // Indique que la publication a été tentée
    } catch (e) {
      print('MQTTService: Error publishing to topic $topic: $e');
      return false;
    }
  }

  // --- Callbacks internes ---

  void _onConnected() {
    _connected = true;
    _isConnecting = false; // Assurer que le flag est reset
    print('MQTTService: *** CONNECTED to broker ***');
    // S'abonner aux sujets nécessaires une fois connecté (si applicable à l'app)
    // Exemple: subscribe('ticks/+/status');
  }

  void _onDisconnected() {
    _connected = false;
    _isConnecting = false; // Assurer que le flag est reset
    print('MQTTService: *** DISCONNECTED from broker ***');
    // Gérer la reconnexion si nécessaire (avec backoff exponentiel)
    // Attention aux boucles de reconnexion infinies si la cause persiste (ex: mauvais certs)
  }

  void _onSubscribed(String topic) {
    print('MQTTService: Subscribed successfully to topic: $topic');
  }

   void _onSubscribeFail(String topic) {
     print('MQTTService: FAILED to subscribe to topic: $topic');
     // Gérer l'échec (ex: réessayer, notifier l'utilisateur)
   }

   void _onUnsubscribed(String? topic) {
      print('MQTTService: Unsubscribed successfully from topic: $topic');
   }

  void _pong() {
    // print('MQTTService: Ping response received (pong)'); // Peut être verbeux
  }

  /// Callback appelé lorsqu'un message est reçu sur un sujet auquel on est abonné.
  void _onMessageReceived(List<MqttReceivedMessage<MqttMessage>> messages) {
     if (_messageStreamController.isClosed) return;

     for (final MqttReceivedMessage<MqttMessage> msg in messages) {
        final MqttPublishMessage payload = msg.payload as MqttPublishMessage;
        final String messagePayload = MqttPublishPayload.bytesToStringAsString(payload.payload.message);
        final String topic = msg.topic;

        print('MQTTService: Received message on topic "$topic": $messagePayload');

        try {
           // Tenter de parser le payload JSON
           final Map<String, dynamic> data = jsonDecode(messagePayload);
           final message = {
              'topic': topic,
              'payload': data,
              'timestamp': DateTime.now().millisecondsSinceEpoch, // Ajouter un timestamp de réception
           };
           // Ajouter le message au Stream pour que les autres parties de l'app puissent écouter
           _messageStreamController.add(message);
        } catch (e) {
           print('MQTTService: Failed to decode JSON message payload: $e');
           // Envoyer l'erreur ou le payload brut si nécessaire
           _messageStreamController.add({
               'topic': topic,
               'payload': messagePayload, // Envoyer le payload brut
               'error': 'JSON Decode Error',
               'timestamp': DateTime.now().millisecondsSinceEpoch,
           });
        }
     }
  }

  // --- Méthodes de Simulation (pour tests UI sans connexion réelle) ---

  /// Simule la réception d'un message MQTT pour le développement/test.
  void simulateMessageReceived(String topic, Map<String, dynamic> data) {
    if (_messageStreamController.isClosed) return;
    print('MQTTService: SIMULATING message on "$topic": $data');
    final message = {
      'topic': topic,
      'payload': data,
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    };
    _messageStreamController.add(message);
  }

  // --- Nettoyage ---

  /// Libère les ressources (client MQTT, StreamController).
  void dispose() {
    print('MQTTService: Disposing...');
    disconnect(); // Assure la déconnexion
    _messageStreamController.close(); // Ferme le StreamController
  }
}

```

--- END OF MODIFIED FILE lib\services\mqtt_service.dart ---

--- START OF MODIFIED FILE lib\services\tick_service.dart ---

```dart
import 'dart:async'; // Pour StreamSubscription si écoute MQTT directe
import 'package:flutter/foundation.dart'; // Pour ChangeNotifier

import '../models/tick_model.dart';
import 'api_service.dart'; // Service pour appeler les Lambdas
import 'auth_service.dart'; // Pour vérifier l'authentification et obtenir l'ID user
// import 'mqtt_service.dart'; // Décommenter si l'app écoute MQTT directement
import '../utils/constants.dart'; // Pour URLs et ErrorMessages

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
       final urlWithParam = Uri.parse(ApiConfig.getTickHistoryFunctionUrl).replace(
          queryParameters: {'tickId': tickId}
       ).toString();
       print("TickService: Getting history for $tickId from URL: $urlWithParam");
       // Appeler l'API
       final response = await _apiService.get(urlWithParam);
       return response; // Retourner la réponse brute (contient success/data ou success/error)
    } catch (e) {
        print("TickService: Exception getting tick history: $e");
        return {'success': false, 'error': ErrorMessages.connectionFailed};
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
      _setError(ErrorMessages.connectionFailed);
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
       // #TODO: Implémenter cet appel API réel
       // final response = await _apiService.post(ApiConfig.ringTickFunctionUrl, body);
       await Future.delayed(AppDurations.shortDelay); // Simulation
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

      // #TODO: Implémenter cet appel API réel (PUT ou POST)
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
      // L'ID pourrait être passé en query param ou dans le body selon l'API
      // Exemple avec query param:
      // final urlWithParam = Uri.parse(ApiConfig.removeTickFunctionUrl).replace(queryParameters: {'tickId': tickId}).toString();
      // print("TickService: Unlinking tick $tickId via URL: $urlWithParam");
      // final response = await _apiService.delete(urlWithParam);

      // Exemple avec body (si méthode POST utilisée pour delete):
      print("TickService: Unlinking tick $tickId via URL: ${ApiConfig.removeTickFunctionUrl}");
      final body = {'tickId': tickId};
      // #TODO: Implémenter cet appel API réel (DELETE ou POST)
      // final response = await _apiService.post(ApiConfig.removeTickFunctionUrl, body); // Ou delete
      await Future.delayed(AppDurations.shortDelay); // Simulation
      final response = {'success': true, 'data': 'Tick unlinked'}; // Placeholder

      if (response['success']) {
        print("TickService: Tick unlinked successfully: $tickId");
        // Supprimer le Tick de la liste locale immédiatement
        final removed = _ticks.removeWhere((tick) => tick.id == tickId);
        if (removed > 0) {
           notifyListeners(); // Notifier l'UI de la suppression
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


  // --- Méthodes Utilitaires ---

  /// Retourne un Tick de la liste locale par son ID, ou `null` si non trouvé.
  Tick? getTickById(String id) {
    try {
      // Utilise firstWhereOrNull de collection pour éviter l'exception
      return _ticks.firstWhere((tick) => tick.id == id);
    } catch (e) { // firstWhere lance une exception si non trouvé
      return null;
    }
    // Alternative avec package collection:
    // import 'package:collection/collection.dart';
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
      // Si le Tick n'existait pas (cas rare), l'ajouter
      print("TickService: Adding new tick locally ${updatedTick.id}");
      _ticks.add(updatedTick);
      notifyListeners();
    }
  }

  /// Vérifie si l'utilisateur est authentifié et si une opération est déjà en cours.
  /// Retourne `false
