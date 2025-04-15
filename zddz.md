Super ! Maintenant que l'authentification est en place et que vous avez réussi à faire fonctionner `fetchTicks` avec une Function URL, on peut établir un **guide générique et TRÈS détaillé** pour que vous et vos amis puissiez ajouter les autres fonctionnalités (associer un Tick, le désassocier, le faire sonner, etc.) en suivant le même modèle.

Ce guide couvre les étapes AWS et Flutter pour ajouter **UNE** nouvelle fonctionnalité via une Lambda Function URL sécurisée par IAM/Cognito.

**Objectif :** Ajouter une nouvelle action (par exemple, "Associer un Tick") à l'application.

**Architecture Cible :**

*   **Flutter App :** Un bouton/action dans l'UI appelle une méthode dans un `Service` (ex: `TickService`).
*   **`TickService` (Flutter) :** Contient la logique métier Flutter, appelle `ApiService`.
*   **`ApiService` (Flutter) :** Prépare la requête HTTP (GET, POST, etc.), récupère le token ID Cognito via `Amplify.Auth`, l'ajoute à l'en-tête `Authorization`, et appelle l'URL de la fonction Lambda spécifique.
*   **AWS Lambda Function URL :** Point d'entrée HTTP pour la fonction Lambda.
    *   **Authentification :** Configurée sur `AWS_IAM`. AWS valide le token Cognito entrant et vérifie si le rôle IAM de l'utilisateur a la permission d'appeler cette URL.
    *   **CORS :** Configuré pour autoriser les requêtes depuis l'application.
*   **AWS Lambda Function :** Contient le code backend (Python, Node.js...) pour l'action spécifique.
    *   **Permissions (Rôle d'Exécution Lambda) :** Doit avoir les permissions IAM pour accéder aux ressources nécessaires (ex: écrire dans DynamoDB `Ticks`).
    *   **Logique :** Extrait l'ID utilisateur (`sub`) du contexte (fourni par l'auth IAM), effectue l'action (ex: ajoute un item dans DynamoDB), retourne une réponse JSON.
*   **AWS DynamoDB :** Base de données (ex: table `Ticks`).
*   **AWS IAM :**
    *   **Rôle d'Exécution Lambda :** Permissions pour la Lambda elle-même (accès DynamoDB, etc.).
    *   **Rôle Authentifié Cognito Identity Pool :** Permission `lambda:InvokeFunctionUrl` pour autoriser les utilisateurs connectés à appeler la Function URL. **(NOTE : Comme vous avez mis `Resource: "*"`, cette partie est *techniquement* déjà couverte pour toutes les Function URLs IAM, mais le guide montre la bonne pratique de spécifier la ressource).**

---

**Guide Étape par Étape pour Ajouter une Nouvelle Fonctionnalité (Exemple : Associer un Tick)**

**(Adaptez les noms et les actions selon la fonctionnalité que vous ajoutez)**

**Partie 1 : Configuration AWS**

1.  **Créer la Fonction Lambda :**
    *   Allez dans la console AWS -> Lambda -> "Créer une fonction".
    *   Choisissez "Author from scratch" (Partir de zéro).
    *   **Nom de la fonction :** Choisissez un nom descriptif, ex: `associateTickFunction`.
    *   **Runtime :** Sélectionnez votre langage préféré (ex: Python 3.11 ou 3.12).
    *   **Architecture :** Laissez `x86_64` par défaut (ou `arm64`).
    *   **Permissions (Rôle d'Exécution) :**
        *   Développez "Change default execution role".
        *   Choisissez "Create a new role with basic Lambda permissions" (Recommandé pour commencer). Lambda créera un rôle avec les permissions pour écrire dans CloudWatch Logs.
        *   *Note :* Si vous avez déjà un rôle avec les permissions de base, vous pouvez choisir "Use an existing role".
    *   Cliquez sur "Créer une fonction".

2.  **Configurer les Permissions du Rôle d'Exécution Lambda :**
    *   *Une fois la fonction créée*, cliquez sur son nom pour ouvrir ses détails.
    *   Allez dans l'onglet **"Configuration"** -> **"Permissions"**.
    *   Cliquez sur le **nom du rôle** (ex: `associateTickFunction-role-xxxxxx`). Cela vous amène à la console IAM.
    *   Dans l'onglet **"Permissions"** du rôle IAM, cliquez sur **"Add permissions"** -> **"Create inline policy"**.
    *   Passez à l'onglet **"JSON"**.
    *   **Collez la policy JSON nécessaire pour CETTE fonction.** Pour associer un Tick, la Lambda doit *écrire* dans la table `Ticks`. La permission requise est `dynamodb:PutItem`.
        ```json
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AllowPutItemIntoTicksTable",
                    "Effect": "Allow",
                    "Action": "dynamodb:PutItem", // Permission d'ajouter/remplacer un item
                    "Resource": "arn:aws:dynamodb:eu-north-1:148430413120:table/Ticks" // ARN de VOTRE table Ticks
                }
                // Vous pouvez ajouter ici la policy pour CloudWatch Logs si elle n'était pas dans le rôle de base
                ,{
                    "Sid": "AllowCloudWatchLoggingAssociate",
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "arn:aws:logs:eu-north-1:148430413120:log-group:/aws/lambda/associateTickFunction:*" // Mettez le nom EXACT de votre fonction
                }
            ]
        }
        ```
        *   **Adaptez** l'ARN de la table et le nom de la fonction dans l'ARN CloudWatch.
    *   Cliquez sur "Review policy", donnez un nom (ex: `DynamoDBWriteTicksPolicy`), puis "Create policy". Le rôle de la Lambda peut maintenant écrire dans la table `Ticks`.

3.  **Écrire/Coller le Code de la Fonction Lambda :**
    *   Retournez à la page de votre fonction Lambda (`associateTickFunction`).
    *   Allez dans l'onglet **"Code"**.
    *   **Remplacez** le code par défaut par le code de votre fonction. Voici un exemple pour `associateTickFunction` (Python 3.x) :

        ```python
        import json
        import boto3
        import os
        import base64
        import uuid # Pour générer un ID unique pour le Tick
        from decimal import Decimal
        import time # Pour timestamp

        # Helper pour JSON (identique à fetchTicks)
        class DecimalEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Decimal):
                    return int(obj) if obj % 1 == 0 else float(obj)
                return super(DecimalEncoder, self).default(obj)

        # Variables d'environnement ou valeurs par défaut
        TABLE_NAME = os.environ.get('TICKS_TABLE_NAME', 'Ticks')

        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(TABLE_NAME)

        def lambda_handler(event, context):
            print("Received event for associateTick:", json.dumps(event, indent=2))

            # --- 1. Extraire l'ID Utilisateur (sub) du Token JWT ---
            user_id = None
            try:
                # Code identique à fetchTicks pour extraire user_id du header Authorization
                auth_header = event.get('headers', {}).get('authorization', event.get('headers', {}).get('Authorization'))
                if not auth_header or not auth_header.startswith('Bearer '):
                    print("Authorization header missing or malformed")
                    return create_response(401, {'error': 'Unauthorized - Missing Token'})
                token = auth_header.split(' ')[1]
                # !!! Validation JWT manquante !!!
                payload_encoded = token.split('.')[1]
                payload_encoded += '=' * (-len(payload_encoded) % 4)
                decoded_payload = base64.b64decode(payload_encoded).decode('utf-8')
                token_claims = json.loads(decoded_payload)
                user_id = token_claims.get('sub')
                if not user_id:
                    return create_response(401, {'error': 'Unauthorized - Invalid Token Payload'})
                print(f"Authenticated user ID (sub): {user_id}")
            except Exception as e:
                print(f"Error extracting user ID from token: {e}")
                return create_response(401, {'error': f'Unauthorized - Token Error: {str(e)}'})

            # --- 2. Extraire les données du corps de la requête ---
            try:
                # Le corps arrive souvent comme une chaîne JSON dans l'event
                if isinstance(event.get('body'), str):
                    body = json.loads(event['body'])
                else:
                     # Si ce n'est pas une chaîne (moins courant avec Function URL simple)
                    body = event.get('body', {})

                tick_name = body.get('tickName')
                mac_address = body.get('macAddress')

                if not tick_name or not mac_address:
                    print("Missing 'tickName' or 'macAddress' in request body")
                    return create_response(400, {'error': 'Bad Request - Missing required fields: tickName, macAddress'})

                print(f"Request to associate: Name='{tick_name}', MAC='{mac_address}' for User='{user_id}'")

            except json.JSONDecodeError:
                 print("Invalid JSON in request body")
                 return create_response(400, {'error': 'Bad Request - Invalid JSON body'})
            except Exception as e:
                 print(f"Error parsing request body: {e}")
                 return create_response(400, {'error': f'Bad Request - Body Error: {str(e)}'})


            # --- 3. Préparer l'item pour DynamoDB ---
            tick_id = str(uuid.uuid4()) # Générer un ID unique pour ce Tick
            current_timestamp = int(time.time()) # Timestamp Unix (secondes)

            item_to_add = {
                'id': tick_id,             # Clé primaire de la table Ticks (ex: UUID)
                'ownerId': user_id,        # Clé du GSI (le 'sub' Cognito)
                'name': tick_name,
                'macAddress': mac_address,
                'createdAt': current_timestamp, # Date de création
                'lastUpdate': current_timestamp, # Dernière mise à jour (initiale)
                'status': 'inactive'       # Statut initial (ou 'active' ?)
                # Ajoutez d'autres attributs par défaut si nécessaire (latitude, longitude, batteryLevel à null ou valeur par défaut)
                # 'latitude': None, # Ou Decimal('0.0') ?
                # 'longitude': None,
                # 'batteryLevel': None # Ou Decimal('100') ?
            }

            # --- 4. Écrire dans DynamoDB ---
            try:
                print(f"Putting item into DynamoDB table '{TABLE_NAME}': {item_to_add}")
                table.put_item(Item=item_to_add)
                print("Successfully added item to DynamoDB.")

                # Retourner succès avec l'item créé (ou juste un message)
                # Convertir l'item pour la réponse JSON (gérer Decimal)
                response_data = {k: (int(v) if isinstance(v, Decimal) and v % 1 == 0 else float(v) if isinstance(v, Decimal) else v) for k, v in item_to_add.items()}
                return create_response(201, {'data': response_data}) # 201 Created

            except boto3.exceptions.botocore.exceptions.ClientError as e:
                # Gérer les erreurs DynamoDB (permissions, etc.)
                error_code = e.response.get('Error', {}).get('Code')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                print(f"DynamoDB PutItem ClientError: {error_code} - {error_message}")
                if error_code == 'AccessDeniedException':
                    return create_response(500, {'error': f"Lambda does not have permission to write to DynamoDB: {error_message}"})
                else:
                    return create_response(500, {'error': f"DynamoDB error during PutItem: {error_message}"})
            except Exception as e:
                print(f"Unexpected error writing to DynamoDB: {e}")
                return create_response(500, {'error': f'Internal server error during database write: {str(e)}'})


        def create_response(status_code, body):
            """Crée une réponse HTTP formatée."""
            return {
                'statusCode': status_code,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,Authorization,X-Amz-Date,X-Api-Key,X-Amz-Security-Token',
                    'Access-Control-Allow-Methods': 'GET,OPTIONS,POST,PUT,DELETE', # Important d'inclure POST ici !
                    'Content-Type': 'application/json'
                },
                'body': json.dumps(body, cls=DecimalEncoder)
            }

        ```
        *   **Adaptez** le nom de la table, le nom de la clé du GSI (`ownerId`), les noms des attributs (`id`, `name`, `macAddress`, etc.) pour correspondre **exactement** à votre modèle de données DynamoDB.
        *   Ce code génère un `uuid` pour la clé primaire du Tick. Assurez-vous que votre table `Ticks` a bien une clé primaire nommée `id` (ou adaptez le code).
        *   Il extrait `tickName` et `macAddress` du corps de la requête POST envoyée par Flutter.

4.  **Configurer la Function URL :**
    *   Allez dans l'onglet "Configuration" -> "Function URL" de votre fonction `associateTickFunction`.
    *   Cliquez sur "Create function URL".
    *   **Auth type : `AWS_IAM`**.
    *   **CORS :** Cochez "Configure CORS".
        *   Allow origin : `*` (pour tester).
        *   Allow methods : Cochez `POST` et `OPTIONS`. (POST car Flutter enverra les données via POST).
        *   Allow headers : Assurez-vous que `Authorization`, `Content-Type`, `X-Amz-*` sont présents.
    *   Cliquez sur "Save".
    *   **Copiez l'URL de la fonction (`Function URL`)** générée.

5.  **Vérifier les Permissions du Rôle Authentifié Cognito :**
    *   Allez dans IAM -> Rôles -> Rôle Authentifié de votre Identity Pool.
    *   Vérifiez la policy attachée (celle avec `lambda:InvokeFunctionUrl`).
    *   **Si vous avez utilisé `Resource: "*"`**, cette policy couvre déjà l'appel à la nouvelle URL `associateTickFunction`. Vous n'avez rien à ajouter *pour l'instant*.
    *   **(Bonne Pratique pour plus tard) :** Si vous n'utilisiez PAS `*`, vous devriez AJOUTER un nouveau `Statement` à cette policy (ou modifier l'existant s'il n'y en a qu'un) pour inclure l'ARN de la nouvelle fonction URL :
        ```json
        // ... (Statement existant pour fetchTicksFunction:url) ...
        ,{
            "Sid": "AllowInvokeAssociateTickLambdaUrlViaIAMAuth",
            "Effect": "Allow",
            "Action": "lambda:InvokeFunctionUrl",
            "Resource": "arn:aws:lambda:eu-north-1:148430413120:function:associateTickFunction:url", // ARN de la nouvelle fonction
            "Condition": {
                "StringEquals": {
                    "lambda:FunctionUrlAuthType": "AWS_IAM"
                }
            }
        }
        ```

6.  **Déployer la Lambda :**
    *   Retournez à la fonction `associateTickFunction` dans la console Lambda.
    *   Cliquez sur **"Deploy"** pour sauvegarder et activer le code Python.

**Partie 2 : Modifications Flutter**

7.  **Ajouter la Constante d'URL :**
    *   Ouvrez `lib/utils/constants.dart`.
    *   Ajoutez la constante pour la nouvelle URL dans `ApiConfig`:
        ```dart
        class ApiConfig {
          static const String getMyTicksURL = 'https://...'; // URL fetchTicks existante
          // AJOUTER CECI (avec votre URL réelle) :
          static const String associateTickFunctionUrl = 'https://VOTRE_URL_ASSOCIATE_TICK_ICI.lambda-url.eu-north-1.on.aws/';
          // ... autres URL ...
        }
        ```

8.  **Ajouter la Méthode dans `TickService` :**
    *   Ouvrez `lib/services/tick_service.dart`.
    *   Ajoutez (ou modifiez si elle existe déjà) la méthode `associateTick` pour utiliser `ApiService.post` et la nouvelle URL :

        ```dart
        // Dans la classe TickService

        Future<bool> associateTick(String name, String macAddress) async {
          // Vérification auth et isLoading (inchangée)
          if (!_authService.isAuthenticated) {
            _error = "Utilisateur non authentifié";
            notifyListeners();
            return false;
          }
          if (isLoading) return false;

          _setLoading(true);
          _error = null;

          try {
            // Préparer le corps de la requête attendu par la Lambda
            final body = {
              'tickName': name,
              'macAddress': macAddress,
            };

            print("TickService: Calling associateTick Function URL: ${ApiConfig.associateTickFunctionUrl}");
            // Appeler ApiService.post avec la nouvelle URL et le body
            final response = await _apiService.post(ApiConfig.associateTickFunctionUrl, body);

            if (response['success']) {
              print("Tick associated successfully via Lambda. Response data: ${response['data']}");
              // Optionnel: Ajouter le nouveau tick localement à partir de response['data']
              // ou simplement rafraîchir toute la liste.
              // final newTick = Tick.fromJson(response['data']);
              // _ticks.add(newTick); // Ajouter localement si la réponse contient le tick créé
              // notifyListeners();
              // OU (plus simple mais moins optimisé) :
              await fetchTicks(); // Recharge toute la liste

              // _setLoading(false); // fetchTicks() gère déjà setLoading(false)
              return true;
            } else {
              _error = response['error'] ?? ErrorMessages.associationFailed;
              _setLoading(false);
              return false;
            }
          } catch (e) {
            print("Exception associating tick: $e");
            _error = ErrorMessages.connectionFailed; // Utiliser erreur générique
            _setLoading(false);
            return false;
          }
        }

        // ... autres méthodes ...
        ```

9.  **Appeler depuis l'UI :**
    *   Ouvrez `lib/screens/tick/add_tick_page.dart`.
    *   Dans la méthode `_triggerAssociationApi`, l'appel existant à `tickService.associateTick(tickName, _tickMacAddress!)` est **déjà correct** car il appelle la méthode `associateTick` que nous venons de modifier dans `TickService`. Vous n'avez rien à changer ici.

**Partie 3 : Tester**

10. **`flutter clean` et `flutter run`**.
11. Connectez-vous à l'application.
12. Allez sur la page "Ajouter un Tick".
13. Suivez les étapes (nommer, scan BT, connexion BT).
14. Lorsque vous cliquez sur "Associer ce Tick", la méthode `_triggerAssociationApi` devrait appeler `tickService.associateTick`, qui appelle `_apiService.post`, qui appelle votre Lambda `associateTickFunction`.
15. **Vérifiez les Logs :**
    *   **Flutter Console :** Regardez si l'appel API POST est logué vers la bonne URL, s'il y a des erreurs Flutter.
    *   **CloudWatch Logs :** Allez dans CloudWatch -> Log groups -> Trouvez le groupe `/aws/lambda/associateTickFunction`. Regardez les derniers logs pour voir si la fonction a été invoquée, si l'ID utilisateur et les données du body ont été reçus correctement, et s'il y a eu des erreurs (permissions DynamoDB, etc.).
16. **Vérifiez DynamoDB :** Allez dans la console DynamoDB, ouvrez votre table `Ticks` et vérifiez si un nouvel item a été ajouté avec les bonnes informations (id, ownerId, name, macAddress...).
17. **Vérifiez l'UI Flutter :** Si l'association réussit, la `TickListPage` devrait se rafraîchir (grâce à `fetchTicks()` appelé après succès) et afficher le nouveau Tick.

---

Vous pouvez maintenant répéter ce processus pour chaque autre fonctionnalité (unlink, updateSettings, ring, locate, history...) en adaptant :

*   Le nom de la fonction Lambda.
*   Les permissions IAM du rôle d'exécution de la Lambda (ex: `dynamodb:DeleteItem`, `dynamodb:UpdateItem`, `iot:Publish` pour MQTT, etc.).
*   Le code Python/Node.js de la Lambda pour effectuer l'action désirée.
*   La configuration de la Function URL (méthode HTTP : GET, POST, PUT, DELETE...).
*   La constante d'URL dans `constants.dart`.
*   La méthode correspondante dans `TickService` pour appeler `ApiService` avec la bonne méthode HTTP (get, post, put, delete), la bonne URL et le bon corps de requête si nécessaire.
*   L'appel à la méthode du service depuis l'UI Flutter.

Bon courage à l'équipe ! C'est répétitif une fois qu'on a compris le flux.
