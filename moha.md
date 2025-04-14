Okay, super ! Le résumé est très clair et montre que vous avez déjà une bonne partie du backend en place. Puisque l'authentification Flutter utilise maintenant Amplify/Cognito, la tâche principale est de **connecter cette authentification Cognito à votre API Gateway existante** et de s'assurer que vos Lambdas utilisent l'identité de l'utilisateur connecté.

On va décomposer ça :

**Objectif Principal :** Faire en sorte que lorsque votre application Flutter appelle votre API Gateway (via `ApiService`), l'API Gateway vérifie que l'utilisateur est bien connecté via Cognito, et transmette l'identité de cet utilisateur à vos fonctions Lambda pour qu'elles puissent agir en conséquence (ex: lister les Ticks de CET utilisateur).

---

**Étape 1 : Sécuriser API Gateway avec un Authorizer Cognito**

C'est l'étape la plus cruciale. On va dire à API Gateway : "Pour ces routes-là (/ticks, /users/me, etc.), tu ne laisses passer que les requêtes qui ont un jeton JWT valide provenant de notre User Pool Cognito".

1.  **Aller sur API Gateway dans la Console AWS :**
    *   Recherchez "API Gateway".
    *   Trouvez et sélectionnez votre API REST existante (celle qui expose `/auth/login`, `/ticks`, etc.).

2.  **Créer un Authorizer Cognito :**
    *   Dans le menu de gauche de votre API, cliquez sur **"Authorizers"**.
    *   Cliquez sur **"Create Authorizer"**.
    *   **Authorizer name :** Donnez-lui un nom clair, par exemple `CognitoUserPoolAuthorizer`.
    *   **Type :** Sélectionnez **"Cognito"**.
    *   **Cognito User Pool :** Dans la liste déroulante, sélectionnez votre User Pool (celui avec l'ID `eu-north-1_EFgbgMufn`). Si vous avez plusieurs User Pools dans cette région, assurez-vous de choisir le bon !
    *   **Token source :** Entrez `Authorization`. C'est le nom de l'en-tête HTTP où votre `ApiService` Flutter enverra le jeton JWT (sous la forme `Bearer <token>`). API Gateway cherchera cet en-tête.
    *   **Token validation :** Laissez vide ou par défaut. API Gateway validera la signature et l'expiration du token automatiquement.
    *   Cliquez sur **"Create Authorizer"**.

3.  **Appliquer l'Authorizer aux Routes :**
    *   Maintenant, il faut dire quelles routes doivent utiliser cet Authorizer. Allez dans **"Resources"** dans le menu de gauche.
    *   Sélectionnez une route que vous voulez protéger (ex: la ressource `/ticks`).
    *   Cliquez sur la méthode HTTP associée (ex: `GET`).
    *   Vous verrez une section "Method Execution" ou un diagramme. Cliquez sur **"Method Request"**.
    *   Dans la section **"Settings"** ou **"Request Validator"**, trouvez le champ **"Authorization"**.
    *   Cliquez sur l'icône crayon (modifier) à côté.
    *   Dans la liste déroulante, sélectionnez l'Authorizer que vous venez de créer (`CognitoUserPoolAuthorizer`).
    *   Cliquez sur l'icône coche (enregistrer/confirmer).
    *   **Répétez cette opération pour TOUTES les méthodes (GET, POST, PUT, DELETE) et TOUTES les ressources qui nécessitent une authentification utilisateur.**
        *   Routes à protéger typiquement : `/ticks` (toutes méthodes), `/ticks/{tickId}` (toutes méthodes), `/ticks/{tickId}/history`, `/ticks/{tickId}/settings`, `/ticks/{tickId}/locate`, `/ticks/{tickId}/ring`, `/users/me`.
        *   Routes à **NE PAS** protéger : `/auth/login`, `/auth/register`, `/auth/forgot-password`, `/auth/reset-password` (car l'utilisateur n'est pas encore authentifié pour ces actions). Pour ces routes, assurez-vous que "Authorization" est réglé sur `NONE`.

---

**Étape 2 : Modifier vos Fonctions Lambda pour Utiliser l'Identité Cognito**

Quand une requête authentifiée via l'Authorizer Cognito arrive à votre Lambda, API Gateway ajoute des informations sur l'utilisateur dans l'objet `event` passé à la fonction Lambda. L'information la plus importante est l'identifiant unique de l'utilisateur (le `sub`), qui correspond au `userId` dans votre `User` model Flutter (et probablement au `user_id` dans vos tables DynamoDB).

1.  **Identifier la Clé :** L'identifiant utilisateur (`sub`) se trouve généralement dans `event.requestContext.authorizer.claims.sub`.

2.  **Adapter le Code Lambda (Exemple en Node.js) :**
    *   Dans **chaque fonction Lambda** qui est derrière une route protégée par l'Authorizer Cognito :
        *   **Extrayez le `userId` :** Au début de votre fonction `handler` :
            ```javascript
            // Exemple en Node.js pour une fonction Lambda
            exports.handler = async (event) => {
                console.log("Received event:", JSON.stringify(event, null, 2)); // Très utile pour déboguer !

                let userId;
                try {
                    // Chemin standard pour récupérer l'ID utilisateur (sub) depuis les claims Cognito
                    userId = event.requestContext.authorizer.claims.sub;

                    if (!userId) {
                        console.error("User ID (sub) not found in claims.");
                        return {
                            statusCode: 403, // Forbidden
                            headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" }, // Ajouter CORS si nécessaire
                            body: JSON.stringify({ success: false, error: "User identifier not found in request." }),
                        };
                    }
                    console.log("Authenticated User ID:", userId);

                } catch (error) {
                    console.error("Error accessing authorizer claims:", error);
                    return {
                        statusCode: 500,
                        headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
                        body: JSON.stringify({ success: false, error: "Internal server error processing authentication." }),
                    };
                }

                // --- Votre logique existante commence ici ---
                // Utilisez la variable 'userId' pour filtrer vos requêtes DynamoDB, etc.

                // Exemple: Pour récupérer les ticks de l'utilisateur
                /*
                const params = {
                    TableName: "Ticks", // Votre nom de table
                    IndexName: "UserIdIndex", // Assurez-vous d'avoir un index secondaire sur user_id si nécessaire
                    KeyConditionExpression: "user_id = :uid",
                    ExpressionAttributeValues: {
                        ":uid": userId
                    }
                };
                const result = await dynamoDbClient.query(params).promise();
                const ticks = result.Items;
                */

                // Exemple: Pour associer un tick
                /*
                const { tickName, macAddress } = JSON.parse(event.body);
                const tickId = macAddress; // Ou une autre logique
                const params = {
                    TableName: "Ticks",
                    Item: {
                        tick_id: tickId, // Clé primaire
                        user_id: userId, // L'ID de l'utilisateur authentifié !
                        tickName: tickName,
                        created_at: new Date().toISOString(),
                        // ... autres attributs
                    }
                };
                await dynamoDbClient.put(params).promise();
                */

                // Retournez votre réponse normale...
                return {
                    statusCode: 200,
                    headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" }, // Ajouter CORS
                    body: JSON.stringify({ success: true, data: { /* ... vos données ... */ } }),
                };
            };
            ```
        *   **Adaptez vos requêtes DynamoDB :** Assurez-vous que toutes les opérations (lecture, écriture, suppression) liées aux `Ticks` ou `TickLocationHistory` utilisent ce `userId` extrait pour filtrer ou associer les données au bon utilisateur. N'utilisez plus un `userId` potentiellement passé dans le corps de la requête (sauf pour des cas d'admin spécifiques, mais pas pour les opérations utilisateur standard).
        *   **Lambda `/users/me` :** Cette fonction peut maintenant simplement utiliser le `userId` extrait pour récupérer les infos de l'utilisateur depuis la table `Users` (ou directement retourner les `claims` Cognito si vous n'avez pas besoin d'infos supplémentaires de votre table `Users`).

---

**Étape 3 : Vérifier/Configurer les Permissions IAM pour les Lambdas**

Chaque fonction Lambda s'exécute avec un "Execution Role" IAM. Ce rôle doit avoir les permissions nécessaires pour faire ce que la fonction doit faire.

1.  **Trouver le Rôle d'Exécution :**
    *   Allez dans le service **Lambda** dans la console AWS.
    *   Sélectionnez une de vos fonctions (ex: celle derrière `GET /ticks`).
    *   Allez dans l'onglet **"Configuration"** puis **"Permissions"**.
    *   Cliquez sur le nom du rôle sous **"Execution role"**. Cela vous mènera à la page IAM de ce rôle.

2.  **Vérifier/Ajouter les Permissions DynamoDB :**
    *   Sur la page du rôle IAM, regardez les politiques attachées (inline ou managed).
    *   Assurez-vous qu'il y a une politique autorisant les actions nécessaires sur vos tables DynamoDB. Au minimum, pour une fonction qui lit/écrit des ticks :
        *   `dynamodb:Query` (si vous utilisez des index)
        *   `dynamodb:GetItem`
        *   `dynamodb:PutItem`
        *   `dynamodb:UpdateItem`
        *   `dynamodb:DeleteItem`
    *   **Important :** La ressource (`Resource`) doit spécifier l'ARN (Amazon Resource Name) de vos tables (`Ticks`, `TickLocationHistory`). Utiliser l'ARN exact est plus sécurisé que d'utiliser `*`.
        *   Vous trouverez l'ARN de vos tables dans la console DynamoDB, en sélectionnant la table, onglet "Overview" ou "Additional info". Il ressemble à `arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/YourTableName`.
    *   Si les permissions manquent, cliquez sur **"Add permissions"** -> **"Attach policies"** (si une politique AWS gérée convient, comme `AmazonDynamoDBFullAccess` - *moins sécurisé*) ou **"Create inline policy"** pour définir des permissions précises.
        *   Exemple de politique inline précise :
            ```json
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "dynamodb:Query",
                            "dynamodb:GetItem",
                            "dynamodb:PutItem",
                            "dynamodb:UpdateItem",
                            "dynamodb:DeleteItem"
                        ],
                        "Resource": [
                            "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/Ticks", // Remplacez ACCOUNT_ID
                            "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/Ticks/index/*", // Si vous avez des index
                            "arn:aws:dynamodb:eu-north-1:ACCOUNT_ID:table/TickLocationHistory" // Remplacez ACCOUNT_ID
                        ]
                    }
                ]
            }
            ```

3.  **Permissions pour la communication avec les Ticks (via IoT Core - voir Étape 4) :**
    *   Les Lambdas qui doivent envoyer des commandes (`/locate`, `/ring`) aux Ticks auront besoin de permissions pour interagir avec AWS IoT Core. Ajoutez ces actions au rôle IAM :
        *   `iot:Publish` (pour envoyer des messages MQTT)
        *   `iot:UpdateThingShadow` (si vous utilisez les Device Shadows)
        *   `iot:GetThingShadow` (potentiellement)
    *   La ressource (`Resource`) sera typiquement l'ARN des topics MQTT (ex: `arn:aws:iot:eu-north-1:ACCOUNT_ID:topic/ticks/*/commands`) ou des Things (ex: `arn:aws:iot:eu-north-1:ACCOUNT_ID:thing/*`).

---

**Étape 4 : Configurer AWS IoT Core pour la Communication Lambda -> Tick**

Vos endpoints `/ticks/{tickId}/locate` et `/ticks/{tickId}/ring` impliquent que le backend (Lambda) doit envoyer une commande à un appareil ESP32 spécifique. Le moyen standard de faire cela dans AWS est via AWS IoT Core et MQTT.

*Votre résumé indique que c'est une "prochaine étape", mais elle est nécessaire pour que ces endpoints fonctionnent.*

1.  **Concepts Clés d'IoT Core :**
    *   **Thing (Objet) :** Une représentation virtuelle de votre appareil physique (un Tick ESP32) dans AWS IoT Core. Chaque Tick aura son propre "Thing". Le nom du Thing pourrait être le `tick_id` (MAC address).
    *   **Certificate (Certificat) :** Chaque Thing a besoin d'un certificat X.509 pour s'authentifier de manière sécurisée auprès d'AWS IoT Core. Ce certificat (et sa clé privée) doit être flashé sur l'ESP32.
    *   **Policy (Stratégie) :** Une stratégie IAM spécifique à IoT qui définit ce qu'un appareil (identifié par son certificat) a le droit de faire (se connecter, publier sur quels topics, s'abonner à quels topics).
    *   **MQTT :** Le protocole de messagerie utilisé. Les appareils publient des données (comme la position GPS) sur des *topics* et s'abonnent à des *topics* pour recevoir des commandes.

2.  **Actions à Réaliser (Résumé pour l'instant) :**
    *   **Créer un "Thing"** dans AWS IoT Core pour chaque Tick lors de son association (peut-être via la Lambda d'association ? Ou manuellement pour les tests).
    *   **Générer un Certificat et une Clé Privée** pour ce Thing. Stocker la clé privée *uniquement* sur l'ESP32. Attacher le certificat au Thing.
    *   **Créer une Stratégie IoT** qui autorise l'appareil à :
        *   Se connecter (`iot:Connect`) avec son `clientId` (souvent le `tick_id`).
        *   Publier ses données GPS sur un topic spécifique (ex: `ticks/{tickId}/data`).
        *   S'abonner à un topic de commandes spécifique (ex: `ticks/{tickId}/commands`).
        *   Recevoir les messages sur ce topic (`iot:Receive`).
        *   (Optionnel) Mettre à jour/Lire son Device Shadow (`iot:UpdateThingShadow`, `iot:GetThingShadow`) si vous utilisez cette méthode pour les commandes.
    *   **Attacher la Stratégie** au Certificat de l'appareil.
    *   **Dans votre code ESP32 :** Utiliser une librairie MQTT (comme `PubSubClient`) avec le certificat et la clé privée pour se connecter à l'endpoint AWS IoT Core (vous le trouverez dans IoT Core -> Settings), s'abonner au topic de commandes, et publier les données GPS.
    *   **Dans vos Lambdas `/locate` et `/ring` :** Utiliser l'AWS SDK (ex: `aws-sdk` pour Node.js) pour appeler `iotdata.publish()` et envoyer un message MQTT sur le topic `ticks/{tickId}/commands` correspondant. Le message contiendrait la commande (ex: `{ "command": "locate" }` ou `{ "command": "ring", "duration": 5 }`).

---

**Étape 5 : Mettre à Jour les Lambdas `/locate` et `/ring` pour Interagir avec IoT Core**

Ces Lambdas, après avoir extrait le `userId` et vérifié que l'utilisateur possède bien le `tickId` demandé (en consultant la table `Ticks`), devront envoyer la commande via IoT.

1.  **Initialiser le Client IoT Data Plane (Exemple Node.js) :**
    ```javascript
    const AWS = require('aws-sdk');
    // L'endpoint spécifique à votre compte AWS IoT Core
    const iotDataEndpoint = 'YOUR_IOT_ENDPOINT.iot.eu-north-1.amazonaws.com'; // Remplacez par votre endpoint réel (IoT Core -> Settings)
    const iotdata = new AWS.IotData({ endpoint: iotDataEndpoint, region: 'eu-north-1' });
    ```

2.  **Publier le Message MQTT (Exemple Node.js dans le handler) :**
    ```javascript
    exports.handler = async (event) => {
        // ... (extraction userId, validation tickId comme à l'étape 2) ...
        const tickId = event.pathParameters.tickId; // Récupérer depuis le chemin de l'URL

        // Vérifier que userId possède tickId (requête DynamoDB sur la table Ticks)
        // ...

        const commandTopic = `ticks/${tickId}/commands`;
        let commandPayload;

        // Déterminer la commande basée sur l'endpoint appelé (ex: event.resource)
        if (event.resource.includes('/locate')) {
            commandPayload = JSON.stringify({ command: 'locate', requestedBy: userId });
        } else if (event.resource.includes('/ring')) {
             commandPayload = JSON.stringify({ command: 'ring', duration: 5, requestedBy: userId }); // Ajouter une durée par exemple
        } else {
            // Gérer erreur endpoint inconnu
            return { statusCode: 400, body: JSON.stringify({ error: 'Invalid command endpoint' })};
        }

        const params = {
            topic: commandTopic,
            payload: commandPayload,
            qos: 1 // Qualité de service (0, 1, ou 2)
        };

        try {
            await iotdata.publish(params).promise();
            console.log(`Command published to ${commandTopic}: ${commandPayload}`);
            return {
                statusCode: 200,
                headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
                body: JSON.stringify({ success: true, message: 'Command sent successfully.' })
            };
        } catch (error) {
            console.error(`Error publishing command to ${commandTopic}:`, error);
            return {
                statusCode: 500,
                 headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
                body: JSON.stringify({ success: false, error: 'Failed to send command to device.' })
            };
        }
    };
    ```

---

**Étape 6 : Déployer API Gateway & Mettre à Jour Flutter**

1.  **Déployer l'API :** Après avoir appliqué les Authorizers Cognito et vérifié/modifié vos Lambdas, vous devez redéployer votre API Gateway pour que les changements prennent effet.
    *   Dans API Gateway, allez dans **"Resources"**.
    *   Cliquez sur le menu **"Actions"** -> **"Deploy API"**.
    *   Sélectionnez un **"Deployment stage"** (ex: `prod`, `dev`, `v1` - vous en avez probablement déjà un). Si vous n'en avez pas, créez-en un.
    *   Cliquez sur **"Deploy"**.

2.  **Obtenir l'URL d'Invocation :**
    *   Une fois déployée, allez dans **"Stages"** dans le menu de gauche.
    *   Sélectionnez le stage que vous venez de déployer (ex: `prod`).
    *   En haut, vous verrez **"Invoke URL"**. C'est l'URL de base de votre API. Elle ressemblera à `https://xxxxx.execute-api.eu-north-1.amazonaws.com/prod`.

3.  **Mettre à Jour `ApiConfig.baseUrl` dans Flutter :**
    *   Ouvrez `lib/utils/constants.dart`.
    *   Remplacez la valeur de `ApiConfig.baseUrl` par l'**Invoke URL** que vous venez de copier.
        ```dart
        class ApiConfig {
          // #TODO: Remplacer par votre URL API Gateway déployée !
          static const String baseUrl = 'https://xxxxx.execute-api.eu-north-1.amazonaws.com/prod'; // Votre URL réelle ici
        }
        ```

---

**Étape 7 : Tester l'Intégration**

1.  **Recompilez et lancez votre application Flutter.**
2.  **Authentification :** Vérifiez à nouveau l'inscription, la confirmation, la connexion, la déconnexion.
3.  **Appels API Protégés :**
    *   Après connexion, allez sur la `TickListPage`. L'appel à `fetchTicks` (qui fait un `GET /ticks`) devrait maintenant fonctionner car `ApiService` envoie le token Cognito et l'Authorizer API Gateway le valide.
    *   Essayez d'associer un nouveau Tick (via `AddTickPage`). L'appel `POST /ticks` devrait fonctionner.
    *   Naviguez vers la page des détails/carte d'un Tick.
    *   Essayez d'appeler les fonctions `Localiser` (`POST /ticks/{tickId}/locate`) et `Faire sonner` (`POST /ticks/{tickId}/ring`). Surveillez les logs de la fonction Lambda correspondante dans CloudWatch (le service de logs AWS) pour voir si elle reçoit la requête et tente de publier sur IoT Core.
    *   Vérifiez l'historique (`GET /ticks/{tickId}/history`).
    *   Allez dans les paramètres du Tick et essayez de changer le nom (`PUT /ticks/{tickId}/settings`) ou de désassocier (`DELETE /ticks/{tickId}`).
4.  **Appels API Non Protégés :** Assurez-vous que `/login` et `/register` fonctionnent toujours sans être connecté.
5.  **Surveillance IoT (si configuré) :** Utilisez le client de test MQTT dans la console AWS IoT Core pour voir si les messages de commande arrivent bien sur les topics attendus lorsque vous cliquez sur "Localiser" ou "Faire sonner".

---

Voilà le plan d'attaque pour connecter votre authentification Amplify/Cognito à votre backend API Gateway/Lambda existant et préparer la communication avec les appareils via IoT Core. C'est dense, mais chaque étape est logique. Prenez votre temps et testez au fur et à mesure !
