# Configuration du Container Registry sur DigitalOcean

## 1. Créer un Container Registry

1. Connectez-vous à votre compte DigitalOcean
2. Accédez à la section "Container Registry" dans le menu latéral
3. Cliquez sur "Create Registry"
4. Choisissez un nom pour votre registre (par exemple "evil2root-registry")
5. Sélectionnez la région la plus proche de vos utilisateurs
6. Choisissez un plan tarifaire adapté à vos besoins
7. Cliquez sur "Create Registry"

## 2. Générer un token d'accès personnel

Pour que GitHub Actions puisse accéder à votre registre:

1. Dans le dashboard DigitalOcean, allez dans "API" (en bas du menu à gauche)
2. Cliquez sur "Generate New Token"
3. Donnez un nom à votre token (ex: "github-actions-token")
4. Sélectionnez une durée d'expiration (ou sans expiration pour plus de simplicité)
5. Assurez-vous que le scope "Write" est activé
6. Cliquez sur "Generate Token"
7. **IMPORTANT**: Copiez immédiatement le token généré et conservez-le dans un endroit sécurisé. Vous ne pourrez plus le voir après avoir quitté cette page.

Ce token sera utilisé comme secret GitHub (`DIGITALOCEAN_ACCESS_TOKEN`). 