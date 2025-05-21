# Stage 1: Build
FROM node:18-alpine as build

WORKDIR /app

# Copier package.json et package-lock.json (ou npm-shrinkwrap.json)
COPY package.json package-lock.json* ./
# Si vous utilisez npm-shrinkwrap.json, ajustez la ligne ci-dessus en conséquence
# COPY package.json npm-shrinkwrap.json ./

# Installer les dépendances de manière déterministe et rapide
# Assurez-vous que package-lock.json ou npm-shrinkwrap.json est présent et à jour
RUN npm ci

# Copier le reste du code source de l'application
COPY . .

# Construire l'application
RUN npm run build

# Stage 2: Production avec Nginx
# Utiliser une version plus récente d'Nginx et alpine
FROM nginx:1.25-alpine

# Créer un utilisateur et un groupe non-root pour Nginx si nécessaire
# Nginx par défaut sur Alpine tourne déjà en tant que 'nginx'.
# Si une configuration spécifique est nécessaire, décommentez et ajustez :
# RUN addgroup -S appgroup && adduser -S appuser -G appgroup
# USER appuser

# Copier la configuration Nginx personnalisée
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

# Copier les fichiers de build depuis l'étape précédente
COPY --from=build /app/build /usr/share/nginx/html

# S'assurer que l'utilisateur nginx a les permissions sur les fichiers servis
# RUN chown -R nginx:nginx /usr/share/nginx/html && chmod -R 755 /usr/share/nginx/html
# Ceci est généralement géré par l'image Nginx de base, mais peut être ajouté pour plus de sécurité.

# Exposer le port
EXPOSE 80

# Commande de démarrage
# Nginx s'exécute en tant que démon par défaut. '-g daemon off;' le fait tourner au premier plan.
CMD ["nginx", "-g", "daemon off;"]