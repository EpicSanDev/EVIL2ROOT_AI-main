import axios from 'axios';

// Configuration de l'URL de l'API
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Création d'une instance axios avec les en-têtes d'authentification
const authAxios = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Intercepteur pour ajouter le token d'authentification à chaque requête
authAxios.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Interface pour les données d'inscription
interface RegisterData {
  email: string;
  password: string;
  full_name: string;
}

// Interface pour les données de connexion
interface LoginData {
  username: string; // L'API utilise username pour l'email
  password: string;
}

// Interface pour la réponse de l'API de connexion
interface LoginResponse {
  access_token: string;
  token_type: string;
  user: {
    id: string;
    email: string;
    full_name: string;
    subscription_tier: string;
    is_active: boolean;
    created_at: string;
    last_login?: string;
  };
}

// Interface pour la mise à jour du profil
interface UserUpdateData {
  full_name?: string;
  email?: string;
}

// Service d'authentification
const authService = {
  // Inscription d'un nouvel utilisateur
  register: async (email: string, password: string, fullName: string) => {
    const data: RegisterData = {
      email,
      password,
      full_name: fullName,
    };
    return axios.post(`${API_URL}/auth/register`, data);
  },
  
  // Connexion d'un utilisateur
  login: async (email: string, password: string) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    
    return axios.post<LoginResponse>(`${API_URL}/auth/login`, formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
  },
  
  // Récupération de l'utilisateur actuel
  getCurrentUser: async () => {
    return authAxios.get(`${API_URL}/auth/me`);
  },
  
  // Mise à jour du profil utilisateur
  updateUser: async (userData: UserUpdateData) => {
    return authAxios.put(`${API_URL}/auth/me`, userData);
  },
  
  // Changement de mot de passe
  changePassword: async (currentPassword: string, newPassword: string) => {
    return authAxios.post(`${API_URL}/auth/change-password`, {
      current_password: currentPassword,
      new_password: newPassword,
    });
  },
  
  // Demande de réinitialisation de mot de passe
  resetPassword: async (email: string) => {
    return axios.post(`${API_URL}/auth/forgot-password`, { email });
  },
  
  // Réinitialisation de mot de passe avec token
  confirmResetPassword: async (token: string, newPassword: string) => {
    return axios.post(`${API_URL}/auth/reset-password`, {
      token,
      new_password: newPassword,
    });
  },
  
  // Suppression du compte
  deleteAccount: async () => {
    return authAxios.delete(`${API_URL}/auth/me`);
  },
};

export default authService; 