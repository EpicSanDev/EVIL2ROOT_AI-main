import axios from 'axios';
import { API_URL, API_ENDPOINTS } from '../config';

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

// Intercepteur pour gérer les erreurs d'authentification (token expiré, etc.)
authAxios.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Si nous recevons une erreur 401, le token est probablement expiré
      localStorage.removeItem('token');
      window.location.href = '/login?session=expired';
    }
    return Promise.reject(error);
  }
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
    try {
      return await axios.post(`${API_URL}${API_ENDPOINTS.AUTH.REGISTER}`, data);
    } catch (error) {
      console.error('Erreur lors de l\'inscription:', error);
      throw error;
    }
  },
  
  // Connexion d'un utilisateur
  login: async (email: string, password: string) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    
    try {
      return await axios.post<LoginResponse>(`${API_URL}${API_ENDPOINTS.AUTH.LOGIN}`, formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
    } catch (error) {
      console.error('Erreur lors de la connexion:', error);
      throw error;
    }
  },
  
  // Récupération de l'utilisateur actuel
  getCurrentUser: async () => {
    try {
      return await authAxios.get(`${API_ENDPOINTS.AUTH.ME}`);
    } catch (error) {
      console.error('Erreur lors de la récupération du profil:', error);
      throw error;
    }
  },
  
  // Mise à jour du profil utilisateur
  updateUser: async (userData: UserUpdateData) => {
    try {
      return await authAxios.put(`${API_ENDPOINTS.AUTH.ME}`, userData);
    } catch (error) {
      console.error('Erreur lors de la mise à jour du profil:', error);
      throw error;
    }
  },
  
  // Changement de mot de passe
  changePassword: async (currentPassword: string, newPassword: string) => {
    try {
      return await authAxios.post(`${API_ENDPOINTS.AUTH.CHANGE_PASSWORD}`, {
        current_password: currentPassword,
        new_password: newPassword,
      });
    } catch (error) {
      console.error('Erreur lors du changement de mot de passe:', error);
      throw error;
    }
  },
  
  // Demande de réinitialisation de mot de passe
  resetPassword: async (email: string) => {
    try {
      return await axios.post(`${API_URL}${API_ENDPOINTS.AUTH.FORGOT_PASSWORD}`, { email });
    } catch (error) {
      console.error('Erreur lors de la demande de réinitialisation:', error);
      throw error;
    }
  },
  
  // Réinitialisation de mot de passe avec token
  confirmResetPassword: async (token: string, newPassword: string) => {
    try {
      return await axios.post(`${API_URL}${API_ENDPOINTS.AUTH.RESET_PASSWORD}`, {
        token,
        new_password: newPassword,
      });
    } catch (error) {
      console.error('Erreur lors de la réinitialisation du mot de passe:', error);
      throw error;
    }
  },
  
  // Suppression du compte
  deleteAccount: async () => {
    try {
      return await authAxios.delete(`${API_ENDPOINTS.AUTH.ME}`);
    } catch (error) {
      console.error('Erreur lors de la suppression du compte:', error);
      throw error;
    }
  },
};

export default authService; 