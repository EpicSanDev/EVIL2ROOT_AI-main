import React, { createContext, useState, useEffect, useContext, ReactNode } from 'react';
import jwt_decode from 'jwt-decode';
import authService from '../services/authService';

interface User {
  id: string;
  email: string;
  full_name: string;
  subscription_tier: string;
  is_active: boolean;
  created_at: string;
  last_login?: string;
}

interface AuthContextType {
  isAuthenticated: boolean;
  user: User | null;
  loading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName: string) => Promise<void>;
  logout: () => void;
  resetPassword: (email: string) => Promise<void>;
  updateUser: (userData: Partial<User>) => Promise<void>;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);

  useEffect(() => {
    // Vérifier si l'utilisateur est déjà authentifié
    const checkAuth = async () => {
      const token = localStorage.getItem('token');
      
      if (token) {
        try {
          // Vérifier si le token est valide
          const decoded = jwt_decode<{ exp: number, sub: string }>(token);
          const currentTime = Date.now() / 1000;
          
          if (decoded.exp < currentTime) {
            // Token expiré
            localStorage.removeItem('token');
            setIsAuthenticated(false);
            setUser(null);
          } else {
            // Token valide, récupérer les informations utilisateur
            const response = await authService.getCurrentUser();
            setUser(response.data);
            setIsAuthenticated(true);
          }
        } catch (error) {
          console.error('Invalid token:', error);
          localStorage.removeItem('token');
          setIsAuthenticated(false);
          setUser(null);
        }
      }
      
      setLoading(false);
    };

    checkAuth();
  }, []);

  const login = async (email: string, password: string) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await authService.login(email, password);
      const { access_token, user } = response.data;
      
      // Stocker le token
      localStorage.setItem('token', access_token);
      
      setUser(user);
      setIsAuthenticated(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Échec de la connexion. Veuillez vérifier vos identifiants.');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const register = async (email: string, password: string, fullName: string) => {
    try {
      setLoading(true);
      setError(null);
      
      await authService.register(email, password, fullName);
      
      // Connexion automatique après inscription
      await login(email, password);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Échec de l\'inscription. Veuillez réessayer.');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
    setIsAuthenticated(false);
  };

  const resetPassword = async (email: string) => {
    try {
      setLoading(true);
      setError(null);
      
      await authService.resetPassword(email);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Échec de la réinitialisation du mot de passe.');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const updateUser = async (userData: Partial<User>) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await authService.updateUser(userData);
      setUser(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Échec de la mise à jour du profil.');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const clearError = () => {
    setError(null);
  };

  const value = {
    isAuthenticated,
    user,
    loading,
    error,
    login,
    register,
    logout,
    resetPassword,
    updateUser,
    clearError
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}; 