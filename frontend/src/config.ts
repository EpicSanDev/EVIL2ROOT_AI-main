/**
 * Configuration globale de l'application frontend
 */

// Déclaration des types pour process.env
declare global {
  interface Window {
    env?: {
      REACT_APP_API_URL?: string;
    };
  }
  namespace NodeJS {
    interface ProcessEnv {
      REACT_APP_API_URL?: string;
    }
  }
}

// URL de base de l'API
// Récupérer d'abord de window.env (si défini dans le HTML), puis de process.env 
// ou utiliser localhost par défaut
const getApiUrl = (): string => {
  if (window.env && window.env.REACT_APP_API_URL) {
    return window.env.REACT_APP_API_URL;
  }
  
  // @ts-ignore - Ignorer l'erreur de process.env pour la compilation TypeScript
  if (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API_URL) {
    // @ts-ignore
    return process.env.REACT_APP_API_URL;
  }
  
  return 'http://localhost:8000/api';
};

export const API_URL = getApiUrl();

// Endpoints de l'API
export const API_ENDPOINTS = {
  // Authentification
  AUTH: {
    LOGIN: '/auth/login',
    REGISTER: '/auth/register',
    ME: '/auth/me',
    FORGOT_PASSWORD: '/auth/forgot-password',
    RESET_PASSWORD: '/auth/reset-password',
    CHANGE_PASSWORD: '/auth/change-password',
  },
  // Trading
  TRADING: {
    POSITIONS: '/trading/positions',
    ORDERS: '/trading/orders',
    BALANCE: '/trading/balance',
    HISTORY: '/trading/history',
  },
  // Dashboard
  DASHBOARD: {
    SUMMARY: '/dashboard/summary',
    PERFORMANCE: '/dashboard/performance',
    ALERTS: '/dashboard/alerts',
  },
  // Settings
  SETTINGS: {
    STRATEGIES: '/settings/strategies',
    RISK: '/settings/risk',
    NOTIFICATIONS: '/settings/notifications',
    PREFERENCES: '/settings/preferences',
  },
  // Backtest
  BACKTEST: {
    RUN: '/backtest/run',
    RESULTS: '/backtest/results',
    HISTORY: '/backtest/history',
  },
};

// Configuration d'authentification
export const AUTH_CONFIG = {
  TOKEN_STORAGE_KEY: 'token',
  TOKEN_EXPIRY_MARGIN: 300, // 5 minutes en secondes
};

// Configuration des notifications
export const NOTIFICATION_CONFIG = {
  DURATION: 5000, // 5 secondes
  POSITION: 'top-right',
};

// Configuration de l'interface utilisateur
export const UI_CONFIG = {
  THEME: {
    LIGHT: 'light',
    DARK: 'dark',
  },
  DEFAULT_THEME: 'dark',
  DEFAULT_LANGUAGE: 'fr',
  DASHBOARD_REFRESH_INTERVAL: 30000, // 30 secondes
};

// Configuration des charts et graphiques
export const CHART_CONFIG = {
  DEFAULT_TIMEFRAME: '1d',
  AVAILABLE_TIMEFRAMES: ['5m', '15m', '1h', '4h', '1d', '1w'],
  DEFAULT_INDICATORS: ['MA', 'RSI'],
};

// Symboles de trading supportés
export const TRADING_SYMBOLS = [
  { id: 'BTCUSDT', name: 'Bitcoin/USDT' },
  { id: 'ETHUSDT', name: 'Ethereum/USDT' },
  { id: 'BNBUSDT', name: 'Binance Coin/USDT' },
  { id: 'ADAUSDT', name: 'Cardano/USDT' },
  { id: 'SOLUSDT', name: 'Solana/USDT' },
];

// Stratégies de trading disponibles
export const TRADING_STRATEGIES = [
  { id: 'technical', name: 'Technique' },
  { id: 'sentiment', name: 'Sentiment' },
  { id: 'rl', name: 'Apprentissage par renforcement' },
  { id: 'hybrid', name: 'Hybride' },
];

// Configuration des niveaux d'abonnement
export const SUBSCRIPTION_TIERS = {
  FREE: 'free',
  BASIC: 'basic',
  PRO: 'pro',
  ENTERPRISE: 'enterprise',
};

export default {
  API_URL,
  API_ENDPOINTS,
  AUTH_CONFIG,
  NOTIFICATION_CONFIG,
  UI_CONFIG,
  CHART_CONFIG,
  TRADING_SYMBOLS,
  TRADING_STRATEGIES,
  SUBSCRIPTION_TIERS,
}; 