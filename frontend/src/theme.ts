import { createTheme } from '@mui/material/styles';

// Thème personnalisé pour l'application
const theme = createTheme({
  palette: {
    primary: {
      main: '#2a3eb1', // Bleu professionnel
      light: '#5e68e2',
      dark: '#001a82',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#19a463', // Vert pour les profits
      light: '#60d694',
      dark: '#007535',
      contrastText: '#ffffff',
    },
    error: {
      main: '#d32f2f', // Rouge pour les pertes et erreurs
      light: '#ff6659',
      dark: '#9a0007',
      contrastText: '#ffffff',
    },
    warning: {
      main: '#f57c00', // Orange pour les avertissements
      light: '#ffad42',
      dark: '#bb4d00',
      contrastText: '#ffffff',
    },
    info: {
      main: '#0288d1', // Bleu clair pour les informations
      light: '#5eb8ff',
      dark: '#005b9f',
      contrastText: '#ffffff',
    },
    success: {
      main: '#388e3c', // Vert foncé pour les succès
      light: '#6abf69',
      dark: '#00600f',
      contrastText: '#ffffff',
    },
    text: {
      primary: '#263238', // Gris foncé pour le texte principal
      secondary: '#546e7a', // Gris moyen pour le texte secondaire
      disabled: '#b0bec5', // Gris clair pour le texte désactivé
    },
    background: {
      default: '#f5f5f5', // Gris très clair pour le fond
      paper: '#ffffff', // Blanc pour les cartes et composants
    },
  },
  typography: {
    fontFamily: [
      'Inter',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
    },
    subtitle1: {
      fontSize: '1rem',
      fontWeight: 500,
    },
    subtitle2: {
      fontSize: '0.875rem',
      fontWeight: 500,
    },
    body1: {
      fontSize: '1rem',
      fontWeight: 400,
    },
    body2: {
      fontSize: '0.875rem',
      fontWeight: 400,
    },
    button: {
      fontSize: '0.875rem',
      fontWeight: 500,
      textTransform: 'none',
    },
    caption: {
      fontSize: '0.75rem',
      fontWeight: 400,
    },
    overline: {
      fontSize: '0.75rem',
      fontWeight: 500,
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '8px 16px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
          },
        },
        contained: {
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 4px rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiTable: {
      styleOverrides: {
        root: {
          borderCollapse: 'separate',
          borderSpacing: '0 4px',
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          '&:hover': {
            backgroundColor: 'rgba(0, 0, 0, 0.03)',
          },
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          padding: '12px 16px',
        },
        head: {
          fontWeight: 600,
          color: '#263238',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          marginBottom: '16px',
        },
      },
    },
  },
});

export default theme; 