import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  Typography,
  AppBar,
  Toolbar,
  Button,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import { useAuth } from '../contexts/AuthContext';

const AuthLayout: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const { isAuthenticated } = useAuth();

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static" color="transparent" elevation={0}>
        <Toolbar>
          <Typography
            variant="h6"
            component={Link}
            to="/"
            sx={{ 
              textDecoration: 'none', 
              color: 'primary.main',
              flexGrow: 1,
              fontWeight: 'bold'
            }}
          >
            EVIL2ROOT Trading Bot
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button component={Link} to="/pricing" color="primary">
              Tarifs
            </Button>
            <Button component={Link} to="/about" color="primary">
              À propos
            </Button>
            <Button component={Link} to="/contact" color="primary">
              Contact
            </Button>
            {!isAuthenticated ? (
              <>
                <Button 
                  component={Link} 
                  to="/login" 
                  color="primary" 
                  variant="outlined"
                  sx={{ display: { xs: 'none', sm: 'block' } }}
                >
                  Connexion
                </Button>
                <Button 
                  component={Link} 
                  to="/register" 
                  color="primary" 
                  variant="contained"
                >
                  S'inscrire
                </Button>
              </>
            ) : (
              <Button 
                component={Link} 
                to="/dashboard" 
                color="primary" 
                variant="contained"
              >
                Tableau de bord
              </Button>
            )}
          </Box>
        </Toolbar>
      </AppBar>

      <Container 
        component="main" 
        maxWidth="xs" 
        sx={{ 
          mt: 4, 
          mb: 4, 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center',
          flex: 1,
          justifyContent: 'center'
        }}
      >
        <Paper 
          elevation={3} 
          sx={{ 
            p: 4, 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            width: '100%',
            borderRadius: 2
          }}
        >
          <Outlet />
        </Paper>
      </Container>

      <Box 
        component="footer" 
        sx={{ 
          p: 2, 
          mt: 'auto', 
          backgroundColor: theme.palette.background.paper,
          borderTop: `1px solid ${theme.palette.divider}`
        }}
      >
        <Container maxWidth="lg">
          <Typography variant="body2" color="text.secondary" align="center">
            © {new Date().getFullYear()} EVIL2ROOT Trading Bot. Tous droits réservés.
          </Typography>
        </Container>
      </Box>
    </Box>
  );
};

export default AuthLayout;