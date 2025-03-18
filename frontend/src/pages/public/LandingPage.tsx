import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  Box,
  Button,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  AppBar,
  Toolbar,
  useTheme,
  useMediaQuery,
  Divider,
  Paper,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Psychology as PsychologyIcon,
  BarChart as BarChartIcon,
  Autorenew as AutorenewIcon,
  AccessTime as AccessTimeIcon,
  Security as SecurityIcon,
} from '@mui/icons-material';
import { useAuth } from '../../contexts/AuthContext';

const LandingPage: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const { isAuthenticated } = useAuth();

  const features = [
    {
      icon: <TrendingUpIcon color="primary" sx={{ fontSize: 40 }} />,
      title: 'Trading automatisé',
      description: 'Automatisez vos stratégies de trading et laissez notre bot opérer 24/7 sur les marchés financiers.',
    },
    {
      icon: <PsychologyIcon color="primary" sx={{ fontSize: 40 }} />,
      title: 'IA avancée',
      description: 'Notre moteur d\'intelligence artificielle analyse continuellement les marchés et s\'adapte aux conditions changeantes.',
    },
    {
      icon: <BarChartIcon color="primary" sx={{ fontSize: 40 }} />,
      title: 'Analyses détaillées',
      description: 'Explorez des données historiques et des analyses prévisionnelles pour prendre des décisions éclairées.',
    },
    {
      icon: <AutorenewIcon color="primary" sx={{ fontSize: 40 }} />,
      title: 'Auto-apprentissage',
      description: 'Notre système apprend de ses performances passées pour s\'améliorer continuellement.',
    },
    {
      icon: <AccessTimeIcon color="primary" sx={{ fontSize: 40 }} />,
      title: 'Réaction rapide',
      description: 'Réagissez en millisecondes aux mouvements du marché et saisissez les opportunités avant les autres.',
    },
    {
      icon: <SecurityIcon color="primary" sx={{ fontSize: 40 }} />,
      title: 'Sécurité maximale',
      description: 'Vos données et stratégies sont protégées par des protocoles de sécurité de niveau bancaire.',
    },
  ];

  const plans = [
    {
      title: 'Gratuit',
      price: '0€',
      period: 'pour toujours',
      features: [
        'Accès limité au bot de trading',
        '3 stratégies prédéfinies',
        'Trading en mode simulé uniquement',
        'Backtesting sur 1 mois de données',
        'Tableau de bord basique',
      ],
      color: theme.palette.grey[200],
      buttonText: 'Essayer gratuitement',
      buttonVariant: 'outlined',
    },
    {
      title: 'Pro',
      price: '49€',
      period: 'par mois',
      features: [
        'Accès complet au bot de trading',
        'Stratégies personnalisées illimitées',
        'Trading en mode réel et simulé',
        'Backtesting sur 5 ans de données',
        'Alertes et notifications avancées',
        'Support prioritaire',
      ],
      color: theme.palette.primary.light,
      buttonText: 'Commencer maintenant',
      buttonVariant: 'contained',
      highlighted: true,
    },
    {
      title: 'Entreprise',
      price: '199€',
      period: 'par mois',
      features: [
        'Tout ce qui est inclus dans Pro',
        'API privée dédiée',
        'Déploiement sur serveur dédié',
        'Personnalisation complète',
        'Formations et webinaires exclusifs',
        'Support dédié 24/7',
      ],
      color: theme.palette.secondary.light,
      buttonText: 'Contacter les ventes',
      buttonVariant: 'outlined',
    },
  ];

  return (
    <Box>
      {/* Header */}
      <AppBar position="static" color="transparent" elevation={0}>
        <Toolbar>
          <Typography
            variant="h5"
            sx={{ 
              flexGrow: 1,
              fontWeight: 'bold',
              color: theme.palette.primary.main
            }}
          >
            EVIL2ROOT Trading Bot
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button component={RouterLink} to="/pricing" color="primary">
              Tarifs
            </Button>
            <Button component={RouterLink} to="/about" color="primary">
              À propos
            </Button>
            <Button component={RouterLink} to="/contact" color="primary">
              Contact
            </Button>
            {!isAuthenticated ? (
              <>
                <Button 
                  component={RouterLink} 
                  to="/login" 
                  color="primary" 
                  variant="outlined"
                  sx={{ display: { xs: 'none', sm: 'block' } }}
                >
                  Connexion
                </Button>
                <Button 
                  component={RouterLink} 
                  to="/register" 
                  color="primary" 
                  variant="contained"
                >
                  S'inscrire
                </Button>
              </>
            ) : (
              <Button 
                component={RouterLink} 
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

      {/* Hero section */}
      <Box
        sx={{
          pt: 8,
          pb: 6,
          backgroundColor: theme.palette.primary.main,
          color: 'white',
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography
                component="h1"
                variant="h2"
                color="inherit"
                gutterBottom
                sx={{ fontWeight: 'bold' }}
              >
                Trading automatisé
                <br />
                propulsé par l'IA
              </Typography>
              <Typography variant="h5" color="inherit" paragraph>
                Découvrez la puissance du trading algorithmique avec notre bot alimenté par l'intelligence artificielle. Performances optimisées, risques maîtrisés.
              </Typography>
              <Box sx={{ mt: 4 }}>
                <Button
                  component={RouterLink}
                  to="/register"
                  variant="contained"
                  color="secondary"
                  size="large"
                  sx={{ mr: 2, fontWeight: 'bold', px: 4, py: 1.5 }}
                >
                  Démarrer maintenant
                </Button>
                <Button
                  component={RouterLink}
                  to="/about"
                  variant="outlined"
                  color="inherit"
                  size="large"
                  sx={{ fontWeight: 'bold', px: 4, py: 1.5 }}
                >
                  En savoir plus
                </Button>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box
                component="img"
                src="/static/images/trading-dashboard.png"
                alt="Trading Dashboard"
                sx={{
                  width: '100%',
                  maxWidth: 500,
                  height: 'auto',
                  borderRadius: 2,
                  boxShadow: 3,
                  display: { xs: 'none', md: 'block' },
                  mx: 'auto',
                }}
              />
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Features section */}
      <Container sx={{ py: 8 }} maxWidth="lg">
        <Typography 
          component="h2" 
          variant="h3" 
          align="center" 
          color="textPrimary" 
          gutterBottom
          sx={{ mb: 6, fontWeight: 'bold' }}
        >
          Fonctionnalités principales
        </Typography>
        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item key={index} xs={12} sm={6} md={4}>
              <Card 
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column',
                  transition: 'transform 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: 6,
                  },
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ mb: 2, display: 'flex', justifyContent: 'center' }}>
                    {feature.icon}
                  </Box>
                  <Typography gutterBottom variant="h5" component="h3" align="center" fontWeight="bold">
                    {feature.title}
                  </Typography>
                  <Typography align="center">
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* Plans/Pricing section preview */}
      <Box sx={{ py: 8, backgroundColor: theme.palette.grey[100] }}>
        <Container maxWidth="lg">
          <Typography 
            component="h2" 
            variant="h3" 
            align="center" 
            color="textPrimary" 
            gutterBottom
            sx={{ mb: 3, fontWeight: 'bold' }}
          >
            Plans tarifaires
          </Typography>
          <Typography 
            variant="h6" 
            align="center" 
            color="textSecondary" 
            paragraph
            sx={{ mb: 6 }}
          >
            Choisissez l'offre qui correspond à vos besoins de trading
          </Typography>
          <Grid container spacing={4} alignItems="flex-end">
            {plans.map((plan, index) => (
              <Grid 
                item 
                key={index} 
                xs={12} 
                sm={plan.highlighted ? 12 : 6} 
                md={4}
              >
                <Card 
                  sx={{ 
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    borderRadius: 2,
                    boxShadow: plan.highlighted ? 10 : 2,
                    transform: plan.highlighted ? 'scale(1.05)' : 'none',
                    position: 'relative',
                    overflow: 'visible',
                    transition: 'transform 0.3s ease',
                    '&:hover': {
                      transform: plan.highlighted ? 'scale(1.08)' : 'scale(1.03)',
                    },
                    zIndex: plan.highlighted ? 1 : 0,
                  }}
                >
                  {plan.highlighted && (
                    <Box 
                      sx={{
                        position: 'absolute',
                        top: -20,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        backgroundColor: theme.palette.secondary.main,
                        color: 'white',
                        py: 1,
                        px: 3,
                        borderRadius: 5,
                        fontWeight: 'bold',
                      }}
                    >
                      RECOMMANDÉ
                    </Box>
                  )}
                  <CardContent sx={{ flexGrow: 1, pt: plan.highlighted ? 4 : 2 }}>
                    <Typography gutterBottom variant="h5" component="h2" align="center" fontWeight="bold">
                      {plan.title}
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'baseline', mb: 2 }}>
                      <Typography component="h3" variant="h3" color="textPrimary" fontWeight="bold">
                        {plan.price}
                      </Typography>
                      <Typography variant="h6" color="textSecondary" sx={{ ml: 1 }}>
                        {plan.period}
                      </Typography>
                    </Box>
                    <Divider sx={{ my: 2 }} />
                    {plan.features.map((feature, featureIndex) => (
                      <Typography 
                        key={featureIndex} 
                        component="li" 
                        variant="subtitle1" 
                        align="center"
                        sx={{ 
                          listStyle: 'none', 
                          py: 0.5,
                          fontWeight: featureIndex === 0 ? 'bold' : 'normal'
                        }}
                      >
                        {feature}
                      </Typography>
                    ))}
                  </CardContent>
                  <CardActions sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
                    <Button 
                      component={RouterLink} 
                      to="/register" 
                      fullWidth 
                      variant={plan.buttonVariant as "outlined" | "contained"}
                      color={plan.highlighted ? "secondary" : "primary"}
                      size="large"
                      sx={{ fontWeight: 'bold' }}
                    >
                      {plan.buttonText}
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
          <Box sx={{ textAlign: 'center', mt: 5 }}>
            <Button 
              component={RouterLink} 
              to="/pricing" 
              variant="outlined" 
              color="primary" 
              size="large"
              sx={{ fontWeight: 'bold', px: 4, py: 1.5 }}
            >
              Voir tous les détails des plans
            </Button>
          </Box>
        </Container>
      </Box>

      {/* Call to action */}
      <Box
        sx={{
          py: 6,
          backgroundColor: theme.palette.background.paper,
        }}
      >
        <Container maxWidth="md">
          <Paper elevation={3} sx={{ p: 5, borderRadius: 3 }}>
            <Typography 
              variant="h4" 
              align="center" 
              color="textPrimary" 
              gutterBottom
              sx={{ fontWeight: 'bold' }}
            >
              Prêt à révolutionner votre trading ?
            </Typography>
            <Typography variant="h6" align="center" color="textSecondary" paragraph>
              Rejoignez des milliers de traders qui utilisent déjà notre bot pour atteindre leurs objectifs financiers.
            </Typography>
            <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
              <Button
                component={RouterLink}
                to="/register"
                variant="contained"
                color="primary"
                size="large"
                sx={{ px: 5, py: 1.5, fontWeight: 'bold' }}
              >
                Créer un compte gratuit
              </Button>
            </Box>
          </Paper>
        </Container>
      </Box>

      {/* Footer */}
      <Box 
        component="footer" 
        sx={{ 
          p: 6, 
          backgroundColor: theme.palette.grey[900],
          color: 'white'
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4}>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="h6" gutterBottom fontWeight="bold">
                EVIL2ROOT Trading Bot
              </Typography>
              <Typography variant="body2" color="inherit">
                Une solution complète de trading algorithmique alimentée par l'intelligence artificielle pour maximiser vos performances sur les marchés financiers.
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="h6" gutterBottom fontWeight="bold">
                Navigation
              </Typography>
              <Box component="ul" sx={{ p: 0, m: 0, listStyle: 'none' }}>
                {[
                  { text: 'Accueil', link: '/' },
                  { text: 'Tarifs', link: '/pricing' },
                  { text: 'À propos', link: '/about' },
                  { text: 'Contact', link: '/contact' },
                ].map((item) => (
                  <li key={item.text}>
                    <Button
                      component={RouterLink}
                      to={item.link}
                      color="inherit"
                      sx={{ pl: 0, textAlign: 'left', textTransform: 'none' }}
                    >
                      {item.text}
                    </Button>
                  </li>
                ))}
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="h6" gutterBottom fontWeight="bold">
                Légal
              </Typography>
              <Box component="ul" sx={{ p: 0, m: 0, listStyle: 'none' }}>
                {[
                  { text: 'Conditions d\'utilisation', link: '#' },
                  { text: 'Politique de confidentialité', link: '#' },
                  { text: 'Mentions légales', link: '#' },
                  { text: 'Risques et avertissements', link: '#' },
                ].map((item) => (
                  <li key={item.text}>
                    <Button
                      component={RouterLink}
                      to={item.link}
                      color="inherit"
                      sx={{ pl: 0, textAlign: 'left', textTransform: 'none' }}
                    >
                      {item.text}
                    </Button>
                  </li>
                ))}
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="h6" gutterBottom fontWeight="bold">
                Contact
              </Typography>
              <Typography variant="body2" paragraph color="inherit">
                Email: contact@evil2root.com
              </Typography>
              <Typography variant="body2" paragraph color="inherit">
                Tel: +33 1 23 45 67 89
              </Typography>
              <Typography variant="body2" paragraph color="inherit">
                75008 Paris, France
              </Typography>
            </Grid>
          </Grid>
          <Box sx={{ mt: 5 }}>
            <Divider sx={{ backgroundColor: theme.palette.grey[700] }} />
            <Typography variant="body2" color="inherit" align="center" sx={{ mt: 2 }}>
              © {new Date().getFullYear()} EVIL2ROOT Trading Bot. Tous droits réservés.
            </Typography>
          </Box>
        </Container>
      </Box>
    </Box>
  );
};

export default LandingPage; 