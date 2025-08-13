import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Box,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Chip,
  Divider,
} from '@mui/material';
import { useMutation } from 'react-query';
import axios from 'axios';

interface PredictionRequest {
  team1: string;
  team2: string;
  venue: string;
  city: string;
  team1_win_percentage: number;
  team2_win_percentage: number;
  team1_recent_form?: number;
  team2_recent_form?: number;
  team1_head_to_head?: number;
  team2_head_to_head?: number;
}

interface PredictionResponse {
  prediction: number;
  probability: number;
  confidence: string;
  features_used: string[];
}

const IPL_TEAMS = [
  'Mumbai Indians',
  'Chennai Super Kings',
  'Royal Challengers Bangalore',
  'Kolkata Knight Riders',
  'Delhi Capitals',
  'Punjab Kings',
  'Rajasthan Royals',
  'Sunrisers Hyderabad',
  'Gujarat Lions',
  'Rising Pune Supergiants',
];

const VENUES = [
  'Wankhede Stadium',
  'M. Chinnaswamy Stadium',
  'Eden Gardens',
  'Arun Jaitley Stadium',
  'Punjab Cricket Association Stadium',
  'Sawai Mansingh Stadium',
  'Rajiv Gandhi International Stadium',
  'Saurashtra Cricket Association Stadium',
];

const CITIES = [
  'Mumbai',
  'Bangalore',
  'Kolkata',
  'Delhi',
  'Mohali',
  'Jaipur',
  'Hyderabad',
  'Rajkot',
  'Pune',
  'Ahmedabad',
];

const PredictionForm: React.FC = () => {
  const [formData, setFormData] = useState<PredictionRequest>({
    team1: '',
    team2: '',
    venue: '',
    city: '',
    team1_win_percentage: 0.5,
    team2_win_percentage: 0.5,
    team1_recent_form: 0.5,
    team2_recent_form: 0.5,
    team1_head_to_head: 0.5,
    team2_head_to_head: 0.5,
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string>('');

  const predictionMutation = useMutation(
    (data: PredictionRequest) => {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      return axios.post<PredictionResponse>(`${apiUrl}/predict`, data);
    },
    {
      onSuccess: (response) => {
        setPrediction(response.data);
        setError('');
      },
      onError: (error: any) => {
        setError(error.response?.data?.detail || 'Prediction failed');
        setPrediction(null);
      },
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (formData.team1 === formData.team2) {
      setError('Please select different teams');
      return;
    }
    predictionMutation.mutate(formData);
  };

  const handleInputChange = (field: keyof PredictionRequest, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const getConfidenceColor = (confidence: string) => {
    switch (confidence.toLowerCase()) {
      case 'high':
        return 'success';
      case 'medium':
        return 'warning';
      case 'low':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        IPL Win Predictor
      </Typography>
      <Typography variant="body1" color="text.secondary" align="center" gutterBottom>
        Predict the outcome of IPL matches using our advanced machine learning model
      </Typography>

      <Paper elevation={3} sx={{ p: 4, mt: 3 }}>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            {/* Team Selection */}
            <Grid item xs={12} md={6}>
              <TextField
                select
                fullWidth
                label="Team 1"
                value={formData.team1}
                onChange={(e) => handleInputChange('team1', e.target.value)}
                required
              >
                {IPL_TEAMS.map((team) => (
                  <option key={team} value={team}>
                    {team}
                  </option>
                ))}
              </TextField>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                select
                fullWidth
                label="Team 2"
                value={formData.team2}
                onChange={(e) => handleInputChange('team2', e.target.value)}
                required
              >
                {IPL_TEAMS.map((team) => (
                  <option key={team} value={team}>
                    {team}
                  </option>
                ))}
              </TextField>
            </Grid>

            {/* Venue and City */}
            <Grid item xs={12} md={6}>
              <TextField
                select
                fullWidth
                label="Venue"
                value={formData.venue}
                onChange={(e) => handleInputChange('venue', e.target.value)}
                required
              >
                {VENUES.map((venue) => (
                  <option key={venue} value={venue}>
                    {venue}
                  </option>
                ))}
              </TextField>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                select
                fullWidth
                label="City"
                value={formData.city}
                onChange={(e) => handleInputChange('city', e.target.value)}
                required
              >
                {CITIES.map((city) => (
                  <option key={city} value={city}>
                    {city}
                  </option>
                ))}
              </TextField>
            </Grid>

            {/* Team Statistics */}
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Team Statistics (0.0 - 1.0)
              </Typography>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label={`${formData.team1} Win Percentage`}
                type="number"
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                value={formData.team1_win_percentage}
                onChange={(e) => handleInputChange('team1_win_percentage', parseFloat(e.target.value))}
                required
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label={`${formData.team2} Win Percentage`}
                type="number"
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                value={formData.team2_win_percentage}
                onChange={(e) => handleInputChange('team2_win_percentage', parseFloat(e.target.value))}
                required
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label={`${formData.team1} Recent Form`}
                type="number"
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                value={formData.team1_recent_form}
                onChange={(e) => handleInputChange('team1_recent_form', parseFloat(e.target.value))}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label={`${formData.team2} Recent Form`}
                type="number"
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                value={formData.team2_recent_form}
                onChange={(e) => handleInputChange('team2_recent_form', parseFloat(e.target.value))}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label={`${formData.team1} Head-to-Head`}
                type="number"
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                value={formData.team1_head_to_head}
                onChange={(e) => handleInputChange('team1_head_to_head', parseFloat(e.target.value))}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label={`${formData.team2} Head-to-Head`}
                type="number"
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                value={formData.team2_head_to_head}
                onChange={(e) => handleInputChange('team2_head_to_head', parseFloat(e.target.value))}
              />
            </Grid>

            {/* Submit Button */}
            <Grid item xs={12}>
              <Box display="flex" justifyContent="center">
                <Button
                  type="submit"
                  variant="contained"
                  size="large"
                  disabled={predictionMutation.isLoading}
                  sx={{ minWidth: 200 }}
                >
                  {predictionMutation.isLoading ? (
                    <CircularProgress size={24} />
                  ) : (
                    'Predict Winner'
                  )}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </form>

        {/* Error Display */}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {/* Prediction Result */}
        {prediction && (
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Prediction Result
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" color="primary">
                    Predicted Winner: {prediction.prediction === 1 ? formData.team1 : formData.team2}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Confidence: {prediction.probability.toFixed(2)} ({prediction.probability * 100}%)
                  </Typography>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Typography variant="body2">Confidence Level:</Typography>
                    <Chip
                      label={prediction.confidence}
                      color={getConfidenceColor(prediction.confidence) as any}
                      size="small"
                    />
                  </Box>
                </Grid>
              </Grid>

              <Box mt={2}>
                <Typography variant="body2" color="text.secondary">
                  Features used: {prediction.features_used.join(', ')}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        )}
      </Paper>
    </Container>
  );
};

export default PredictionForm;