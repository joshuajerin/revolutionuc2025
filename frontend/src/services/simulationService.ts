import axios from 'axios';

interface SimulationResponse {
  image: string;
  molecularStructure: string;
}

const API_URL = 'http://localhost:3001';

export const runSimulation = async (prompt: string): Promise<SimulationResponse> => {
  try {
    const response = await axios.post<SimulationResponse>(`${API_URL}/api/simulate`, { prompt });
    return response.data;
  } catch (error) {
    console.error('Error running simulation:', error);
    throw new Error('Failed to run simulation. Please try again.');
  }
}; 