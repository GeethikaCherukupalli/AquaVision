import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const runFullInvestigation = async (lat = 28.932, lon = -88.974, date = null) => {
  let url = `/api/v1/pipeline/full-investigation?lat=${lat}&lon=${lon}`;
  if (date) {
    url += `&date=${date}`;
  }
  const response = await apiClient.get(url);
  return response.data;
};