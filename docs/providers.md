# Providers and data sources

## Sentinel-1
- CDSE provider abstraction
- Requires CDSE credentials when live queries are used
- Search and metadata access use the official interfaces in a provider-aware way

## Ocean current data
- Copernicus Marine adapter
- Requires credentials for live retrieval

## Weather data
- Open-Meteo adapter
- Demo-safe fallback is available

## AIS
- Historical AIS provider abstraction
- Mock provider for demo/testing
