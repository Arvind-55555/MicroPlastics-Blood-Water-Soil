# Microplastic Detection & Classification ML Pipeline

A production-ready machine learning pipeline for detecting, classifying, and estimating microplastic concentrations in Blood, Water, and Soil samples.

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw ingested data
│   ├── processed/        # Cleaned and preprocessed data
│   └── sample_data.json  # Sample data for dashboard
├── models/
│   ├── checkpoints/      # Model checkpoints
│   └── saved/            # Final trained models
├── src/
│   ├── ingestion/        # Data acquisition modules
│   ├── preprocessing/    # Data cleaning and transformation
│   ├── models/           # Model architectures and training
│   ├── api/              # FastAPI prediction endpoints
│   ├── monitoring/       # Monitoring and alerting
│   └── utils/            # Utility functions
├── dashboard/            # Streamlit dashboard
├── docs/                 # GitHub Pages dashboard
│   ├── index.html       # Main dashboard page
│   └── dashboard.js     # Dashboard JavaScript
├── notebooks/            # Jupyter notebooks for EDA
├── config/               # Configuration files
└── scripts/              # Utility scripts

```

## Features

- **Real-time Data Ingestion**: Automated discovery and ingestion from multiple sources
- **Multimodal Models**: Spectra (FTIR/Raman), Images (microscopy), and Tabular data
- **Production API**: FastAPI endpoints for predictions
- **Monitoring**: Real-time anomaly detection and alerting
- **Web Dashboard**: Interactive visualizations (Streamlit + GitHub Pages)

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

3. Run data ingestion:
```bash
python -m src.ingestion.main
```

4. Train models:
```bash
python -m src.models.train
```

5. Start API server:
```bash
uvicorn src.api.main:app --reload
```

6. Launch Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

## GitHub Pages Dashboard

The dashboard is available at: https://arvind-55555.github.io/MicroPlastics-Blood-Water-Soil

## License

MIT
