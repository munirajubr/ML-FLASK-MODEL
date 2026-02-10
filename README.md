# AgriVision Application

> A lightweight web application for crop recommendation and plant disease detection using machine learning models.

## Features

- Crop recommendation based on input soil and environmental parameters.
- Plant disease detection from images using a trained model.
- Simple Flask web interface with templates and static assets.

## Repo Structure

- `app.py` - Main Flask app (run the web server)
- `newapp.py`, `app copy.py` - alternate/dev versions of the app
- `crop_model.py` - crop recommendation model utilities
- `disease_model.py` - plant disease detection model utilities
- `crop_details.py`, `crop_model.py` - crop data and helper routines
- `Dataset/` - CSV datasets used by the models (eggplant CSVs included)
- `templates/` - HTML templates (`index.html`, `crop.html`, `disease.html`)
- `static/` - Static files like `style.css`

## Setup

1. Create a Python virtual environment (recommended):

```
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

## Run

Start the Flask app (example):

```
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

If your project uses a different entrypoint (for example `newapp.py`), run that file instead.

## Dataset

The `Dataset/` folder contains CSV files used by the models, including:

- `eggplant_details.csv`
- `eggplant_diseases.csv`
- `eggplant_varieties.csv`

## Team

- [Muniraju B R](https://github.com/munirajubr)
- [Harshita Sakaray](https://github.com/harshita-sakaray26)
- [Gunashree H M](https://github.com/gunashree-hm)
- [Disha S Kashipati](https://github.com/Disha-S-Kashipati)

## Notes

- This README is a concise overview. If you'd like, I can expand sections (example: API endpoints, detailed model training steps, or add a LICENSE file).
