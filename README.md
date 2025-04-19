# Fall Detection Cloud App

This is a cloud-ready fall detection system that supports:

✅ Real-time camera stream input  
✅ Roboflow object detection integration  
✅ Webhook alert system  
✅ Streamlit UI  
✅ Cloud settings saved to file  

## Setup & Run

### Requirements
- Python 3.9+
- Streamlit
- Roboflow API key
- Optional: RTSP-compatible camera

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

### Xano Integration

To connect to Xano:
1. Set up a Xano workspace with `auth` and `settings` tables.
2. Replace `settings.json` logic with API calls:
    - POST `/auth/login` and `/auth/signup`
    - GET/POST `/settings`

### Deployment

Use [Streamlit Community Cloud](https://streamlit.io/cloud) for free and easy deployment.
Just upload the project and set up required secrets.
