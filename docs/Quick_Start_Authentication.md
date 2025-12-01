# Quick Start: Authentication Setup

## 🚀 5-Minute Setup Guide

### Step 1: Earth Engine Authentication (Required)

**Option A: Service Account (Recommended for Production)**

1. Download service account JSON from Google Cloud Console
2. Save as `auth/service-account.json`
3. In EpistemX, go to "Earth Engine Authentication"
4. Upload the JSON file
5. Click "Authenticate"

**Option B: Personal Account (For Development)**

1. Run in terminal:
```python
import ee
ee.Authenticate()
ee.Initialize()
```
2. Follow browser prompts
3. Restart EpistemX

### Step 2: Google Drive Export (Optional)

**Only needed if you want to export results to Drive**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable "Google Drive API"
3. Create OAuth 2.0 Client ID (Web application)
4. Add redirect URI: `http://localhost:8501`
5. Download JSON as `secrets/oauth_client_secret.json`
6. In EpistemX, go to "Google Drive Export Setup"
7. Click "Sign in with Google"
8. Authorize the application

---

## 📋 Checklist

### Earth Engine Setup
- [ ] Google Cloud Project created
- [ ] Earth Engine API enabled
- [ ] Service account created (or personal account registered)
- [ ] Service account JSON downloaded
- [ ] Authenticated in EpistemX
- [ ] ✅ Status shows "Earth Engine is authenticated and ready"

### Drive Export Setup (Optional)
- [ ] Google Drive API enabled
- [ ] OAuth 2.0 credentials created
- [ ] Redirect URI configured
- [ ] OAuth JSON downloaded
- [ ] Authenticated in EpistemX
- [ ] ✅ Status shows "Authenticated as: your-email@gmail.com"

---

## 🔧 File Structure

```
your-project/
├── auth/                          # Service account files
│   └── service-account.json       # Earth Engine service account
├── secrets/                       # OAuth credentials
│   └── oauth_client_secret.json   # Drive OAuth credentials
├── .streamlit/
│   └── secrets.toml              # Streamlit Cloud secrets (optional)
└── .gitignore                    # MUST include auth/ and secrets/
```

---

## 🔒 Security Checklist

- [ ] Added `auth/` to `.gitignore`
- [ ] Added `secrets/` to `.gitignore`
- [ ] Never committed credentials to Git
- [ ] Using environment variables in production
- [ ] OAuth redirect URIs match deployment URL

---

## 🧪 Quick Test

### Test Earth Engine
```python
from epistemx import is_ee_initialized
import ee

if is_ee_initialized():
    print("✅ Earth Engine ready")
    image = ee.Image('USGS/SRTMGL1_003')
    print(f"Bands: {image.bandNames().getInfo()}")
else:
    print("❌ Earth Engine not authenticated")
```

### Test Drive Export
```python
import streamlit as st
from epistemx import DriveHelper, DriveAuthManager

if 'drive_credentials' in st.session_state:
    print("✅ Drive authenticated")
    creds = DriveAuthManager.dict_to_credentials(
        st.session_state['drive_credentials']
    )
    helper = DriveHelper(creds)
    folder_id = helper.create_folder('Test')
    print(f"Created folder: {folder_id}")
else:
    print("❌ Drive not authenticated")
```

---

## 🆘 Common Issues

### "Earth Engine not authenticated"
→ Go to "Earth Engine Authentication" page and authenticate

### "OAuth2 client secrets file not found"
→ Create `secrets/oauth_client_secret.json` with your OAuth credentials

### "Redirect URI mismatch"
→ Check that redirect URI in Google Cloud Console matches your app URL

### "Permission denied"
→ Ensure service account has Earth Engine permissions
→ Ensure Drive API is enabled for OAuth

---

## 📚 Next Steps

1. ✅ Complete authentication setup
2. 📖 Read [Drive Export Authentication Guide](Drive_Export_Authentication.md)
3. 🧪 Try the example workflows
4. 🚀 Start processing Earth Engine data!

---

## 🔗 Quick Links

- [Google Cloud Console](https://console.cloud.google.com/)
- [Earth Engine Code Editor](https://code.earthengine.google.com/)
- [Manage OAuth Permissions](https://myaccount.google.com/permissions)
- [Full Documentation](Drive_Export_Authentication.md)
