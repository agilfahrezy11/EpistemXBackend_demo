# EpistemX Authentication System

## Overview

EpistemX uses a dual authentication system to provide secure access to Google Earth Engine and enable exports to user's personal Google Drive.

## Two Authentication Methods

### 1. 🔐 Service Account (Earth Engine API)
- **Purpose:** Backend access to Earth Engine for processing
- **Type:** Application-level credentials
- **Setup:** One-time configuration
- **File:** `auth/service-account.json`

### 2. ☁️ OAuth2 (Google Drive Export)
- **Purpose:** Export results to user's personal Drive
- **Type:** User-specific credentials
- **Setup:** Each user must authorize
- **File:** `secrets/oauth_client_secret.json`

## Why Two Systems?

This separation provides:
- ✅ Reliable Earth Engine processing (service account)
- ✅ Personal Drive exports (OAuth2 per user)
- ✅ User privacy and data control
- ✅ Scalability for multiple users

## Quick Setup

### Earth Engine (Required)

```bash
# 1. Place service account JSON
mkdir -p auth
cp /path/to/your-service-account.json auth/service-account.json

# 2. Authenticate in app
# Go to: Earth Engine Authentication page
# Upload the JSON file
```

### Drive Export (Optional)

```bash
# 1. Place OAuth credentials
mkdir -p secrets
cp /path/to/oauth_client_secret.json secrets/

# 2. Authenticate in app
# Go to: Google Drive Export Setup page
# Click "Sign in with Google"
```

## File Structure

```
epistemx/
├── auth/                          # Service accounts (gitignored)
│   └── service-account.json
├── secrets/                       # OAuth credentials (gitignored)
│   └── oauth_client_secret.json
├── src/epistemx/
│   ├── ee_config.py              # Earth Engine auth
│   └── drive_utils.py            # Drive OAuth utilities
├── pages/
│   ├── 0_Earth_Engine_Authentication.py
│   └── 7_Google_Drive_Export_Setup.py
└── docs/
    ├── Authentication_Guide.md
    ├── Drive_Export_Authentication.md
    └── Quick_Start_Authentication.md
```

## Security

### ⚠️ Important
- **Never commit** `auth/` or `secrets/` directories
- **Always use** `.gitignore` to exclude credentials
- **Use environment variables** in production
- **Rotate keys** regularly

### .gitignore
```gitignore
# Authentication files
auth/
secrets/
*.json
!package.json

# Environment variables
.env
.env.local
```

## Documentation

- 📖 [Complete Authentication Guide](Authentication_Guide.md)
- 🚀 [Quick Start Guide](Quick_Start_Authentication.md)
- ☁️ [Drive Export Guide](Drive_Export_Authentication.md)
- 🔗 [Visio and Code Tags](Visio_and_code_tags.md)

## API Usage

### Earth Engine
```python
from epistemx import initialize_with_service_account, is_ee_initialized

# Initialize
initialize_with_service_account('auth/service-account.json')

# Check status
if is_ee_initialized():
    print("✅ Ready to process Earth Engine data")
```

### Drive Export
```python
from epistemx import DriveAuthManager, DriveHelper, export_to_drive_with_oauth
import ee

# Assuming user is authenticated via Streamlit UI
credentials = DriveAuthManager.dict_to_credentials(
    st.session_state['drive_credentials']
)

# Create Drive helper
drive_helper = DriveHelper(credentials)

# Export Earth Engine image
task = export_to_drive_with_oauth(
    image=ee_image,
    description='My Export',
    folder='EpistemX_Exports',
    file_name_prefix='result',
    region=region,
    scale=30
)
task.start()
```

## Troubleshooting

### Earth Engine Issues
- ❌ "Not authenticated" → Upload service account JSON
- ❌ "Permission denied" → Check service account has EE permissions
- ❌ "Project not found" → Verify project ID in JSON

### Drive Export Issues
- ❌ "OAuth file not found" → Create `secrets/oauth_client_secret.json`
- ❌ "Redirect URI mismatch" → Check Google Cloud Console settings
- ❌ "Token expired" → Sign out and sign in again

## Support

For detailed help:
1. Check the [Authentication Guide](Authentication_Guide.md)
2. Review [Quick Start](Quick_Start_Authentication.md)
3. Read [Drive Export Guide](Drive_Export_Authentication.md)
4. Check Google Cloud Console settings

## Resources

- [Google Earth Engine](https://earthengine.google.com/)
- [Google Cloud Console](https://console.cloud.google.com/)
- [Drive API Documentation](https://developers.google.com/drive/api)
- [OAuth 2.0 Guide](https://developers.google.com/identity/protocols/oauth2)
