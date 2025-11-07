# Earth Engine Authentication Guide

This guide explains how to authenticate with Google Earth Engine in the EpistemX platform.

## Overview

The EpistemX platform provides two authentication methods:

1. **Personal Account (OAuth)** - Use your personal Google account
2. **Service Account** - Use a service account JSON key file

## Authentication Page

Access the authentication page by navigating to **"Earth Engine Authentication"** in the sidebar of the Streamlit app.

## Method 1: Personal Account Authentication

### Prerequisites
- Google account with Earth Engine access
- Account registered for Earth Engine (commercial or non-commercial use)
- Visit [Google Earth Engine](https://earthengine.google.com/) to register

### Steps

#### Option A: Terminal Authentication (Recommended)
1. Open a terminal or command prompt
2. Activate your Python environment
3. Run:
   ```python
   import ee
   ee.Authenticate()
   ee.Initialize()
   ```
4. Follow the browser prompts to authenticate
5. Restart the Streamlit application

#### Option B: Python Script Authentication
1. Create a Python script with:
   ```python
   from epistemx import authenticate_manually
   authenticate_manually()
   ```
2. Run the script
3. Follow the authentication flow
4. Restart the Streamlit application

### Troubleshooting
- **"Not authenticated" error**: Complete the authentication flow first
- **"No project" error**: Specify your Google Cloud Project ID
- **"Access denied" error**: Ensure your account is registered for Earth Engine

## Method 2: Service Account Authentication

### Prerequisites
- Google Cloud Project with Earth Engine API enabled
- Service account created with Earth Engine permissions
- Service account JSON key file downloaded

### Steps to Create Service Account

1. **Create Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Note your Project ID

2. **Enable Earth Engine API**
   - Navigate to "APIs & Services" > "Library"
   - Search for "Earth Engine API"
   - Click "Enable"

3. **Create Service Account**
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Enter name and description
   - Click "Create and Continue"

4. **Grant Permissions**
   - Add role "Earth Engine Resource Admin"
   - Click "Continue" then "Done"

5. **Create and Download Key**
   - Click on the service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose "JSON" format
   - Click "Create" (file downloads automatically)

6. **Register with Earth Engine**
   - Go to [Earth Engine Code Editor](https://code.earthengine.google.com/)
   - Register your service account email (from JSON file)

### Using Service Account in EpistemX

The authentication page provides three options:

#### Option 1: Upload JSON File
1. Click "Upload JSON File" tab
2. Upload your service account JSON key file
3. Optionally enter Project ID
4. Click "Authenticate with Uploaded File"

#### Option 2: Paste JSON Content
1. Click "Paste JSON Content" tab
2. Copy entire content of JSON file
3. Paste into text area
4. Optionally enter Project ID
5. Click "Authenticate with JSON Content"

#### Option 3: Use Existing File
1. Place JSON file in `auth/` directory
2. Click "Use Existing File" tab
3. Select your file from dropdown
4. Optionally enter Project ID
5. Click "Authenticate with Existing File"

## Environment Variables

For automated deployment, you can use environment variables:

### Service Account JSON (Base64 Encoded) - Recommended
```bash
export GOOGLE_SERVICE_ACCOUNT_JSON_B64="<base64-encoded-json>"
```

To create Base64 encoded JSON:
```bash
# Linux/Mac
base64 -w 0 service-account.json

# Windows PowerShell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("service-account.json"))
```

### Service Account JSON (Direct)
```bash
export GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
```

### Service Account File Path
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

## Verification

After authentication, the status indicator will show:
- ✅ **Green**: Earth Engine authenticated and ready
- ⚠️ **Yellow**: Authentication required

## Security Best Practices

1. **Never commit service account keys to version control**
2. **Use environment variables for production deployments**
3. **Rotate service account keys regularly**
4. **Use least-privilege permissions**
5. **Store keys securely (e.g., secret managers)**

## Additional Resources

- [Google Earth Engine Homepage](https://earthengine.google.com/)
- [Earth Engine Python Installation Guide](https://developers.google.com/earth-engine/guides/python_install)
- [Service Account Documentation](https://developers.google.com/earth-engine/guides/service_account)
- [Google Cloud Console](https://console.cloud.google.com/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Earth Engine documentation
3. Contact your system administrator
