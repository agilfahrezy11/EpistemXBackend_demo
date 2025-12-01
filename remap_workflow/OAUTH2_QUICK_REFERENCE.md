# OAuth2 Quick Reference - REMAP

## The Big Picture

REMAP uses OAuth2 to let users export GeoTIFF files to **their own Google Drive**. This requires user permission, which is what OAuth2 provides.

---

## The 3 Main Components

### 1. **Frontend (Vue.js)** - `app/src/components/Export.vue`
- Loads Google's JavaScript API
- Shows "Sign in with Google" button
- Opens OAuth consent popup
- Gets authorization code from Google
- Sends code to backend
- Polls for export completion

### 2. **Backend (Python)** - `backend/main/auth.py`
- Receives authorization code
- Exchanges code for credentials (access token + refresh token)
- Stores credentials in session
- Uses credentials to access user's Drive
- Manages token refresh automatically

### 3. **Session Storage** - `backend/main/shared.py`
- Stores credentials server-side (secure)
- Automatically refreshes expired tokens
- Persists across requests using GAE Datastore

---

## The Flow in 5 Steps

```
1. User clicks "Sign in with Google"
   └─> Opens Google consent popup

2. User approves access
   └─> Google returns authorization code

3. Frontend sends code to backend
   └─> Backend exchanges code for tokens

4. Backend stores tokens in session
   └─> User can now export to Drive

5. Export uses stored tokens
   └─> Files appear in user's Drive
```

---

## Key Code Locations

### Frontend: Initialize Google Auth

**File:** `app/src/components/Export.vue`

```javascript
// Load Google API and initialize
mounted() {
  var script = document.createElement('script')
  script.src = 'https://apis.google.com/js/api:client.js'
  script.onload = () => this.auth()
  document.head.appendChild(script)
}

auth() {
  window.gapi.load('auth2', () => {
    const g = window.gapi.auth2.init({
      client_id: 'YOUR_CLIENT_ID.apps.googleusercontent.com'
    })
    this.setGoogleObject(g)
  })
}
```

### Frontend: Sign In

```javascript
signInClick() {
  window.gapi.auth2.getAuthInstance().grantOfflineAccess()
    .then(authResult => {
      // Send code to backend
      this.$http.post('/oauth2callback', authResult)
        .then(() => this.signIn())
    })
}
```

### Backend: Exchange Code for Tokens

**File:** `backend/main/auth.py`

```python
class OAuth(BaseHandler):
    def post(self):
        code = json.loads(self.request.body)['code']
        
        # Exchange code for credentials
        credentials = oauth2client.client.credentials_from_clientsecrets_and_code(
            filename=config.OAUTH_FILE,
            code=code,
            scope='https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/userinfo.email'
        )
        
        # Get user info
        http = credentials.authorize(httplib2.Http())
        ui = build('plus', 'v1', http=http)
        userinfo = ui.people().get(userId='me').execute(http=http)
        
        # Store in session
        self.session['email'] = userinfo['emails'][0]['value']
        self.session['credentials'] = credentials.to_json()
```

### Backend: Use Credentials for Drive Export

```python
class ExportWorker(GetMapData):
    def post(self):
        # ... classification code ...
        
        # Recreate credentials from session
        credentials = oauth2client.client.OAuth2Credentials.from_json(
            self.request.get('credentials')
        )
        
        # Create Drive helper with user credentials
        user_drive_helper = drive.DriveHelper(credentials)
        
        # Create folder in user's Drive
        folder = user_drive_helper.CreateFolder('REMAP Export Folder')
        
        # Copy files to user's Drive
        user_drive_helper.CopyFile(file_id, 'export.tif', folder)
```

### Backend: Automatic Token Refresh

**File:** `backend/main/shared.py`

```python
class BaseHandler(webapp2.RequestHandler):
    def dispatch(self):
        if 'credentials' in self.session:
            credentials = OAuth2Credentials.from_json(self.session['credentials'])
            
            # Refresh if expired
            if credentials.access_token_expired:
                credentials.refresh(httplib2.Http())
                self.session['credentials'] = credentials.to_json()
        
        # Process request
        webapp2.RequestHandler.dispatch(self)
```

---

## What Gets Stored in Session

```python
self.session['email']        # "user@example.com"
self.session['user_id']      # "1234567890"
self.session['credentials']  # JSON string containing:
                             # - access_token (expires in 1 hour)
                             # - refresh_token (long-lived)
                             # - token_expiry
                             # - scopes
```

---

## Two Types of Drive Access

### Service Account (App's Drive)
```python
# Used for temporary storage
APP_CREDENTIALS = ServiceAccountCredentials.from_json_keyfile_name(
    'secrets/gee_service_account_secrets.json',
    scopes='https://www.googleapis.com/auth/drive'
)
APP_DRIVE_HELPER = drive.DriveHelper(APP_CREDENTIALS)
```

### User OAuth2 (User's Drive)
```python
# Used for final export location
credentials = OAuth2Credentials.from_json(self.session['credentials'])
user_drive_helper = drive.DriveHelper(credentials)
```

---

## The Export Process

```
1. User clicks "Drive export GeoTIFF"
   │
2. Backend queues background task
   │  └─ Stores credentials in task params
   │
3. Background worker runs
   │  ├─ Classifies region
   │  ├─ Exports to temp Drive (service account)
   │  ├─ Waits for EE export
   │  ├─ Grants user access to temp files
   │  ├─ Uses user credentials to:
   │  │  ├─ Create folder in user's Drive
   │  │  ├─ Copy GeoTIFF to user's folder
   │  │  └─ Create metadata CSV
   │  └─ Deletes temp files
   │
4. Frontend polls for status
   │  └─ Shows "Completed!" when done
   │
5. Files appear in user's Drive
```

---

## Required Setup Files

### 1. OAuth Client Credentials
**File:** `secrets/oauth_secret.json`

```json
{
  "web": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "client_secret": "YOUR_CLIENT_SECRET",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "redirect_uris": ["http://localhost:8080/oauth2callback"]
  }
}
```

**How to get:**
1. Go to Google Cloud Console
2. Create project
3. Enable Drive API
4. Create OAuth 2.0 Client ID (Web application)
5. Download JSON

### 2. Service Account Credentials
**File:** `secrets/gee_service_account_secrets.json`

```json
{
  "type": "service_account",
  "project_id": "your-project",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...",
  "client_email": "your-service-account@your-project.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token"
}
```

**How to get:**
1. Go to Google Cloud Console
2. Create service account
3. Grant Earth Engine and Drive permissions
4. Create key (JSON)
5. Download JSON

### 3. Session Secret
**File:** `secrets/wsgi.txt`

```
your-random-secret-key-here
```

**How to generate:**
```python
import os
import base64
print(base64.b64encode(os.urandom(64)).decode('utf-8'))
```

---

## Common Questions

### Q: Why "offline access"?
**A:** To get a refresh token that allows the backend to access Drive even when the user is not actively logged in. Essential for long-running export tasks.

### Q: Where are credentials stored?
**A:** Server-side in Google App Engine Datastore (encrypted). Never sent to frontend.

### Q: What if the access token expires?
**A:** Automatically refreshed using the refresh token in `BaseHandler.dispatch()`.

### Q: What permissions does the app request?
**A:** Only Drive access and email address. Shown in consent screen.

### Q: Can users revoke access?
**A:** Yes, at any time via Google Account settings. App will detect this and require re-authentication.

### Q: Why two Drive helpers?
**A:** 
- **APP_DRIVE_HELPER** (service account) - For temp storage and cleanup
- **user_drive_helper** (OAuth2) - For accessing user's personal Drive

---

## Debugging Tips

### Check if user is signed in:
```javascript
// Frontend
console.log(this.$store.getters.isSignedIn)
```

```python
# Backend
if 'credentials' in self.session:
    print("User is authenticated")
    print("Email:", self.session['email'])
```

### Check token expiry:
```python
credentials = OAuth2Credentials.from_json(self.session['credentials'])
print("Expired:", credentials.access_token_expired)
print("Expiry:", credentials.token_expiry)
```

### Check export status:
```javascript
// Frontend
this.$http.get('/api/exportstatus').then(response => {
  console.log("Status:", response.body)
})
```

### View session contents:
```python
# Backend
print("Session keys:", self.session.keys())
print("Email:", self.session.get('email'))
```

---

## Security Best Practices

✅ **DO:**
- Store credentials server-side only
- Use HTTPS for all OAuth flows
- Request minimal scopes needed
- Refresh tokens automatically
- Clear session on sign out

❌ **DON'T:**
- Send credentials to frontend
- Store tokens in localStorage
- Request unnecessary permissions
- Hardcode client secrets in frontend
- Keep expired sessions

---

## Summary

OAuth2 in REMAP enables:

1. ✅ User consent for Drive access
2. ✅ Secure credential storage
3. ✅ Automatic token refresh
4. ✅ Long-running background exports
5. ✅ Files delivered to user's Drive

**The key insight:** Frontend handles UI and initial auth, backend handles token management and Drive operations, session storage keeps everything secure.
