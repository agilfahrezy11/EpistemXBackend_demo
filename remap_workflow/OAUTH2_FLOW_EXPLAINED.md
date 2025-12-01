# OAuth2 Flow in REMAP - Complete Explanation

## Overview

REMAP uses **two separate authentication mechanisms**:

1. **Service Account** (Backend) - For Earth Engine API access
2. **OAuth2 User Authentication** (Frontend + Backend) - For Google Drive access

This document focuses on the **OAuth2 flow** for Drive export functionality.

---

## Why OAuth2 is Needed

The application needs to:
- Export GeoTIFF files to the **user's personal Google Drive**
- Create folders in the user's Drive
- Copy files from a temporary location to the user's Drive

This requires **user consent** and **user-specific credentials**, which is what OAuth2 provides.

---

## The Complete OAuth2 Flow - Step by Step

### Phase 1: Frontend Initialization (When Page Loads)

**File:** `app/src/components/Export.vue`

```javascript
mounted() {
  // 1. Dynamically load Google API client library
  var script = document.createElement('script')
  script.onload = () => {
    if (!window.gapi) {
      return void (console.error('No gapi included'))
    }
    this.auth()  // Initialize auth
  }
  script.src = 'https://apis.google.com/js/api:client.js'
  script.async = true
  script.defer = true
  document.getElementsByTagName('head')[0].appendChild(script)
}
```

**What happens:**
- Loads Google's JavaScript API client library
- Once loaded, calls `auth()` method

---

### Phase 2: Initialize Google Auth2 Client

**File:** `app/src/components/Export.vue`

```javascript
auth() {
  window.gapi.load('auth2', () => {
    // Initialize Google Auth2 with OAuth2 Client ID
    const g = window.gapi.auth2.init({
      client_id: '705714878286-qbg7sf892td0gkkeorv0m0frlu7qhmgv.apps.googleusercontent.com'
    })
    
    // Store the auth object in Vuex store
    this.setGoogleObject(g)
    this.authReady = true  // Show "Sign in with Google" button
  })
}
```

**What happens:**
- Initializes Google's OAuth2 client with the application's Client ID
- The Client ID is registered in Google Cloud Console
- Stores the auth object in Vuex state for later use
- UI now shows "Sign in with Google" button

**Vuex Store Update:** `app/src/store/modules/AuthModule.js`

```javascript
mutations: {
  setGoogleObject(state, gObj) {
    state.googleObject = gObj  // Store Google auth instance
  }
}
```

---

### Phase 3: User Clicks "Sign in with Google"

**File:** `app/src/components/Export.vue`

```javascript
signInClick() {
  // Request offline access (to get refresh token)
  window.gapi.auth2.getAuthInstance().grantOfflineAccess().then(authResult => {
    if (authResult.code) {
      // Send the authorization code to the server
      this.$http.post('/oauth2callback', authResult)
        .then(_ => {
          this.signIn()  // Update Vuex state
        })
        .catch(err => {
          console.error('Error with oauth2callback.')
          console.error(err)
          this.signOut()
        })
    }
  }, err => {
    if (err.error !== 'popup_closed_by_user') {
      throw err
    }
  })
}
```

**What happens:**

1. **`grantOfflineAccess()`** - Opens Google's OAuth consent screen in a popup
   - User sees: "REMAP wants to access your Google Drive"
   - User clicks "Allow"
   
2. **Google returns an authorization code** (NOT an access token)
   - This is a one-time code that can be exchanged for tokens
   - Format: `authResult = { code: "4/0AY0e-g7..." }`

3. **Send code to backend** via POST to `/oauth2callback`
   - The backend will exchange this code for actual credentials

**Why offline access?**
- Gets a **refresh token** that allows the backend to access Drive even when user is not actively logged in
- Necessary for long-running export tasks

---

### Phase 4: Backend Receives Authorization Code

**File:** `backend/server.py` (Routing)

```python
# Routing
app = webapp2.WSGIApplication([
    # ... other routes
    ('/oauth2callback', OAuth),  # Maps to OAuth class
    # ...
])
```

**File:** `backend/main/auth.py`

```python
class OAuth(BaseHandler):
    def post(self):
        # 1. Extract the authorization code from request
        code = json.loads(self.request.body)['code']
        
        self.session['reload'] = True
        
        if code != '':
            # 2. Exchange code for credentials (access token + refresh token)
            credentials = oauth2client.client.credentials_from_clientsecrets_and_code(
                filename=config.OAUTH_FILE,  # secrets/oauth_secret.json
                code=code,
                scope='https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/userinfo.email'
            )
            
            # 3. Use credentials to get user info
            http = credentials.authorize(httplib2.Http())
            ui = build('plus', 'v1', http=http)
            userinfo = ui.people().get(userId='me').execute(http=http)
            
            # 4. Store everything in server-side session
            self.session['email'] = userinfo['emails'][0]['value']
            self.session['user_id'] = userinfo['id']
            self.session['credentials'] = credentials.to_json()
```

**What happens:**

1. **Receives authorization code** from frontend
2. **Exchanges code for credentials** using `oauth_secret.json` (contains Client ID + Client Secret)
3. **Gets user information** (email, user ID) using the credentials
4. **Stores in session:**
   - `email` - User's email address
   - `user_id` - Google user ID
   - `credentials` - Full OAuth2 credentials (access token, refresh token, expiry)

**Session Storage:** Uses webapp2 sessions backed by Google App Engine Datastore

---

### Phase 5: Frontend Updates State

**File:** `app/src/components/Export.vue`

```javascript
this.$http.post('/oauth2callback', authResult)
  .then(_ => {
    this.signIn()  // Update Vuex to mark user as signed in
  })
```

**File:** `app/src/store/modules/AuthModule.js`

```javascript
mutations: {
  signIn(state) {
    state.serverSignedIn = true  // Backend has valid credentials
  }
}

getters: {
  isSignedIn: state => 
    state.googleObject !== null &&           // Google client initialized
    state.googleObject.isSignedIn.get() &&   // User signed in on frontend
    state.serverSignedIn                     // Backend has credentials
}
```

**What happens:**
- Vuex state updated to reflect successful authentication
- UI now shows "Drive export GeoTIFF" button as enabled
- User can now export to Drive

---

### Phase 6: Session Management (Automatic)

**File:** `backend/main/shared.py`

Every request goes through `BaseHandler.dispatch()`:

```python
class BaseHandler(webapp2.RequestHandler):
    def dispatch(self):
        # 1. Get session store
        self.session_store = sessions.get_store(request=self.request)
        
        # 2. Check if credentials exist and are valid
        if 'credentials' in self.session:
            credentials = oauth2client.client.OAuth2Credentials.from_json(
                self.session['credentials']
            )
            
            # 3. Refresh if expired
            if credentials.access_token_expired:
                try:
                    credentials.refresh(httplib2.Http())
                    self.session['credentials'] = credentials.to_json()
                except:
                    # Refresh failed, clear session
                    self.session.pop('email', None)
                    self.session.pop('user_id', None)
                    self.session.pop('credentials', None)
        
        try:
            # 4. Process the actual request
            webapp2.RequestHandler.dispatch(self)
        finally:
            # 5. Save session back to datastore
            self.session_store.save_sessions(self.response)

    @webapp2.cached_property
    def session(self):
        return self.session_store.get_session(backend="datastore")
```

**What happens:**
- **Every request** automatically checks credentials
- **Refreshes access token** if expired (using refresh token)
- **Clears session** if refresh fails (user needs to re-authenticate)
- **Saves session** after each request

---

### Phase 7: Using Credentials for Drive Export

**File:** `backend/main/auth.py`

When user clicks "Drive export GeoTIFF":

```python
class Export(GetMapData):
    def post(self):
        # 1. Mark export as in progress (stored in datastore)
        datastoreProgress(self.session['email'], 'IN_PROGRESS')
        
        # 2. Store training data temporarily
        dataKey = datastoreTraining(self.request.body)
        
        # 3. Queue background task (GAE has 60-second request limit)
        taskqueue.add(
            url='/exportworker',
            params={
                'dataKey': dataKey.urlsafe(),
                'ip': self.request.remote_addr,
                'credentials': self.session['credentials'],  # Pass credentials
                'email': self.session['email'],              # Pass email
                'User-Agent': self.request.headers.get('User-Agent')
            }
        )
        
        # 4. Return immediately (don't wait for export)
        self.response.write(json.dumps({}))
```

**Background Worker:**

```python
class ExportWorker(GetMapData):
    def post(self):
        # ... classification code ...
        
        # 1. Export to Earth Engine's temporary Drive location
        task = ee.batch.Export.image(
            image=classified,
            description=file_name,
            config={
                'driveFileNamePrefix': temp_file_prefix,  # UUID
                'maxPixels': 1e10,
                'scale': 30,
                'region': json.dumps([[x['lng'], x['lat']] for x in self.data['region']])
            }
        )
        
        task.start()
        while task.active():
            time.sleep(10)  # Wait for EE export
        
        # 2. If successful, copy to user's Drive
        if task.status()['state'] == ee.batch.Task.State.COMPLETED:
            # Get files from temp location (using app service account)
            files = APP_DRIVE_HELPER.GetExportedFiles(temp_file_prefix)
            
            # Recreate user credentials from JSON
            credentials = oauth2client.client.OAuth2Credentials.from_json(
                self.request.get('credentials')
            )
            
            # Grant app access to files
            for f in files:
                APP_DRIVE_HELPER.GrantAccess(f['id'], self.request.get('email'))
            
            # Create Drive helper with USER credentials
            user_drive_helper = drive.DriveHelper(credentials)
            
            if len(files) > 0:
                # Create folder in user's Drive
                folder = user_drive_helper.CreateFolder(
                    'REMAP Export Folder ' + export_time
                )
                
                # Create metadata file
                user_drive_helper.CreateFile(
                    'REMAP metadata ' + export_time + '.csv', 
                    meta_file, 
                    folder
                )
                
                # Copy GeoTIFF files to user's Drive
                for f in files:
                    try:
                        user_drive_helper.CopyFile(f['id'], file_name, folder)
                        APP_DRIVE_HELPER.DeleteFile(f['id'])  # Clean up temp
                    except:
                        pass
            
            # Mark as completed
            datastoreProgress(self.request.get('email'), 'COMPLETED')
```

**What happens:**

1. **Earth Engine exports** to a temporary Drive location (using service account)
2. **App grants user access** to the temporary files
3. **User's credentials** are used to:
   - Create a folder in their Drive
   - Copy files from temp location to their folder
   - Create metadata CSV
4. **Cleanup** - Delete temporary files
5. **Update status** in datastore

---

### Phase 8: Frontend Polls for Completion

**File:** `app/src/components/Export.vue`

```javascript
downloadDrive() {
  this.exportReady = false
  this.$http.post('/api/export', this.mapData()).then(data => {
    // Start polling every 10 seconds
    setTimeout(this.poll, 10000)
    this.exportStatus = 'IN_PROGRESS'
  })
}

poll() {
  this.$http.get('/api/exportstatus')
    .then(response => {
      this.exportStatus = response.body
      console.log(`polling result ${response.body}`)
      
      if (this.exportStatus === 'IN_PROGRESS') {
        // Keep polling
        setTimeout(this.poll, 10000)
      } else {
        // Show completion message
        if (this.exportStatus === 'COMPLETED') {
          this.$toasted.show('Drive download completed!', { duration: 5000 })
        } else if (this.exportStatus === 'ERROR') {
          this.$toasted.show('Error downloading from Drive.', { duration: 5000 })
        }
        this.exportReady = true
      }
    })
}
```

**Backend Status Check:**

```python
class ExportStatus(BaseHandler):
    def get(self):
        # Look up status in datastore by email
        status = checkDatastoreProgress(self.session['email'])
        self.response.write(status)
```

**File:** `backend/main/datastore.py`

```python
class Progress(ndb.Model):
    emailAddress = ndb.StringProperty()
    exportProgress = ndb.StringProperty()  # 'IN_PROGRESS', 'COMPLETED', 'ERROR'

def checkDatastoreProgress(email):
    query = Progress.query(Progress.emailAddress == email).fetch()
    if len(query) > 0:
        return query[0].exportProgress
    else:
        return 'NOT_STARTED'

def datastoreProgress(email, progress):
    query = Progress.query(Progress.emailAddress == email).fetch()
    if len(query) > 0:
        p = query[0]
        p.exportProgress = progress
    else:
        p = Progress(emailAddress=email, exportProgress=progress)
    p.put()
```

**What happens:**
- Frontend polls `/api/exportstatus` every 10 seconds
- Backend looks up status in datastore by user's email
- Returns: `'NOT_STARTED'`, `'IN_PROGRESS'`, `'COMPLETED'`, or `'ERROR'`
- Frontend shows appropriate message when complete

---

### Phase 9: Sign Out

**File:** `app/src/components/Export.vue`

```javascript
preSignOut() {
  this.exportReady = false
  this.$http.post('/api/signout').then(this.signOut)
}
```

**Backend:**

```python
class SignOut(BaseHandler):
    def post(self):
        self.session.pop('email', None)
        self.session.pop('user_id', None)
        self.session.pop('credentials', None)
```

**Frontend Vuex:**

```javascript
mutations: {
  signOut(state) {
    state.googleObject.signOut().then(_ => console.log('signed out'))
    state.serverSignedIn = false
  }
}
```

**What happens:**
- Clears server-side session
- Signs out from Google on frontend
- Updates Vuex state

---

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1-2: INITIALIZATION                                       │
└─────────────────────────────────────────────────────────────────┘

Frontend (Export.vue)
    │
    ├─ Load Google API script
    │
    ├─ Initialize gapi.auth2 with Client ID
    │
    └─ Store auth object in Vuex
         │
         └─ Show "Sign in with Google" button


┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3-5: AUTHENTICATION                                       │
└─────────────────────────────────────────────────────────────────┘

User clicks "Sign in"
    │
    ├─ Frontend: gapi.auth2.grantOfflineAccess()
    │       │
    │       └─ Opens Google OAuth consent popup
    │
    ├─ User approves
    │
    ├─ Google returns authorization code
    │       │
    │       └─ { code: "4/0AY0e-g7..." }
    │
    ├─ Frontend: POST /oauth2callback with code
    │
    ├─ Backend: Exchange code for credentials
    │       │
    │       ├─ Uses oauth_secret.json (Client ID + Secret)
    │       │
    │       ├─ Gets access token + refresh token
    │       │
    │       └─ Gets user info (email, user_id)
    │
    ├─ Backend: Store in session
    │       │
    │       ├─ session['email']
    │       ├─ session['user_id']
    │       └─ session['credentials']
    │
    └─ Frontend: Update Vuex state (serverSignedIn = true)


┌─────────────────────────────────────────────────────────────────┐
│ PHASE 6-8: DRIVE EXPORT                                         │
└─────────────────────────────────────────────────────────────────┘

User clicks "Drive export GeoTIFF"
    │
    ├─ Frontend: POST /api/export with training data
    │
    ├─ Backend: Queue background task
    │       │
    │       ├─ Store training data in datastore
    │       │
    │       ├─ Pass credentials to worker
    │       │
    │       └─ Return immediately
    │
    ├─ Frontend: Start polling /api/exportstatus
    │
    ├─ Background Worker:
    │       │
    │       ├─ Classify region
    │       │
    │       ├─ Export to EE temp Drive (service account)
    │       │
    │       ├─ Wait for EE export to complete
    │       │
    │       ├─ Grant user access to temp files
    │       │
    │       ├─ Use USER credentials to:
    │       │   ├─ Create folder in user's Drive
    │       │   ├─ Copy GeoTIFF to user's folder
    │       │   └─ Create metadata CSV
    │       │
    │       ├─ Delete temp files
    │       │
    │       └─ Update status: 'COMPLETED'
    │
    └─ Frontend: Poll detects completion
            │
            └─ Show success message


┌─────────────────────────────────────────────────────────────────┐
│ SESSION MANAGEMENT (Automatic on every request)                 │
└─────────────────────────────────────────────────────────────────┘

Every Request
    │
    ├─ BaseHandler.dispatch()
    │       │
    │       ├─ Load session from datastore
    │       │
    │       ├─ Check if credentials exist
    │       │
    │       ├─ If access token expired:
    │       │   ├─ Refresh using refresh token
    │       │   └─ Update session
    │       │
    │       ├─ Process request
    │       │
    │       └─ Save session back to datastore
    │
    └─ Continue with request handler
```

---

## Key Files Summary

| File | Purpose |
|------|---------|
| `app/src/components/Export.vue` | Frontend OAuth UI and flow initiation |
| `app/src/store/modules/AuthModule.js` | Vuex state management for auth |
| `backend/main/auth.py` | OAuth callback, export endpoints |
| `backend/main/shared.py` | BaseHandler with session management |
| `backend/lib/drive.py` | Drive API helper methods |
| `backend/main/datastore.py` | Progress tracking in datastore |
| `backend/server.py` | Route configuration |
| `secrets/oauth_secret.json` | OAuth Client ID + Secret |

---

## OAuth2 Credentials Structure

**What's stored in `session['credentials']`:**

```json
{
  "access_token": "ya29.a0AfH6SMB...",
  "client_id": "705714878286-qbg7sf892td0gkkeorv0m0frlu7qhmgv.apps.googleusercontent.com",
  "client_secret": "...",
  "refresh_token": "1//0gXXXXXXXXXXXX",
  "token_expiry": "2023-11-19T12:34:56Z",
  "token_uri": "https://oauth2.googleapis.com/token",
  "user_agent": null,
  "revoke_uri": "https://oauth2.googleapis.com/revoke",
  "id_token": null,
  "id_token_jwt": null,
  "token_response": {...},
  "scopes": [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/userinfo.email"
  ],
  "token_info_uri": "https://oauth2.googleapis.com/tokeninfo",
  "invalid": false,
  "_class": "OAuth2Credentials",
  "_module": "oauth2client.client"
}
```

**Key components:**
- **access_token** - Short-lived token (1 hour) for API calls
- **refresh_token** - Long-lived token to get new access tokens
- **token_expiry** - When access token expires
- **scopes** - What permissions were granted

---

## Two Types of Credentials

### 1. Service Account (App-level)

**File:** `backend/main/auth.py`

```python
APP_CREDENTIALS = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_name(
    config.EE_PRIVATE_KEY_FILE,
    scopes='https://www.googleapis.com/auth/drive'
)
APP_DRIVE_HELPER = drive.DriveHelper(APP_CREDENTIALS)
```

**Used for:**
- Earth Engine API access
- Temporary Drive storage
- Cleanup operations

**Characteristics:**
- No user interaction needed
- Fixed credentials
- Limited to app's own resources

### 2. User OAuth2 (User-level)

**File:** `backend/main/auth.py`

```python
credentials = oauth2client.client.OAuth2Credentials.from_json(
    self.request.get('credentials')
)
user_drive_helper = drive.DriveHelper(credentials)
```

**Used for:**
- Accessing user's Drive
- Creating folders in user's Drive
- Copying files to user's Drive

**Characteristics:**
- Requires user consent
- User-specific
- Can access user's resources

---

## Security Considerations

1. **Credentials in Session**
   - Stored server-side in Google App Engine Datastore
   - Not accessible to frontend JavaScript
   - Encrypted by GAE

2. **Refresh Token**
   - Allows long-term access without re-authentication
   - Stored securely in session
   - Can be revoked by user at any time

3. **Scopes**
   - Limited to Drive and email only
   - User sees exactly what permissions are requested
   - Can't access other Google services

4. **HTTPS Only**
   - All OAuth flows require HTTPS
   - Tokens never transmitted over HTTP

---

## Common Issues & Solutions

### Issue: "Access token expired"
**Solution:** Automatic refresh in `BaseHandler.dispatch()`

### Issue: "User not signed in"
**Solution:** Check `isSignedIn` getter (requires both frontend and backend auth)

### Issue: "Export stuck in IN_PROGRESS"
**Solution:** Background worker may have failed, check logs

### Issue: "Can't access user's Drive"
**Solution:** User needs to re-authenticate (refresh token may be revoked)

---

## Summary

The OAuth2 flow enables REMAP to:

1. ✅ Get user consent for Drive access
2. ✅ Store credentials securely server-side
3. ✅ Automatically refresh expired tokens
4. ✅ Export files to user's personal Drive
5. ✅ Handle long-running exports via background tasks
6. ✅ Track export progress per user

The key insight is the **separation of concerns**:
- **Frontend** handles UI and initial OAuth popup
- **Backend** handles token exchange and storage
- **Background worker** uses stored credentials for Drive operations
- **Polling** keeps frontend updated on progress
