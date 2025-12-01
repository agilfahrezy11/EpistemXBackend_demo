# Service Account Flow in REMAP - Complete Explanation

## Overview

REMAP uses **Google Earth Engine Service Account** credentials for backend Earth Engine API access. This is separate from the OAuth2 user authentication used for Drive exports.

**Two Authentication Systems:**

1. **Service Account (Backend)** - For Earth Engine API access ← **THIS DOCUMENT**
2. **OAuth2 User Authentication (Frontend + Backend)** - For Google Drive access

This document focuses on the **Service Account flow** for Earth Engine operations.

---

## Why Service Account is Needed

The application needs to:
- Access Google Earth Engine API to process satellite imagery
- Perform land cover classification using Landsat composites
- Generate map tiles for visualization
- Export classified images to temporary Drive storage
- Run long-running Earth Engine tasks without user interaction

Service accounts provide **application-level credentials** that don't require user consent and work continuously in the background.

---

## Service Account vs OAuth2 - Key Differences

| Aspect | Service Account | OAuth2 User Auth |
|--------|----------------|------------------|
| **Purpose** | Earth Engine API access | User's Google Drive access |
| **Scope** | Application-level | User-specific |
| **Authentication** | Private key file | User consent popup |
| **Lifetime** | Permanent (until revoked) | Requires refresh tokens |
| **User Interaction** | None required | Requires user approval |
| **Used For** | Classification, map tiles, EE exports | Copying files to user's Drive |

---

## The Complete Service Account Flow - Step by Step

### Phase 1: Application Startup (Server Initialization)

**File:** `backend/config.py`

```python
import ee
from google.appengine.api import urlfetch

# 1. Read service account email from file
EE_ACCOUNT = open('secrets/ee_account.txt').read().strip()
# Example: 'remap-service@your-project.iam.gserviceaccount.com'

# 2. Path to service account private key JSON
EE_PRIVATE_KEY_FILE = 'secrets/gee_service_account_secrets.json'

# 3. Create Earth Engine credentials object
EE_CREDENTIALS = ee.ServiceAccountCredentials(EE_ACCOUNT, EE_PRIVATE_KEY_FILE)

# 4. Initialize Earth Engine with service account
ee.Initialize(EE_CREDENTIALS)

# 5. Set timeouts for long-running operations
urlfetch.set_default_fetch_deadline(120000)  # 120 seconds
ee.data.setDeadline(60000)  # 60 seconds
```

**What happens:**

1. **Reads service account email** - Identifies which service account to use
2. **Loads private key JSON** - Contains cryptographic keys for authentication
3. **Creates credentials object** - Wraps the service account details
4. **Initializes Earth Engine** - Authenticates with Google Earth Engine servers
5. **Sets timeouts** - Configures how long to wait for EE operations

**This happens ONCE when the server starts** - All subsequent requests use these credentials automatically.

---

### Phase 2: Service Account JSON Structure

**File:** `secrets/gee_service_account_secrets.json`

```json
{
  "type": "service_account",
  "project_id": "your-gcp-project-id",
  "private_key_id": "abc123...",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC...\n-----END PRIVATE KEY-----\n",
  "client_email": "remap-service@your-project.iam.gserviceaccount.com",
  "client_id": "1234567890",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/..."
}
```

**Key components:**

- **type** - Always "service_account"
- **project_id** - Your Google Cloud Project ID
- **private_key** - RSA private key for signing JWT tokens
- **client_email** - Service account email address
- **token_uri** - Where to exchange JWT for access tokens

**Security Note:** This file contains sensitive credentials and should NEVER be committed to version control.

---

### Phase 3: How Service Account Authentication Works

**Behind the scenes when `ee.Initialize()` is called:**

```
1. Earth Engine Python library reads the private key
   │
2. Creates a JWT (JSON Web Token) signed with private key
   │  
   ├─ Header: { "alg": "RS256", "typ": "JWT" }
   ├─ Payload: {
   │    "iss": "remap-service@project.iam.gserviceaccount.com",
   │    "scope": "https://www.googleapis.com/auth/earthengine",
   │    "aud": "https://oauth2.googleapis.com/token",
   │    "exp": 1234567890,  # Expiration time
   │    "iat": 1234567000   # Issued at time
   │  }
   └─ Signature: RSA-SHA256(header + payload, private_key)
   │
3. Sends JWT to Google's token endpoint
   │
4. Google validates the JWT signature
   │
5. Google returns an access token
   │  
   └─ { "access_token": "ya29.c.Kl6iB...", "expires_in": 3600 }
   │
6. Earth Engine library stores the access token
   │
7. All subsequent EE API calls include this token
   │
8. Token automatically refreshed when expired
```

**This is all handled automatically by the Earth Engine Python library** - you don't need to manage tokens manually.

---

### Phase 4: Using Service Account for Earth Engine Operations

#### Example 1: Loading Predictor Images

**File:** `backend/remap/predictor_image.py`

```python
import ee

def predictor_image(past=False):
    """Returns an ee.Image with predictor bands for classification."""
    return base_predictor_layer(past).unmask(-20000)

def base_predictor_layer(past=False):
    # 1. Load climate data (uses service account credentials)
    bioclim = ee.Image('WORLDCLIM/V1/BIO').select(
        ['bio01', 'bio12'],
        ['Mean Annual Temperature', 'Annual Precipitation']
    )
    
    # 2. Load elevation data (uses service account credentials)
    elevation = ee.Image('USGS/SRTMGL1_003').rename(['Elevation'])
    slope = ee.Terrain.slope(elevation).rename(['Slope'])
    
    # 3. Load Landsat composite (uses service account credentials)
    if not past:
        ls = ee.Image('projects/remap-app/ls_8_cflte1_2k14to17_at_30m_ui8')
    else:
        ls = ee.Image('projects/remap-app/ls_7_self_masked_99_03_at_30m')
    
    # 4. Calculate spectral indices
    ndvi = ls.normalizedDifference(['NIR', 'Red']).rename(['NDVI'])
    ndwi = ls.normalizedDifference(['Green', 'NIR']).rename(['NDWI'])
    
    # 5. Combine all bands
    return ee.Image([bioclim, elevation, slope, ls, ndvi, ndwi])
```

**What happens:**
- All `ee.Image()` calls use the service account credentials initialized at startup
- No additional authentication needed
- Service account has read access to public Earth Engine datasets
- Service account has read access to private `projects/remap-app/` assets

---

#### Example 2: Classification Workflow

**File:** `backend/remap/classification.py`

```python
import ee
from parameters import *
from predictor_image import *

def get_classified_from_fc(train_fc, predictors, past=False):
    """Classifies a region using training points and selected predictors."""
    
    # 1. Get predictor image (uses service account)
    composite = predictor_image(past).select(predictors)
    
    # 2. Sample predictor values at training points (uses service account)
    training = composite.reduceRegions(
        train_fc,
        reducer=ee.Reducer.first(),
        scale=parameters['reduce_to_vector_scale']
    )
    
    # 3. Train Random Forest classifier (uses service account)
    classifier = ee.Classifier.randomForest(
        numberOfTrees=parameters['number_of_trees'],
        minLeafPopulation=parameters['min_leaf_pop'],
        outOfBagMode=parameters['oob_mode']
    ).train(training, "label", composite.bandNames())
    
    # 4. Classify the entire image (uses service account)
    return composite.classify(classifier)
```

**What happens:**
- All Earth Engine operations use service account credentials
- `reduceRegions()` - Samples pixel values at training points
- `train()` - Trains classifier on Earth Engine servers
- `classify()` - Applies classifier to entire image
- All computations happen server-side on Google's infrastructure

---

#### Example 3: Generating Map Tiles

**File:** `backend/main/remap_api.py`

```python
class GetMapData(BaseHandler):
    def post(self):
        """Receives training data and returns a classified raster."""
        
        # 1. Create training FeatureCollection (uses service account)
        self.train_fc = self.get_train(self.classes)
        
        # 2. Classify region (uses service account)
        classified = remap.get_classified_from_fc(
            self.train_fc, 
            self.predictors, 
            self.past
        ).clip(self.region)
        
        # 3. Get map tile URL (uses service account)
        m = classified.getMapId({
            'min': 1,
            'max': len(self.classes),
            'palette': ",".join([label['colour'] for label in self.classes])
        })
        
        # 4. Return tile URL to frontend
        self.response.write(json.dumps({
            'mapid': m['mapid'],
            'token': m['token']
        }))
```

**What happens:**
- `getMapId()` generates a unique map tile URL
- Earth Engine servers render tiles on-demand
- Frontend can request tiles using the mapid and token
- All tile generation uses service account credentials

---

#### Example 4: Exporting to Drive

**File:** `backend/main/auth.py`

```python
class ExportWorker(GetMapData):
    def post(self):
        # 1. Classify region (uses service account)
        classified = remap.get_classified_from_fc(
            self.train_fc, 
            self.predictors, 
            self.past
        ).clip(self.region)
        
        # 2. Export to temporary Drive location (uses service account)
        temp_file_prefix = str(uuid.uuid4())
        task = ee.batch.Export.image(
            image=classified,
            description='REMAP Export',
            config={
                'driveFileNamePrefix': temp_file_prefix,
                'maxPixels': 1e10,
                'scale': 30,
                'region': json.dumps([[x['lng'], x['lat']] for x in self.data['region']])
            }
        )
        
        # 3. Start export task (uses service account)
        task.start()
        
        # 4. Wait for completion
        while task.active():
            time.sleep(10)
        
        # 5. If successful, copy to user's Drive (uses USER OAuth2 credentials)
        if task.status()['state'] == ee.batch.Task.State.COMPLETED:
            # ... copy files to user's Drive using OAuth2 credentials
```

**What happens:**
- `ee.batch.Export.image()` creates an export task
- Export runs on Earth Engine servers using service account
- Files are saved to service account's Drive (temporary location)
- Later copied to user's Drive using OAuth2 credentials

---

### Phase 5: Service Account for Drive Operations

**File:** `backend/main/auth.py`

```python
# Create service account credentials for Drive API
APP_CREDENTIALS = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_name(
    config.EE_PRIVATE_KEY_FILE,  # Same JSON file as Earth Engine
    scopes='https://www.googleapis.com/auth/drive'
)

# Create Drive helper with service account credentials
APP_DRIVE_HELPER = drive.DriveHelper(APP_CREDENTIALS)
```

**Used for:**

1. **Cleanup temporary files**
```python
class Clean(webapp2.RequestHandler):
    def get(self):
        # Find old exported files
        files = APP_DRIVE_HELPER.GetExportedFiles('')
        now = datetime.utcnow()
        
        # Delete files older than 2 hours
        for f in files:
            cDate = datetime.strptime(f['createdDate'], "%Y-%m-%dT%H:%M:%S.%fZ")
            if now - cDate > timedelta(hours=2):
                APP_DRIVE_HELPER.DeleteFile(f['id'])
```

2. **Grant user access to exported files**
```python
# Grant user write access to temp files
for f in files:
    APP_DRIVE_HELPER.GrantAccess(f['id'], user_email)
```

3. **Delete temp files after copying**
```python
# Clean up after successful copy
APP_DRIVE_HELPER.DeleteFile(temp_file_id)
```

---

### Phase 6: Two Drive Helpers - Service Account vs User OAuth2

```python
# SERVICE ACCOUNT Drive Helper (app's Drive)
APP_CREDENTIALS = ServiceAccountCredentials.from_json_keyfile_name(
    'secrets/gee_service_account_secrets.json',
    scopes='https://www.googleapis.com/auth/drive'
)
APP_DRIVE_HELPER = drive.DriveHelper(APP_CREDENTIALS)

# USER OAuth2 Drive Helper (user's Drive)
credentials = OAuth2Credentials.from_json(session['credentials'])
user_drive_helper = drive.DriveHelper(credentials)
```

**Why two helpers?**

| Operation | Uses | Reason |
|-----------|------|--------|
| Earth Engine export | Service Account | EE exports to service account's Drive |
| Temporary file storage | Service Account | App controls temp files |
| Grant access to user | Service Account | App owns the files |
| Create folder in user's Drive | User OAuth2 | Accessing user's Drive |
| Copy files to user's Drive | User OAuth2 | Accessing user's Drive |
| Delete temp files | Service Account | App owns the files |

---

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: SERVER STARTUP (ONE TIME)                              │
└─────────────────────────────────────────────────────────────────┘

Server starts
    │
    ├─ Load config.py
    │       │
    │       ├─ Read secrets/ee_account.txt
    │       │   └─ "remap-service@project.iam.gserviceaccount.com"
    │       │
    │       ├─ Read secrets/gee_service_account_secrets.json
    │       │   └─ { private_key, client_email, ... }
    │       │
    │       ├─ Create EE_CREDENTIALS
    │       │   └─ ee.ServiceAccountCredentials(email, key_file)
    │       │
    │       └─ ee.Initialize(EE_CREDENTIALS)
    │               │
    │               ├─ Generate JWT signed with private key
    │               ├─ Exchange JWT for access token
    │               ├─ Store access token in ee.data module
    │               └─ Set up automatic token refresh
    │
    └─ Earth Engine is now ready for all requests


┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: USER REQUEST - CLASSIFICATION                          │
└─────────────────────────────────────────────────────────────────┘

User draws region and adds training points
    │
    ├─ Frontend: POST /api/map with training data
    │
    ├─ Backend: GetMapData.post()
    │       │
    │       ├─ Convert training points to ee.FeatureCollection
    │       │   └─ Uses service account (automatic)
    │       │
    │       ├─ Load predictor image
    │       │   ├─ ee.Image('WORLDCLIM/V1/BIO')
    │       │   ├─ ee.Image('USGS/SRTMGL1_003')
    │       │   └─ ee.Image('projects/remap-app/ls_8_...')
    │       │   └─ All use service account (automatic)
    │       │
    │       ├─ Sample predictors at training points
    │       │   └─ composite.reduceRegions(train_fc, ...)
    │       │   └─ Uses service account (automatic)
    │       │
    │       ├─ Train Random Forest classifier
    │       │   └─ ee.Classifier.randomForest().train(...)
    │       │   └─ Uses service account (automatic)
    │       │
    │       ├─ Classify region
    │       │   └─ composite.classify(classifier)
    │       │   └─ Uses service account (automatic)
    │       │
    │       └─ Generate map tiles
    │           └─ classified.getMapId(vis_params)
    │           └─ Returns: { mapid, token }
    │
    └─ Frontend: Display map tiles
            └─ Tiles served by Earth Engine using service account


┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: EXPORT TO DRIVE                                        │
└─────────────────────────────────────────────────────────────────┘

User clicks "Export to Drive"
    │
    ├─ Frontend: POST /api/export
    │
    ├─ Backend: Queue background task
    │
    ├─ Background Worker: ExportWorker.post()
    │       │
    │       ├─ Classify region (uses service account)
    │       │   └─ remap.get_classified_from_fc(...)
    │       │
    │       ├─ Export to EE temp Drive (uses service account)
    │       │   ├─ ee.batch.Export.image(...)
    │       │   ├─ task.start()
    │       │   └─ Exports to service account's Drive
    │       │
    │       ├─ Wait for EE export to complete
    │       │   └─ while task.active(): sleep(10)
    │       │
    │       ├─ Find exported files (uses service account)
    │       │   └─ APP_DRIVE_HELPER.GetExportedFiles(prefix)
    │       │
    │       ├─ Grant user access (uses service account)
    │       │   └─ APP_DRIVE_HELPER.GrantAccess(file_id, user_email)
    │       │
    │       ├─ Copy to user's Drive (uses USER OAuth2)
    │       │   ├─ user_drive_helper.CreateFolder(...)
    │       │   ├─ user_drive_helper.CopyFile(...)
    │       │   └─ Accesses user's personal Drive
    │       │
    │       └─ Delete temp files (uses service account)
    │           └─ APP_DRIVE_HELPER.DeleteFile(file_id)
    │
    └─ Frontend: Poll for completion
            └─ Shows success message


┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: AUTOMATIC TOKEN REFRESH (CONTINUOUS)                   │
└─────────────────────────────────────────────────────────────────┘

Every Earth Engine API call
    │
    ├─ Check if access token expired
    │       │
    │       └─ If expired:
    │           ├─ Generate new JWT
    │           ├─ Exchange for new access token
    │           └─ Update stored token
    │
    └─ Make API call with current token
```

---

## Key Files Summary

| File | Purpose |
|------|---------|
| `backend/config.py` | Initialize Earth Engine with service account |
| `backend/main/auth.py` | Service account Drive operations, export worker |
| `backend/remap/predictor_image.py` | Load Earth Engine datasets |
| `backend/remap/classification.py` | Train classifier and classify regions |
| `backend/main/remap_api.py` | API endpoints for classification and map tiles |
| `backend/lib/drive.py` | Drive API helper methods |
| `secrets/ee_account.txt` | Service account email |
| `secrets/gee_service_account_secrets.json` | Service account private key |

---

## Required Setup Files

### 1. Service Account Email

**File:** `secrets/ee_account.txt`

```
remap-service@your-project.iam.gserviceaccount.com

```

**Note:** Must have empty second line

### 2. Service Account Private Key

**File:** `secrets/gee_service_account_secrets.json`

```json
{
  "type": "service_account",
  "project_id": "your-gcp-project",
  "private_key_id": "abc123...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "remap-service@your-project.iam.gserviceaccount.com",
  "client_id": "1234567890",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/..."
}
```

---

## How to Create a Service Account

### Step 1: Create Service Account in Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (or create a new one)
3. Navigate to **IAM & Admin** → **Service Accounts**
4. Click **Create Service Account**
5. Enter details:
   - **Name:** `remap-service` (or your choice)
   - **Description:** `Service account for REMAP Earth Engine access`
6. Click **Create and Continue**

### Step 2: Grant Permissions

Grant the following roles:

1. **Earth Engine Resource Admin** (for Earth Engine access)
2. **Service Account Token Creator** (for authentication)
3. **Storage Admin** (if using Google Cloud Storage)

Click **Continue** → **Done**

### Step 3: Create and Download Key

1. Find your service account in the list
2. Click the three dots (⋮) → **Manage keys**
3. Click **Add Key** → **Create new key**
4. Select **JSON** format
5. Click **Create**
6. Save the downloaded JSON file as `secrets/gee_service_account_secrets.json`

### Step 4: Register with Earth Engine

1. Go to [Earth Engine](https://code.earthengine.google.com/)
2. Register your service account email
3. Or use the command line:
   ```bash
   earthengine authenticate --service-account=remap-service@your-project.iam.gserviceaccount.com
   ```

### Step 5: Grant Access to Earth Engine Assets

If you have private Earth Engine assets (like `projects/remap-app/ls_8_...`):

1. Go to [Earth Engine Code Editor](https://code.earthengine.google.com/)
2. Navigate to **Assets** tab
3. Select your asset
4. Click **Share**
5. Add your service account email with **Reader** access

---

## Security Best Practices

### ✅ DO:

- **Store credentials securely** - Never commit to version control
- **Use `.gitignore`** - Add `secrets/` to `.gitignore`
- **Rotate keys regularly** - Create new keys periodically
- **Limit permissions** - Grant only necessary roles
- **Monitor usage** - Check Cloud Console for unusual activity
- **Use separate service accounts** - One per environment (dev, prod)

### ❌ DON'T:

- **Commit private keys** - Never push to GitHub
- **Share keys publicly** - Keep credentials private
- **Use personal accounts** - Use service accounts for apps
- **Grant excessive permissions** - Follow principle of least privilege
- **Hardcode credentials** - Always load from files
- **Reuse keys across projects** - Create separate keys

---

## Common Issues & Solutions

### Issue: "Earth Engine not initialized"

**Cause:** `ee.Initialize()` not called or failed

**Solution:**
```python
# Check if initialized
if not ee.data._initialized:
    ee.Initialize(EE_CREDENTIALS)
```

### Issue: "Service account does not have access to Earth Engine"

**Cause:** Service account not registered with Earth Engine

**Solution:**
1. Go to https://signup.earthengine.google.com/
2. Register service account email
3. Wait a few minutes for propagation

### Issue: "Permission denied" when accessing assets

**Cause:** Service account doesn't have access to private assets

**Solution:**
1. Share asset with service account email
2. Grant "Reader" permission
3. Or make asset public

### Issue: "Invalid JWT signature"

**Cause:** Private key file corrupted or incorrect

**Solution:**
1. Download new key from Cloud Console
2. Replace `secrets/gee_service_account_secrets.json`
3. Restart server

### Issue: "Token expired" errors

**Cause:** Automatic refresh failing

**Solution:**
- Check network connectivity
- Verify token_uri in JSON file
- Check Cloud Console for service account status

---

## Debugging Tips

### Check if Earth Engine is initialized:

```python
import ee
print("Initialized:", ee.data._initialized)
print("API Base URL:", ee.data._api_base_url)
```

### Test service account credentials:

```python
import ee

# Initialize
credentials = ee.ServiceAccountCredentials(
    'remap-service@project.iam.gserviceaccount.com',
    'secrets/gee_service_account_secrets.json'
)
ee.Initialize(credentials)

# Test with simple operation
image = ee.Image('USGS/SRTMGL1_003')
print("Elevation min:", image.reduceRegion(
    reducer=ee.Reducer.min(),
    geometry=ee.Geometry.Point([0, 0]).buffer(1000),
    scale=30
).getInfo())
```

### Check service account permissions:

```python
# Try to access a private asset
try:
    image = ee.Image('projects/remap-app/ls_8_cflte1_2k14to17_at_30m_ui8')
    print("Access granted:", image.bandNames().getInfo())
except Exception as e:
    print("Access denied:", str(e))
```

### Monitor API calls:

```python
# Enable logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Earth Engine will log API calls
```

---

## Comparison: Service Account vs OAuth2

### When to Use Service Account:

✅ Backend server operations  
✅ Automated tasks  
✅ Earth Engine API access  
✅ No user interaction needed  
✅ Application-level resources  

### When to Use OAuth2:

✅ User-specific operations  
✅ Accessing user's Drive  
✅ Requires user consent  
✅ User-level resources  
✅ Frontend authentication  

### REMAP Uses Both:

```python
# Service Account - Earth Engine operations
EE_CREDENTIALS = ee.ServiceAccountCredentials(email, key_file)
ee.Initialize(EE_CREDENTIALS)

# Service Account - Temporary Drive storage
APP_CREDENTIALS = ServiceAccountCredentials.from_json_keyfile_name(
    key_file, scopes='https://www.googleapis.com/auth/drive'
)
APP_DRIVE_HELPER = drive.DriveHelper(APP_CREDENTIALS)

# OAuth2 - User's Drive access
user_credentials = OAuth2Credentials.from_json(session['credentials'])
user_drive_helper = drive.DriveHelper(user_credentials)
```

---

## Summary

The Service Account flow enables REMAP to:

1. ✅ Access Earth Engine API without user interaction
2. ✅ Load satellite imagery and climate data
3. ✅ Perform land cover classification
4. ✅ Generate map tiles for visualization
5. ✅ Export classified images to temporary storage
6. ✅ Run long-running background tasks
7. ✅ Manage temporary files automatically

**Key Insight:** Service account credentials are initialized once at server startup and used automatically for all Earth Engine operations. No per-request authentication needed.

**The Flow:**
```
Server starts → Load credentials → Initialize EE → All requests use same credentials
```

**Contrast with OAuth2:**
```
User signs in → Get authorization code → Exchange for tokens → Store in session → Use per-user
```

Both systems work together to provide a seamless experience:
- **Service Account** handles the heavy lifting (Earth Engine processing)
- **OAuth2** handles user-specific operations (Drive exports)
