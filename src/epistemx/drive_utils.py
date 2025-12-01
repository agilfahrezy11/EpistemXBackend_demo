"""
Google Drive Utilities Module

Provides OAuth2 authentication and Google Drive operations for exporting
Earth Engine results to user's personal Google Drive.

This module implements the OAuth2 flow similar to REMAP's approach, adapted for Streamlit.
"""

import logging
import json
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from io import BytesIO

# Configure logging
logger = logging.getLogger(__name__)

# OAuth2 scopes required for Drive operations
SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/userinfo.email'
]

class DriveAuthManager:
    """
    Manages OAuth2 authentication for Google Drive access.
    
    This class handles the OAuth2 flow for obtaining user consent
    to access their Google Drive for exporting Earth Engine results.
    """
    
    def __init__(self, client_secrets_file: str, redirect_uri: str = 'http://localhost:8501'):
        """
        Initialize Drive authentication manager.
        
        Parameters
        ----------
        client_secrets_file : str
            Path to OAuth2 client secrets JSON file
        redirect_uri : str
            OAuth2 redirect URI (default: http://localhost:8501 for Streamlit)
        """
        self.client_secrets_file = client_secrets_file
        self.redirect_uri = redirect_uri
        
    def get_authorization_url(self) -> tuple[str, str]:
        """
        Generate OAuth2 authorization URL for user to visit.
        
        Returns
        -------
        tuple[str, str]
            Authorization URL and state parameter
            
        Example
        -------
        >>> auth_manager = DriveAuthManager('oauth_secrets.json')
        >>> auth_url, state = auth_manager.get_authorization_url()
        >>> print(f"Visit: {auth_url}")
        """
        try:
            flow = Flow.from_client_secrets_file(
                self.client_secrets_file,
                scopes=SCOPES,
                redirect_uri=self.redirect_uri
            )
            
            auth_url, state = flow.authorization_url(
                access_type='offline',  # Get refresh token
                include_granted_scopes='true',
                prompt='consent'  # Force consent to ensure refresh token
            )
            
            return auth_url, state
            
        except Exception as e:
            logger.error(f"Failed to generate authorization URL: {e}")
            raise
    
    def exchange_code_for_credentials(self, auth_code: str) -> tuple[Optional[Credentials], Optional[Dict]]:
        """
        Exchange authorization code for OAuth2 credentials.
        
        Parameters
        ----------
        auth_code : str
            Authorization code from OAuth2 callback
            
        Returns
        -------
        tuple[Credentials, dict]
            OAuth2 credentials and user info, or (None, None) if failed
            
        Example
        -------
        >>> credentials, user_info = auth_manager.exchange_code_for_credentials(code)
        >>> print(f"Authenticated as: {user_info['email']}")
        """
        try:
            flow = Flow.from_client_secrets_file(
                self.client_secrets_file,
                scopes=SCOPES,
                redirect_uri=self.redirect_uri
            )
            
            flow.fetch_token(code=auth_code)
            credentials = flow.credentials
            
            # Get user information
            service = build('oauth2', 'v2', credentials=credentials)
            user_info = service.userinfo().get().execute()
            
            logger.info(f"Successfully authenticated user: {user_info.get('email')}")
            return credentials, user_info
            
        except Exception as e:
            logger.error(f"Failed to exchange code for credentials: {e}")
            return None, None
    
    @staticmethod
    def credentials_to_dict(credentials: Credentials) -> Dict[str, Any]:
        """
        Convert credentials object to dictionary for storage.
        
        Parameters
        ----------
        credentials : Credentials
            OAuth2 credentials object
            
        Returns
        -------
        dict
            Serializable credentials dictionary
        """
        return {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
    
    @staticmethod
    def dict_to_credentials(creds_dict: Dict[str, Any]) -> Credentials:
        """
        Convert dictionary back to credentials object.
        
        Parameters
        ----------
        creds_dict : dict
            Credentials dictionary
            
        Returns
        -------
        Credentials
            OAuth2 credentials object
        """
        return Credentials(
            token=creds_dict.get('token'),
            refresh_token=creds_dict.get('refresh_token'),
            token_uri=creds_dict.get('token_uri'),
            client_id=creds_dict.get('client_id'),
            client_secret=creds_dict.get('client_secret'),
            scopes=creds_dict.get('scopes')
        )
    
    @staticmethod
    def refresh_credentials(credentials: Credentials) -> Credentials:
        """
        Refresh expired OAuth2 credentials.
        
        Parameters
        ----------
        credentials : Credentials
            Potentially expired credentials
            
        Returns
        -------
        Credentials
            Refreshed credentials
        """
        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
                logger.info("Credentials refreshed successfully")
            except Exception as e:
                logger.error(f"Failed to refresh credentials: {e}")
                raise
        return credentials


class DriveHelper:
    """
    Helper class for Google Drive operations.
    
    Provides methods for creating folders, uploading files, and managing
    Drive resources for Earth Engine export results.
    """
    
    def __init__(self, credentials: Credentials):
        """
        Initialize Drive helper with OAuth2 credentials.
        
        Parameters
        ----------
        credentials : Credentials
            OAuth2 credentials for Drive access
        """
        self.credentials = credentials
        self.service = build('drive', 'v3', credentials=credentials)
        
    def create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> str:
        """
        Create a folder in Google Drive.
        
        Parameters
        ----------
        folder_name : str
            Name of the folder to create
        parent_id : str, optional
            Parent folder ID (None for root)
            
        Returns
        -------
        str
            Created folder ID
            
        Example
        -------
        >>> helper = DriveHelper(credentials)
        >>> folder_id = helper.create_folder('REMAP Exports')
        """
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            folder = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            folder_id = folder.get('id')
            logger.info(f"Created folder '{folder_name}' with ID: {folder_id}")
            return folder_id
            
        except Exception as e:
            logger.error(f"Failed to create folder: {e}")
            raise
    
    def upload_file(
        self, 
        file_name: str, 
        file_content: bytes, 
        mime_type: str = 'application/octet-stream',
        folder_id: Optional[str] = None
    ) -> str:
        """
        Upload a file to Google Drive.
        
        Parameters
        ----------
        file_name : str
            Name of the file
        file_content : bytes
            File content as bytes
        mime_type : str
            MIME type of the file
        folder_id : str, optional
            Parent folder ID (None for root)
            
        Returns
        -------
        str
            Uploaded file ID
            
        Example
        -------
        >>> content = b"Sample content"
        >>> file_id = helper.upload_file('data.csv', content, 'text/csv', folder_id)
        """
        try:
            file_metadata = {'name': file_name}
            
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            media = MediaIoBaseUpload(
                BytesIO(file_content),
                mimetype=mime_type,
                resumable=True
            )
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            logger.info(f"Uploaded file '{file_name}' with ID: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise
    
    def copy_file(self, file_id: str, new_name: str, folder_id: Optional[str] = None) -> str:
        """
        Copy a file in Google Drive.
        
        Parameters
        ----------
        file_id : str
            Source file ID
        new_name : str
            Name for the copied file
        folder_id : str, optional
            Destination folder ID
            
        Returns
        -------
        str
            Copied file ID
        """
        try:
            file_metadata = {'name': new_name}
            
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            copied_file = self.service.files().copy(
                fileId=file_id,
                body=file_metadata
            ).execute()
            
            copied_id = copied_file.get('id')
            logger.info(f"Copied file to '{new_name}' with ID: {copied_id}")
            return copied_id
            
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            raise
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from Google Drive.
        
        Parameters
        ----------
        file_id : str
            File ID to delete
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            self.service.files().delete(fileId=file_id).execute()
            logger.info(f"Deleted file with ID: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False
    
    def list_files(self, folder_id: Optional[str] = None, query: Optional[str] = None) -> List[Dict]:
        """
        List files in Google Drive.
        
        Parameters
        ----------
        folder_id : str, optional
            Folder ID to list files from
        query : str, optional
            Custom query string
            
        Returns
        -------
        list[dict]
            List of file metadata dictionaries
        """
        try:
            if query is None:
                if folder_id:
                    query = f"'{folder_id}' in parents and trashed=false"
                else:
                    query = "trashed=false"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType, createdTime, modifiedTime)"
            ).execute()
            
            files = results.get('files', [])
            logger.info(f"Found {len(files)} files")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def grant_access(self, file_id: str, email: str, role: str = 'writer') -> bool:
        """
        Grant access to a file for a specific user.
        
        Parameters
        ----------
        file_id : str
            File ID to share
        email : str
            Email address to grant access to
        role : str
            Permission role ('reader', 'writer', 'owner')
            
        Returns
        -------
        bool
            True if successful
        """
        try:
            permission = {
                'type': 'user',
                'role': role,
                'emailAddress': email
            }
            
            self.service.permissions().create(
                fileId=file_id,
                body=permission,
                fields='id'
            ).execute()
            
            logger.info(f"Granted {role} access to {email} for file {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to grant access: {e}")
            return False


def ensure_valid_credentials(creds_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Ensure credentials are valid, refresh if needed.
    
    Parameters
    ----------
    creds_dict : dict
        Credentials dictionary
        
    Returns
    -------
    dict or None
        Updated credentials dictionary, or None if refresh failed
        
    Example
    -------
    >>> updated_creds = ensure_valid_credentials(st.session_state['drive_credentials'])
    >>> if updated_creds:
    >>>     st.session_state['drive_credentials'] = updated_creds
    """
    try:
        credentials = DriveAuthManager.dict_to_credentials(creds_dict)
        
        if credentials.expired and credentials.refresh_token:
            credentials = DriveAuthManager.refresh_credentials(credentials)
            return DriveAuthManager.credentials_to_dict(credentials)
        
        return creds_dict
        
    except Exception as e:
        logger.error(f"Failed to ensure valid credentials: {e}")
        return None
