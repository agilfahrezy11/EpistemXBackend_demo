

import ee
import streamlit as st
from pathlib import Path

class EarthEngineAuth:
    """Handle Earth Engine authentication and initialization"""
    
    @staticmethod
    def initialize():
        """
        Try multiple authentication methods in order of preference
        """
        # Method 1: Already initialized
        if EarthEngineAuth._check_initialized():
            return True
        
        # Method 2: Service Account (Production)
        if EarthEngineAuth._try_service_account():
            return True
        
        # Method 3: Persistent credentials (Development)
        if EarthEngineAuth._try_persistent_credentials():
            return True
        
        # Method 4: Interactive authentication
        return EarthEngineAuth._interactive_authentication()
    
    @staticmethod
    def _check_initialized():
        """Check if EE is already initialized"""
        try:
            ee.Initialize()
            st.success("✅ Earth Engine initialized")
            return True
        except:
            return False
    
    @staticmethod
    def _try_service_account():
        """Try service account authentication"""
        try:
            # From Streamlit secrets
            if 'gcp_service_account' in st.secrets:
                service_account_info = dict(st.secrets["gcp_service_account"])
                credentials = ee.ServiceAccountCredentials(
                    email=service_account_info['client_email'],
                    key_data=service_account_info['private_key']
                )
                ee.Initialize(credentials)
                st.success("✅ Initialized with service account")
                return True
        except Exception as e:
            st.warning(f"Service account auth failed: {e}")
            return False
    
    @staticmethod
    def _try_persistent_credentials():
        """Try using persistent credentials"""
        try:
            credentials_path = Path.home() / '.config' / 'earthengine' / 'credentials'
            if credentials_path.exists():
                ee.Initialize()
                st.success("✅Initialized with stored credentials")
                return True
        except Exception as e:
            st.warning(f"Stored credentials failed: {e}")
            return False
    
    @staticmethod
    def _interactive_authentication():
        """Guide user through interactive authentication"""
        st.error("Earth Engine Authentication Required")
        
        with st.expander("📖 Authentication Instructions", expanded=True):
            st.markdown("""
            ### How to Authenticate:
            
            **Option 1: Command Line (Recommended)**
```bash
            earthengine authenticate
    """)