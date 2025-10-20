import pandas as pd
import streamlit as st
# ----- System response 2.1.b -----
class LULCSchemeClass:
    """
    Module 2: Land Cover Classification Scheme Manager
    Allows users to define, edit, and manage land cover classification schemes
    """
    #adapted from def init 
    #use st.session state to store result, instead of using self.
    #line 14 - 18
    def __init__(self):
        # Initialize session state if not exists
        if 'lulc_classes' not in st.session_state:
            st.session_state.lulc_classes = []
        if 'lulc_next_id' not in st.session_state:
            st.session_state.lulc_next_id = 1
        if 'lulc_edit_mode' not in st.session_state:
            st.session_state.lulc_edit_mode = False
        if 'lulc_edit_idx' not in st.session_state:
            st.session_state.lulc_edit_idx = None
    
    #Adapted from widget based layout (jupyter notebook) to streamlit session
    #@ is decorator or wrapper, so that when called, it does not need parenthesis
    #it provide clean access to each variable in streamlit session state
    #The original code (faza) use jupyter notebook UI, which is incompatible with streamlit
    @property
    def classes(self):
        """Get classes from session state"""
        return st.session_state.lulc_classes
    @classes.setter
    def classes(self, value):
        """Set classes in session state"""
        st.session_state.lulc_classes = value
    
    #adapter from line 46-51 (self.class_id_input)
    @property 
    def next_id(self):
        """Get next ID from session state"""
        return st.session_state.lulc_next_id
    
    @next_id.setter
    def next_id(self, value):
        """Set next ID in session state"""
        st.session_state.lulc_next_id = value
    
    #adapted from line 184, but tailored with streamlit session state
    def add_class(self, class_id, class_name, color_code):
        """Add a new class to the classification scheme"""
        class_name = class_name.strip()
        
        #Validate the class name input, must not empty
        if not class_name:
            st.error("❌ Class name cannot be empty!")
            return False
        #check if ID already exist
        if not st.session_state.lulc_edit_mode:
            if any(c['ID'] == class_id for c in self.classes):
                st.error(f"❌ Class ID {class_id} already exists!")
                return False
        
        #Class updating 
        if st.session_state.lulc_edit_mode and st.session_state.lulc_edit_idx is not None:
            #Update existing class. Adapted from line 205 - 209
            self.classes[st.session_state.lulc_edit_idx] = {
                'ID': class_id,
                'Class Name': class_name,
                'Color Code': color_code
            }
            st.success(f"✅ Class '{class_name}' (ID: {class_id}) updated successfully!")
            st.session_state.lulc_edit_mode = False
            st.session_state.lulc_edit_idx = None
        else:
            # Add new class
            self.classes.append({
                'ID': class_id,
                'Class Name': class_name,
                'Color Code': color_code
            })
            st.success(f"✅ Class '{class_name}' (ID: {class_id}) added successfully!")
        
        #Sort classes by ID (adapted from line 211)
        self.classes = sorted(self.classes, key=lambda x: x['ID'])
        
        # Update next ID (adapted from line 214)
        self.next_id = max([c['ID'] for c in self.classes]) + 1
        
        return True
    #adapted from line 229 onward
    def EditClass(self, idx):
        """Edit an existing class"""
        if 0 <= idx < len(self.classes):
            st.session_state.lulc_edit_mode = True
            st.session_state.lulc_edit_idx = idx
            return self.classes[idx]
        return None
    #adapted from line 247

    def DeleteClass(self, idx):
        """Delete a class from the scheme"""
        if 0 <= idx < len(self.classes):
            class_to_delete = self.classes[idx]
            del self.classes[idx]
            st.success(f"Class '{class_to_delete['Class Name']}' (ID: {class_to_delete['ID']}) deleted successfully!")
            return True
        return False
    
    #new additional function to cancel edit if needed
    def cancel_edit(self):
        """Cancel edit mode"""
        st.session_state.lulc_edit_mode = False
        st.session_state.lulc_edit_idx = None

    #adapted from line 266 - 320
    def processCsvUpload(self, uploaded_file):
        """Process uploaded CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check for required columns
            required_cols = ['ID', 'Kelas Tutupan/Penggunaan Lahan', 'Color Palette']
            alt_cols = ['ID', 'Class Name', 'Color Code']
            alt_cols2 = ['ID', 'Class Name', 'Color']
            
            if all(col in df.columns for col in required_cols):
                # Standard format
                self.classes = []
                for _, row in df.iterrows():
                    self.classes.append({
                        'ID': int(row['ID']),
                        'Class Name': row['Kelas Tutupan/Penggunaan Lahan'],
                        'Color Code': row['Color Palette']
                    })
            #adapted from line 289 (add more tolerance to the template)
            elif all(col in df.columns for col in alt_cols) or all(col in df.columns for col in alt_cols2):
                # Unified alternative format
                color_field = 'Color Code' if 'Color Code' in df.columns else 'Color'
                self.classes = [
                    {
                        'ID': int(row['ID']),
                        'Class Name': row['Class Name'],
                        'Color Code': row[color_field]
                    }
                    for _, row in df.iterrows()
                ]

            else:
                st.error(f"❌ CSV must contain one of these column sets:\n- {required_cols}\n- {alt_cols}\n- {alt_cols2}")
                return False
            
            # Sort by ID
            self.classes = sorted(self.classes, key=lambda x: x['ID'])
            
            # Update next ID
            if self.classes:
                self.next_id = max([c['ID'] for c in self.classes]) + 1
            
            st.success(f"✅ Successfully loaded {len(self.classes)} classes from CSV!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")
            return False
    #adapted from line 327
    def download_csv(self):
        """Generate CSV for download"""
        if not self.classes:
            st.warning("⚠️ No classes to download! Please add classes first.")
            return None
        
        df = self.get_dataframe()
        return df.to_csv(index=False).encode('utf-8')    
    #adapted from line 370
    def get_dataframe(self):
        """Get the classification scheme as a DataFrame"""
        if not self.classes:
            return pd.DataFrame(columns=['ID', 'Land Cover Class', 'Color Palette'])
        
        df = pd.DataFrame(self.classes)
        df = df.rename(columns={
            'ID': 'ID',
            'Class Name': 'Land Cover Class',
            'Color Code': 'Color Palette'
        })
        return df
    #Adapted from line 407
    def load_default_scheme(self, default_classes):
        """Load a default classification scheme"""
        self.classes = default_classes.copy()
        self.classes = sorted(self.classes, key=lambda x: x['ID'])
        
        if self.classes:
            self.next_id = max([c['ID'] for c in self.classes]) + 1
        
        st.success(f"✅ Loaded default scheme with {len(self.classes)} classes!")
        return True
    #new function  to load the user interface in streamlit envinronment 
 
    @staticmethod
    def get_default_schemes():
        """Return available default classification schemes"""
        return {
            "RESTORE+ Project": [
                {'ID': 1, 'Class Name': 'Natural Forest', 'Color Code': "#0E6D0E"},
                {'ID': 2, 'Class Name': 'Agroforestry', 'Color Code': "#F08306"},
                {'ID': 3, 'Class Name': 'Monoculture Plantation', 'Color Code': "#38E638"},
                {'ID': 4, 'Class Name': 'Grassland or Savanna', 'Color Code': "#80DD80"},
                {'ID': 5, 'Class Name': 'Shrub', 'Color Code': "#5F972A"},
                {'ID': 6, 'Class Name': 'Paddy Field', 'Color Code': "#777907"},
                {'ID': 7, 'Class Name': 'Cropland (Palawija, Horticulture)', 'Color Code': "#E8F800"},
                {'ID': 8, 'Class Name': 'Settlement', 'Color Code': "#F81D00"},
                {'ID': 9, 'Class Name': 'Cleared Land', 'Color Code': "#E9B970"},
                {'ID': 10, 'Class Name': 'Waterbody', 'Color Code': "#1512F3"},
            ]
        } 