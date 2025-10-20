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
        # Initialize session state if not exists - use consistent naming
        if 'lulc_classes' not in st.session_state:
            st.session_state.lulc_classes = []
        if 'lulc_next_id' not in st.session_state:
            st.session_state.lulc_next_id = 1
        if 'lulc_edit_mode' not in st.session_state:
            st.session_state.lulc_edit_mode = False
        if 'lulc_edit_idx' not in st.session_state:
            st.session_state.lulc_edit_idx = None
        if 'csv_temp_classes' not in st.session_state:
            st.session_state.csv_temp_classes = []
    
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
        
        # Validate class_id type
        try:
            class_id = int(class_id)
        except (ValueError, TypeError):
            st.error("❌ Class ID must be a valid number!")
            return False
        
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
        if self.classes:
            self.next_id = max([c['ID'] for c in self.classes]) + 1
        else:
            self.next_id = 1
        
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

    def process_csv_upload(self, df, id_col, name_col):
        """Process CSV upload - only ID and Name columns, colors will be assigned later"""
        try:
            # Build class records without colors
            class_list = []
            used_ids = set()

            for _, row in df.iterrows():
                class_id = row[id_col]
                class_name = row[name_col]

                # Skip empty rows
                if pd.isna(class_id) or pd.isna(class_name):
                    continue

                # Convert class_id to int if possible for consistency
                try:
                    class_id = int(class_id)
                except (ValueError, TypeError):
                    st.error(f"Invalid Class ID format: {class_id}. Must be a number.")
                    return False

                # Check duplicates
                if class_id in used_ids:
                    st.error(f"Duplicate Class ID found: {class_id}")
                    return False
                used_ids.add(class_id)

                class_list.append({
                    "ID": class_id,
                    "Class Name": class_name,
                    "Color Code": "#2e8540"  # Default color, will be changed by user
                })

            # Store in session state for color assignment
            if 'csv_temp_classes' not in st.session_state:
                st.session_state.csv_temp_classes = []
            st.session_state.csv_temp_classes = class_list
            return True

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return False
    
    def finalize_csv_upload(self, color_assignments):
        """Finalize CSV upload with user-assigned colors"""
        try:
            # Update colors based on user assignments
            for i, class_data in enumerate(st.session_state.csv_temp_classes):
                if i < len(color_assignments):
                    class_data["Color Code"] = color_assignments[i]
            
            # Save to main classes
            self.classes = st.session_state.csv_temp_classes.copy()
            self.classes = sorted(self.classes, key=lambda x: x['ID'])
            
            # Update next ID
            if self.classes:
                self.next_id = max([c['ID'] for c in self.classes]) + 1
            
            # Clear temporary storage
            st.session_state.csv_temp_classes = []
            return True
            
        except Exception as e:
            st.error(f"Error finalizing CSV upload: {e}")
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
        """Get the classification scheme as a DataFrame, normalized for UI rendering"""
        if not self.classes:
            return pd.DataFrame(columns=["ID", "Land Cover Class", "Color Palette"])

        df = pd.DataFrame(self.classes)

        # Normalize column names from manual, default, or CSV uploads
        rename_map = {
            "ID": "ID",
            "Class ID": "ID",
            "Class Name": "Land Cover Class",
            "Land Cover Class": "Land Cover Class",
            "Color": "Color Palette",
            "Color Code": "Color Palette",
            "Color Palette": "Color Palette"
        }

        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Ensure final column order
        final_cols = ["ID", "Land Cover Class", "Color Palette"]
        df = df[[col for col in final_cols if col in df.columns]]

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
    #default classification scheme (RESTORE+ Project)
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