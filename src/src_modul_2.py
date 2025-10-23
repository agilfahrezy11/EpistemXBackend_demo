import pandas as pd
import streamlit as st
import random
from typing import List, Dict, Optional, Any, Tuple


class LULC_Scheme_Manager:
    """
    Class for facilitating land cover classification scheme definition.
    
    This class manages the creation, editing, and validation of land use/land cover
    classification schemes through manual input, CSV upload, or default templates.
    All data is stored in Streamlit session state for persistence across page interactions.
    
    Attributes
    ----------
    classes : List[Dict[str, Any]]
        List of classification classes with ID, name, and color information
    next_id : int
        Next available ID for new classes
    """
    
    def __init__(self):
        """Initialize session state variables for LULC scheme management. Sets up all required session state variables"""
        self._init_session_state()
    #adapted from def init 
    #use st.session state to store result, instead of using self.
    #line 14 - 18
    def _init_session_state(self) -> None:
        """Initialize all required session state variables."""
        session_vars = {
            'lulc_classes': [],
            'lulc_next_id': 1,
            'lulc_edit_mode': False,
            'lulc_edit_idx': None,
            'csv_temp_classes': []
        }
        
        for var, default_value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default_value
    #Adapted from widget based layout (jupyter notebook) to streamlit session
    #@ is decorator or wrapper, so that when called, it does not need parenthesis
    #it provide clean access to each variable in streamlit session state
    #The original code (faza) use jupyter notebook UI, which is incompatible with streamlit
    @property
    def classes(self) -> List[Dict[str, Any]]:
        """
        Get classes from session state.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of classification classes, each containing 'ID', 'Class Name', and 'Color Code'
        """
        return st.session_state.lulc_classes
    #adapter from line 46-51 (self.class_id_input)
    @classes.setter
    def classes(self, value: List[Dict[str, Any]]) -> None:
        """
        Set classes in session state.
        
        Parameters
        ----------
        value : List[Dict[str, Any]]
            List of classification classes to store in session state
        """
        st.session_state.lulc_classes = value
    
    @property 
    def next_id(self) -> int:
        """
        Get next available ID from session state.
        
        Returns
        -------
        int
            Next available class ID for new classes
        """
        return st.session_state.lulc_next_id
    
    @next_id.setter
    def next_id(self, value: int) -> None:
        """
        Set next available ID in session state.
        
        Parameters
        ----------
        value : int
            Next available class ID to set
        """
        st.session_state.lulc_next_id = value

    def validate_class_input(self, class_id: Any, class_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate class input parameters.
        
        Parameters
        ----------
        class_id : Any
            Class ID to validate (will be converted to int)
        class_name : str
            Class name to validate
            
        Returns
        -------
        Tuple[bool, Optional[str]]
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        # Validate class_id type
        try:
            class_id = int(class_id)
        except (ValueError, TypeError):
            return False, "Class ID must be a valid number!"
        
        # Validate class name
        class_name = class_name.strip()
        if not class_name:
            return False, "Class name cannot be empty!"
        
        # Check if ID already exists (only for new classes)
        if not st.session_state.lulc_edit_mode:
            if any(c['ID'] == class_id for c in self.classes):
                return False, f"Class ID {class_id} already exists!"
        
        return True, None
    #adapted from line 184, but tailored with streamlit compability    
    def add_class(self, class_id: Any, class_name: str, color_code: str) -> Tuple[bool, str]:
        """
        Add or update a class in the classification scheme.
        
        Parameters
        ----------
        class_id : Any
            Unique identifier for the class (will be converted to int)
        class_name : str
            Human-readable name for the class
        color_code : str
            Hex color code for visualization (e.g., '#FF0000')
            
        Returns
        -------
        Tuple[bool, str]
            Tuple of (success, message). Message contains success or error details.
        """
        # Validate input
        is_valid, error_msg = self.validate_class_input(class_id, class_name)
        if not is_valid:
            return False, error_msg
        
        class_id = int(class_id)
        class_name = class_name.strip()
        
        class_data = {
            'ID': class_id,
            'Class Name': class_name,
            'Color Code': color_code
        }
        
        # Update existing class
        if st.session_state.lulc_edit_mode and st.session_state.lulc_edit_idx is not None:
            self.classes[st.session_state.lulc_edit_idx] = class_data
            success_msg = f"Class '{class_name}' (ID: {class_id}) updated successfully!"
            self._reset_edit_mode()
        else:
            # Add new class
            self.classes.append(class_data)
            success_msg = f"Class '{class_name}' (ID: {class_id}) added successfully!"
        
        self._sort_and_update_next_id()
        return True, success_msg

    #functions for manual input options
    def _reset_edit_mode(self) -> None:
        """
        Reset edit mode session state.
        
        Clears the edit mode flags to return to normal add mode.
        """
        st.session_state.lulc_edit_mode = False
        st.session_state.lulc_edit_idx = None
    
    def _sort_and_update_next_id(self) -> None:
        """
        Sort classes by ID and update next available ID.
        
        Maintains classes in ascending ID order and calculates the next
        available ID for new classes.
        """
        self.classes = sorted(self.classes, key=lambda x: x['ID'])
        
        if self.classes:
            self.next_id = max([c['ID'] for c in self.classes]) + 1
        else:
            self.next_id = 1
    
    def _generate_random_color(self) -> str:
        """
        Generate a random hex color code.
        
        Returns
        -------
        str
            Random hex color code in format '#RRGGBB'
        """
        return f"#{random.randint(0, 0xFFFFFF):06x}"
    
    def _generate_distinct_colors(self, count: int) -> List[str]:
        """
        Generate a list of visually distinct colors.
        
        Parameters
        ----------
        count : int
            Number of distinct colors to generate
            
        Returns
        -------
        List[str]
            List of hex color codes that are visually distinct
        """
        if count <= 0:
            return []
        
        # Predefined distinct colors for better visual separation
        distinct_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
            "#F8C471", "#82E0AA", "#F1948A", "#85C1E9", "#D7BDE2",
            "#A3E4D7", "#F9E79F", "#D5A6BD", "#AED6F1", "#A9DFBF"
        ]
        
        if count <= len(distinct_colors):
            return distinct_colors[:count]
        
        # If we need more colors than predefined, generate random ones
        colors = distinct_colors.copy()
        while len(colors) < count:
            colors.append(self._generate_random_color())
        
        return colors[:count]
    #adapted from line 229 onward
    def edit_class(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Set class for editing mode.
        
        Parameters
        ----------
        idx : int
            Index of the class to edit in the classes list
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Class data if valid index, None otherwise
        """
        if 0 <= idx < len(self.classes):
            st.session_state.lulc_edit_mode = True
            st.session_state.lulc_edit_idx = idx
            return self.classes[idx]
        return None
    #adapted from line 247
    def delete_class(self, idx: int) -> Tuple[bool, str]:
        """
        Delete a class from the scheme.
        
        Parameters
        ----------
        idx : int
            Index of the class to delete
            
        Returns
        -------
        Tuple[bool, str]
            Tuple of (success, message). Message contains operation details.
        """
        if 0 <= idx < len(self.classes):
            class_to_delete = self.classes[idx]
            del self.classes[idx]
            success_msg = f"Class '{class_to_delete['Class Name']}' (ID: {class_to_delete['ID']}) deleted successfully!"
            return True, success_msg
        return False, "Invalid class index"
    
    def cancel_edit(self) -> None:
        """
        Cancel edit mode.
        
        Resets the edit mode flags and returns to normal add mode.
        """
        self._reset_edit_mode()

    #adapted from line 266 - 320
    #Change so that csv is more tolaratable 
    def process_csv_upload(self, df: pd.DataFrame, id_col: str, name_col: str, 
                          color_col: Optional[str] = None) -> Tuple[bool, str]:
        """
        Process CSV upload - validate and prepare for color assignment.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the CSV data
        id_col : str
            Column name containing class IDs
        name_col : str
            Column name containing class names
        color_col : Optional[str], default=None
            Column name containing color codes (optional)
            
        Returns
        -------
        Tuple[bool, str]
            Tuple of (success, message). Message contains details about the operation.
        """
        try:
            class_list = []
            used_ids = set()

            for _, row in df.iterrows():
                class_id = row[id_col]
                class_name = row[name_col]

                # Skip empty rows
                if pd.isna(class_id) or pd.isna(class_name):
                    continue

                # Validate and convert class_id
                try:
                    class_id = int(class_id)
                except (ValueError, TypeError):
                    return False, f"Invalid Class ID format: {class_id}. Must be a number."

                # Check for duplicates
                if class_id in used_ids:
                    return False, f"Duplicate Class ID found: {class_id}"
                used_ids.add(class_id)

                # Handle color assignment
                if color_col and not pd.isna(row[color_col]):
                    color_code = str(row[color_col]).strip()
                    # Validate hex color format
                    if not color_code.startswith('#'):
                        color_code = f"#{color_code}"
                    if len(color_code) != 7:
                        color_code = "#2e8540"  # Fallback to default
                else:
                    color_code = "#2e8540"  # Default color

                class_list.append({
                    "ID": class_id,
                    "Class Name": str(class_name).strip(),
                    "Color Code": color_code
                })

            # If no color column was provided, assign distinct random colors
            if not color_col:
                distinct_colors = self._generate_distinct_colors(len(class_list))
                for i, class_data in enumerate(class_list):
                    class_data["Color Code"] = distinct_colors[i]

            st.session_state.csv_temp_classes = class_list
            color_msg = "with colors from CSV" if color_col else "with auto-generated colors"
            return True, f"Successfully loaded {len(class_list)} classes from CSV {color_msg}"

        except Exception as e:
            return False, f"Error processing CSV: {str(e)}"
    
    def finalize_csv_upload(self, color_assignments: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Finalize CSV upload with user-assigned colors.
        
        Parameters
        ----------
        color_assignments : Optional[List[str]], default=None
            List of hex color codes assigned by user. If None, uses existing colors.
            
        Returns
        -------
        Tuple[bool, str]
            Tuple of (success, message). Message contains operation details.
        """
        try:
            # Update colors based on user assignments if provided
            if color_assignments:
                for i, class_data in enumerate(st.session_state.csv_temp_classes):
                    if i < len(color_assignments):
                        class_data["Color Code"] = color_assignments[i]
            
            # Save to main classes and sort
            self.classes = st.session_state.csv_temp_classes.copy()
            self._sort_and_update_next_id()
            
            # Clear temporary storage
            st.session_state.csv_temp_classes = []
            
            return True, f"Classification scheme created with {len(self.classes)} classes"
            
        except Exception as e:
            return False, f"Error finalizing CSV upload: {str(e)}"
    def get_csv_data(self) -> Optional[bytes]:
        """
        Generate CSV data for download.
        
        Returns
        -------
        Optional[bytes]
            CSV data as bytes for download, None if no classes exist
        """
        if not self.classes:
            return None
        
        df = self.get_dataframe()
        return df.to_csv(index=False).encode('utf-8')
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the classification scheme as a normalized DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized column names for display and export
        """
        if not self.classes:
            return pd.DataFrame(columns=["ID", "Land Cover Class", "Color Palette"])

        df = pd.DataFrame(self.classes)

        # Normalize column names
        column_mapping = {
            "ID": "ID",
            "Class ID": "ID", 
            "Class Name": "Land Cover Class",
            "Land Cover Class": "Land Cover Class",
            "Color": "Color Palette",
            "Color Code": "Color Palette",
            "Color Palette": "Color Palette"
        }

        # Apply column renaming
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Ensure consistent column order
        expected_columns = ["ID", "Land Cover Class", "Color Palette"]
        available_columns = [col for col in expected_columns if col in df.columns]
        
        return df[available_columns]

    #Adapted from line 407
    def load_default_scheme(self, scheme_name: str) -> Tuple[bool, str]:
        """
        Load a predefined classification scheme.
        
        Parameters
        ----------
        scheme_name : str
            Name of the default scheme to load
            
        Returns
        -------
        Tuple[bool, str]
            Tuple of (success, message). Message contains operation details.
        """
        default_schemes = self.get_default_schemes()
        
        if scheme_name not in default_schemes:
            return False, f"Unknown scheme: {scheme_name}"
        
        self.classes = default_schemes[scheme_name].copy()
        self._sort_and_update_next_id()
        
        return True, f"Loaded {scheme_name} with {len(self.classes)} classes"
    #Add RESTORE+ classification scheme
    @staticmethod
    def get_default_schemes() -> Dict[str, List[Dict[str, Any]]]:
        """
        Return available default classification schemes.
        
        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Dictionary mapping scheme names to lists of class definitions.
            Each class contains 'ID', 'Class Name', and 'Color Code'.
        """
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
    
    @staticmethod
    def auto_detect_csv_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Auto-detect ID, Name, and Color columns in CSV.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze for column detection
            
        Returns
        -------
        Tuple[Optional[str], Optional[str], Optional[str]]
            Tuple of (id_column, name_column, color_column). Any can be None if not found.
        """
        columns_lower = [c.lower().replace(" ", "").replace("_", "") for c in df.columns]
        
        def find_column(keywords: List[str]) -> Optional[str]:
            for keyword in keywords:
                for i, col in enumerate(columns_lower):
                    if keyword in col:
                        return df.columns[i]
            return None
        
        id_col = find_column(["id", "classid", "kode", "code"])
        name_col = find_column(["classname", "class", "kelas", "name", "nama"])
        color_col = find_column(["color", "colorcode", "warna", "colour", "hex", "palette"])
        
        return id_col, name_col, color_col 