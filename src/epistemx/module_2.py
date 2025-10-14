import pandas as pd
import ipywidgets as widgets
from ipywidgets import VBox, HBox, Button, Text, Output, HTML, ColorPicker, IntText, FileUpload
from IPython.display import display, clear_output
import io

# ----- System response 2.1.b -----
class LULCSchemeClass:
    """
    Module 2: Land Cover Classification Scheme Manager
    Allows users to define, edit, and manage land cover classification schemes
    """
    
    def __init__(self):
        self.classes = []
        self.next_id = 1
        self.output = Output()
        self.scheme_table = VBox()
        
        # File upload widget
        self.file_upload = FileUpload(
            accept='.csv',
            multiple=False,
            description='Upload CSV',
            layout=widgets.Layout(width='200px')
        )
        self.file_upload.observe(self.ProcessCsvUpload, names='value')
        
        self.SetupUI()
    
    def SetupUI(self):
        """Setup the user interface"""
        
        # Title
        title = HTML("""
        <h3 style='color: #000000; padding-bottom: 5px;'>
            Define Land Cover/Use Classes
        </h3>
        """)
        
        # Add new class section
        add_title = HTML("""
        <h4 style='color: #000000; margin-top: 20px;'>Add a new class</h4>
        """)
        
        self.class_id_input = IntText(
            value=self.next_id,
            description='',
            placeholder='Class ID',
            disabled=False,
            layout=widgets.Layout(width='100px')
        )
        
        self.class_name_input = Text(
            value='',
            description='',
            placeholder='Enter class name (e.g., Hutan, Permukiman)',
            disabled=False,
            layout=widgets.Layout(width='300px')
        )
        
        self.color_picker = ColorPicker(
            concise=False,
            description='',
            value='#2e8540',
            disabled=False,
            layout=widgets.Layout(width='150px')
        )
        
        self.add_button = Button(
            description='Add Class',
            button_style='success',
            icon='plus',
            layout=widgets.Layout(width='120px')
        )
        self.add_button.on_click(self.AddClass)
        
        input_row = HBox([
            self.class_id_input,
            self.class_name_input,
            self.color_picker,
            self.add_button
        ], layout=widgets.Layout(margin='5px 0'))
        
        # Download button
        self.download_button = Button(
            description='Download as CSV',
            button_style='primary',
            icon='download',
            layout=widgets.Layout(width='200px', height='40px', margin='20px 0')
        )
        self.download_button.on_click(self.DownloadCsv)
        
        # Layout
        scheme_title = HTML("""
        <h4 style='color: #2c3e50; margin-top: 15px;'>Defined classes</h4>
        """)
        
        self.ui = VBox([
            title,
            HTML("<hr style='border: 1px solid #ddd;'>"),
            add_title,
            input_row,
            HTML("<hr style='border: 1px solid #ddd;'>"),
            scheme_title,
            self.scheme_table,
            self.download_button,
            self.output
        ], layout=widgets.Layout(padding='20px'))
        
        self.UpdateSchemeDisplay()
    
    def UpdateSchemeDisplay(self):
        """Update the classification scheme display"""
        rows = []
        
        for i, class_data in enumerate(self.classes):
            color_display = ColorPicker(
                value=class_data["Color Code"],
                layout=widgets.Layout(width='150px'),
                description=''
            )
            color_display.observe(lambda change, idx=i: self.UpdateColor(idx, change), names='value')
            
            edit_button = Button(
                description="Edit",
                button_style="info",
                icon='edit',
                layout=widgets.Layout(width='80px', margin='2px')
            )
            edit_button.on_click(lambda b, idx=i: self.EditClass(idx))
            
            delete_button = Button(
                description="Delete",
                button_style="danger",
                icon='trash',
                layout=widgets.Layout(width='80px', margin='2px')
            )
            delete_button.on_click(lambda b, idx=i: self.DeleteClass(idx))
            
            actions = HBox([edit_button, delete_button])
            
            row = widgets.GridBox(
                children=[
                    HTML(f"<div style='text-align:center; padding: 8px;'><strong>{class_data['ID']}</strong></div>"),
                    HTML(f"<div style='padding: 8px;'>{class_data['Class Name']}</div>"),
                    HBox([color_display, HTML(f"<div style='padding: 8px;'>{class_data['Color Code']}</div>")]),
                    actions
                ],
                layout=widgets.Layout(
                    grid_template_columns='80px 300px 350px 200px',
                    width='100%',
                    padding='4px',
                    border_bottom='1px solid #eee'
                )
            )
            rows.append(row)
        
        if not rows:
            rows = [HTML("""
            <div style='background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;'>
                <p style='margin: 0;'>No classes defined yet. Add your first class above!</p>
            </div>
            """)]
        
        header = widgets.GridBox(
            children=[
                HTML("<div style='padding: 8px;'><strong>ID</strong></div>"),
                HTML("<div style='padding: 8px;'><strong>Class Name</strong></div>"),
                HTML("<div style='padding: 8px;'><strong>Color Code</strong></div>"),
                HTML("<div style='padding: 8px;'><strong>Actions</strong></div>")
            ],
            layout=widgets.Layout(
                grid_template_columns='80px 300px 350px 200px',
                width='100%',
                padding='8px',
                background_color='#3498db',
                border_bottom='2px solid #2980b9'
            )
        )
        
        self.scheme_table.children = [header] + rows
    
    def AddClass(self, b=None):
        """Add a new class to the classification scheme"""
        class_id = self.class_id_input.value
        class_name = self.class_name_input.value.strip()
        color_code = self.color_picker.value
        
        # Validation
        if not class_name:
            with self.output:
                clear_output()
                print("Error: Class name cannot be empty!")
            return
        
        # Check if ID already exists
        if any(c['ID'] == class_id for c in self.classes):
            with self.output:
                clear_output()
                print(f"Error: Class ID {class_id} already exists!")
            return
        
        # Add the class
        self.classes.append({
            'ID': class_id,
            'Class Name': class_name,
            'Color Code': color_code
        })
        
        # Sort classes by ID
        self.classes.sort(key=lambda x: x['ID'])
        
        # Update next ID
        self.next_id = max([c['ID'] for c in self.classes]) + 1
        self.class_id_input.value = self.next_id
        
        # Clear inputs
        self.class_name_input.value = ''
        self.color_picker.value = '#2e8540'
        
        # Update display
        self.UpdateSchemeDisplay()
        
        with self.output:
            clear_output()
            print(f"Class '{class_name}' (ID: {class_id}) added successfully!")
    
    def EditClass(self, idx):
        """Edit an existing class"""
        if 0 <= idx < len(self.classes):
            class_to_edit = self.classes[idx]
            
            # Pre-fill the input fields
            self.class_id_input.value = class_to_edit['ID']
            self.class_name_input.value = class_to_edit['Class Name']
            self.color_picker.value = class_to_edit['Color Code']
            
            # Remove the class (will be re-added with updated values)
            del self.classes[idx]
            self.UpdateSchemeDisplay()
            
            with self.output:
                clear_output()
                print(f"Editing class ID {class_to_edit['ID']}. Modify the fields above and click 'Add Class' to save changes.")
    
    def DeleteClass(self, idx):
        """Delete a class from the scheme"""
        if 0 <= idx < len(self.classes):
            class_to_delete = self.classes[idx]
            del self.classes[idx]
            self.UpdateSchemeDisplay()
            
            with self.output:
                clear_output()
                print(f"Class '{class_to_delete['Class Name']}' (ID: {class_to_delete['ID']}) deleted successfully!")
    
    def UpdateColor(self, idx, change):
        """Update color when color picker is changed"""
        if change['name'] == 'value' and 0 <= idx < len(self.classes):
            self.classes[idx]['Color Code'] = change['new']
            with self.output:
                clear_output()
                print(f"Color updated for class '{self.classes[idx]['Class Name']}'")
    
    def ProcessCsvUpload(self, change):
        """Process uploaded CSV file"""
        if not self.file_upload.value:
            return
        
        try:
            uploaded_file = list(self.file_upload.value.values())[0]
            content = uploaded_file['content']
            df = pd.read_csv(io.BytesIO(content))
            
            # Check for required columns
            required_cols = ['ID', 'Kelas Tutupan/Penggunaan Lahan', 'Color Palette']
            alt_cols = ['ID', 'Class Name', 'Color']
            
            if all(col in df.columns for col in required_cols):
                # Standard format
                self.classes = []
                for _, row in df.iterrows():
                    self.classes.append({
                        'ID': int(row['ID']),
                        'Class Name': row['Kelas Tutupan/Penggunaan Lahan'],
                        'Color Code': row['Color Palette']
                    })
            elif all(col in df.columns for col in alt_cols):
                # Alternative format
                self.classes = []
                for _, row in df.iterrows():
                    self.classes.append({
                        'ID': int(row['ID']),
                        'Class Name': row['Class Name'],
                        'Color Code': row['Color']
                    })
            else:
                with self.output:
                    clear_output()
                    print(f"CSV must contain columns: {required_cols} or {alt_cols}")
                return
            
            # Sort by ID
            self.classes.sort(key=lambda x: x['ID'])
            
            # Update next ID
            if self.classes:
                self.next_id = max([c['ID'] for c in self.classes]) + 1
                self.class_id_input.value = self.next_id
            
            self.UpdateSchemeDisplay()
            
            with self.output:
                clear_output()
                print(f"Successfully loaded {len(self.classes)} classes from CSV!")
            
            # Reset uploader
            self.file_upload.value.clear()
            self.file_upload._counter = 0
            
        except Exception as e:
            with self.output:
                clear_output()
                print(f"Error processing CSV: {e}")
    
    def DownloadCsv(self, b):
        """Download the classification scheme as CSV"""
        if not self.classes:
            with self.output:
                clear_output()
                print("No classes to download! Please add classes first.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.classes)
        df = df.rename(columns={
            'ID': 'ID',
            'Class Name': 'Kelas Tutupan/Penggunaan Lahan',
            'Color Code': 'Color Palette'
        })
        
        # Generate CSV
        csv_string = df.to_csv(index=False)
        
        # Save to file
        filename = 'classification_scheme.csv'
        df.to_csv(filename, index=False)
        
        with self.output:
            clear_output()
            display(HTML(f"""
            <div style='background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745;'>
                <p style='margin: 0; color: #155724;'>
                    <strong>Classification scheme saved successfully!</strong><br>
                    File saved as: <code>{filename}</code><br>
                    Total classes: {len(self.classes)}
                </p>
            </div>
            """))
            
            print("\nCSV Preview:")
            print(csv_string)
            
            print("\n" + "="*70)
            print("DataFrame Preview:")
            print("="*70)
            display(df)
    
    def GetDataframe(self):
        """Get the classification scheme as a DataFrame for use in Module 3"""
        if not self.classes:
            return pd.DataFrame(columns=['ID', 'Kelas Tutupan/Penggunaan Lahan', 'Color Palette'])
        
        df = pd.DataFrame(self.classes)
        df = df.rename(columns={
            'ID': 'ID',
            'Class Name': 'Kelas Tutupan/Penggunaan Lahan',
            'Color Code': 'Color Palette'
        })
        return df
    
    def Display(self):
        """Display the UI"""
        display(self.ui)


# Helper functions for save/load
def SaveClassificationScheme(classifier, filename='lc_scheme.csv'):
    """Save the current classification scheme to a CSV file"""
    df = classifier.GetDataframe()
    if len(df) > 0:
        df.to_csv(filename, index=False)
        print(f"Classification scheme saved to {filename}")
        return True
    else:
        print("No classes to save!")
        return False


def LoadClassificationScheme(classifier, filename='lc_scheme.csv'):
    """Load a classification scheme from a CSV file"""
    try:
        df = pd.read_csv(filename)
        classifier.classes = []
        
        for idx, row in df.iterrows():
            classifier.classes.append({
                'ID': int(row['ID']),
                'Class Name': row['Kelas Tutupan/Penggunaan Lahan'],
                'Color Code': row['Color Palette']
            })
        
        classifier.next_id = max([c['ID'] for c in classifier.classes]) + 1
        classifier.class_id_input.value = classifier.next_id
        classifier.UpdateSchemeDisplay()
        
        print(f"Classification scheme loaded from {filename}")
        print(f"Loaded {len(classifier.classes)} classes")
        return True
    except FileNotFoundError:
        print(f"File {filename} not found!")
        return False
    except Exception as e:
        print(f"Error loading file: {e}")
        return False