import os
import shutil

# Specify the path to the folder containing the files and subfolders
parent_folder = 'scans/'

# Loop through all items in the parent folder
for item_name in os.listdir(parent_folder):
    item_path = os.path.join(parent_folder, item_name)
    
    # Check if the item is a directory or file and matches the pattern for numbers greater than 1
    if (os.path.isdir(item_path) or os.path.isfile(item_path)) and item_name[0].isdigit() and "_" in item_name:
        number_part = item_name.split('_')[0]
        
        # Only delete items where the number is greater than 1
        if int(number_part) > 1:
            try:
                # Delete the item (file or folder)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Removes a directory and all its contents
                    print(f"Deleted folder: {item_path}")
                else:
                    os.remove(item_path)  # Removes a file
                    print(f"Deleted file: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
