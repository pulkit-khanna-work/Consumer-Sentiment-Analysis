import os
import zipfile


def zip_da_files(file_names_to_zip, zip_file_name):
  # Open the zip file in write mode
  with zipfile.ZipFile(zip_file_name, 'w') as zipf:
      # Loop through the list of file names
      for file_name in file_names_to_zip:
          # Check if the file exists in the current directory
          if os.path.exists(file_name):
              zip_file_path = os.path.basename(file_name)

              # Add the file to the zip archive
              zipf.write(file_name, arcname=zip_file_path)
          else:
              print(f"Warning: File '{file_name}' not found. Skipping...")