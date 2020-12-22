from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()

drive = GoogleDrive(gauth)

# Upload a json file (will be identical for a pickle file)
json_file = drive.CreateFile()
json_file.SetContentFile('test.json')
json_file.Upload() # Files.insert()
print json_file['title'], json_file['id']