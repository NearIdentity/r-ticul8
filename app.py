'''

r-ticul8 0.0

Web Application for Insight Health Data Science Fellowship, Boston, MA -- 2018A

Author: Abdullah Al Rashid

Last Revision: 2018.01.26


'''

import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import backend
import random

# Flags for Debugging if Needed
dev_status = True
prod_status = not(dev_status)

# App Initialisation
app = Flask(__name__)

# Uploads Directory
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Images Directory
app.config['IMAGES_FOLDER'] = 'images/'

# Admissible File Extensions for Uploading
app.config['ALLOWED_EXTENSIONS'] = set(['wav'])	

def allowed_file(filename):	# For a given file, returning allowed type
	return '.' in filename and \
			filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# App Root
@app.route('/')
def index():
	return render_template('index.html')

# Route for Processing File Upload and Displaying Results
@app.route('/upload', methods=['POST'])
def upload():
	''' '''
	backend.delete_uploaded_files()
	backend.delete_result_image()	
	''' '''

	# Parsing Names of Files to be Uploaded
	uploaded_files = request.files.getlist("file[]")
	filenames = []
	for file_handle in uploaded_files:
		# Checking for File Extension Admissibility
		if file_handle and allowed_file(file_handle.filename):
			# Securing File-Name 
			filename = secure_filename(file_handle.filename)
			# Saving File in Uploads Folder
			file_handle.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
			# Adding File-Name into a List for Later Use
			filenames.append(filename)
	''' 

	Internal Processing of Data from Uploaded Files		

	'''
	''' '''
	X_test, app_feature_data_dict, sample_feat_data_dict = backend.process_sample()
	y_pred = backend.sample_prediction(X_test)

	img_token = random.randint(1,1001)
	imagename = "Results_" + str(img_token) + ".png"

	flags = backend.generate_results(app_feature_data_dict, sample_feat_data_dict, y_pred, image_token=img_token)
	
	prediction = "Unimpaired"
	if y_pred == 1:
		prediction = "Impaired"
	''' '''	
	# Redirection to the Next App Route
	return render_template('upload.html', prediction=prediction, filenames=filenames, flags=flags, imagename=imagename)
 
# App Route for Serving Uploaded Files BAck
@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Routing of Images from Designated Folder
@app.route('/images/<imagefilename>')
def image_file(imagefilename):
	return send_from_directory(app.config['IMAGES_FOLDER'], imagefilename)

if __name__ == '__main__':
	app.run(debug=True, port=5003)
