import random

from flask import render_template, request

from project import app
from project.controllers import home_controller
from project.models.common import Common

project_name = 'motorbike-license-plate-detection'
page_title = 'Mô hình dự đoán biển số xe máy'
page_dataset_url = 'https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection'
page_dataset_name = 'Chars74K'
page_class_names = [
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
	'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

@app.route(f'/{project_name}', methods=['GET'], endpoint=str(random.getrandbits(128)))
def index():
	page = home_controller.license_plate_index()
	page.update(
		{
			'project_url': project_name,
			'title': page_title,
			'dataset_url': page_dataset_url,
			'dataset_name': page_dataset_name
		}
	)
	return render_template(
		'license_plate_index.html',
		page=page
	)

@app.route(f'/{project_name}', methods=['POST'], endpoint=str(random.getrandbits(128)))
def pred_temp_output():
	secret_response = request.form['g-recaptcha-response']
	if home_controller.check_captcha(secret_response):
		page = home_controller.license_plate_index()
		img_output = Common(
			project_name,
			request,
			page_class_names
		).pred_for_license_plate()
		page.update(
			{
				'project_url': project_name,
				'title': page_title,
				'dataset_url': page_dataset_url,
				'dataset_name': page_dataset_name,
				'image': img_output
			}
		)
		return render_template(
			'license_plate_index.html',
			page=page
		)
