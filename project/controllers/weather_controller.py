import random

from flask import render_template, request

from project import app
from project.controllers import home_controller
from project.models.common import Common

project_name = 'weather'
page_title = 'Mô hình dự đoán thời tiết'
page_dataset_url = 'https://universe.roboflow.com/my-workspace-jlq0f/weather-classification-w5xug'
page_dataset_name = 'Weather Classification'

@app.route(f'/{project_name}', methods=['GET'], endpoint=str(random.getrandbits(128)))
def index():
	page = home_controller.default_index()
	page.update(
		{
			'project_url': project_name,
			'title': page_title,
			'dataset_url': page_dataset_url,
			'dataset_name': page_dataset_name
		}
	)
	return render_template(
		'index.html',
		page=page
	)

@app.route(f'/{project_name}', methods=['POST'], endpoint=str(random.getrandbits(128)))
def pred_temp_output():
	secret_response = request.form['g-recaptcha-response']
	if home_controller.check_captcha(secret_response):
		page = home_controller.default_index()
		pred, img_output = Common(
			project_name,
			request
		).pred_for_weather()
		page.update(
			{
				'project_url': project_name,
				'title': page_title,
				'dataset_url': page_dataset_url,
				'dataset_name': page_dataset_name,
				'image': img_output,
				'result': pred[0],
				'acc': pred[1],
				'loss': pred[2],
				'val_acc': {
					'data': pred[3],
					'label': pred[4]
				}
			}
		)
		return render_template(
			'index.html',
			page=page
		)
