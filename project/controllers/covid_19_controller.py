import random

from flask import render_template, request

from project import app
from project.controllers import home_controller
from project.models.common import Common

project_name = 'covid-19'
page_title = 'Mô hình dự đoán viêm phổi do Covid 19 qua ảnh chụp X-quang'
page_dataset_url = 'https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images'
page_dataset_name = 'Lung Diseases by Covid-19'
page_class_names = sorted(['COVID', 'Non-COVID'])

@app.route(f'/{project_name}', methods=['GET'], endpoint=str(random.getrandbits(128)))
def index():
	page = home_controller.default_index()
	page.update(
		{
			'project_url': project_name,
			'title': page_title,
			'dataset_url': page_dataset_url,
			'dataset_name': page_dataset_name,
			'val_acc': {
				'label': page_class_names
			}
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
			request,
			page_class_names
		).pred_for_covid19()
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
					'label': page_class_names
				}
			}
		)
		return render_template(
			'index.html',
			page=page
		)
