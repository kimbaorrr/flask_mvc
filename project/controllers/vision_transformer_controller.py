import random

from flask import render_template, request

from project import app
from project.controllers import home_controller
from project.models.common import Common

project_name = 'vision-transformer'
page_title = 'Mô hình phân loại hình ảnh Vision Transformer (ViT)'
page_dataset_url = 'https://www.kaggle.com/competitions/imagenet-object-localization-challenge'
page_dataset_name = 'ImageNet'

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
		).pred_for_vit()
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
