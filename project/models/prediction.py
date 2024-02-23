import os

import numpy as np
import tensorflow_hub as hub
from keras.models import load_model

from project.models.logs import log_errors
from project.models.tools import image_processing
import gc

model_loc = '/models_h5'

def get_results_from_result_of_pred(result_of_pred=None, class_names=None):
	"""
	Trả kết quả từ tập result_of_pred (All Default)
	"""
	try:
		top_acc = np.round(100 * np.max(result_of_pred), 2)
		top_loss = np.round(100 - top_acc, 2)
		top_result = class_names[np.argmax(result_of_pred)]
		acc_of_pred = np.round(result_of_pred * 100, 2).tolist()
		return top_result, top_acc, top_loss, acc_of_pred, class_names
	except Exception as ex:
		log_errors(ex)

class TFDefault:
	"""
	Default Constructor for Tensorflow
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None,
			img_size=(224, 224)
	):
		self.img_save_path = img_save_path
		self.tf_model_name = f'{tf_model_name}.h5'
		self.custom_object = False
		self.class_names = class_names
		self.img_size = img_size
		self.tf_results = None

		try:
			# Thiết lập chung
			tf_model_loc = os.path.join(model_loc, self.tf_model_name)
			# Nạp mô hình
			model = load_model(
				tf_model_loc, custom_objects={'KerasLayer': hub.KerasLayer}
			) if self.custom_object else load_model(tf_model_loc)
			# Xử lý ảnh đầu vào
			output_img = image_processing(self.img_save_path, self.img_size)
			# Dự đoán giá trị ảnh
			result_of_pred = np.array(model.predict(output_img)[0], dtype=float)
			# Clear memory
			_ = gc.collect()
			self.tf_results = get_results_from_result_of_pred(result_of_pred, self.class_names)
		except Exception as ex:
			log_errors(ex)
			_ = gc.collect()


# class RoboflowDefault:
# 	"""
# 	Default Constructor for Roboflow
# 	Args:
# 		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
# 		rb_project_name: Tên của dự án trên Roboflow
# 		rb_api_key: Mã API thông báo
# 		rb_version: Phiên bản của model
# 	"""
#
# 	def __init__(
# 		self,
# 		img_save_path=None,
# 		rb_project_name=None,
# 		rb_api_key=None,
# 		rb_version=None
# 	):
# 		self.rb_project_name = rb_project_name
# 		self.rb_api_key = rb_api_key
# 		self.rb_version = rb_version
# 		self.model = None
# 		self.img_save_path = img_save_path
# 		self.class_names = []
# 		self.rb_results = None
#
# 	def __call__(self, *args, **kwargs):
# 		try:
# 			list_of_pred = []
# 			self.model_download()
# 			du_doan = self.model.predict(self.img_save_path)
# 			for a in du_doan[0]['predictions']:
# 				list_of_pred.append(float(a['confidence']))
# 				self.class_names.append(str(a['class']).capitalize())
# 			result_of_pred = np.asarray(list_of_pred, dtype=float)
# 			self.rb_results = get_results_from_result_of_pred(result_of_pred, self.class_names)
# 		except Exception as ex:
# 			log_errors(ex)
#
# 	def model_download(self):
# 		rf = Roboflow(api_key=self.rb_api_key)
# 		project = rf.workspace().project(self.rb_project_name)
# 		model = project.version(self.rb_version).model
# 		self.model = model

# class HuggingPageDefault:
# 	def __init__(
# 		self,
# 		img_save_path=None,
# 		hg_api_token=None,
# 		hg_model_url=None
# 	):
# 		self.img_save_path = img_save_path
# 		self.hg_api_token = hg_api_token
# 		self.hg_model_url = hg_model_url
# 		self.class_names = []
# 		self.hg_results = None
#
# 	def __call__(self, *args, **kwargs):
# 		try:
# 			list_of_pred = []
# 			hg_url = f'https://api-inference.huggingface.co/models/{self.hg_model_url}'
# 			headers = {"Authorization": f"Bearer {self.hg_api_token}"}
# 			with open(self.img_save_path, "rb") as f:
# 				data = f.read()
# 				du_doan = requests.post(hg_url, headers=headers, data=data).json()
# 			for a in du_doan:
# 				list_of_pred.append(float(a['score']))
# 				self.class_names.append(str(a['label']).capitalize())
# 			result_of_pred = np.asarray(list_of_pred, dtype=float)
# 			self.hg_results = get_results_from_result_of_pred(result_of_pred, self.class_names)
# 		except Exception as ex:
# 			log_errors(ex)
class BrainTumor(TFDefault):
	"""
	Mô hình dự đoán bệnh u não qua ảnh chụp cộng hưởng từ (MRI)
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	Returns:
		List/Tuple chứa kết quả, độ chính xác, sai số của giá trị dự đoán
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None
	):
		super().__init__(img_save_path, class_names, tf_model_name)

class Covid19(TFDefault):
	"""
	Mô hình dự đoán bệnh viêm phổi qua ảnh chụp X-quang
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	Returns:
		List/Tuple chứa kết quả, độ chính xác, sai số của giá trị dự đoán
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None
	):
		super().__init__(img_save_path, class_names, tf_model_name, (128, 128))

class TyreQuality(TFDefault):
	"""
	Mô hình dự đoán chất lượng lốp xe
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	Returns:
		List/Tuple chứa kết quả, độ chính xác, sai số của giá trị dự đoán
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None
	):
		super().__init__(img_save_path, class_names, tf_model_name, (128, 128))

