import os.path
import random

from project.models.prediction import *

class Common:
	"""
	Mô hình chung cho mọi loại dự đoán
	Args:
		project_name: Tên của mô hình dự đoán
		my_request: Các yêu cầu từ người dùng từ Internet
		class_names: List/Tuple chứa nhãn của tập dữ liệu
	"""

	def __init__(
		self,
		project_name=None,
		my_request=None,
		class_names=None
	):

		self.save_dir = 'static' if not os.path.exists('/imgs') else '/imgs'

		if not os.path.exists(self.save_dir):
			os.mkdir(self.save_dir)

		self.project_name = project_name
		self.full_path = os.path.join(self.save_dir, self.project_name)

		if not os.path.exists(self.full_path):
			os.mkdir(self.full_path)

		self.my_request = my_request
		self.file_name = str(random.getrandbits(64))
		self.file_name_with_ex = f'{self.file_name}.jpg'
		self.full_path_of_file = os.path.join(self.full_path, self.file_name_with_ex)
		self.my_request.files['file'].save(os.path.join(self.full_path, self.file_name_with_ex))
		self.file_over_cdn = f'https://cdn.baoit.xyz/ai_project/{self.project_name}/{self.file_name_with_ex}'
		self.class_names = class_names

	def pred_for_brain_tumor(self):
		"""
		Chạy mô hình dự đoán bệnh u não
		Returns:
			List/Tuple kết quả dự đoán & xuất ảnh để hiển thị lên màn hình người dùng
		"""
		try:
			a = BrainTumor(
				self.full_path_of_file,
				self.class_names,
				'brain_tumor_model'
			)
			return a.tf_results, self.file_over_cdn
		except Exception as ex:
			log_errors(ex)

	def pred_for_covid19(self):
		"""
		Chạy mô hình dự đoán Covid19
		Returns:
			List/Tuple kết quả dự đoán & xuất ảnh để hiển thị lên màn hình người dùng
		"""
		try:
			a = Covid19(
				self.full_path_of_file,
				self.class_names,
				'covid_19_model'
			)
			return a.tf_results, self.file_over_cdn
		except Exception as ex:
			log_errors(ex)

	# def pred_for_weather(self):
	# 	"""
	# 	Chạy mô hình dự đoán thời tiết
	# 	Returns:
	# 		List/Tuple kết quả dự đoán & xuất ảnh để hiển thị lên màn hình người dùng
	# 	"""
	# 	try:
	# 		a = RoboflowDefault(
	# 			self.full_path_of_file,
	# 			'weather-classification-w5xug',
	# 			'nbernU2pi3cCX5AH4VTi',
	# 			8
	# 		)
	# 		a.__call__()
	# 		return a.rb_results, self.file_over_cdn
	# 	except Exception as ex:
	# 		log_errors(ex)

	def pred_for_tyre_quality(self):
		"""
		Chạy mô hình dự đoán chất lượng lốp xe
		Returns:
			List/Tuple kết quả dự đoán & xuất ảnh để hiển thị lên màn hình người dùng
		"""
		try:
			a = TyreQuality(
				self.full_path_of_file,
				self.class_names,
				'tyre_quality_model'
			)
			return a.tf_results, self.file_over_cdn
		except Exception as ex:
			log_errors(ex)
