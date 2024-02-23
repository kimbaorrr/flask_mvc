import secrets

from flask_minify import minify

from project import app

# Cấu hình Web Server
if __name__ == '__main__':
	app.config['MAX_CONTENT_LENGTH'] = int(15e6)
	app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
	app.config['SECRET_KEY'] = secrets.token_hex()
	minify(app, html=True, js=True, cssless=True)
	app.run(host='0.0.0.0', port=5003)
