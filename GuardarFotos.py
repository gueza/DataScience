from flask import Flask, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = "C:/Users/steza/Documentos/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH 


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           

@app.route('/subir', methods = ['GET', 'POST'])
def upload_file():
    categoria = request.form.get('categoria')

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
            
        file = request.files['file']

        if file.filename == '':
            return 'Imagen no seleccionada'
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],categoria, filename))
            return 'Imagen guardada'
        else:
            return 'Extension no permitida' 
            #return format(categoria)

if __name__ == '__main__':
    app.run()
