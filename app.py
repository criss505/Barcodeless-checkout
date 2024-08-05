from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import numpy as np
import os
import requests
from io import BytesIO
import shutil
from datetime import datetime
from psycopg2 import OperationalError
from sqlalchemy import text

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:pass@localhost:5433/barcodeless'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define the Product model
class Product(db.Model):
    __tablename__ = 'Products'  # Capitalized table name

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    price = db.Column(db.Integer)
    description = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        return f'<Product {self.name}>'

# Define the Issues model
class Issue(db.Model):
    __tablename__ = 'Issues'

    id = db.Column(db.Integer, primary_key=True)
    prediction = db.Column(db.String(50))
    path = db.Column(db.String(255))
    checked = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<Issue {self.prediction}>'


# Define config folders
UPLOAD_FOLDER = 'static/uploads'
CROPPED_FOLDER = 'static/cropped'
REPORTS_FOLDER = 'static/reports'
for folder in [UPLOAD_FOLDER, CROPPED_FOLDER, REPORTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CROPPED_FOLDER'] = CROPPED_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER

# Set roboflow model
from roboflow import Roboflow
rf = Roboflow(api_key="mGwEA9bmy0nOwOEAfDZY")
project = rf.workspace().project("barcodeless-project")
model = project.version(2).model


@app.before_request
def check_service_health():
    # Check the product identification service health
    try:
        response = requests.get('http://localhost:5001/health-check')
        if response.status_code != 200:
            raise Exception("Service returned non-OK status.")
    except requests.ConnectionError:
        flash('Product identification service is down', 'error')
        return render_template('service_down.html')

    # Check the database connection health
    try:
        exists = db.session.query(db.exists().where(Product.id == 1)).scalar()
        if not exists:
            raise Exception("Database query returned no results. Database might be empty.")
    except Exception as e:
        flash(f'Database service is down: {str(e)}', 'error')
        return render_template('service_down.html')


@app.cli.command('db_create')
def db_create():
    db.create_all()
    print('Database created!')

@app.route('/')
def index():
    uploaded_image_url = session.get('uploaded_image_url', None)
    return render_template('index.html', uploaded_image_url=uploaded_image_url)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file and file.filename:
        filename = file.filename
        filepath = app.config['UPLOAD_FOLDER'] + '/' + filename
        file.save(filepath)
        session['uploaded_image_url'] = url_for('static', filename='uploads/' + filename)
        session['filename'] = filename

        # make predictions
        img = Image.open(filepath)
        img.thumbnail((1024, 1024))
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)
        img_array = np.array(Image.open(img_byte_arr))
        detections = model.predict(img_array, confidence=40, overlap=30).json()
        simple_detections = [{'x': det['x'], 'y': det['y'], 'width': det['width'], 'height': det['height'],
                        'class': det['class'], 'class_id': det['class_id']} for det in detections['predictions']]
        print(simple_detections)
        session['detections'] = simple_detections

        return redirect(url_for('index')) 
    flash('No file selected or file is empty.', 'error')
    return redirect(url_for('index'))

@app.route('/confirmation')
def confirmation():
    filename = session.get('filename')
    if not filename:
        flash('No file processed or file missing.', 'error')
        return redirect(url_for('index'))

    file_path = app.config['UPLOAD_FOLDER'] + '/' + filename
    img = Image.open(file_path)
    img.thumbnail((1024, 1024))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=85)
    img_byte_arr.seek(0)
    img_array = np.array(Image.open(img_byte_arr))

    cropped_files = []

    # crop and save images
    for i, detection in enumerate(session['detections']):
        x_center = int(detection['x'])
        y_center = int(detection['y'])
        width = int(detection['width'])
        height = int(detection['height'])
        x = x_center - width // 2
        y = y_center - height // 2

        # crop and save each image
        cropped_img = img_array[y:y+height, x:x+width]
        cropped_pil = Image.fromarray(cropped_img)
        cropped_filename = f'cropped_{i}.jpg'
        cropped_filepath = app.config['CROPPED_FOLDER'] + '/' + cropped_filename
        cropped_pil.save(cropped_filepath)
        cropped_files.append(cropped_filepath)
        detection['image_url'] = cropped_filepath

    identified_objects = []

    # send each cropped image for prediction
    for cropped_filepath in cropped_files:
        print(cropped_filepath)
        with open(cropped_filepath, 'rb') as img_file:
            response = requests.post('http://localhost:5001/predict', files={'file': img_file})
            if response.status_code == 200:
                predicted_class_id = response.json()['class_id']

                # fetch product details from the database using the predicted class id
                product = Product.query.filter_by(id=predicted_class_id+1).first()
                if product:
                    identified_objects.append({
                        'class': product.name,
                        'image_url': cropped_filepath
                    })
                else:
                    identified_objects.append({
                        'class': 'Unknown',
                        'image_url': cropped_filepath
                    })
            else:
                identified_objects.append({
                    'class': 'Error',
                    'image_url': cropped_filepath
                })
    
    session['identified_objects'] = identified_objects

    return render_template('confirmation.html', results=identified_objects)

@app.route('/checkout')
def checkout():
    identified_objects = session.get('identified_objects', [])
    product_counts = {}
    products_list = []
    total_price = 0

    # count the occurrences of each identified product
    for obj in identified_objects:
        product_name = obj['class'].lower()
        if product_name in product_counts:
            product_counts[product_name] += 1
        else:
            product_counts[product_name] = 1

    # fetch product details and calculate total price
    for product_name, count in product_counts.items():
        product = Product.query.filter(db.func.lower(Product.name) == product_name).first()
        if product:
            total_price += product.price * count
            products_list.append({
                'name': product.name,
                'quantity': count,
                'unit_price': product.price,
                'total_price': product.price * count
            })

    return render_template('checkout.html', results=products_list, total_price=total_price)

@app.route('/pay', methods=['POST'])
def pay():
    return render_template('pay.html')

@app.route('/end_transaction', methods=['GET', 'POST'])
def end_transaction():
    session.clear()
    return render_template('end_transaction.html')

@app.route('/report_problem', methods=['POST'])
def report_problem():
    try:
        data = request.get_json()
        image_url = data.get('imageUrl', None)
        print(image_url)
        image_class = data.get('imageClass', None)
        print(image_class)
        if image_url:
            # copy image in reports folder
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{image_class}_{current_time}.jpeg"
            target_directory = app.config['REPORTS_FOLDER'] + '/' + new_filename
            shutil.copy(image_url, target_directory)

            # create new issue
            new_issue = Issue(
                prediction = image_class,
                path = target_directory,
                checked = False
            ) 
            
            # add issue to database
            try:
                db.session.add(new_issue)
                db.session.commit()
                # remove reported image from session detections
                new_detections = [det for det in session.get('detections', []) if det['image_url'] != image_url]
                session['detections'] = new_detections

                return jsonify({'message': 'Report successful'}), 200
            except Exception as e:
                db.session.rollback()
                print(f"Error adding issue to DB: {e}")
                return jsonify({'error': str(e)}), 500

        else:
            return jsonify({'error': 'No image specified'}), 400
    except Exception as e:
        print("Error processing the request:", str(e))
        return jsonify({'error': 'Server encountered an error'}), 500

@app.route('/abort_basket')
def abort_basket():
    session.clear()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host='127.0.0.24', port=5000, debug=True)
