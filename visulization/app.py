from flask import Flask, jsonify, request, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///images.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(100), nullable=False)
    
    x_error_close_mean = db.Column(db.Float(precision=20), nullable=False)
    x_error_far_mean = db.Column(db.Float(precision=20), nullable=False)
    z_error_close_mean = db.Column(db.Float(precision=20), nullable=False)
    z_error_far_mean = db.Column(db.Float(precision=20), nullable=False)
    
    x_error_close_mean_large = db.Column(db.Boolean, nullable=False)
    x_error_far_mean_large = db.Column(db.Boolean, nullable=False)
    z_error_close_mean_large = db.Column(db.Boolean, nullable=False)
    z_error_far_mean_large = db.Column(db.Boolean, nullable=False)
    
    x_error_close_mean_20 = db.Column(db.Boolean, nullable=False)
    x_error_far_mean_20 = db.Column(db.Boolean, nullable=False)
    z_error_close_mean_20 = db.Column(db.Boolean, nullable=False)
    z_error_far_mean_20 = db.Column(db.Boolean, nullable=False)
    
    scene = db.Column(db.String(20), nullable=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/images', methods=['GET'])
def get_images():
    images = Image.query.all()
    result = [
        {
            'id': img.id,
            'image_path': img.image_path,
            'x_error_close_mean': img.x_error_close_mean,
            'x_error_far_mean': img.x_error_far_mean,
            'z_error_close_mean': img.z_error_close_mean,
            'z_error_far_mean': img.z_error_far_mean,
            'x_error_close_mean_large': img.x_error_close_mean_large,
            'x_error_far_mean_large': img.x_error_far_mean_large,
            'z_error_close_mean_large': img.z_error_close_mean_large,
            'z_error_far_mean_large': img.z_error_far_mean_large,
            'x_error_close_mean_20': img.x_error_close_mean_20,
            'x_error_far_mean_20': img.x_error_far_mean_20,
            'z_error_close_mean_20': img.z_error_close_mean_20,
            'z_error_far_mean_20': img.z_error_far_mean_20,
            'scene': img.scene
        }
        for img in images
    ]
    return jsonify(result)

@app.route('/api/images/filter', methods=['POST'])
def filter_images():
    filters = request.json
    query = Image.query

    if filters['scene']:
        query = query.filter_by(scene=filters['scene'])
    if filters['x_error_close_mean_large']:
        query = query.filter_by(x_error_close_mean_large=bool(int(filters['x_error_close_mean_large'])))
    if filters['x_error_far_mean_large']:
        query = query.filter_by(x_error_far_mean_large=bool(int(filters['x_error_far_mean_large'])))
    if filters['z_error_close_mean_large']:
        query = query.filter_by(z_error_close_mean_large=bool(int(filters['z_error_close_mean_large'])))
    if filters['z_error_far_mean_large']:
        query = query.filter_by(z_error_far_mean_large=bool(int(filters['z_error_far_mean_large'])))
    if filters['x_error_close_mean_20']:
        query = query.filter_by(x_error_close_mean_20=bool(int(filters['x_error_close_mean_20'])))
    if filters['x_error_far_mean_20']:
        query = query.filter_by(x_error_far_mean_20=bool(int(filters['x_error_far_mean_20'])))
    if filters['z_error_close_mean_20']:
        query = query.filter_by(z_error_close_mean_20=bool(int(filters['z_error_close_mean_20'])))
    if filters['z_error_far_mean_20']:
        query = query.filter_by(z_error_far_mean_20=bool(int(filters['z_error_far_mean_20'])))

    images = query.all()
    result = [
        {
            'id': img.id,
            'image_path': img.image_path,
            'x_error_close_mean': img.x_error_close_mean,
            'x_error_far_mean': img.x_error_far_mean,
            'z_error_close_mean': img.z_error_close_mean,
            'z_error_far_mean': img.z_error_far_mean,
            'x_error_close_mean_large': img.x_error_close_mean_large,
            'x_error_far_mean_large': img.x_error_far_mean_large,
            'z_error_close_mean_large': img.z_error_close_mean_large,
            'z_error_far_mean_large': img.z_error_far_mean_large,
            'x_error_close_mean_20': img.x_error_close_mean_20,
            'x_error_far_mean_20': img.x_error_far_mean_20,
            'z_error_close_mean_20': img.z_error_close_mean_20,
            'z_error_far_mean_20': img.z_error_far_mean_20,
            'scene': img.scene
        }
        for img in images
    ]
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
