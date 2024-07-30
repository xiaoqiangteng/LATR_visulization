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
    position_deviation = db.Column(db.String(20), nullable=False)
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
            'position_deviation': img.position_deviation,
            'scene': img.scene
        }
        for img in images
    ]
    return jsonify(result)

@app.route('/api/images/filter', methods=['POST'])
def filter_images():
    data = request.json
    position_deviation = data.get('position_deviation')
    scene = data.get('scene')

    query = Image.query
    if position_deviation:
        query = query.filter_by(position_deviation=position_deviation)
    if scene:
        query = query.filter_by(scene=scene)

    images = query.all()
    result = [
        {
            'id': img.id,
            'image_path': img.image_path,
            'position_deviation': img.position_deviation,
            'scene': img.scene
        }
        for img in images
    ]
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
