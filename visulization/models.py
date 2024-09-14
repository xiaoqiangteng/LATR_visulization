from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(256), nullable=False)
    label = db.Column(db.String(256), nullable=False)

    def __repr__(self):
        return f'<Image {self.id}>'
