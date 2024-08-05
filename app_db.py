from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Configuration
class Config:
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:pass@localhost:5433/barcodeless'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize SQLAlchemy
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

# Define the route
@app.route('/')
def index():
    products = Product.query.all()
    issues = Issue.query.all()
    
    if products:
        for product in products:
            print(product)
    else:
        print('No products found in the database.')
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print('No issues found in the database.')

    return 'Check your console for the list of products and issues!'

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create tables if they don't exist

        print('Inserting test data into Products table...')
        products = [
            Product(name='chips', price=13, description='salt flavored'),
            Product(name='corn', price=6),
            Product(name='gummies', price=7, description='haribo sour fizz'),
            Product(name='knoppers', price=3),
            Product(name='lemon', price=2),
            Product(name='lime', price=2),
            Product(name='milk', price=7, description='cow milk 3%'),
            Product(name='milka-brownie', price=8),
            Product(name='milka-hazelnut', price=8),
            Product(name='orange', price=3),
            Product(name='oreo', price=7, description='6-pack'),
            Product(name='water', price=4)
        ]
        db.session.bulk_save_objects(products)
        db.session.commit()
        print('Test data inserted into Products table.')

    app.run(host='0.0.0.0', port=5000, debug=True)
