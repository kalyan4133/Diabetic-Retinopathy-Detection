import os
from flask import Flask, request, render_template, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Path for saving uploaded images
UPLOAD_FOLDER = 'K:\\DR\\static\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model path and device configuration
model_path = r'K:\DR\data\model_vit.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained Vision Transformer model
model = models.vit_b_16(pretrained=False)
num_ftrs = model.heads.head.in_features
model.heads.head = nn.Linear(num_ftrs, 5)  # 5 classes
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model = model.to(device)

# Define label mapping
classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Home route to render index.html
@app.route('/')
def home():
    return render_template('index.html')

# Route to render predict.html (the image upload page)
@app.route('/predict')
def show_predict_page():
    return render_template('predict.html')

@app.route('/submit_image', methods=['POST'])
def submit_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the file to static/uploads directory
            filename = file.filename
            filepath = os.path.join('static', 'uploads', filename)
            file.save(filepath)

            img = Image.open(file)
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img)
                _, preds = torch.max(outputs, 1)
                prediction = classes[preds.item()]

            # Pass filename to display uploaded image
            return render_template('result.html', prediction=prediction, filename=filename)
    return redirect(url_for('show_predict_page'))

if __name__ == '__main__':
    app.run(debug=True)
