import os
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template_string, send_from_directory
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Load Advanced AI Model & Real Labels
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()
categories = weights.meta["categories"] # Gets real English names!

# 2. Professional Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

photo_data = []

# 3. Enhanced AI Prediction (Returns Name & Confidence %)
def predict_tag(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        probabilities = F.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    tag_name = categories[predicted_idx.item()]
    conf_score = round(confidence.item() * 100, 1)
    return tag_name.capitalize(), conf_score

# 4. Premium UI (HTML + CSS + Animations)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroGallery | AI Vision</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root { --primary: #6366f1; --secondary: #a855f7; --bg: #f8fafc; --surface: #ffffff; --text: #1e293b; }
        body { font-family: 'Poppins', sans-serif; background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%); color: var(--text); margin: 0; padding: 40px 20px; min-height: 100vh; }
        .container { max-width: 1000px; margin: 0 auto; }
        
        /* Header */
        .header { text-align: center; margin-bottom: 40px; animation: fadeInDown 0.8s ease; }
        .header h1 { font-size: 2.5rem; font-weight: 600; background: -webkit-linear-gradient(right, var(--primary), var(--secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        
        /* Glassmorphism Panels */
        .panel { background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.5); border-radius: 16px; padding: 25px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); animation: fadeInUp 0.8s ease; }
        
        /* Forms & Inputs */
        .input-group { display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; }
        input[type="text"], input[type="file"] { flex: 1; padding: 12px 15px; border: 2px solid #e2e8f0; border-radius: 8px; outline: none; transition: border-color 0.3s ease; font-family: 'Poppins', sans-serif; background: var(--surface); }
        input[type="text"]:focus { border-color: var(--primary); }
        button { padding: 12px 25px; background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; transition: transform 0.2s ease, box-shadow 0.2s ease; display: flex; align-items: center; gap: 8px; }
        button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3); }
        
        /* Gallery Grid */
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 25px; margin-top: 30px; }
        .card { background: var(--surface); border-radius: 16px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.3s ease, box-shadow 0.3s ease; animation: fadeIn 1s ease; position: relative; overflow: hidden; }
        .card:hover { transform: translateY(-8px); box-shadow: 0 20px 25px rgba(0,0,0,0.1); }
        .card img { width: 100%; height: 200px; object-fit: cover; border-radius: 12px; transition: transform 0.5s ease; }
        .card:hover img { transform: scale(1.03); }
        
        /* Card Data */
        .card-info { padding-top: 15px; }
        .card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 5px; }
        .badges { display: flex; justify-content: space-between; align-items: center; margin-top: 10px; }
        .tag-badge { background: #e0e7ff; color: var(--primary); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
        .conf-badge { background: #dcfce7; color: #16a34a; padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
        
        /* Animations */
        @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fa-solid fa-brain"></i> NeuroGallery</h1>
            <p>Intelligent Image Recognition & Storage</p>
        </div>

        <div class="panel">
            <h3><i class="fa-solid fa-cloud-arrow-up"></i> Upload & Analyze</h3>
            <form method="POST" enctype="multipart/form-data">
                <div class="input-group">
                    <input type="text" name="title" placeholder="Give your photo a title..." required>
                    <input type="file" name="photo" accept="image/*" required>
                    <button type="submit"><i class="fa-solid fa-wand-magic-sparkles"></i> Process Image</button>
                </div>
            </form>
        </div>

        <div class="panel">
            <h3><i class="fa-solid fa-magnifying-glass"></i> AI Tag Search</h3>
            <form method="POST">
                <div class="input-group">
                    <input type="text" name="search" placeholder="Search by AI tag (e.g., 'car', 'dog', 'keyboard')...">
                    <button type="submit"><i class="fa-solid fa-filter"></i> Filter</button>
                </div>
            </form>
        </div>

        <div class="gallery">
            {% for photo in photos %}
                <div class="card">
                    <img src="{{ url_for('uploaded_file', filename=photo.filename) }}" alt="{{ photo.title }}">
                    <div class="card-info">
                        <div class="card-title">{{ photo.title }}</div>
                        <div class="badges">
                            <span class="tag-badge"><i class="fa-solid fa-tag"></i> {{ photo.tag }}</span>
                            <span class="conf-badge"><i class="fa-solid fa-check-circle"></i> {{ photo.confidence }}%</span>
                        </div>
                    </div>
                </div>
            {% else %}
                <p style="text-align: center; width: 100%; color: #64748b; margin-top: 20px;">
                    <i class="fa-regular fa-folder-open" style="font-size: 2rem;"></i><br>
                    No photos in the gallery yet.
                </p>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

# 5. Route for Image Rendering
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 6. Main App Logic
@app.route("/", methods=["GET", "POST"])
def index():
    results = photo_data

    if request.method == "POST":
        # Handle Upload
        if "photo" in request.files:
            file = request.files["photo"]
            title = request.form.get("title", "Untitled")
            
            if file.filename != '':
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)
                
                # Get Tag and Confidence Percentage
                tag, confidence = predict_tag(filepath)
                
                photo_data.append({
                    "title": title,
                    "filename": file.filename, 
                    "tag": tag,
                    "confidence": confidence
                })
                results = photo_data

        # Handle Search
        if "search" in request.form:
            search_tag = request.form["search"].lower()
            results = [p for p in photo_data if search_tag in p["tag"].lower() or search_tag in p["title"].lower()]

    # Reverse results so newest photos show up first
    return render_template_string(HTML_TEMPLATE, photos=list(reversed(results)))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)