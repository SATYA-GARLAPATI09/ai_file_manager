import os
from flask import Flask, request, render_template_string, send_from_directory

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

photo_data = []

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroGallery | Smart Vision</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://unpkg.com/ml5@0.12.2/dist/ml5.min.js"></script>
    <style>
        :root { --primary: #6366f1; --secondary: #a855f7; --bg: #f8fafc; --surface: #ffffff; --text: #1e293b; }
        body { font-family: 'Poppins', sans-serif; background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%); color: var(--text); margin: 0; padding: 40px 20px; min-height: 100vh; }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 2.5rem; background: -webkit-linear-gradient(right, var(--primary), var(--secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .panel { background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px); border-radius: 16px; padding: 25px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }
        .input-group { display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; }
        input[type="text"], input[type="file"] { flex: 1; padding: 12px; border: 2px solid #e2e8f0; border-radius: 8px; font-family: 'Poppins'; }
        button { padding: 12px 25px; background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; transition: transform 0.2s; }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.7; cursor: not-allowed; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 25px; margin-top: 30px; }
        .card { background: white; border-radius: 16px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .card img { width: 100%; height: 200px; object-fit: cover; border-radius: 12px; }
        .badges { display: flex; justify-content: space-between; margin-top: 10px; }
        .tag-badge { background: #e0e7ff; color: var(--primary); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
        .conf-badge { background: #dcfce7; color: #16a34a; padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fa-solid fa-bolt"></i> NeuroGallery Fast</h1>
        </div>

        <div class="panel">
            <h3><i class="fa-solid fa-cloud-arrow-up"></i> Upload & Analyze</h3>
            <form id="uploadForm" method="POST" enctype="multipart/form-data">
                <div class="input-group">
                    <input type="text" name="title" placeholder="Give your photo a title..." required>
                    <input type="file" id="photoInput" name="photo" accept="image/*" required>
                    
                    <input type="hidden" name="ai_tag" id="ai_tag">
                    <input type="hidden" name="ai_conf" id="ai_conf">
                    
                    <button type="submit" id="submitBtn"><i class="fa-solid fa-wand-magic-sparkles"></i> Process Image</button>
                </div>
            </form>
            <img id="hiddenPreview" style="display:none; max-width: 200px; margin-top: 10px; border-radius: 8px;">
        </div>

        <div class="gallery">
            {% for photo in photos %}
                <div class="card">
                    <img src="{{ url_for('uploaded_file', filename=photo.filename) }}">
                    <div style="padding-top: 15px;">
                        <strong>{{ photo.title }}</strong>
                        <div class="badges">
                            <span class="tag-badge"><i class="fa-solid fa-tag"></i> {{ photo.tag }}</span>
                            <span class="conf-badge"><i class="fa-solid fa-check"></i> {{ photo.confidence }}%</span>
                        </div>
                    </div>
                </div>
            {% endfor %}
        </div>
    </div>

    <script>
        const form = document.getElementById('uploadForm');
        const fileInput = document.getElementById('photoInput');
        const preview = document.getElementById('hiddenPreview');
        const submitBtn = document.getElementById('submitBtn');

        // Show preview image
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                preview.src = URL.createObjectURL(file);
                preview.style.display = 'block';
            }
        });

        // Intercept form submission to run AI first
        form.addEventListener('submit', (e) => {
            if(!document.getElementById('ai_tag').value) {
                e.preventDefault(); 
                submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Scanning AI...';
                submitBtn.disabled = true;

                // Run AI prediction
                ml5.imageClassifier('MobileNet').then(classifier => {
                    classifier.classify(preview, (error, results) => {
                        if (error) {
                            console.error(error);
                            submitBtn.innerHTML = 'Error!';
                            return;
                        }
                        
                        // Clean up tag string and calculate confidence
                        let bestMatch = results[0].label.split(',')[0]; 
                        let confidence = Math.round(results[0].confidence * 100);
                        
                        // Inject results into hidden form fields
                        document.getElementById('ai_tag').value = bestMatch.charAt(0).toUpperCase() + bestMatch.slice(1);
                        document.getElementById('ai_conf').value = confidence;
                        
                        // Send data to Python server
                        form.submit();
                    });
                });
            }
        });
    </script>
</body>
</html>
"""

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    global photo_data
    
    if request.method == "POST" and "photo" in request.files:
        file = request.files["photo"]
        if file.filename != '':
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            
            # Retrieve the AI data calculated by the browser
            photo_data.append({
                "title": request.form.get("title", "Untitled"),
                "filename": file.filename, 
                "tag": request.form.get("ai_tag", "Unknown"),
                "confidence": request.form.get("ai_conf", "0")
            })

    return render_template_string(HTML_TEMPLATE, photos=list(reversed(photo_data)))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)