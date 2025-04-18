import streamlit as st
import torch
import pandas as pd
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import numpy as np
import cv2
import re
import base64


import os
# path = r"C:\Users\ARKADIP GHOSH\Downloads\WhatsApp Image 2025-04-06 at 01.27.37.jpeg"
# st.write("File exists:", os.path.exists(path))
# st.image(r"WhatsApp Image 2025-04-06 at 01.27.37.jpeg", width=200)

import streamlit as st
import base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# bg_image = get_base64_image("wmremove-transformed (4).jpeg")

bg_image = get_base64_image("Screenshot 2025-04-18 130716.png")

# ✅ Use f-string so {bg_image} is replaced with actual content
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .main {{
        background-image: url("data:image/jpeg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        padding: 20px;
        border-radius: 12px;
    }}

    .prediction-card {{
        background-color: #e6ffe6;
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 16px;
        line-height: 1.6;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}

    .remedy-box {{
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        border-left: 6px solid #fb8c00;
    }}

    .medicine-box {{
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        border-left: 6px solid #43a047;
    }}

    img {{
        border-radius: 10px;
        border: 2px solid #ddd;
    }}

    .header-title {{
        font-size: 36px !important;
        color: #2e7d32;
        font-weight: bold;
    }}

    .remedy-box, .medicine-box {{
        background-color: #e6ffe6;
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 16px;
        line-height: 1.6;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    </style>
""", unsafe_allow_html=True)

# Load the Excel file for recommendations
df = pd.read_excel("Book2.xlsx")

# Recommendation Finder
def find_closest_precaution_medicine(predicted_class, temp, humidity, ph, light, rain):
    # Filter only the predicted class
    class_df = df[df["Class Name"] == predicted_class].copy()

    # Normalize rain input
    if isinstance(rain, str):
        rain = rain.strip().lower()
        rain = 1 if rain in ["yes", "1", "true"] else 0
    else:
        rain = int(rain)

    # Ensure Rain column is integer
    class_df["Rain"] = pd.to_numeric(class_df["Rain"], errors='coerce').fillna(0).astype(int)

    # Define distance function
    def distance(row):
        temp_diff = abs(row["Temperature"] - temp)
        humidity_diff = abs(row["Humidity"] - humidity)
        ph_diff = abs(row["Soil pH"] - ph)
        light_diff = abs(row["Light Intensity"] - light)
        rain_diff = abs(row["Rain"] - rain) * 1000  # Large penalty for mismatch
        return temp_diff + humidity_diff + ph_diff + light_diff + rain_diff

    # Apply and get best match
    class_df["distance_score"] = class_df.apply(distance, axis=1)
    best_match = class_df.sort_values("distance_score").iloc[0]

    return {
        "Precautions": best_match["Precautions"],
        "Medicines": best_match["Medicines"]
    }


# # Format remedies text
# def format_remedies(text):
#     formatted = re.sub(r'(?<!^)(?=\d\.)', r'\n', text.strip())
#     return formatted

def format_remedies(text):
    # Add newline before each numbered point (1. 2. 3.)
    formatted = re.sub(r'(?<!^)(?=\d\.)', r'\n', text.strip())
    return formatted


# --- Define U-Net model ---
class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_c, out_c, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_c, out_c, 3, padding=1),
                torch.nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = torch.nn.MaxPool2d(2)

        self.up2 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = torch.nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = torch.sigmoid(self.final(d1))
        return out

# --- Set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load models ---
unet = UNet()
unet.load_state_dict(torch.load('leaf_unett_model.pth', map_location=device))
unet.to(device).eval()

resnet = models.resnet18(pretrained=False)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 41)
resnet.load_state_dict(torch.load('resnet18_plant_disease.pth', map_location=device))
resnet.to(device).eval()

# --- Class labels ---
class_names = [
    'Apple___Apple_scab', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'AshGourd_Downey_mildew', 'AshGourd___healthy', 'AshGourd___K_N_deficiency',
    'BitterGourd___Downey_mildew', 'BitterGourd___Fusarium_wilt', 'BitterGourd___Healthy',
    'BottleGourd___Downy_Mildew', 'BottleGourd___Healthy', 'BottleGourd___Immature_Gourd', 
    'BottleGourd___Nutrition_Deficiency', 'Brinjal_Healthy', 'Brinjal___begomovirus', 
    'Brinjal___verticillium_wilt', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight', 'Cucumber_healthy', 'Cucumber___Anthracnose_lesions', 
    'Cucumber___Downy_mildew', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Pumpkin___Downy_Mildew', 
    'Pumpkin___Healthy_Leaf', 'Pumpkin___Mosaic_Disease', 'RidgeGourd_Healthy', 'RidgeGourd_Leaf_Eating_Insect', 
    'RidgeGourd_N_K_deficiency', 'SnakeGourd_healthy', 'SnakeGourd_Leafspot', 'SnakeGourd_N_K_deficiency', 
    'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Septoria_leaf_spot'
]

# --- Transforms ---
transform_resnet = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_unet = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# --- Streamlit App ---
def main():
    st.markdown('<h1 class="header-title">🌿 Plant Disease Detection & Remedies</h1>', unsafe_allow_html=True)


    # Environmental Inputs
    temperature = st.number_input("Temperature (°C)", min_value=0, max_value=100, value=30, key="main_temp")
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
    soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0)
    light_intensity = st.number_input("Light Intensity (Lux)", min_value=0, max_value=20000, value=5000)
    rain_status = st.selectbox("Rain Status", ["0", "1"])

    uploaded_image = st.file_uploader("Upload Plant Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Make Prediction"):
            top_classes, top_confidences = classify_with_resnet(image)
            segmented_image = segment_with_unet(image)

            st.markdown("### 🧠 Top 3 Predictions:")
            for label, conf in zip(top_classes, top_confidences):
                st.markdown(f'<div class="prediction-card"><strong>{label}</strong><br>Confidence: {conf:.2f}%</div>', unsafe_allow_html=True)


            st.image(segmented_image, caption="Segmented Disease Region", use_column_width=True)

            predicted_class = top_classes[0]
            remedies = find_closest_precaution_medicine(predicted_class, temperature, humidity, soil_ph, light_intensity, rain_status)

            if remedies is not None:
                st.markdown("### 💊 Disease Remedies & Care")
                # Prepare formatted precautions
                precautions_html = format_remedies(remedies["Precautions"]).replace("\n", "<br>")
                st.markdown(f'<div class="remedy-box"><strong>🛡️ Precautions:</strong><br>{precautions_html}</div>', unsafe_allow_html=True)

                # Prepare formatted medicines
                medicines_html = format_remedies(remedies["Medicines"]).replace("\n", "<br>")
                st.markdown(f'<div class="medicine-box"><strong>💉 Medicines:</strong><br>{medicines_html}</div>', unsafe_allow_html=True)

                # st.markdown("### 💊 Disease Remedies & Care")
                # st.markdown("**🛡️ Precautions:**")
                # st.markdown(format_remedies(remedies["Precautions"]))
                # st.markdown("**💉 Medicines:**")
                # st.markdown(format_remedies(remedies["Medicines"]))
            else:
                st.warning("No matching recommendations found for these conditions.")

# --- ResNet Prediction ---
def classify_with_resnet(image):
    input_tensor = transform_resnet(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = resnet(input_tensor)
        probs = F.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, 3)
    top_classes = [class_names[i] for i in top_indices[0]]
    top_confidences = [top_probs[0][i].item() * 100 for i in range(3)]
    return top_classes, top_confidences

# --- U-Net Segmentation ---
def segment_with_unet(image, top_class_name=None):
    input_tensor = transform_unet(image).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = unet(input_tensor)[0][0].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

    img_np = np.array(image.resize((256, 256)))
    circled_img = img_np.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    largest_center = None

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)


    # Resize original to 256x256
    original_resized = np.array(image.resize((256, 256)))

    # Create 3-channel mask
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply blending
    blended = cv2.addWeighted(original_resized, 0.8, colored_mask, 0.2, 0)

    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area < 300:  # change threshold from 20 to something bigger
    #         continue
    for cnt in contours:
        if cv2.contourArea(cnt) < 40:
            continue
    

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(circled_img, center, radius, (255, 0, 0), 2)  # green circles

        if cv2.contourArea(cnt) > largest_area:
            largest_area = cv2.contourArea(cnt)
            largest_center = center

    # Draw top class name near largest region
    if top_class_name and largest_center:
        x, y = largest_center
        cv2.putText(
            circled_img, top_class_name, (x - 30, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
        )

    return circled_img


# Run the app
if __name__ == "__main__":
    main()

