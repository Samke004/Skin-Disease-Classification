import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0, efficientnet_b3
import plotly.express as px
import pandas as pd

MODEL_PATHS = {
    "EfficientNet-B0": "best_efficient_net.pth",
    "EfficientNet-B3": "best_efficientb3_net.pth",
    "EfficientNet-B0 (novo preprocesiranje)": "best_efficient_net-new_pretprocesing.pth",
    "ResNet50": "best_resnet50.pth"
}

CLASS_FULLNAMES = {
    'bkl':  'Benign keratosis-like lesions',
    'nv':   'Melanocytic nevi',
    'df':   'Dermatofibroma',
    'mel':  'Melanoma',
    'vasc': 'Vascular lesions',
    'bcc':  'Basal cell carcinoma',
    'akiec':'Actinic keratoses and intraepithelial carcinoma'
}
NUM_CLASSES = len(CLASS_FULLNAMES)

def get_model(model_key):
    if "ResNet50" in model_key:
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        state_dict = torch.load(MODEL_PATHS[model_key], map_location=torch.device('cpu'), weights_only=False)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("base_model."):
                new_key = k.replace("base_model.", "")
                new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)
    elif "B3" in model_key:
        model = efficientnet_b3(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        state_dict = torch.load(MODEL_PATHS[model_key], map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
    else:
        model = efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        state_dict = torch.load(MODEL_PATHS[model_key], map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
    model.eval()
    return model

def get_preprocessing(model_key):
    if "novo preprocesiranje" in model_key:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

st.title("🩺 Skin Disease Diagnosis")
st.write("""
Upload a photo of a skin lesion and select the deep learning model for diagnosis. The predicted class and confidence will be shown.
""")

model_key = st.selectbox("Choose a model:", list(MODEL_PATHS.keys()))
uploaded_file = st.file_uploader("Upload a skin image (JPG, PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded image", use_column_width=True)
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            model = get_model(model_key)
            preprocess = get_preprocessing(model_key)
            input_tensor = preprocess(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_idx = probs.argmax()
                pred_class_short = list(CLASS_FULLNAMES.keys())[pred_idx]
                pred_class = CLASS_FULLNAMES[pred_class_short]
                confidence = probs[pred_idx]

            st.success(f"**Prediction:** {pred_class}")
            st.info(f"**Confidence:** {confidence:.2%}")

            st.subheader("Probabilities for all classes:")
            df = pd.DataFrame({
                "Diagnosis": [CLASS_FULLNAMES[k] for k in CLASS_FULLNAMES.keys()],
                "Probability": [float(v) for v in probs]
            }).sort_values("Probability", ascending=False)

            fig = px.bar(
                df,
                x="Diagnosis",
                y="Probability",
                text="Probability",
                labels={"Probability": "Probability", "Diagnosis": "Diagnosis"},
                color="Probability",
                color_continuous_scale="Blues",
                range_y=[0, 1]
            )
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-30, yaxis_title="Probability", xaxis_title="Diagnosis", uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig, use_container_width=True)


            st.caption("⚠️ This tool is not a replacement for a professional medical diagnosis.")

else:
    st.warning("Please upload an image.")