import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import joblib

# ------------------------------
# 1. Load Pretrained Components
# ------------------------------
@st.cache_resource
def load_model_and_utils():
    model = DeepCSATNet(input_dim=397, hidden_dim=256, output_dim=5)
    model.load_state_dict(torch.load("best_model_balanced.pth", map_location=torch.device("cpu")))
    model.eval()

    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

# ------------------------------
# 2. Define Model Architecture
# ------------------------------
class DeepCSATNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepCSATNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ------------------------------
# 3. Streamlit Page Config
# ------------------------------
st.set_page_config(
    page_title="DeepCSAT v1",
    page_icon="🤖",
    layout="centered"
)
st.title("DeepCSAT v1 – Ecommerce Customer Satisfaction Prediction")
st.write("Baseline MLP Model trained on ecommerce support interactions.")


# ------------------------------
# 4. Load Model + Utilities
# ------------------------------
model, scaler, label_encoder = load_model_and_utils()


# ------------------------------
# 5. User Input Section
# ------------------------------
st.subheader("Enter Customer Interaction Details")

# Example input fields – adjust as per your dataset
channel_name = st.selectbox("Channel", ["Chat", "Email", "Voice", "Social"])
category = st.selectbox("Category", ["Delivery", "Product Issue", "Payment", "Others"])
sub_category = st.text_input("Sub-Category (optional)", "NA")
item_price = st.number_input("Item Price", min_value=0.0, value=500.0, step=10.0)
response_delay_hrs = st.number_input("Response Delay (hrs)", min_value=0.0, value=2.5, step=0.1)
agent_shift = st.selectbox("Agent Shift", ["Morning", "Evening", "Night"])
remarks = st.text_area("Customer Remarks", "The delivery was delayed but resolved quickly.")

# You can extend with more fields if necessary.


# ------------------------------
# 6. Feature Vector Preparation
# ------------------------------
def prepare_input():
    # Dummy example — normally you’d one-hot encode or embed text
    # Here we assume embeddings already handled during training.
    input_data = np.zeros((1, 397))  # Match your training feature size
    input_data[0, 0] = 1 if channel_name == "Chat" else 0
    input_data[0, 1] = 1 if category == "Delivery" else 0
    input_data[0, 2] = item_price
    input_data[0, 3] = response_delay_hrs
    # ... (other dummy encodings, or real preprocessing logic)

    scaled = scaler.transform(input_data)
    return torch.tensor(scaled, dtype=torch.float32)


# ------------------------------
# 7. Prediction Logic
# ------------------------------
if st.button("Predict CSAT Score"):
    with st.spinner("Analyzing interaction..."):
        X_input = prepare_input()
        with torch.no_grad():
            output = model(X_input)
            probs = torch.softmax(output, dim=1).numpy()[0]
            pred_class = np.argmax(probs)
            pred_label = label_encoder.inverse_transform([pred_class])[0]

        st.success(f"Predicted CSAT Score: **{pred_label}** 🌟")
        st.progress(int(probs[pred_class] * 100))
        st.write(f"Confidence: {probs[pred_class]*100:.2f}%")

        # Optional: show probability distribution
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "CSAT Class": label_encoder.classes_,
            "Probability": probs
        })
        st.bar_chart(prob_df.set_index("CSAT Class"))

st.markdown("---")
st.caption("Version: DeepCSAT v1 | Model: best_model_balanced.pth | Framework: PyTorch")
