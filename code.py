import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def extract_floor(floor_text):
    if floor_text == "Ground":
        return 0
    elif floor_text == "Upper Basement":
        return -1
    elif pd.isnull(floor_text):
        return -2
    else:
        try:
            return int(floor_text.split()[0])
        except:
            return -2

def preprocess_data(df):
    df = df.copy()
    df.drop(columns=["Posted On"], inplace=True)
    df["Floor_Num"] = df["Floor"].apply(extract_floor)
    df.drop(columns=["Floor"], inplace=True)
    df_encoded = pd.get_dummies(df, columns=[
        "Area Type", "City", "Furnishing Status", "Tenant Preferred", "Area Locality"
    ], drop_first=True)
    x = df_encoded.drop(columns=["Rent", "Point of Contact"]).copy()
    y = df_encoded["Rent"].values.reshape(-1, 1)
    return x, y, df_encoded

df = pd.read_csv("/Users/kanishkk/Documents/House_Rent_Dataset.csv")
x1, y1, df_encoded = preprocess_data(df)

scaler_x = StandardScaler()
scaler_y = StandardScaler()
x1 = scaler_x.fit_transform(x1)
y1 = scaler_y.fit_transform(y1)

x = torch.tensor(x1, dtype=torch.float32)
y = torch.tensor(y1, dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

class HouseRentModel(nn.Module):
    def __init__(self, input_features):
        super(HouseRentModel, self).__init__()
        self.linear = nn.Linear(input_features, 1)
    def forward(self, x):
        return self.linear(x)

features = x_train.shape[1]
model = HouseRentModel(features)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

torch.manual_seed(42)
epochs = 1000
for epoch in range(epochs):
    model.train()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        model(x_test)

torch.save(model.state_dict(), "house_rent_model.pth")

print("Please enter the values for the following features:")

input_order = list(df_encoded.drop(columns=["Rent", "Point of Contact"]).columns)
input_vector = [0.0] * len(input_order)
input_map = {feature: idx for idx, feature in enumerate(input_order)}

continuous_fields = ["Size", "Bathroom", "BHK", "Floor_Num"]
for field in continuous_fields:
    val = input(f"{field}: ")
    try:
        input_vector[input_map[field]] = float(val)
    except:
        input_vector[input_map[field]] = 0.0

categorical_fields = {
    "Area Type": "Area Type_",
    "City": "City_",
    "Furnishing Status": "Furnishing Status_",
    "Tenant Preferred": "Tenant Preferred_",
    "Area Locality": "Area Locality_"
}
for cat_field, prefix in categorical_fields.items():
    val = input(f"{cat_field}: ").strip()
    key = prefix + val
    if key in input_map:
        input_vector[input_map[key]] = 1.0

input_tensor = torch.tensor([input_vector], dtype=torch.float32)
input_scaled = torch.tensor(scaler_x.transform(input_tensor), dtype=torch.float32)

model = HouseRentModel(len(input_order))
model.load_state_dict(torch.load("house_rent_model.pth"))
model.eval()

with torch.inference_mode():
    prediction = model(input_scaled)
    rent = scaler_y.inverse_transform(prediction.numpy())
    print(f"Predicted Rent: ₹{rent[0][0]:,.2f}")

