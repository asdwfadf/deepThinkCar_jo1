from tensorflow.keras.models import load_model

model = load_model(
    "output/lane_navigation_final.h5",
    compile=False
)

print(f"Total Parameters: {model.count_params():,}")