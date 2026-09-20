from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json

app = Flask(__name__)

# =================================================
# Load CNN model
# =================================================

MODEL_FILE = "imu_activity_cnn.keras"

model = tf.keras.models.load_model(MODEL_FILE)

print("CNN model loaded successfully!")


# =================================================
# Load normalization values
# =================================================

mean = np.load("imu_mean.npy")
std = np.load("imu_std.npy")

print("Normalization values loaded!")


# =================================================
# Load activity names
# =================================================

with open("activity_names.json", "r") as f:
    activity_names = json.load(f)

print("Activity labels loaded!")


# =================================================
# IMU sample storage
# =================================================

samples = []

WINDOW_SIZE = 100


# =================================================
# Receive IMU data
# =================================================

@app.route("/imu", methods=["POST"])
def receive_imu():

    global samples

    try:

        data = request.get_json()

        # Get IMU values
        ax = float(data["ax"])
        ay = float(data["ay"])
        az = float(data["az"])

        gx = float(data["gx"])
        gy = float(data["gy"])
        gz = float(data["gz"])

        # Add sample
        samples.append([
            ax,
            ay,
            az,
            gx,
            gy,
            gz
        ])

        print(
            f"Sample {len(samples)}/{WINDOW_SIZE} | "
            f"ax={ax:.0f} "
            f"ay={ay:.0f} "
            f"az={az:.0f} "
            f"gx={gx:.0f} "
            f"gy={gy:.0f} "
            f"gz={gz:.0f}"
        )

        # =================================================
        # When 100 samples are collected
        # =================================================

        if len(samples) >= WINDOW_SIZE:

            # Convert to NumPy
            window = np.array(
                samples[-WINDOW_SIZE:],
                dtype=np.float32
            )

            # Shape:
            # (100, 6)

            # Add batch dimension
            window = np.expand_dims(window, axis=0)

            # Shape:
            # (1, 100, 6)

            # =================================================
            # Normalize using training values
            # =================================================

            window_normalized = (
                window - mean
            ) / (std + 1e-8)

            # =================================================
            # CNN prediction
            # =================================================

            probabilities = model.predict(
                window_normalized,
                verbose=0
            )[0]

            prediction = int(np.argmax(probabilities))

            confidence = float(
                probabilities[prediction] * 100
            )

            # Get activity name
            activity = activity_names[str(prediction)]

            print()
            print("================================")
            print("       ACTIVITY DETECTED")
            print("================================")
            print(f"Activity   : {activity}")
            print(f"Confidence : {confidence:.2f}%")
            print("================================")
            print()

            # Clear samples
            samples.clear()

            return jsonify({
                "status": "prediction",
                "activity": activity,
                "confidence": round(confidence, 2)
            })

        # Still collecting
        return jsonify({
            "status": "collecting",
            "samples": len(samples)
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# =================================================
# Start server
# =================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print("        IMU ACTIVITY CNN SERVER")
    print("==========================================")
    print("Model      : 1D CNN")
    print("Activities : 6")
    print("Window     : 100 samples")
    print("Server     : http://10.43.155.161:5000")
    print("==========================================")
    print("Waiting for ESP32...")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )