from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI(
    title="Crop Yield Prediction API",
    description="A Machine Learning project that estimates crop yield.",
    version="1.0.0"
)

# Sample training data for demonstration
# Inputs: rainfall (mm), temperature (°C), fertilizer (kg/hectare)
X = np.array([
    [400, 20, 50],
    [500, 22, 70],
    [600, 25, 80],
    [700, 27, 100],
    [800, 28, 120],
    [900, 30, 140],
    [550, 24, 75],
    [750, 26, 110],
    [650, 23, 90],
    [850, 29, 130]
], dtype=float)

# Illustrative yields, not real-world validated data
y = np.array([2.0, 2.5, 3.0, 3.6, 4.0, 4.3, 2.8, 3.8, 3.2, 4.1])

model = LinearRegression()
model.fit(X, y)


class CropData(BaseModel):
    rainfall_mm: float = Field(gt=0, le=3000)
    temperature_c: float = Field(ge=-10, le=60)
    fertilizer_kg_per_hectare: float = Field(ge=0, le=1000)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crop Yield Prediction</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #eef7ed;
                margin: 0;
                padding: 30px 15px;
            }
            .card {
                max-width: 520px;
                margin: auto;
                background: white;
                padding: 28px;
                border-radius: 16px;
                box-shadow: 0 4px 18px #0002;
            }
            h1 { color: #246b36; }
            label {
                display: block;
                margin-top: 18px;
                font-weight: bold;
            }
            input {
                box-sizing: border-box;
                width: 100%;
                padding: 12px;
                margin-top: 7px;
                border: 1px solid #bbb;
                border-radius: 7px;
            }
            button {
                width: 100%;
                margin-top: 24px;
                padding: 13px;
                background: #287a3d;
                color: white;
                border: none;
                border-radius: 7px;
                font-size: 16px;
                cursor: pointer;
            }
            #result {
                margin-top: 20px;
                padding: 12px;
                border-radius: 7px;
                background: #f0f6ee;
                white-space: pre-wrap;
            }
            .note { color: #555; font-size: 13px; }
            a { color: #246b36; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🌱 Crop Yield Prediction</h1>
            <p>Estimate crop yield using a Machine Learning regression model.</p>

            <form id="cropForm">
                <label>Rainfall (mm)</label>
                <input id="rainfall" type="number" min="0.1"
                    max="3000" value="600" step="any" required>

                <label>Temperature (°C)</label>
                <input id="temperature" type="number"
                    min="-10" max="60" value="25" step="any" required>

                <label>Fertilizer (kg/hectare)</label>
                <input id="fertilizer" type="number" min="0"
                    max="1000" value="80" step="any" required>

                <button type="submit">Predict Crop Yield</button>
            </form>

            <div id="result" aria-live="polite">
                Enter the values above to get an estimate.
            </div>

            <p class="note">
                Demonstration only: the model uses illustrative sample data.
                Its predictions are not validated agricultural advice.
            </p>

            <p>🔌 <a href="/docs">Open Swagger API Documentation</a></p>
        </div>

        <script>
            document.getElementById("cropForm").addEventListener("submit",
                async function(event) {
                    event.preventDefault();
                    const result = document.getElementById("result");
                    result.textContent = "Calculating...";

                    const data = {
                        rainfall_mm: Number(document.getElementById("rainfall").value),
                        temperature_c: Number(document.getElementById("temperature").value),
                        fertilizer_kg_per_hectare:
                            Number(document.getElementById("fertilizer").value)
                    };

                    try {
                        const response = await fetch("/predict", {
                            method: "POST",
                            headers: {"Content-Type": "application/json"},
                            body: JSON.stringify(data)
                        });
                        const output = await response.json();

                        if (!response.ok) {
                            throw new Error(output.detail || "Prediction failed");
                        }

                        result.textContent =
                            "Estimated crop yield: " +
                            output.predicted_yield_tonnes_per_hectare +
                            " tonnes/hectare";
                    } catch (error) {
                        result.textContent = "Error: " + error.message;
                    }
                }
            );
        </script>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: CropData):
    features = np.array([[
        data.rainfall_mm,
        data.temperature_c,
        data.fertilizer_kg_per_hectare
    ]])

    prediction = float(model.predict(features)[0])

    return {
        "project": "Crop Yield Prediction",
        "predicted_yield_tonnes_per_hectare": round(max(0, prediction), 2),
        "unit": "tonnes/hectare",
        "notice": "Experimental estimate from illustrative sample data."
    }