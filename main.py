from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ad_simulation import ADNeuronSimulation
import uvicorn
import os

app = FastAPI(title="AIVC Neuron Viewer")
sim = ADNeuronSimulation()

# Mount the static folder to serve the frontend interface
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    """Serves the main HTML page when you visit the root URL."""
    return FileResponse("static/index.html")

@app.get("/api/neuron-state/{t}")
def get_neuron_state(t: float):
    """The API endpoint that returns the biological JSON state for slider value t."""
    return sim.get_state(t)

if __name__ == "__main__":
    # Automatically uses Render's assigned port, or defaults to 8000 when running locally on your computer
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)