from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn, io, cv2, numpy as np
from src.models.infer import InferenceModel

app = FastAPI(title="Multi-Crop API")
model = None

class PredictResponse(BaseModel):
    top1_class: str
    top1_conf: float
    topk: list

@app.on_event("startup")
def load_model():
    global model
    model = InferenceModel("weights/best.pt", model_name="efficientnet_b0", device="cpu")

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    preds = model.predict_image(rgb, topk=5)
    top1_class, top1_conf = preds[0]  # Get first prediction (class, confidence)
    return PredictResponse(top1_class=top1_class, top1_conf=top1_conf, topk=preds)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
