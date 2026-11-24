from fastapi import FastAPI
from pydantic import BaseModel
from pii_detector import PiiDetector

app = FastAPI(title="Simple Agent Workflow")

detector = PiiDetector()  # created once at startup, reused for every request

class TextIn(BaseModel):
    text: str

@app.get("/")
def hello():
    return {"message": "Hello World"}


@app.post("/detect-pii")
def detect(body: TextIn):
    """Show what would be sent to the LLM after masking personal data."""
    masked, mapping = detector.mask(body.text)

    return {
        "pii_detected": bool(mapping),
        "types": sorted({key.strip("[]").rsplit("_", 1)[0] for key in mapping}),
        "masked_text": masked,
        # The mapping holds real personal data. It is NOT returned here:
        # in a real agent it stays on the server and never goes to the LLM.
    }
