
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# ----------- Config -----------
MODEL_NAME = "bert-base-chinese"
MODEL_PATH = "bert_reply_classifier.pt"
RESPONSE_CSV = "responses_chatstyle.csv"
MAX_LEN = 32
TOP_K = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- Load Model & Tokenizer -----------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
responses_df = pd.read_csv(RESPONSE_CSV, encoding="big5")

class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))

model = BertClassifier(num_classes=len(responses_df))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ----------- FastAPI App -----------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class Query(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response_list": [], "user_input": ""})

@app.post("/reply", response_class=HTMLResponse)
async def get_reply(request: Request, message: str = Form(...)):
    encoded = tokenizer(message, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        topk = torch.topk(probs, k=TOP_K, dim=1)
        top_indices = topk.indices.squeeze(0).tolist()

    replies = [responses_df.loc[i, 'text'] for i in top_indices]
    return templates.TemplateResponse("index.html", {"request": request, "response_list": replies, "user_input": message})

# ----------- Run Server -----------
# uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
