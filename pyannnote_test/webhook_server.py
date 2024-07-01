from fastapi import FastAPI, Request

app = FastAPI()

@app.webhooks.post("/webhooks")
def webhook(request: Request):
    data = request.json()
    print("Webhook received data:", data)
    return {"message": "Webhook received successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)