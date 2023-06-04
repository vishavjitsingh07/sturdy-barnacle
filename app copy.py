
from service import predict_result
from fastapi import FastAPI , UploadFile

route = FastAPI()

@route.get("/")
async def testing():
    return "sucess"

@route.post("/image")
async def testing(image : UploadFile):
    """Image results"""

    return {"success" : True, "payload" : predict_result(image)}