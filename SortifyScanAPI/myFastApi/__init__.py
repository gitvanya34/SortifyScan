from io import BytesIO

from PIL import Image
import io
from fastapi import FastAPI
from sortifyscan.cargo import *
from sortifyscan.export import ExportMedia
from typing import Annotated
from sortifyscan.Isortifyscan import ISortifyScan
from fastapi import FastAPI, File, UploadFile
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/gabarity/calc")
async def create_file(image: Annotated[bytes, File()]):
    pil_image = Image.open(io.BytesIO(image))
    return ISortifyScan.sortify_scan_image_api(pil_image)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
