from pydantic import BaseModel, Field

class ImageRequest(BaseModel):
    image: bytes = Field(..., description="Image file in bytes")

    class Config:
        schema_extra = {
            "example": {
                "image": "<image_bytes>"
            }
        }

class ImagePath(BaseModel):
    image : str