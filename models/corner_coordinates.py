from pydantic import BaseModel, Field
class CornerCoordinate(BaseModel):
    left_coordinate: tuple = None
    right_coordinate: tuple = None
    top_coordinate: tuple = None
    bottom_coordinate: tuple = None