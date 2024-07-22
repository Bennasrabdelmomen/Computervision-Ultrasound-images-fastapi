from fastapi import FastAPI, HTTPException, status, Depends, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Union, Annotated, List
from jose import jwt
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import os
from model import load_model, prepare_image, predict
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import os
from app.ultrasoundclass import load_model_f, predict_image
from fastapi.responses import JSONResponse



#setting up loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

SECRET_KEY = "aaa"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

fake_users_db = {
    "ahmed": {
        "username": "ahmed",
        "full_name": "ahmed abdesmad",
        "email": "ahmedabdessmad@zzz.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    },
    "mounir": {
        "username": "mounir",
        "full_name": "mounir abdessmad 2",
        "email": "mounirabdessmad2@zzz.com",
        "hashed_password": "fakehashedsecret2",
        "disabled": True,
    },
}

model = load_model()
model_loaded = load_model_f(model_path="resnet50_binary_classification.pth")
model_breast_cancer=load_model_f(model_path="resnet50_breast_cancer_classification.pth")
class Token(BaseModel):
    access_token: str
    token_type: str


class Predictiona(BaseModel):
    filename: str
    content_type: str
    prediction: float
    predicted_class: str

class TokenData(BaseModel):
    username: Union[str, None] = None

class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None

class UserInDB(User):
    hashed_password: str

class UserCredentials(BaseModel):
    username: str
    password: str

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

bearer_scheme = HTTPBearer()

async def get_current_user(token: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/generate_token", response_model=Token)
async def generate_token(username: str, password: str):
    user = get_user(fake_users_db, username)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    return current_user

class Prediction(BaseModel):
    filename: str
    content_type: str
    predictions: List[dict] = []


class Patient(BaseModel):
    id: int
    name: str
    age: int
    medical_history: List[str] = []

fake_patients_db = []

@app.post("/register_patient", response_model=Patient)
async def register_patient(patient: Patient):
    for existing_patient in fake_patients_db:
        if existing_patient["id"] == patient.id:
            raise HTTPException(status_code=400, detail="Patient already exists")
    fake_patients_db.append(patient.dict())
    return patient

def get_patient_by_name(name: str):
    for patient in fake_patients_db:
        if patient["name"] == name:
            return patient
    return None

def get_patient_by_id(id: int):
    for patient in fake_patients_db:
        if patient["id"] == id:
            return patient
    return None
@app.post("/Ultrasound Image check", response_model=Predictiona)
async def prediction(current_user: Annotated[User, Depends(get_current_user)],
file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_bytes = await file.read()
        prediction_score = predict_image(image_bytes, model_loaded)
        predicted_class = "True" if prediction_score > 0.5 else "False"

        return JSONResponse(content={
            "filename": file.filename,
            "content_type": file.content_type,
            "prediction": prediction_score,
            "Ultrasound Image ?": predicted_class
        })
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
@app.post("/Malignant breast cancer detection", response_model=Predictiona)
async def prediction(current_user: Annotated[User, Depends(get_current_user)],
file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_bytes = await file.read()
        prediction_score = predict_image(image_bytes, model_loaded)
        predicted_class = "True" if prediction_score > 0.5 else "False"

        return JSONResponse(content={
            "filename": file.filename,
            "content_type": file.content_type,
            "prediction": prediction_score,
            "Is the breast cancer Malignant ?": predicted_class
        })
    except Exception as e:
        logger.error(f"Error in /Breast cancer classification endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict", response_model=Prediction)
async def prediction(
    current_user: Annotated[User, Depends(get_current_user)],
    file: UploadFile = File(...),
    patient_id: int = 0
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    patient = get_patient_by_id(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Save the uploaded file to a directory
    upload_dir = "uploaded_images"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Open the image
    image = Image.open(file_path).convert("RGB")
    image = prepare_image(image, target=(224, 224))
    print(f"Prepared image for prediction: {image}")  # Log the prepared image
    response = predict(image, model)

    prediction_result = f"Filename: {file.filename}, Predictions: {response}"
    patient["medical_history"].append(prediction_result)

    print(f"Medical history for patient {patient_id} so far:")
    for record in patient["medical_history"]:
        print(record)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "predictions": response
    }

@app.get("/patients_medical_history")
async def get_medical_history(
    current_user: Annotated[User, Depends(get_current_user)],
    patient_id: int = ""
):
    patient = get_patient_by_id(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {
        "Patient name": patient["name"],
        "Patient age": patient["age"],
        "medical record": patient["medical_history"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5001)
