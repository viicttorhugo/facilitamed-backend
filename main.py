import os, os.path, logging
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import firebase_admin
from firebase_admin import auth as fb_auth, credentials

log = logging.getLogger("uvicorn.error")

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "").strip()
CREDS_PATH = os.getenv("FIREBASE_CREDENTIALS_JSON", "").strip()
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app = FastAPI(title="FacilitaMed API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not firebase_admin._apps:
    if CREDS_PATH and os.path.exists(CREDS_PATH):
        cred = credentials.Certificate(CREDS_PATH)
        firebase_admin.initialize_app(cred, {"projectId": FIREBASE_PROJECT_ID or cred.project_id})
        log.info("Firebase Admin inicializado com %s", CREDS_PATH)
    else:
        firebase_admin.initialize_app()
        log.warning("Firebase Admin inicializado sem CREDS_PATH")

bearer = HTTPBearer(auto_error=False)

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Authorization: Bearer <ID_TOKEN> requerido")
    token = creds.credentials
    try:
        decoded = fb_auth.verify_id_token(token, check_revoked=True)
        aud = decoded.get("aud")
        iss = decoded.get("iss", "")
        if FIREBASE_PROJECT_ID and (aud != FIREBASE_PROJECT_ID and FIREBASE_PROJECT_ID not in iss):
            raise HTTPException(status_code=401, detail=f"Token de projeto incorreto (aud={aud}, iss={iss})")
        return decoded
    except Exception as e:
        log.exception("Falha ao verificar token Firebase: %s", e)
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

@app.get("/me/status")
def me_status(user = Depends(get_current_user)):
    return {"uid": user["uid"], "email": user.get("email"), "active": True}

@app.get("/debug/token")
def debug_token(user = Depends(get_current_user)):
    return user
