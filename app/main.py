
import os
from io import BytesIO
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Session, create_engine, select
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from openai import OpenAI

# Firebase Admin
import firebase_admin
from firebase_admin import auth as fb_auth, credentials

# Stripe
import stripe

# --------- Environment ---------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///facilitamed.db")
ASSETS_LOGO = os.getenv("ASSETS_LOGO", "assets/logo.png")
CORS_ORIGINS = (os.getenv("CORS_ORIGINS", "http://localhost:5173") or "").split(",")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Firebase
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON", "")  # path to service account JSON

# Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")  # monthly price id
APP_SITE_URL = os.getenv("APP_SITE_URL", "http://localhost:5173")  # frontend url

# Allowlist optional
ALLOWED_EMAILS = set(e.strip().lower() for e in os.getenv("ALLOWED_EMAILS", "").split(",") if e.strip())
ALLOWED_DOMAINS = set(d.strip().lower() for d in os.getenv("ALLOWED_DOMAINS", "").split(",") if d.strip())

engine = create_engine(DATABASE_URL, echo=False)

app = FastAPI(title="FacilitaMed API (Free Deploy Ready)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- DB ---------
from sqlmodel import SQLModel, Field as SQLField

class User(SQLModel, table=True):
    email: str = SQLField(primary_key=True, index=True)
    is_active: bool = False
    plan_expires_at: Optional[datetime] = None
    created_at: datetime = SQLField(default_factory=datetime.utcnow)

class Visit(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    cpf: str = SQLField(index=True)
    timestamp: datetime = SQLField(default_factory=datetime.utcnow, index=True)
    hda: str = ""
    antecedentes: str = ""
    alergias: str = ""
    meds_em_uso: str = ""
    exame_fisico: str = ""
    vitals: str = ""     # JSON string
    conduta: str = ""
    medicamentos: str = ""

class Patient(SQLModel, table=True):
    cpf: str = SQLField(primary_key=True, index=True)
    nome: str
    created_at: datetime = SQLField(default_factory=datetime.utcnow)

def init_db():
    SQLModel.metadata.create_all(engine)

@app.on_event("startup")
def on_startup():
    init_db()
    # Stripe
    if STRIPE_SECRET_KEY:
        stripe.api_key = STRIPE_SECRET_KEY
    # Firebase Admin
    if not firebase_admin._apps:
        cred = None
        if FIREBASE_CREDENTIALS_JSON and os.path.exists(FIREBASE_CREDENTIALS_JSON):
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_JSON)
        else:
            try:
                cred = credentials.ApplicationDefault()
            except Exception:
                cred = None
        if cred:
            firebase_admin.initialize_app(cred, {'projectId': FIREBASE_PROJECT_ID or None})
        else:
            firebase_admin.initialize_app()

# --------- Auth (Firebase ID Token) ---------
def verify_firebase_token(authorization: Optional[str]) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authorization: Bearer <ID_TOKEN> requerido")
    token = authorization.split(" ", 1)[1].strip()
    try:
        decoded = fb_auth.verify_id_token(token, check_revoked=True)
        return decoded
    except Exception as e:
        raise HTTPException(401, f"Token inválido: {e}")

def domain_allowed(email: str) -> bool:
    if ALLOWED_EMAILS and email.lower() in ALLOWED_EMAILS:
        return True
    if ALLOWED_DOMAINS and email.lower().split("@")[-1] in ALLOWED_DOMAINS:
        return True
    return not (ALLOWED_EMAILS or ALLOWED_DOMAINS)

def get_user(authorization: Optional[str] = Header(default=None)) -> User:
    decoded = verify_firebase_token(authorization)
    email = decoded.get("email")
    if not email:
        raise HTTPException(401, "Token sem e-mail")
    if not domain_allowed(email):
        raise HTTPException(403, "E-mail não autorizado")
    with Session(engine) as s:
        u = s.get(User, email.lower())
        if not u:
            u = User(email=email.lower(), is_active=False, plan_expires_at=None)
            s.add(u); s.commit()
        return u

def require_active_user(user: User = Depends(get_user)) -> User:
    if user.is_active and (not user.plan_expires_at or user.plan_expires_at > datetime.utcnow()):
        return user
    raise HTTPException(402, "Assinatura inativa/expirada")

# --------- OpenAI ---------
def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY não configurado")
    return OpenAI(api_key=OPENAI_API_KEY)

# --------- Schemas ---------
class Vitals(BaseModel):
    pa: Optional[str] = ""
    fc: Optional[str] = ""
    fr: Optional[str] = ""
    temp: Optional[str] = ""
    spo2: Optional[str] = ""
    peso: Optional[str] = ""
    altura: Optional[str] = ""
    imc: Optional[str] = ""

class VisitIn(BaseModel):
    hda: str = ""
    antecedentes: str = ""
    alergias: str = ""
    meds_em_uso: str = ""
    exame_fisico: str = ""
    vitals: Vitals = Field(default_factory=Vitals)
    conduta: str = ""
    medicamentos: str = ""

class PatientIn(BaseModel):
    cpf: str
    nome: str

class PatientOut(BaseModel):
    cpf: str
    nome: str
    visitas: list

# --------- Public status ---------
@app.get("/me/status")
def me_status(user: User = Depends(get_user)):
    active = user.is_active and (not user.plan_expires_at or user.plan_expires_at > datetime.utcnow())
    until = user.plan_expires_at.isoformat() if user.plan_expires_at else None
    return {"email": user.email, "active": active, "plan_expires_at": until}

# --------- Billing (Stripe) ---------
class CheckoutBody(BaseModel):
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None

@app.post("/billing/create-checkout-session")
def create_checkout_session(body: CheckoutBody, user: User = Depends(get_user)):
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        raise HTTPException(500, "Stripe não configurado")
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            customer_email=user.email,
            success_url=(body.success_url or f"{APP_SITE_URL}/?session_id={{CHECKOUT_SESSION_ID}}"),
            cancel_url=(body.cancel_url or f"{APP_SITE_URL}/?canceled=1"),
            allow_promotion_codes=True,
        )
        return {"url": session.url}
    except Exception as e:
        raise HTTPException(500, f"Stripe error: {e}")

# Verificação sem webhook (para Render Free, etc.)
class VerifyBody(BaseModel):
    session_id: str

@app.post("/billing/verify-session")
def verify_session(body: VerifyBody, user: User = Depends(get_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe não configurado")
    try:
        s = stripe.checkout.Session.retrieve(body.session_id)
        status = s.get("status")
        paid = s.get("payment_status")
        if paid == "paid" or status == "complete":
            with Session(engine) as db:
                u = db.get(User, user.email.lower())
                if u:
                    u.is_active = True
                    u.plan_expires_at = datetime.utcnow() + timedelta(days=30)
                    db.add(u); db.commit()
            return {"ok": True, "status": status, "payment_status": paid}
        return {"ok": False, "status": status, "payment_status": paid}
    except Exception as e:
        raise HTTPException(400, f"Stripe verify error: {e}")

# --------- Pacientes / Consultas ---------
@app.post("/api/patients")
def upsert_patient(p: PatientIn, user: User = Depends(require_active_user)):
    with Session(engine) as s:
        pat = s.get(Patient, p.cpf)
        if pat:
            pat.nome = p.nome
        else:
            pat = Patient(cpf=p.cpf, nome=p.nome)
            s.add(pat)
        s.commit()
    return {"ok": True}

@app.get("/api/patients/{cpf}", response_model=PatientOut)
def get_patient(cpf: str, user: User = Depends(require_active_user)):
    with Session(engine) as s:
        pat = s.get(Patient, cpf)
        if not pat:
            raise HTTPException(404, "Paciente não encontrado")
        visits = s.exec(select(Visit).where(Visit.cpf == cpf).order_by(Visit.timestamp.desc())).all()
        out_vis = []
        for v in visits:
            out_vis.append({
                "ts": v.timestamp.isoformat(),
                "hda": v.hda,
                "antecedentes": v.antecedentes,
                "alergias": v.alergias,
                "meds_em_uso": v.meds_em_uso,
                "exame_fisico": v.exame_fisico,
                "vitals": v.vitals,
                "conduta": v.conduta,
                "medicamentos": v.medicamentos,
            })
        return {"cpf": pat.cpf, "nome": pat.nome, "visitas": out_vis}

@app.post("/api/visits/{cpf}")
def add_visit(cpf: str, body: Dict[str, Any], user: User = Depends(require_active_user)):
    from json import dumps
    with Session(engine) as s:
        pat = s.get(Patient, cpf)
        if not pat:
            raise HTTPException(404, "Paciente não encontrado")
        v = Visit(
            cpf=cpf,
            hda=body.get("hda",""),
            antecedentes=body.get("antecedentes",""),
            alergias=body.get("alergias",""),
            meds_em_uso=body.get("meds_em_uso",""),
            exame_fisico=body.get("exame_fisico",""),
            vitals=dumps(body.get("vitals",{})),
            conduta=body.get("conduta",""),
            medicamentos=body.get("medicamentos",""),
        )
        s.add(v); s.commit()
        return {"ok": True, "id": v.id}

# --------- IA ---------
class IAContext(BaseModel):
    cpf: Optional[str] = None
    nome: Optional[str] = None
    hda: Optional[str] = ""
    antecedentes: Optional[str] = ""
    alergias: Optional[str] = ""
    exame_fisico: Optional[str] = ""
    vitals: Optional[Dict[str, Any]] = {}

class IAAsk(BaseModel):
    context: IAContext
    ask: str

def chat_once(system: str, user: str) -> str:
    client = get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content": system},
                      {"role":"user","content": user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=[{"role":"system","content": system},
                       {"role":"user","content": user}],
                temperature=0.2,
            )
            text = getattr(resp, "output_text", "") or ""
            return text.strip() or f"Sem resposta no momento. ({e})"
        except Exception as e2:
            return f"Falha IA: {e} / {e2}"

@app.post("/api/ia/diagnostico")
def ia_diagnostico(body: IAContext, user: User = Depends(require_active_user)):
    system = ("Você é um assistente clínico para APS (SUS). "
              "Liste 3-5 hipóteses iniciais com breve justificativa. "
              "A decisão final é do médico.")
    user_c = (f"Paciente: {body.nome} CPF {body.cpf}\n"
              f"HDA: {body.hda}\nAntecedentes: {body.antecedentes}\nAlergias: {body.alergias}\n"
              f"Exame Físico: {body.exame_fisico}\nSinais Vitais: {body.vitals}")
    return {"text": chat_once(system, user_c)}

@app.post("/api/ia/prescricao")
def ia_prescricao(body: IAContext, user: User = Depends(require_active_user)):
    system = ("Sugira 2-3 alternativas de prescrição (nomes genéricos), formato: "
              "'PRINCÍPIO — DOSE — POSOLOGIA — DURAÇÃO — OBS'. Sem diagnosticar. "
              "Decisão final é do médico.")
    user_c = (f"Paciente: {body.nome} CPF {body.cpf}\nHDA: {body.hda}\n"
              f"Antecedentes: {body.antecedentes}\nAlergias: {body.alergias}\n"
              f"Exame Físico: {body.exame_fisico}\nSinais Vitais: {body.vitals}")
    return {"text": chat_once(system, user_c)}

@app.post("/api/ia/laudo")
def ia_laudo(body: IAAsk, user: User = Depends(require_active_user)):
    system = ("Laudo objetivo para APS: Identificação; Achados/Contexto; Conclusão; Recomendações.")
    return {"text": chat_once(system, f"{body.context.model_dump()}\nPedido: {body.ask}")}

@app.post("/api/ia/interpretacao-exame")
def ia_interpret_exame(body: IAAsk, user: User = Depends(require_active_user)):
    system = ("Interprete exame para APS/SUS: Resumo; Interpretação; Condutas; Alertas; Plano terapêutico inicial.")
    return {"text": chat_once(system, f"{body.context.model_dump()}\nExame: {body.ask}")}

# --------- PDF ---------
class DocRequest(BaseModel):
    cpf: str
    nome: str
    corpo: Optional[str] = ""
    medico: Optional[str] = ""
    crm: Optional[str] = ""
    unidade: Optional[str] = ""

def draw_header(c: canvas.Canvas):
    try:
        if ASSETS_LOGO and os.path.exists(ASSETS_LOGO):
            c.drawImage(ImageReader(ASSETS_LOGO), 20*mm, 270*mm, width=24*mm, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50*mm, 285*mm, "UBS - Secretaria Municipal de Saúde")
    c.setFont("Helvetica", 10)
    c.drawString(50*mm, 279*mm, "FacilitaMed — Atenção Primária")

def pdf_response(builder, filename="documento.pdf"):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    draw_header(c)
    builder(c)
    c.showPage()
    c.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})

@app.post("/api/pdf/atestado")
def pdf_atestado(req: DocRequest, user: User = Depends(require_active_user)):
    def build(c: canvas.Canvas):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20*mm, 260*mm, "ATESTADO MÉDICO")
        c.setFont("Helvetica", 11)
        y = 245*mm
        lines = [
            f"Paciente: {req.nome} — CPF {req.cpf}",
            req.corpo or "Atesto, para os devidos fins, que o(a) paciente necessita de afastamento...",
            f"Médico: {req.medico} — CRM {req.crm} — {req.unidade}",
            f"Data: {datetime.now().strftime('%d/%m/%Y')}",
        ]
        for ln in lines:
            c.drawString(20*mm, y, ln); y -= 8*mm
    return pdf_response(build, "atestado.pdf")

@app.post("/api/pdf/receita")
def pdf_receita(req: DocRequest, user: User = Depends(require_active_user)):
    def build(c: canvas.Canvas):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20*mm, 260*mm, "RECEITUÁRIO SIMPLES")
        c.setFont("Helvetica", 11)
        y = 245*mm
        meds = (req.corpo or "").split("\n")
        hdr = f"Paciente: {req.nome} — CPF {req.cpf}"
        c.drawString(20*mm, y, hdr); y -= 10*mm
        for ln in meds:
            ln = ln.strip()
            if not ln: continue
            c.drawString(25*mm, y, f"- {ln}"); y -= 8*mm
        y -= 10*mm
        c.drawString(20*mm, y, f"Médico: {req.medico} — CRM {req.crm} — {req.unidade}"); y -= 8*mm
        c.drawString(20*mm, y, f"Data: {datetime.now().strftime('%d/%m/%Y')}")
    return pdf_response(build, "receita.pdf")

@app.post("/api/pdf/laudo")
def pdf_laudo(req: DocRequest, user: User = Depends(require_active_user)):
    def build(c: canvas.Canvas):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(20*mm, 260*mm, "LAUDO MÉDICO")
        c.setFont("Helvetica", 11)
        y = 245*mm
        lines = [
            f"Paciente: {req.nome} — CPF {req.cpf}",
            *(req.corpo or "").split("\n")
        ]
        for ln in lines:
            if not ln: continue
            c.drawString(20*mm, y, ln); y -= 8*mm
        y -= 10*mm
        c.drawString(20*mm, y, f"Médico: {req.medico} — CRM {req.crm} — {req.unidade}"); y -= 8*mm
        c.drawString(20*mm, y, f"Data: {datetime.now().strftime('%d/%m/%Y')}")
    return pdf_response(build, "laudo.pdf")
