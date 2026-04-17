from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("sqlite:///./data/aria.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    import os
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)


class Employee(Base):
    __tablename__ = "employees"
    id             = Column(Integer, primary_key=True)
    name           = Column(String(100), nullable=False)
    department     = Column(String(100), default="")
    position       = Column(String(100), default="")
    phone          = Column(String(20),  default="")
    email          = Column(String(100), default="")
    telegram_id    = Column(String(50),  default="")
    face_embedding = Column(LargeBinary, nullable=True)
    face_image     = Column(String(255), default="")
    is_active      = Column(Boolean, default=True)
    created_at     = Column(DateTime, default=datetime.utcnow)


class Guest(Base):
    __tablename__ = "guests"
    id             = Column(Integer, primary_key=True)
    name           = Column(String(100), default="")
    phone          = Column(String(20),  default="")
    company        = Column(String(100), default="")
    face_embedding = Column(LargeBinary, nullable=True)
    face_image     = Column(String(255), default="")
    visit_count    = Column(Integer, default=1)
    last_visit     = Column(DateTime, default=datetime.utcnow)
    notes          = Column(Text, default="")
    created_at     = Column(DateTime, default=datetime.utcnow)


class KnowledgeDoc(Base):
    __tablename__ = "knowledge_docs"
    id           = Column(Integer, primary_key=True)
    filename     = Column(String(255))
    file_path    = Column(String(255))
    file_type    = Column(String(20))
    status       = Column(String(20), default="pending")
    chunk_count  = Column(Integer, default=0)
    uploaded_at  = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)


class Appointment(Base):
    __tablename__ = "appointments"
    id            = Column(Integer, primary_key=True)
    guest_name    = Column(String(100), default="")
    employee_name = Column(String(100), default="")
    employee_id   = Column(Integer, default=0)
    scheduled_at  = Column(DateTime)
    purpose       = Column(Text, default="")
    status        = Column(String(20), default="confirmed")
    created_at    = Column(DateTime, default=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"
    id         = Column(Integer, primary_key=True)
    guest_id   = Column(Integer, default=0)
    role       = Column(String(10))
    content    = Column(Text)
    timestamp  = Column(DateTime, default=datetime.utcnow)
