from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# ============================
# USER TABLE
# ============================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    debates = relationship("DebateHistory", back_populates="user")


# ============================
# DEBATE HISTORY TABLE
# ============================

class DebateHistory(Base):
    __tablename__ = "debate_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question = Column(Text)
    final_answer = Column(Text)

    user = relationship("User", back_populates="debates")