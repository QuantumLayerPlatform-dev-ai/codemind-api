"""
User model
"""

from sqlalchemy import Column, String, Boolean, Text
try:
    from .base import BaseModel
except ImportError:
    from models.base import BaseModel

class User(BaseModel):
    """User model"""
    __tablename__ = "users"

    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)

    def __repr__(self):
        return f"<User(email='{self.email}', full_name='{self.full_name}')>"