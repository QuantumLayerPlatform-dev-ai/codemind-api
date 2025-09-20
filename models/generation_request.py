"""
Generation request model
"""

from sqlalchemy import Column, String, Text, Float, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
try:
    from .base import BaseModel
except ImportError:
    from models.base import BaseModel

class GenerationRequest(BaseModel):
    """Generation request model"""
    __tablename__ = "generation_requests"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    request_id = Column(String(255), unique=True, nullable=False, index=True)
    business_description = Column(Text, nullable=False)
    complexity = Column(Float, default=0.5, nullable=False)
    requirements = Column(JSON, default=dict, nullable=False)
    status = Column(String(50), default="pending", nullable=False)
    task_type = Column(String(100), nullable=False)

    # Relationships
    user = relationship("User", backref="generation_requests")

    def __repr__(self):
        return f"<GenerationRequest(id='{self.request_id}', status='{self.status}')>"