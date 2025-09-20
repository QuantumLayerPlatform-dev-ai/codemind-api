"""
Generation response model
"""

from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
try:
    from .base import BaseModel
except ImportError:
    from models.base import BaseModel

class GenerationResponse(BaseModel):
    """Generation response model"""
    __tablename__ = "generation_responses"

    request_id = Column(String(255), ForeignKey("generation_requests.request_id"), nullable=False)
    content = Column(Text, nullable=False)
    model_used = Column(String(100), nullable=False)
    tokens_used = Column(Integer, nullable=False)
    cost_usd = Column(Float, nullable=False)
    duration_ms = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=True)
    status = Column(String(50), default="completed", nullable=False)

    # Relationships
    generation_request = relationship("GenerationRequest", backref="responses")

    def __repr__(self):
        return f"<GenerationResponse(request_id='{self.request_id}', model='{self.model_used}')>"