from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Obligation(BaseModel):
    industry: Optional[str] = None
    regulator: Optional[str] = None
    instrument_name: Optional[str] = None
    instrument_type: Optional[str] = None
    citation: Optional[str] = None
    actor: Optional[str] = None
    obligation_text: str
    trigger: Optional[str] = None
    deadline: Optional[str] = None
    penalty: Optional[str] = None
    effective_date: Optional[str] = None
    source_file: Optional[str] = None
    source_page: Optional[int] = None


class ExtractionResult(BaseModel):
    obligations: List[Obligation] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str
    similarity_top_k: int = 10
    industry: Optional[str] = None
    regulator: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class ExtractRequest(BaseModel):
    query: str
    similarity_top_k: int = 12
    max_return: int = 30
    industry: Optional[str] = None
    regulator: Optional[str] = None


class ExtractResponse(BaseModel):
    obligations: List[Obligation] = Field(default_factory=list)
    used_sources: List[Dict[str, Any]] = Field(default_factory=list)
