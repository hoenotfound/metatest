# models_health.py
from pydantic import BaseModel
from typing import List, Optional


class AccountStats(BaseModel):
    spend: float
    impressions: int
    clicks: int
    ctr: float
    cpc: float
    results: Optional[float] = None
    cpr: Optional[float] = None


class Issue(BaseModel):
    level: str
    campaign_id: str
    campaign_name: str
    metric: str
    value: float
    benchmark: Optional[float] = None
    reason: str
    suggestion: str


class HealthCheckResponse(BaseModel):
    summary: str
    account_stats: AccountStats
    issues: List[Issue]
