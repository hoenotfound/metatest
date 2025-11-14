from typing import List, Dict, Any
from models_health import AccountStats, Issue


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


# Tuned benchmarks (generic, no industry references)
MIN_CTR = 0.015            # 1.5 percent
MAX_CPC = 2.00             # RM2.00 per click
MAX_CPR = 20.0             # RM20 per result
MAX_FREQUENCY = 5.0
MAX_CPM_MULTIPLIER = 1.6
MIN_SPEND_TO_JUDGE = 20.0
MIN_RESULTS_FOR_CPR = 5.0


def aggregate_account_stats(rows: List[Dict[str, Any]]) -> AccountStats:
    """
    Aggregate account level stats and define "results" in a flexible way:
    1) Prefer hard results (purchase / lead / complete_registration)
    2) If none, use messages
    3) If none, use landing page views
    """
    total_spend = 0.0
    total_impressions = 0
    total_clicks = 0

    total_hard_results = 0.0     # purchase / lead / complete_registration
    total_lpv = 0.0              # landing page views
    total_msg = 0.0              # message conversations

    for r in rows:
        spend = _safe_float(r.get("spend", 0))
        impressions = int(r.get("impressions", 0) or 0)
        clicks = int(r.get("clicks", 0) or 0)

        total_spend += spend
        total_impressions += impressions
        total_clicks += clicks

        for a in r.get("actions") or []:
            at = (a.get("action_type") or "").strip()
            val = _safe_float(a.get("value"))
            if at in ("purchase", "lead", "complete_registration"):
                total_hard_results += val
            elif at == "landing_page_view":
                total_lpv += val
            elif at.startswith("onsite_conversion.messaging_conversation_started"):
                total_msg += val

    # Decide what we call "results" for CPR:
    # 1) hard results, else 2) messages, else 3) landing page views
    if total_hard_results > 0:
        total_results = total_hard_results
    elif total_msg > 0:
        total_results = total_msg
    elif total_lpv > 0:
        total_results = total_lpv
    else:
        total_results = 0.0

    ctr = (total_clicks / total_impressions) if total_impressions else 0.0
    cpc = (total_spend / total_clicks) if total_clicks else 0.0
    cpr = (total_spend / total_results) if total_results else 0.0

    return AccountStats(
        spend=round(total_spend, 2),
        impressions=total_impressions,
        clicks=total_clicks,
        ctr=ctr,
        cpc=round(cpc, 4) if cpc else 0.0,
        results=total_results or None,
        cpr=round(cpr, 2) if cpr else None,
    )


def detect_issues(rows: List[Dict[str, Any]]) -> List[Issue]:
    issues: List[Issue] = []

    # Compute account median CPM
    cpm_values = []
    for r in rows:
        spend = _safe_float(r.get("spend"))
        impressions = int(r.get("impressions", 0) or 0)
        if spend > 0 and impressions > 0:
            cpm_values.append(spend * 1000.0 / impressions)

    account_median_cpm = None
    if cpm_values:
        sorted_cpm = sorted(cpm_values)
        mid = len(sorted_cpm) // 2
        account_median_cpm = (
            sorted_cpm[mid]
            if len(sorted_cpm) % 2
            else 0.5 * (sorted_cpm[mid - 1] + sorted_cpm[mid])
        )

    # Loop through campaigns
    for r in rows:
        cid = r.get("campaign_id", "")
        cname = r.get("campaign_name", "Unnamed campaign")

        spend = _safe_float(r.get("spend", 0))
        impressions = int(r.get("impressions", 0) or 0)
        clicks = int(r.get("clicks", 0) or 0)
        freq = _safe_float(r.get("frequency", 0))

        actions = r.get("actions") or []

        # RULE 0: Delivery Issue
        if 0 < spend < MIN_SPEND_TO_JUDGE and impressions < 1000:
            issues.append(Issue(
                level="low",
                campaign_id=cid,
                campaign_name=cname,
                metric="Delivery",
                value=impressions,
                benchmark=1000,
                reason="There is spend but impressions are still low.",
                suggestion="Check audience size, learning phase and delivery restrictions."
            ))

        if spend < MIN_SPEND_TO_JUDGE:
            continue

        ctr = (clicks / impressions) if impressions else 0.0
        cpc = (spend / clicks) if clicks else 0.0
        cpm = (spend * 1000.0 / impressions) if impressions else 0.0

        # RULE 1: Low CTR
        if ctr < MIN_CTR:
            issues.append(Issue(
                level="high",
                campaign_id=cid,
                campaign_name=cname,
                metric="CTR",
                value=round(ctr * 100, 2),
                benchmark=MIN_CTR * 100,
                reason="CTR is below baseline.",
                suggestion="Try a stronger hook, shorter text or fresher visuals."
            ))

        # RULE 2: High CPC
        if cpc > MAX_CPC:
            issues.append(Issue(
                level="medium",
                campaign_id=cid,
                campaign_name=cname,
                metric="CPC",
                value=round(cpc, 2),
                benchmark=MAX_CPC,
                reason="Clicks are more expensive than expected.",
                suggestion="Test broader audiences or simpler creative elements."
            ))

        # RULE 3: High Frequency
        if freq > MAX_FREQUENCY:
            issues.append(Issue(
                level="medium",
                campaign_id=cid,
                campaign_name=cname,
                metric="Frequency",
                value=round(freq, 2),
                benchmark=MAX_FREQUENCY,
                reason="Ad has been shown many times and may be overexposed.",
                suggestion="Rotate new creatives or reduce budget."
            ))

        # RULE 4: High CPM
        if account_median_cpm is not None and cpm > account_median_cpm * MAX_CPM_MULTIPLIER:
            issues.append(Issue(
                level="medium",
                campaign_id=cid,
                campaign_name=cname,
                metric="CPM",
                value=round(cpm, 2),
                benchmark=round(account_median_cpm, 2),
                reason="Cost per 1000 views is higher than usual.",
                suggestion="Review targeting, placements or creative relevance."
            ))

        # Gather deeper actions
        lpv = 0.0
        msg = 0.0
        results = 0.0

        for a in actions:
            at = (a.get("action_type") or "").strip()
            val = _safe_float(a.get("value"))
            if at == "landing_page_view":
                lpv += val
            if at.startswith("onsite_conversion.messaging_conversation_started"):
                msg += val
            if at in ("purchase", "lead", "complete_registration"):
                results += val

        # RULE 5: Clicks but no deeper intent
        if clicks >= 100 and (lpv + msg) < 1:
            issues.append(Issue(
                level="high",
                campaign_id=cid,
                campaign_name=cname,
                metric="Conversions",
                value=0.0,
                benchmark=None,
                reason="There are clicks but almost no deeper actions.",
                suggestion="Check link accuracy and page loading performance."
            ))

        # RULE 6: High CPR from hard results only
        if results >= MIN_RESULTS_FOR_CPR:
            cpr = spend / results
            if cpr > MAX_CPR:
                issues.append(Issue(
                    level="high",
                    campaign_id=cid,
                    campaign_name=cname,
                    metric="CPR",
                    value=round(cpr, 2),
                    benchmark=MAX_CPR,
                    reason="Cost per result is higher than baseline.",
                    suggestion="Improve offer clarity or target warmer audiences."
                ))

        # EXTRA RULE 7: Creative fatigue (CTR drop 3 days)
        ctr_trend = r.get("ctr_trend")
        if ctr_trend and len(ctr_trend) >= 3:
            if ctr_trend[-1] < ctr_trend[-2] < ctr_trend[-3]:
                issues.append(Issue(
                    level="medium",
                    campaign_id=cid,
                    campaign_name=cname,
                    metric="Creative Fatigue",
                    value=round(ctr_trend[-1] * 100, 2),
                    benchmark=None,
                    reason="CTR has declined over the last few days.",
                    suggestion="Refresh creative elements to avoid drop in engagement."
                ))

    return issues


def build_summary(account_stats: AccountStats, issues: List[Issue]) -> str:
    if not issues:
        return (
            f"Account looks stable. Spend RM{account_stats.spend:.2f} "
            f"across {account_stats.impressions} impressions and {account_stats.clicks} clicks."
        )

    high = sum(1 for i in issues if i.level == "high")
    med = sum(1 for i in issues if i.level == "medium")

    return (
        f"Found {len(issues)} issues this period "
        f"({high} high, {med} medium). Total spend RM{account_stats.spend:.2f}."
    )
