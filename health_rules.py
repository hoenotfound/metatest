# health_rules.py
from typing import List, Dict, Any
from models_health import AccountStats, Issue


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def aggregate_account_stats(rows: List[Dict[str, Any]]) -> AccountStats:
    """
    Aggregate account level stats from campaign insights.
    """
    total_spend = 0.0
    total_impressions = 0
    total_clicks = 0
    total_results = 0.0

    for r in rows:
        spend = _safe_float(r.get("spend", 0))
        impressions = int(r.get("impressions", 0) or 0)
        clicks = int(r.get("clicks", 0) or 0)

        total_spend += spend
        total_impressions += impressions
        total_clicks += clicks

        actions = r.get("actions") or []
        for a in actions:
            at = (a.get("action_type") or "").strip()
            val = _safe_float(a.get("value"))
            if at in ("purchase", "lead", "complete_registration"):
                total_results += val

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
    """
    Run simple health rules on campaign level insights.
    """
    issues: List[Issue] = []

    # Global benchmarks. You can tune these by client or objective later.
    MIN_CTR = 0.01             # 1 percent
    MAX_CPC = 1.50             # RM 1.50 per click
    MAX_FREQUENCY = 5.0        # above this: fatigue risk
    MAX_CPM_MULTIPLIER = 1.6   # CPM > 1.6 times account median
    MIN_SPEND_TO_JUDGE = 20.0  # below this we ignore most checks
    MIN_RESULTS_FOR_CPR = 5.0  # need enough results before judging CPR
    MAX_CPR = 20.0             # RM per result, adjust per client

    # Pre compute account CPM median
    cpm_values = []
    for r in rows:
        spend = _safe_float(r.get("spend", 0))
        impressions = int(r.get("impressions", 0) or 0)
        if spend > 0 and impressions > 0:
            cpm_values.append(spend * 1000.0 / impressions)

    account_median_cpm = None
    if cpm_values:
        sorted_cpm = sorted(cpm_values)
        mid = len(sorted_cpm) // 2
        if len(sorted_cpm) % 2 == 1:
            account_median_cpm = sorted_cpm[mid]
        else:
            account_median_cpm = 0.5 * (sorted_cpm[mid - 1] + sorted_cpm[mid])

    # Per campaign checks
    for r in rows:
        cid = r.get("campaign_id", "")
        cname = r.get("campaign_name", "Unnamed campaign")

        spend = _safe_float(r.get("spend", 0))
        impressions = int(r.get("impressions", 0) or 0)
        clicks = int(r.get("clicks", 0) or 0)
        freq = _safe_float(r.get("frequency", 0))

        actions = r.get("actions") or []

        # 0. Delivery problem: has spend but low impressions
        if 0 < spend < MIN_SPEND_TO_JUDGE and impressions < 1000:
            issues.append(Issue(
                level="low",
                campaign_id=cid,
                campaign_name=cname,
                metric="Delivery",
                value=impressions,
                benchmark=1000,
                reason="Campaign has started spending but impressions are still low.",
                suggestion=(
                    "Check audience size, learning phase, and any bid or budget limits. "
                    "Make sure there are no tight rules blocking delivery."
                ),
            ))

        if spend < MIN_SPEND_TO_JUDGE:
            # Too small to judge for deeper performance
            continue

        ctr = (clicks / impressions) if impressions else 0.0
        cpc = (spend / clicks) if clicks else 0.0
        cpm = (spend * 1000.0 / impressions) if impressions else 0.0

        # 1. Low CTR
        if ctr < MIN_CTR:
            issues.append(Issue(
                level="high",
                campaign_id=cid,
                campaign_name=cname,
                metric="CTR",
                value=round(ctr * 100, 2),     # store as percent
                benchmark=MIN_CTR * 100,
                reason="CTR is lower than the simple baseline for this account.",
                suggestion=(
                    "Refresh creatives and test stronger hooks in the main text. "
                    "Use more eye catching images or short videos."
                ),
            ))

        # 2. High CPC
        if cpc > MAX_CPC:
            issues.append(Issue(
                level="medium",
                campaign_id=cid,
                campaign_name=cname,
                metric="CPC",
                value=round(cpc, 3),
                benchmark=MAX_CPC,
                reason="Cost per click is high for the current spend.",
                suggestion=(
                    "Review targeting and placements. Test broader audiences or cheaper placements, "
                    "and pause weak ads with very low CTR."
                ),
            ))

        # 3. High frequency (fatigue risk)
        if freq > MAX_FREQUENCY:
            issues.append(Issue(
                level="medium",
                campaign_id=cid,
                campaign_name=cname,
                metric="Frequency",
                value=round(freq, 2),
                benchmark=MAX_FREQUENCY,
                reason="People have seen this ad many times. There is a risk of ad fatigue.",
                suggestion=(
                    "Rotate fresh creatives or reduce budget for this campaign. "
                    "Consider opening new audiences instead of pushing the same group."
                ),
            ))

        # 4. High CPM vs account median
        if account_median_cpm and cpm > account_median_cpm * MAX_CPM_MULTIPLIER:
            issues.append(Issue(
                level="medium",
                campaign_id=cid,
                campaign_name=cname,
                metric="CPM",
                value=round(cpm, 2),
                benchmark=round(account_median_cpm, 2),
                reason="This campaign is paying much more per 1000 views than the account median.",
                suggestion=(
                    "Tighten targeting or adjust bidding. Check if the creative fits the audience, "
                    "or move spend into better performing campaigns."
                ),
            ))

        # 5. Clicks but no deeper results (LPV or messages)
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

        if clicks >= 100 and (lpv + msg) < 1:
            issues.append(Issue(
                level="high",
                campaign_id=cid,
                campaign_name=cname,
                metric="Conversions",
                value=0.0,
                benchmark=None,
                reason="Campaign has spend and clicks but almost no landing page views or messages.",
                suggestion=(
                    "Check the URL and event setup. Make sure the landing page loads fast "
                    "and the link is correct in every ad."
                ),
            ))

        # 6. Expensive results (CPL / CPA) using results from actions
        if results >= MIN_RESULTS_FOR_CPR:
            cpr = spend / results if results else 0.0
            if cpr > MAX_CPR:
                issues.append(Issue(
                    level="high",
                    campaign_id=cid,
                    campaign_name=cname,
                    metric="CPR",
                    value=round(cpr, 2),
                    benchmark=MAX_CPR,
                    reason="Cost per result is high compared to a simple benchmark.",
                    suggestion=(
                        "Narrow targeting to higher intent audiences and improve the offer on the landing page. "
                        "Pause ad sets with very high cost per result."
                    ),
                ))

    return issues


def build_summary(account_stats: AccountStats, issues: List[Issue]) -> str:
    """
    Short human summary string for the account.
    """
    if not issues:
        return (
            f"Account looks healthy. Spend RM{account_stats.spend:.2f} over "
            f"{account_stats.impressions} impressions and {account_stats.clicks} clicks."
        )

    high = sum(1 for i in issues if i.level == "high")
    med = sum(1 for i in issues if i.level == "medium")

    return (
        f"Found {len(issues)} issues in this period "
        f"({high} high, {med} medium). Total spend RM{account_stats.spend:.2f}."
    )
