from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
import importlib
import importlib.util
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from urllib.parse import urlencode
from i18n import TRANSLATIONS, metric_keys, t
from models_health import HealthCheckResponse
from health_rules import aggregate_account_stats, detect_issues, build_summary


# Facebook Ads API imports
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.adsinsights import AdsInsights
from facebook_business.adobjects.ad import Ad
from facebook_business.adobjects.adcreative import AdCreative



# ========== Theming & Palette (Light/Dark aware) ==========
import plotly.express as px
from plotly.colors import diverging, sequential


def safe_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a dataframe column filled with defaults if missing."""

    if column in df.columns:
        return df[column].fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def format_int(series: pd.Series) -> pd.Series:
    """Pretty print integer-like columns for Streamlit tables."""

    return series.apply(lambda value: f"{int(value):,}" if pd.notna(value) else "0")


def format_currency(series: pd.Series, prefix: str = "", decimals: int = 2) -> pd.Series:
    """Format numeric values as currency strings."""

    return series.apply(
        lambda value: f"{prefix}{value:.{decimals}f}" if pd.notna(value) else f"{prefix}0.00"
    )


def format_percentage(series: pd.Series, decimals: int = 2, suffix: str = "%") -> pd.Series:
    """Format numeric values as percentage strings."""

    default = f"0.{decimals * '0'}{suffix}"
    return series.apply(
        lambda value: f"{value:.{decimals}f}{suffix}" if pd.notna(value) else default
    )


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to a 0-1 range."""

    max_value = series.max()
    if pd.isna(max_value) or max_value <= 0:
        return pd.Series(0, index=series.index, dtype="float64")
    return series / max_value


def format_frequency(series: pd.Series, decimals: int = 1, suffix: str = "x") -> pd.Series:
    """Format ad frequency values with a trailing multiplier suffix."""

    default = f"0.{decimals * '0'}{suffix}"
    return series.apply(
        lambda value: f"{value:.{decimals}f}{suffix}" if pd.notna(value) else default
    )


def get_streamlit_theme_base() -> str:
    try:
        return (st.get_option("theme.base") or "light").lower()
    except Exception:
        return "light"


def inject_global_styles(is_dark: bool, palette: dict) -> None:
    """Inject bespoke dashboard styling to elevate the visual hierarchy."""

    bg_color = "#020617" if is_dark else "#f1f5f9"
    surface_color = "rgba(15, 23, 42, 0.72)" if is_dark else "rgba(255, 255, 255, 0.86)"
    border_color = "rgba(148, 163, 184, 0.35)" if is_dark else "rgba(100, 116, 139, 0.25)"
    subtle_text = "#cbd5f5" if is_dark else "#475569"
    accent_color = palette["series"].get("clicks", "#3B82F6")
    highlight_color = palette["series"].get("lpv", "#22C55E")

    st.markdown(
        f"""
        <style>
        :root {{
            --dashboard-bg: {bg_color};
            --dashboard-surface: {surface_color};
            --dashboard-border: {border_color};
            --dashboard-muted: {subtle_text};
            --dashboard-accent: {accent_color};
            --dashboard-highlight: {highlight_color};
        }}

        [data-testid="stAppViewContainer"] > .main {{
            background: radial-gradient(circle at 0% 0%, rgba(59, 130, 246, 0.1), transparent 45%),
                        radial-gradient(circle at 100% 0%, rgba(16, 185, 129, 0.08), transparent 40%),
                        var(--dashboard-bg);
        }}

        .dashboard-hero {{
            background: linear-gradient(135deg, rgba(30, 64, 175, 0.85), rgba(16, 185, 129, 0.85));
            border-radius: 28px;
            padding: 32px 40px;
            display: flex;
            flex-wrap: wrap;
            gap: 32px;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 25px 50px -12px rgba(30, 41, 59, 0.65);
            color: #ffffff;
            position: relative;
            overflow: hidden;
        }}

        .dashboard-hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at top right, rgba(255,255,255,0.35), transparent 55%);
            pointer-events: none;
        }}

        .dashboard-hero h1 {{
            margin: 0;
            font-size: 2.4rem;
            line-height: 1.1;
            font-weight: 700;
        }}

        .dashboard-hero p {{
            margin-top: 12px;
            font-size: 1.05rem;
            max-width: 540px;
            opacity: 0.88;
        }}

        .hero-main {{
            position: relative;
            z-index: 2;
            max-width: 620px;
        }}

        .hero-kicker {{
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.16em;
            font-weight: 600;
            opacity: 0.75;
            margin-bottom: 12px;
        }}

        .hero-meta {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            position: relative;
            z-index: 2;
        }}

        .hero-badge {{
            backdrop-filter: blur(14px);
            background: rgba(15, 23, 42, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.28);
            border-radius: 18px;
            padding: 16px 20px;
            min-width: 220px;
        }}

        .hero-badge span {{
            display: block;
        }}

        .hero-badge .badge-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            opacity: 0.7;
        }}

        .hero-badge .badge-value {{
            font-weight: 600;
            margin-top: 6px;
            font-size: 1rem;
        }}

        .card-divider {{
            height: 1px;
            background: rgba(148, 163, 184, 0.35);
            margin: 28px 0;
        }}

        .section-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.35), transparent);
            margin: 48px 0 32px;
        }}

        [data-testid="stMetric"] {{
            background: var(--dashboard-surface);
            border-radius: 20px;
            padding: 22px;
            border: 1px solid var(--dashboard-border);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.15);
            backdrop-filter: blur(14px);
        }}

        [data-testid="stMetric"] [data-testid="stMetricLabel"] {{
            color: var(--dashboard-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.75rem;
        }}

        [data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: inherit;
            font-size: 1.95rem;
            font-weight: 700;
        }}

        [data-baseweb="tab-list"] {{
            gap: 0.75rem;
        }}

        button[role="tab"] {{
            border-radius: 999px !important;
            padding: 0.65rem 1.35rem !important;
            background: rgba(15, 23, 42, 0.35) !important;
            border: 1px solid var(--dashboard-border) !important;
            color: inherit !important;
            font-weight: 600;
            transition: all 0.25s ease;
        }}

        button[role="tab"][aria-selected="true"] {{
            background: linear-gradient(135deg, var(--dashboard-accent), var(--dashboard-highlight)) !important;
            border-color: transparent !important;
            color: #fff !important;
            box-shadow: 0 10px 25px -12px rgba(16, 185, 129, 0.8);
        }}

        div[data-testid="stNotification"] {{
            border-radius: 16px;
            border: 1px solid var(--dashboard-border);
            box-shadow: 0 18px 38px -24px rgba(15, 23, 42, 0.6);
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 22px;
            border: 1px solid var(--dashboard-border);
            overflow: hidden;
            box-shadow: 0 22px 45px -30px rgba(15, 23, 42, 0.65);
            background: var(--dashboard-surface);
        }}

        div[data-testid="stDataFrame"] > div:nth-child(2) {{
            border: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ====== Facebook OAuth Config (keep secrets in env or Streamlit secrets) ======
def _cfg(key: str, default: str | None = None) -> str | None:
    # prefer Streamlit secrets, then env var; never hard-code secrets
    val = None
    try:
        val = st.secrets.get(key, None)
    except Exception:
        pass
    return val or os.getenv(key, default)

FB_APP_ID = _cfg("FB_APP_ID")              # None if not set
FB_APP_SECRET = _cfg("FB_APP_SECRET")      # None if not set
# IMPORTANT: Set this to the exact public URL of your Streamlit app page
# e.g. "https://your-domain.app/app" or when testing locally "http://localhost:8501"
FB_REDIRECT_URI = _cfg("FB_REDIRECT_URI", "https://metatest.streamlit.app/")
FB_SCOPES = ["ads_read"]  # add "business_management" if you need broader listing/asset mgmt

GEMINI_API_KEY = _cfg("GEMINI_API_KEY")  # from st.secrets or env
genai = None
if GEMINI_API_KEY and importlib.util.find_spec("google.generativeai") is not None:
    genai = importlib.import_module("google.generativeai")
    genai.configure(api_key=GEMINI_API_KEY)

def style_fig(fig, *, height=400, showlegend=True):
    """Apply consistent background, grid, fonts, and sizing."""
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"]),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=PALETTE["grid"], zeroline=False)
    return fig

# --- Secrets-based profiles ---
def load_secret_profiles():
    """
    Reads profiles from .streamlit/secrets.toml
    Expected shape:
    [profiles.<name>]
    APP_ID=""; APP_SECRET=""; ACCESS_TOKEN=""; AD_ACCOUNT_ID=""
    """
    profs = st.secrets.get("profiles", {})
    out = {}
    for name, vals in profs.items():
        out[name] = {
            "app_id": vals.get("APP_ID", ""),
            "app_secret": vals.get("APP_SECRET", ""),
            "access_token": vals.get("ACCESS_TOKEN", ""),
            "ad_account_id": vals.get("AD_ACCOUNT_ID", ""),
        }
    return out

def to_numeric(df, cols):
    """Convert specified columns to numeric"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def derive_landing_page_views(df):
    """Derive landing page views from actions"""
    if 'actions' not in df.columns:
        df['landing_page_view'] = 0
        return df

    def pull(actions):
        total = 0.0
        try:
            arr = actions if isinstance(actions, list) else json.loads(actions)
            for a in arr or []:
                at = (a.get('action_type') or "")
                if at == 'landing_page_view':
                    v = a.get('value')
                    if v is not None:
                        try:
                            total += float(v)
                        except (TypeError, ValueError):
                            pass
        except Exception:
            pass
        return total

    df['landing_page_view'] = df['actions'].apply(pull)
    return df
def derive_msg_starts(df):
    """Derive messaging_conversation_starts from actions"""
    if 'actions' not in df.columns:
        df['messaging_conversation_starts'] = 0
        return df

    def pull(actions):
        total = 0.0
        try:
            arr = actions if isinstance(actions, list) else json.loads(actions)
            for a in arr or []:
                at = (a.get('action_type') or "")
                if at.startswith('onsite_conversion.messaging_conversation_started'):
                    v = a.get('value')
                    if v is not None:
                        try:
                            total += float(v)
                        except (TypeError, ValueError):
                            pass
        except Exception:
            pass
        return total

    df['messaging_conversation_starts'] = df['actions'].apply(pull)
    return df

def fetch_insights(ad_account_id, level, fields, start_date, end_date, 
                   time_increment=None, breakdowns=None):
    """Fetch insights from Facebook Ads API"""
    acct = AdAccount(ad_account_id)
    params = {
        'time_range': {'since': start_date, 'until': end_date},
        'level': level,
        'limit': 500,
        'action_report_time': 'conversion',
        'action_attribution_windows': ['7d_click', '1d_view'],
    }
    
    if time_increment:
        params['time_increment'] = time_increment
    if breakdowns:
        params['breakdowns'] = breakdowns

    data = list(acct.get_insights(fields=fields, params=params))
    return pd.DataFrame(data)

# ========== Data Fetching Functions ==========

@st.cache_data(ttl=3600)
def fetch_campaign_data(_api_init_params, ad_account_id, start_date, end_date):
    """Fetch campaign-level data"""
    fields = [
        AdsInsights.Field.campaign_id,
        AdsInsights.Field.campaign_name,
        AdsInsights.Field.spend,
        AdsInsights.Field.reach,
        AdsInsights.Field.impressions,
        AdsInsights.Field.clicks,
        AdsInsights.Field.ctr,
        AdsInsights.Field.cpc,
        "actions",
    ]

    
    df = fetch_insights(ad_account_id, "campaign", fields, start_date, end_date)
    df = derive_msg_starts(df)
    df = derive_landing_page_views(df)
    df = to_numeric(df, ["spend","reach","impressions","clicks","ctr","cpc","landing_page_view", "messaging_conversation_starts"])

    def calc_cpp(row):
        lpv = row.get("landing_page_view", 0)
        return (row["spend"] / lpv) if lpv not in (0, None) else None

    df["cost_per_lpv"] = df.apply(calc_cpp, axis=1)
    return df

@st.cache_data(ttl=3600)
def fetch_ad_data(_api_init_params, ad_account_id, start_date, end_date):
    fields = [
        AdsInsights.Field.ad_id,
        AdsInsights.Field.ad_name,
        AdsInsights.Field.adset_name,
        AdsInsights.Field.campaign_name,
        AdsInsights.Field.impressions,
        AdsInsights.Field.reach,          # <-- add
        AdsInsights.Field.clicks,
        AdsInsights.Field.ctr,
        AdsInsights.Field.cpc,
        AdsInsights.Field.spend,          # <-- add
        "actions",
    ]
    df = fetch_insights(ad_account_id, "ad", fields, start_date, end_date)
    df = derive_msg_starts(df)
    df = derive_landing_page_views(df)
    df = to_numeric(df, ["impressions","reach","clicks","ctr","cpc","spend",
                         "landing_page_view","messaging_conversation_starts"])
    # compute ad-level cost_per_lpv (optional but improves scoring)
    df["cost_per_lpv"] = np.where(
        (df["landing_page_view"] > 0),
        df["spend"] / df["landing_page_view"],
        np.nan
    )
    return df

@st.cache_data(ttl=3600)
def fetch_daily_data(_api_init_params, ad_account_id, start_date, end_date):
    fields = [
        AdsInsights.Field.date_start,
        AdsInsights.Field.clicks,
        AdsInsights.Field.impressions,
        AdsInsights.Field.spend,      # new
        "actions",
    ]
    df = fetch_insights(ad_account_id, "account", fields, start_date, end_date, time_increment=1)
    df = derive_landing_page_views(df)
    df = derive_msg_starts(df)
    df = to_numeric(df, ["clicks", "impressions", "landing_page_view",
                         "messaging_conversation_starts", "spend"])
    return df


@st.cache_data(ttl=3600)
def fetch_breakdown_data(_api_init_params, ad_account_id, start_date, end_date, breakdown_type):
    """Fetch data with specific breakdown"""
    fields = [AdsInsights.Field.clicks, AdsInsights.Field.impressions, "actions"]
    df = fetch_insights(ad_account_id, "account", fields, start_date, end_date, breakdowns=[breakdown_type])
    df = derive_landing_page_views(df)
    df = to_numeric(df, ["clicks","impressions","landing_page_view"])
    return df

@st.cache_data(ttl=3600)
def _best_preview_url_for_ad(ad_id: str) -> str | None:
    """
    Returns a preview image URL for a given ad:
    1) Try creative.thumbnail_url / image_url
    2) Fallback to Ad.get_previews() and parse the first <img src=...>
    """
    try:
        # 1) Try pulling creative thumbnail directly (fast & stable)
        creatives = Ad(ad_id).get_ad_creatives(
            fields=[
                AdCreative.Field.thumbnail_url,
                AdCreative.Field.image_url,
                AdCreative.Field.object_story_id,
            ],
            params={"limit": 1},
        )
        for c in creatives:
            url = c.get(AdCreative.Field.thumbnail_url) or c.get(AdCreative.Field.image_url)
            if url:
                return url
    except Exception:
        pass

    # 2) Fallback: generated preview HTML -> extract first <img src="...">
    try:
        previews = Ad(ad_id).get_previews(params={"ad_format": "DESKTOP_FEED_STANDARD"})
        for p in previews:
            html = p.get("body") or ""
            # tiny, permissive img-src extractor
            import re
            m = re.search(r'<img[^>]+src="([^"]+)"', html)
            if m:
                return m.group(1)
    except Exception:
        pass

    return None


import secrets


def fb_login_url(state: str | None = None) -> str:
    """Generate Facebook OAuth login URL with CSRF state token."""

    params = {
        "client_id": FB_APP_ID,
        "redirect_uri": FB_REDIRECT_URI,  # Must match EXACTLY in token exchange
        "response_type": "code",
        "scope": ",".join(FB_SCOPES),
    }
    if state:
        params["state"] = state
    return f"https://www.facebook.com/v20.0/dialog/oauth?{urlencode(params)}"


def fb_exchange_code_for_token(code: str) -> dict:
    """Exchange authorization code for access token"""

    try:
        response = requests.get(
            "https://graph.facebook.com/v20.0/oauth/access_token",
            params={
                "client_id": FB_APP_ID,
                "redirect_uri": FB_REDIRECT_URI,  # MUST match the OAuth dialog request
                "client_secret": FB_APP_SECRET,
                "code": code,
            },
            timeout=20,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError("Failed to exchange Facebook authorization code") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("Facebook authorization response was not valid JSON") from exc


def fb_long_lived_token(short_token: str) -> dict:
    """Exchange short-lived token for long-lived token (~60 days)"""

    try:
        response = requests.get(
            "https://graph.facebook.com/v20.0/oauth/access_token",
            params={
                "grant_type": "fb_exchange_token",
                "client_id": FB_APP_ID,
                "client_secret": FB_APP_SECRET,
                "fb_exchange_token": short_token,
            },
            timeout=20,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError("Failed to upgrade to a long-lived Facebook token") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("Facebook long-lived token response was not valid JSON") from exc


def fb_api_get(path: str, token: str, **params) -> dict:
    """Make authenticated request to Facebook Graph API"""

    try:
        response = requests.get(
            f"https://graph.facebook.com/v20.0/{path.lstrip('/')}",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Facebook API request failed for '{path}'") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Facebook API response for '{path}' was not valid JSON") from exc

def detect_anomalies(daily_df: pd.DataFrame, campaigns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple rule based checks for obvious problems.
    Returns a small dataframe with columns:
    level, metric, when, detail, suggestion
    """

    issues: list[dict[str, str]] = []

    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=["level", "metric", "when", "detail", "suggestion"])

    df = daily_df.copy()
    if "date_start" not in df.columns:
        return pd.DataFrame(columns=["level", "metric", "when", "detail", "suggestion"])
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df = df[df["date_start"].notna()].sort_values("date_start")

    if df.empty or "impressions" not in df.columns or "clicks" not in df.columns:
        return pd.DataFrame(columns=["level", "metric", "when", "detail", "suggestion"])

    # Compute daily CTR and CPM
    df["ctr"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0.0)
    if "spend" in df.columns:
        df["cpm"] = np.where(df["impressions"] > 0, df["spend"] * 1000.0 / df["impressions"], 0.0)
    else:
        df["cpm"] = 0.0

    # 1) Sudden CTR drop vs previous period
    if len(df) >= 5:
        last_row = df.iloc[-1]
        prev = df.iloc[:-1]
        prev_ctr_mean = prev["ctr"].mean()
        if prev_ctr_mean > 0 and last_row["ctr"] < prev_ctr_mean * 0.6:
            drop_pct = (1.0 - last_row["ctr"] / prev_ctr_mean) * 100.0
            issues.append({
                "level": "high",
                "metric": "CTR",
                "when": last_row["date_start"].strftime("%Y-%m-%d"),
                "detail": f"Daily CTR dropped about {drop_pct:.0f} percent vs recent average.",
                "suggestion": "Refresh creatives and test new hooks or thumbnails. Also review targeting.",
            })

    # 2) Spend spike vs recent average
    if "spend" in df.columns and df["spend"].sum() > 0 and len(df) >= 5:
        last_row = df.iloc[-1]
        prev = df.iloc[:-1]
        prev_spend_mean = prev["spend"].mean()

        if prev_spend_mean > 0 and last_row["spend"] > prev_spend_mean * 1.8:
            spike_pct = (last_row["spend"] / prev_spend_mean - 1.0) * 100.0
            issues.append({
                "level": "medium",
                "metric": "Spend",
                "when": last_row["date_start"].strftime("%Y-%m-%d"),
                "detail": f"Daily spend spiked about {spike_pct:.0f} percent vs recent average.",
                "suggestion": "Check if this is intentional. If not, review budget caps and bid strategy.",
            })

    # 3) Sudden zero conversions (LPV or messages) after having some
    conv_cols = []
    if "landing_page_view" in df.columns:
        conv_cols.append("landing_page_view")
    if "messaging_conversation_starts" in df.columns:
        conv_cols.append("messaging_conversation_starts")

    if conv_cols and len(df) >= 5:
        last_rows = df.tail(2)
        prev = df.iloc[:-2]
        for col in conv_cols:
            prev_avg = prev[col].mean()
            recent_sum = last_rows[col].sum()
            if prev_avg > 0.5 and recent_sum == 0:
                pretty_name = "Landing page views" if col == "landing_page_view" else "Message conversations"
                issues.append({
                    "level": "high",
                    "metric": pretty_name,
                    "when": f"{last_rows.iloc[0]['date_start'].strftime('%Y-%m-%d')} to {last_rows.iloc[-1]['date_start'].strftime('%Y-%m-%d')}",
                    "detail": f"{pretty_name} dropped to zero after having consistent activity before.",
                    "suggestion": "Check pixel or event tracking, landing page, and ad links. Something may be broken.",
                })

    # Campaign level checks
    if campaigns_df is not None and not campaigns_df.empty:
        camp = campaigns_df.copy()
        camp = to_numeric(camp, ["impressions", "reach", "spend", "ctr"])

        # 4) High frequency campaigns
        if "impressions" in camp.columns and "reach" in camp.columns:
            camp["frequency"] = np.where(camp["reach"] > 0, camp["impressions"] / camp["reach"], 0.0)
            noisy = camp[camp["frequency"] >= 4.0]
            for _, row in noisy.iterrows():
                cname = row.get("campaign_name", "Unnamed campaign")
                freq = row["frequency"]
                issues.append({
                    "level": "medium",
                    "metric": "Frequency",
                    "when": "whole period",
                    "detail": f"Campaign \"{cname}\" has high frequency around {freq:.1f}.",
                    "suggestion": "Consider refreshing creative or expanding audience to avoid fatigue.",
                })

        # 5) CPM too high vs account median
        if "spend" in camp.columns and "impressions" in camp.columns:
            camp["cpm"] = np.where(camp["impressions"] > 0, camp["spend"] * 1000.0 / camp["impressions"], np.nan)
            valid = camp[camp["cpm"].notna() & (camp["cpm"] > 0)]
            if not valid.empty:
                median_cpm = valid["cpm"].median()
                high_cpm = valid[valid["cpm"] > median_cpm * 1.6]
                for _, row in high_cpm.iterrows():
                    cname = row.get("campaign_name", "Unnamed campaign")
                    cpm_val = row["cpm"]
                    issues.append({
                        "level": "medium",
                        "metric": "CPM",
                        "when": "whole period",
                        "detail": f"Campaign \"{cname}\" CPM is about RM {cpm_val:.2f}, higher than account median.",
                        "suggestion": "Tighten targeting or adjust bidding. Also check if creative is relevant to the audience.",
                    })

    if not issues:
        return pd.DataFrame(columns=["level", "metric", "when", "detail", "suggestion"])

    return pd.DataFrame(issues)

def build_local_health_from_campaigns(campaigns_df) -> dict:
    """
    Turn campaigns_df into the same JSON shape as the backend HealthCheckResponse.
    So Gemini prompt stays the same.
    """
    if campaigns_df is None or campaigns_df.empty:
        return {
            "summary": "No data for this period.",
            "account_stats": {
                "spend": 0.0,
                "impressions": 0,
                "clicks": 0,
                "ctr": 0.0,
                "cpc": 0.0,
                "results": None,
                "cpr": None,
            },
            "issues": [],
        }

    # Work on a copy so we do not mutate the global df
    df = campaigns_df.copy()

    # 1) Compute frequency if possible
    if "frequency" not in df.columns and {"impressions", "reach"}.issubset(df.columns):
        df["frequency"] = np.where(df["reach"] > 0, df["impressions"] / df["reach"], 0.0)

    rows = []
    for _, row in df.iterrows():
        actions = row.get("actions", [])

        # Make sure actions is always a list, not a JSON string
        if isinstance(actions, str):
            try:
                actions = json.loads(actions)
            except Exception:
                actions = []

        if not isinstance(actions, list):
            actions = []

        rows.append({
            "campaign_id": row.get("campaign_id", ""),
            "campaign_name": row.get("campaign_name", ""),
            "spend": float(row.get("spend", 0) or 0),
            "impressions": float(row.get("impressions", 0) or 0),
            "clicks": float(row.get("clicks", 0) or 0),
            "frequency": float(row.get("frequency", 0) or 0),
            # extra fields for smarter rules (backend can ignore if it does not use them)
            "landing_page_view": float(row.get("landing_page_view", 0) or 0),
            "messaging_conversation_starts": float(row.get("messaging_conversation_starts", 0) or 0),
            "ctr": float(row.get("ctr", 0) or 0),
            "cpc": float(row.get("cpc", 0) or 0),
            "actions": actions,
        })

    account_stats = aggregate_account_stats(rows)
    issues = detect_issues(rows)
    summary = build_summary(account_stats, issues)

    resp = HealthCheckResponse(
        summary=summary,
        account_stats=account_stats,
        issues=issues,
    )
    return resp.model_dump()


def explain_health_with_gemini(
    health_data: dict,
    anomalies_df: pd.DataFrame | None,
    lang: str = "en",
) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key not configured."

    if genai is None:
        return "Gemini integration is unavailable because google-generativeai is not installed."

    model = genai.GenerativeModel("gemini-2.5-flash")

    summary = health_data.get("summary", "")
    stats = health_data.get("account_stats", {})
    issues = health_data.get("issues", [])

    # Convert anomalies to list
    anomalies = []
    if anomalies_df is not None and not anomalies_df.empty:
        try:
            anomalies = anomalies_df.to_dict(orient="records")
        except Exception:
            anomalies = []

    if lang == "zh":
        language_hint = (
            "Reply in Chinese, using simple, natural Malaysian style. "
            "Write like you are explaining to a busy business owner on WhatsApp."
        )
    else:
        language_hint = (
            "Reply in simple English for Malaysian SME owners. "
            "No heavy marketing jargon. Sounds like a friendly consultant."
        )

    prompt = f"""
You are a Meta ads performance strategist.

Your job is to turn the backend JSON into a clear, client-ready explanation that a Malaysian SME owner can understand quickly.

Below is the health data:

SUMMARY:
{summary}

ACCOUNT_STATS:
{json.dumps(stats, ensure_ascii=False, indent=2)}

BACKEND_ISSUES:
{json.dumps(issues, ensure_ascii=False, indent=2)}

ANOMALIES_FROM_DASHBOARD:
{json.dumps(anomalies, ensure_ascii=False, indent=2)}


=====================
### STRICT OUTPUT FORMAT
Follow this EXACT structure and spacing. Do not merge points into the same line.

---

### 1. TL;DR
Write 2–3 friendly, simple sentences.
• Summarize overall performance  
• Mention the single biggest issue  
• No long paragraphs  

---

### 2. Key Issues
List each issue on its own line. Do not merge them.

Format exactly like this:
🔴 **Issue Title** – One short sentence only  
🟡 **Issue Title** – One short sentence only  
🟢 **Issue Title** – One short sentence only  

Group issues by themes only if relevant:
• Engagement (CTR, CPC, creatives)  
• Spend Efficiency (CPM, cost spikes, wastage)  
• Audience (reach, fatigue, frequency)  
• Conversion/Tracking (LPV, messages, pixel problems, zero events)

Each line must be separate.

---

### 3. Opportunities
Write 2–4 bullets.
Each bullet must be one line:

• Positive signal 1  
• Positive signal 2  
• Positive signal 3  

Keep bullets short and encouraging.

---

### 4. Action Plan
Give 3–6 numbered action steps, each on its own line.

Format exactly like this:
1️⃣ Action step in 1–2 sentences.  
2️⃣ Action step in 1–2 sentences.  
3️⃣ Action step in 1–2 sentences.  
4️⃣ Action step in 1–2 sentences.  

Do not combine several steps into a single line.

---

### Tone Rules:
{language_hint}
• No jargon  
• No technical fluff  
• Sound like a friendly consultant on WhatsApp  
• Keep total length under 250–300 words  
• Do not include anything outside the 4 sections above.

Generate the final answer now.
"""

    try:
        resp = model.generate_content(prompt)
        return resp.text or "(No response from Gemini)"
    except Exception as e:
        return f"Error calling Gemini: {e}"



def main():
    st.set_page_config(
        page_title="Facebook Ads Dashboard",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---- Theme & Plotly defaults ----
    THEME_BASE = get_streamlit_theme_base()
    IS_DARK = THEME_BASE == "dark"

    global PALETTE, SERIES_COLOR_MAP
    PALETTE = {
        "text": "#e5e7eb" if IS_DARK else "#0f172a",
        "grid": "rgba(148, 163, 184, 0.25)" if IS_DARK else "rgba(100, 116, 139, 0.25)",
        "bg": "rgba(0,0,0,0)",
        "series": {
            "clicks":        "#3B82F6",
            "impressions":   "#F59E0B",
            "lpv":           "#22C55E",
            "conversations": "#A855F7",
            "spend":         "#EF4444",
            "reach":         "#06B6D4",
            "ctr":           "#10B981",
            "cpc":           "#F97316",
        },
        "categorical": [
            "#3B82F6","#22C55E","#F59E0B","#EF4444","#A855F7",
            "#06B6D4","#84CC16","#F97316","#EAB308","#10B981",
        ],
        "sequential": sequential.Viridis,
        "diverging":  diverging.RdBu,
    }

    px.defaults.template = "plotly_dark" if IS_DARK else "plotly"
    px.defaults.color_discrete_sequence = PALETTE["categorical"]
    px.defaults.color_continuous_scale = PALETTE["sequential"]

    SERIES_COLOR_MAP = {
        "Clicks":        PALETTE["series"]["clicks"],
        "Impressions":   PALETTE["series"]["impressions"],
        "Landing Page Views": PALETTE["series"]["lpv"],
        "Conversations": PALETTE["series"]["conversations"],
        "Spend":         PALETTE["series"]["spend"],
        "Reach":         PALETTE["series"]["reach"],
        "CTR":           PALETTE["series"]["ctr"],
        "CPC":           PALETTE["series"]["cpc"],
    }

    inject_global_styles(IS_DARK, PALETTE)

    # Optional mapping for platform-specific colors (left empty to use default palette)
    platform_colors = {}

    # ====== OAuth Session State ======
    if "fb_user_token" not in st.session_state:
        st.session_state.fb_user_token = None
    if "fb_user_name" not in st.session_state:
        st.session_state.fb_user_name = None
    if "fb_ad_accounts" not in st.session_state:
        st.session_state.fb_ad_accounts = []
    if "fb_oauth_state" not in st.session_state:
        st.session_state.fb_oauth_state = None

    # ====== Handle OAuth redirect ======
    qp = st.query_params
    code_param = qp.get("code")
    error_param = qp.get("error")

    if error_param:
        st.error(f"Facebook login error: {error_param}")
        error_desc = qp.get("error_description")
        if error_desc:
            st.error(f"Details: {error_desc}")
        st.stop()

    # Process the OAuth code if present
    state_param = qp.get("state")

    if st.session_state.fb_user_token is None and code_param:
        expected_state = st.session_state.get("fb_oauth_state")
        if expected_state and state_param != expected_state:
            st.error("State mismatch during Facebook login. Please try signing in again.")
            st.session_state.fb_oauth_state = None
            st.stop()

        try:
            with st.spinner("Completing sign-in..."):
                short = fb_exchange_code_for_token(code_param)
                longl = fb_long_lived_token(short["access_token"])
                st.session_state.fb_user_token = longl["access_token"]

                me = fb_api_get("/me", st.session_state.fb_user_token, fields="name,id")
                st.session_state.fb_user_name = me.get("name", "Facebook user")

                accs = fb_api_get(
                    "/me/adaccounts",
                    st.session_state.fb_user_token,
                    fields="id,account_id,name,currency,timezone_name",
                    limit=200,
                )
                st.session_state.fb_ad_accounts = accs.get("data", [])

                st.session_state.fb_oauth_state = None
                st.query_params.clear()
                st.success(f"✅ Signed in as {st.session_state.fb_user_name}")
                st.rerun()
        except RuntimeError as e:
            st.error(f"❌ Login failed: {e}")
            st.session_state.fb_oauth_state = None
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Facebook API error: {e}")
            try:
                error_detail = e.response.json()
                st.json(error_detail)
            except Exception:
                st.code(e.response.text)
            st.session_state.fb_oauth_state = None
            st.stop()
        except Exception as e:
            st.error(f"❌ Login failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.fb_oauth_state = None
            st.stop()

    # ====== SIDEBAR: Authentication & Settings ======
    with st.sidebar:
        # If not logged in, show login options
        if not st.session_state.fb_user_token:
            # Optional logo above authentication header - try local file first, then LOGO_URL from config/secrets
            try:
                logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
            except Exception:
                logo_path = "logo.png"

            try:
                if os.path.exists(logo_path):
                    st.image(logo_path, width=100)
                else:
                    # prefer configured LOGO_URL, otherwise use this default URL
                    logo_url = _cfg("LOGO_URL", "")
                    if not logo_url:
                        logo_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFVK7PWJ_oV1dYs3YBZJEN2sv0Xw-Gcf-grA&s"
                    if logo_url:
                        st.image(logo_url, width=100)
            except Exception:
                # Don't break the sidebar if image loading fails
                pass

            st.header("🔐 Authentication")
            
            # Try to load profiles
            try:
                profiles = load_secret_profiles()
            except Exception:
                profiles = {}
            
            # Show login method selector
            if profiles:
                login_method = st.radio(
                    "Choose login method:",
                    ["Facebook OAuth", "Saved Profile"],
                    label_visibility="collapsed"
                )
                
                if login_method == "Facebook OAuth":
                    st.markdown("### Sign in with Facebook")
                    st.info("Login to access all your ad accounts")

                    if not st.session_state.fb_oauth_state:
                        st.session_state.fb_oauth_state = secrets.token_urlsafe(16)

                    # with st.expander("🔧 OAuth Debug Info"):
                    #     st.code(f"Redirect URI: {FB_REDIRECT_URI}")
                    #     st.caption("This must match your Facebook App settings")

                    st.link_button(
                        "🔑 Sign in with Facebook",
                        fb_login_url(st.session_state.fb_oauth_state),
                        use_container_width=True,
                        type="primary"
                    )
                
                else:  # Saved Profile
                    st.markdown("### Use Saved Profile")
                    st.info("Quick access with pre-configured tokens")
                    
                    profile_names = list(profiles.keys())
                    selected_profile = st.selectbox(
                        "Select Profile", 
                        profile_names,
                        label_visibility="collapsed"
                    )
                    
                    if st.button("✓ Use This Profile", type="primary", use_container_width=True):
                        profile = profiles[selected_profile]
                        
                        if not profile.get("access_token") or not profile.get("ad_account_id"):
                            st.error("⚠️ Profile missing ACCESS_TOKEN or AD_ACCOUNT_ID")
                            st.stop()
                        
                        st.session_state.fb_user_token = profile["access_token"]
                        st.session_state.fb_user_name = f"Profile: {selected_profile}"
                        
                        # mark session as using a saved profile and persist the profile blob
                        st.session_state.is_profile_mode = True
                        st.session_state.selected_profile = {
                            "app_id": profile.get("app_id"),
                            "app_secret": profile.get("app_secret"),
                            "access_token": profile.get("access_token"),
                            "ad_account_id": profile.get("ad_account_id"),
                        }

                        # populate ad accounts in the same shape as OAuth path
                        st.session_state.fb_ad_accounts = [{
                            "id": profile["ad_account_id"],
                            "account_id": profile["ad_account_id"].replace("act_", ""),
                            "name": selected_profile,
                            "currency": "USD",
                            "timezone_name": "UTC"
                        }]
                        
                        st.success(f"✅ Loaded: {selected_profile}")
                        st.rerun()
                    
                    with st.expander("ℹ️ How to create profiles"):
                        st.markdown("""
                        **Send to Caden at: hoeyinf1@yahoo.com**
                        
                        ```toml
                        [profiles.my_account]
                        APP_ID = "your_app_id"
                        APP_SECRET = "your_app_secret"
                        ACCESS_TOKEN = "your_token"
                        AD_ACCOUNT_ID = "act_123456"
                        ```
                        
                        Get tokens at:
                        - [Graph API Explorer](https://developers.facebook.com/tools/explorer/)
                        - [Token Debugger](https://developers.facebook.com/tools/debug/accesstoken/)
                        """)
            
            else:
                # No profiles, show only OAuth
                st.markdown("### Sign in with Facebook")
                st.info("Login to access your ad accounts")

                with st.expander("🔧 OAuth Debug Info"):
                    st.code(f"Redirect URI: {FB_REDIRECT_URI}")
                    st.caption("This must match your Facebook App settings")

                if not st.session_state.fb_oauth_state:
                    st.session_state.fb_oauth_state = secrets.token_urlsafe(16)

                st.link_button(
                    "🔑 Sign in with Facebook",
                    fb_login_url(st.session_state.fb_oauth_state),
                    use_container_width=True,
                    type="primary"
                )
                
                st.markdown("---")
                st.caption("💡 **Tip:** Add profiles to `secrets.toml` for quick access without OAuth")
            
            st.stop()
        
        # ===== Sidebar styling (light card look) =====
        st.markdown("""
        <style>
        /* Sidebar section cards */
        .sb-row { display: flex; gap: 6px; align-items: center; }
        .sb-kicker { font-size: 0.8rem; opacity: .7; }
        .sb-dot {
            width: 5px; height: 5px;
            border-radius: 50%;
            background:#10b981;
            display:inline-block;
            margin-right:5px;
        }

        /* If the quick picks are in the MAIN area */
        div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
            font-size: 0.70rem !important;
            padding: 2px 6px !important;
            height: 24px !important;
            min-width: 48px !important;
            line-height: 1.1 !important;
            border-radius: 4px !important;
        }

        /* If they are in the SIDEBAR instead, use this stricter scope */
        div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
            font-size: 0.70rem !important;
            padding: 2px 6px !important;
            height: 24px !important;
            min-width: 48px !important;
            line-height: 1.1 !important;
            border-radius: 4px !important;
        }
        
        </style>
        """, unsafe_allow_html=True)



        # ===== Defaults =====
        if "data_ready" not in st.session_state:
            st.session_state.data_ready = False
        if "last_params" not in st.session_state:
            st.session_state.last_params = None


        # ===== Header / account area =====
        st.header("⚙️ Settings")

        # top row: user + logout
        u1, u2 = st.columns([4, 1])
        with u1:
            st.caption(f"**{st.session_state.get('fb_user_name','')}**")
        with u2:
            if st.button("🚪", help="Logout", use_container_width=True):
                for k in ("fb_user_token", "fb_user_name", "fb_ad_accounts", "data_ready", "last_params"):
                    st.session_state.pop(k, None)
                st.query_params.clear()
                st.rerun()


        # ===== Ad Account =====
        st.markdown('<div class="sb-card">', unsafe_allow_html=True)
        st.markdown('<h4>📊 Ad Account</h4>', unsafe_allow_html=True)

        acct_opts, acct_id_map = [], {}
        for a in st.session_state.fb_ad_accounts:
            label = f"{a.get('name','(no name)')} — {a.get('account_id','?')} ({a.get('currency','')})"
            acct_opts.append(label)
            acct_id_map[label] = a.get("id")

        if not acct_opts:
            st.warning("No ad accounts available")
            st.stop()

        selected_label = st.selectbox(
            "Select Account", acct_opts, index=0, label_visibility="collapsed",
            help="Choose the ad account to report on"
        )
        selected_ad_account_id = acct_id_map[selected_label]
        st.markdown('</div>', unsafe_allow_html=True)


        # ===== Language =====
        st.markdown('<div class="sb-card">', unsafe_allow_html=True)
        st.markdown('<h4>🌐 Language</h4>', unsafe_allow_html=True)
        lang = st.selectbox("Language", options=['en', 'zh'], index=0, label_visibility="collapsed")

        translations = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
        def L_current(key: str) -> str:
            return translations.get(key, t('en', key, key))
        st.markdown('</div>', unsafe_allow_html=True)


        # ===== Date range + quick picks =====
        st.markdown('<div class="sb-card">', unsafe_allow_html=True)
        st.markdown(f'<h4>📅 {L_current("date_range")}</h4>', unsafe_allow_html=True)

        # --- wrap the buttons inside a <div class="quick-ranges"> container ---

        q1, q2, q3, q4 = st.columns(4)
        quick_pick = None
        if q1.button("7d", use_container_width=True): quick_pick = 7
        if q2.button("30d", use_container_width=True): quick_pick = 30
        if q3.button("90d", use_container_width=True): quick_pick = 90
        if q4.button("YTD", use_container_width=True): quick_pick = "ytd"


        default_end = datetime.now().date()
        default_start = default_end - timedelta(days=30)

        if quick_pick == 7:
            default_start, default_end = default_end - timedelta(days=7), default_end
        elif quick_pick == 30:
            default_start, default_end = default_end - timedelta(days=30), default_end
        elif quick_pick == 90:
            default_start, default_end = default_end - timedelta(days=90), default_end
        elif quick_pick == "ytd":
            default_start, default_end = datetime(default_end.year, 1, 1).date(), default_end

        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start", value=default_start, label_visibility="collapsed")
        with c2:
            end_date = st.date_input("End", value=default_end, label_visibility="collapsed")

        # simple validation
        if start_date > end_date:
            st.error("Start date must be on or before end date.")

        st.markdown('</div>', unsafe_allow_html=True)   # close sb-card (move this here)

        # ===== Metric selection =====
        st.markdown(f'<h4>🔎 {L_current("metric_select")}</h4>', unsafe_allow_html=True)
        metric_keys_dict = metric_keys(lang)
        metric_options = list(metric_keys_dict.values())
        selected_metric = st.selectbox("Metric", metric_options, index=0, label_visibility="collapsed",
                                    help="Choose the primary KPI for charts")
        metric_key = [k for k, v in metric_keys_dict.items() if v == selected_metric][0]
        st.markdown('</div>', unsafe_allow_html=True)

        # ===== Load data CTA (sticky feel by placing last) =====
        st.markdown('<div class="sb-row"><span class="sb-dot"></span><span class="sb-kicker">Ready to fetch data</span></div>', unsafe_allow_html=True)
        load_clicked = st.button("📥 " + L_current('load_data'), type="primary", use_container_width=True,
                                disabled=(start_date > end_date))

        # update params & readiness
        params_tuple = (selected_ad_account_id, str(start_date), str(end_date), metric_key, lang)
        if st.session_state.last_params != params_tuple:
            st.session_state.data_ready = False
            st.session_state.last_params = params_tuple
        if load_clicked:
            st.session_state.data_ready = True
        st.markdown('</div>', unsafe_allow_html=True)


    # Header
    account_badge = selected_label
    date_summary = f"{start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}"
    st.markdown(
        f"""
        <div class="dashboard-hero">
            <div class="hero-main">
                <div class="hero-kicker" style="display: flex; align-items: center; gap: 0.5rem;">
                    <img src="https://i.imgur.com/m25fSv6.png" 
                        alt="Company Logo" 
                        style="height: 28px; border-radius: 6px; margin-right: 6px;">
                    <span>DA Smarketing Solutions</span>
                </div>
                <h1>Facebook Ads Manager Dashboard</h1>
                <p>Track spend, engagement, and conversion velocity with a polished executive overview.</p>
            </div>
            <div class="hero-meta">
                <div class="hero-badge">
                    <span class="badge-label">Active Account</span>
                    <span class="badge-value">{account_badge}</span>
                </div>
                <div class="hero-badge">
                    <span class="badge-label">Reporting Window</span>
                    <span class="badge-value">{date_summary}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    if not st.session_state.data_ready:
        st.info(L_current('select_profile_prompt'))
        
        with st.expander(L_current('getting_started'), expanded=True):
            st.markdown(f"""
            ### {L_current('getting_started_title')}
            
            1. **{L_current('getting_started_step1')}**: {L_current('getting_started_step1_details')}
            2. **{L_current('getting_started_step2')}**: {L_current('getting_started_step2_details')}
            3. **{L_current('getting_started_step3')}**: {L_current('getting_started_step3_details')}
            4. **{L_current('getting_started_step4')}**: {L_current('getting_started_step4_details')}
            
            ### {L_current('getting_started_creds_title')}
            - **{L_current('getting_started_creds_app')}**     [{L_current('WhatsApp')}](https://wa.link/k0zyxq)
            """)
        st.stop()
    
    
    # Initialize API using the logged-in user's token
    try:
        # Check if using saved profile or OAuth
        if st.session_state.get("is_profile_mode", False):
            # Use profile credentials
            profile = st.session_state.selected_profile
            FacebookAdsApi.init(
                profile["app_id"],
                profile["app_secret"],
                profile["access_token"]
            )
            api_params = (profile["app_id"], profile["app_secret"], profile["access_token"])
        else:
            # Use OAuth credentials
            FacebookAdsApi.init(
                FB_APP_ID,
                FB_APP_SECRET,
                st.session_state.fb_user_token
            )
            api_params = (FB_APP_ID, FB_APP_SECRET, st.session_state.fb_user_token)
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Facebook Ads API: {str(e)}")
        st.stop()

    
    # Fetch Data
    with st.spinner(L_current('fetching_data')):
        try:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            campaigns_df = fetch_campaign_data(api_params, selected_ad_account_id, start_str, end_str)
            ads_df = fetch_ad_data(api_params, selected_ad_account_id, start_str, end_str)
            # --- Build preview map keyed by ad_id ---
            preview_map = {}
            if 'ad_id' in ads_df.columns:
                unique_ids = [aid for aid in ads_df['ad_id'].dropna().unique().tolist() if str(aid)]
                # Be gentle with API: dedup + simple loop (cached by @st.cache_data)
                for aid in unique_ids:
                    url = _best_preview_url_for_ad(str(aid))
                    if url:
                        preview_map[str(aid)] = url

            daily_df = fetch_daily_data(api_params, selected_ad_account_id, start_str, end_str)
            # Ensure types and required columns for plotting
            if 'date_start' in daily_df.columns:
                daily_df['date_start'] = pd.to_datetime(daily_df['date_start'], errors='coerce')
            if 'messaging_conversation_starts' not in daily_df.columns:
                daily_df['messaging_conversation_starts'] = 0
            
            # Breakdowns
            gender_df = fetch_breakdown_data(api_params, selected_ad_account_id, start_str, end_str, "gender")
            age_df = fetch_breakdown_data(api_params, selected_ad_account_id, start_str, end_str, "age")
            device_df = fetch_breakdown_data(api_params, selected_ad_account_id, start_str, end_str, "impression_device")
            platform_df = fetch_breakdown_data(api_params, selected_ad_account_id, start_str, end_str, "publisher_platform")
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.stop()
    
    st.success(L_current('data_loaded_success'))

    # ========== Key Metrics ==========
    st.header(L_current('key_metrics'))

    # Calculate totals
    total_spend = campaigns_df['spend'].sum() if 'spend' in campaigns_df.columns else 0
    total_clicks = campaigns_df['clicks'].sum() if 'clicks' in campaigns_df.columns else 0
    total_impressions = campaigns_df['impressions'].sum() if 'impressions' in campaigns_df.columns else 0
    total_reach = campaigns_df['reach'].sum() if 'reach' in campaigns_df.columns else 0
    total_landing_page_views = campaigns_df['landing_page_view'].sum() if 'landing_page_view' in campaigns_df.columns else 0
    avg_ctr = campaigns_df['ctr'].mean() if 'ctr' in campaigns_df.columns else 0
    avg_cpc = campaigns_df['cpc'].mean() if 'cpc' in campaigns_df.columns else 0
    
    total_messaging_conversation_starts = campaigns_df['messaging_conversation_starts'].sum() if 'messaging_conversation_starts' in campaigns_df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=L_current('total_spend'),
            value=f"RM {total_spend:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label=L_current('total_clicks'),
            value=f"{total_clicks:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label=L_current('landing_page_views'),
            value=f"{int(total_landing_page_views):,}",
            delta=None
        )
    
    with col4:
        st.metric(
            label=L_current('total_reach'),
            value=f"{total_reach:,}",
            delta=None
        )
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            label=L_current('total_conversations'),
            value=f"{int(total_messaging_conversation_starts):,}",
            delta=None
        )
    
    with col6:
        st.metric(
            label=L_current('total_impressions'),
            value=f"{total_impressions:,}",
            delta=None
        )
    
    with col7:
        st.metric(
            label=L_current('avg_ctr'),
            value=f"{avg_ctr:.2f}%",
            delta=None
        )
    
    with col8:
        st.metric(
            label=L_current('avg_cpc'),
            value=f"RM {avg_cpc:.2f}",
            delta=None
        )

    # ========== AI Account Health Summary (Gemini) ==========
    # Add custom CSS for the AI button
    st.markdown("""
        <style>
        /* Target the AI analysis button specifically */
        div[data-testid="stHorizontalBlock"] > div:first-child div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(59, 130, 246, 0.2)) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1.5px solid rgba(139, 92, 246, 0.4) !important;
            box-shadow: 
                0 8px 32px 0 rgba(139, 92, 246, 0.25),
                inset 0 1px 0 0 rgba(255, 255, 255, 0.1) !important;
            color: inherit !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        div[data-testid="stHorizontalBlock"] > div:first-child div[data-testid="stButton"] > button:hover {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.35), rgba(59, 130, 246, 0.35)) !important;
            border-color: rgba(139, 92, 246, 0.6) !important;
            box-shadow: 
                0 12px 40px 0 rgba(139, 92, 246, 0.4),
                inset 0 1px 0 0 rgba(255, 255, 255, 0.2) !important;
            transform: translateY(-2px) !important;
        }
        
        div[data-testid="stHorizontalBlock"] > div:first-child div[data-testid="stButton"] > button:active {
            transform: translateY(0px) !important;
        }
        
        /* Shimmer effect on hover */
        div[data-testid="stHorizontalBlock"] > div:first-child div[data-testid="stButton"] > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        div[data-testid="stHorizontalBlock"] > div:first-child div[data-testid="stButton"] > button:hover::before {
            left: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    ai_col1, ai_col2 = st.columns([1, 8])
    with ai_col1:
        run_ai_summary = st.button(
            "✨ Analyze with AI",
            help="Calls your Meta health backend + Gemini to summarize performance."
        )

    with ai_col2:
        st.caption("Get a client-friendly explanation and action list for this account and date range.")

    if run_ai_summary:
        with st.spinner("🤖 Running health check and asking Gemini..."):
            # Call backend health check
            health = build_local_health_from_campaigns(campaigns_df)

            # Use the same anomalies logic as in Overview
            anomalies_df = detect_anomalies(daily_df, campaigns_df)

            if health:
                ai_text = explain_health_with_gemini(
                    health_data=health,
                    anomalies_df=anomalies_df,
                    lang=lang,
                )

                st.markdown("---")
                st.markdown("#### 📊 AI-Generated Summary for Client")
                
                # Styled container for AI response
                st.markdown(f"""
                    <div style="
                        background: var(--dashboard-surface);
                        border: 1px solid var(--dashboard-border);
                        border-radius: 16px;
                        padding: 24px;
                        margin: 16px 0;
                        backdrop-filter: blur(10px);
                    ">
                        {ai_text}
                    </div>
                """, unsafe_allow_html=True)

                with st.expander("🔍 Raw health data (debug)"):
                    st.json(health)

                if anomalies_df is not None and not anomalies_df.empty:
                    with st.expander("⚠️ Anomalies used in AI analysis"):
                        st.dataframe(anomalies_df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    
    # Helper function for ad scoring
    def score_ads(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite performance scores for ads with sensible defaults."""

        if df.empty:
            return df.copy()

        scores = pd.DataFrame(index=df.index)
        scores['ad_name'] = df.get('ad_name', pd.Series('', index=df.index))
        scores['campaign_name'] = df.get('campaign_name', pd.Series('', index=df.index))

        clicks = safe_series(df, 'clicks')
        impressions = safe_series(df, 'impressions')
        reach = safe_series(df, 'reach')
        landing_page_view = safe_series(df, 'landing_page_view')
        conversations = safe_series(df, 'messaging_conversation_starts')

        scores['clicks'] = clicks
        scores['impressions'] = impressions
        scores['landing_page_view'] = landing_page_view
        scores['messaging_conversation_starts'] = conversations
        scores['reach'] = reach.replace(0, np.nan).fillna(impressions)

        scores['ctr'] = (100 * clicks / impressions).replace([np.inf, -np.inf], 0).fillna(0)
        scores['lpv_rate'] = (100 * landing_page_view / clicks).replace([np.inf, -np.inf], 0).fillna(0)
        scores['conv_rate'] = (100 * conversations / clicks).replace([np.inf, -np.inf], 0).fillna(0)
        scores['reach_rate'] = (100 * scores['reach'] / impressions).replace([np.inf, -np.inf], 0).fillna(0)

        scores['cost_per_click'] = safe_series(df, 'cpc')
        scores['cost_per_lpv'] = safe_series(df, 'cost_per_lpv')

        scores['frequency'] = (impressions / scores['reach']).replace([np.inf, -np.inf], 0).fillna(0)
        frequency_max = scores['frequency'].max()
        if pd.isna(frequency_max) or frequency_max <= 0:
            scores['frequency_score'] = 0
        else:
            scores['frequency_score'] = 1 - (scores['frequency'] / frequency_max)
            scores['frequency_score'] = scores['frequency_score'].fillna(0)

        for column in ['clicks', 'impressions', 'reach', 'ctr', 'lpv_rate', 'conv_rate', 'reach_rate']:
            scores[f'{column}_norm'] = normalize_series(scores[column])

        for column in ['cost_per_click', 'cost_per_lpv']:
            norm = normalize_series(scores[column])
            scores[f'{column}_norm'] = 1 - norm

        engagement_score = (
            scores['ctr_norm'] * 0.4 +
            scores['lpv_rate_norm'] * 0.3 +
            scores['conv_rate_norm'] * 0.3
        )

        efficiency_score = (
            scores['cost_per_click_norm'] * 0.5 +
            scores['cost_per_lpv_norm'] * 0.5
        )

        volume_score = (
            scores['clicks_norm'] * 0.4 +
            scores['reach_norm'] * 0.3 +
            scores['reach_rate_norm'] * 0.3
        )

        scores['composite_score'] = (
            engagement_score * 0.4 +
            efficiency_score * 0.3 +
            volume_score * 0.2 +
            scores['frequency_score'] * 0.1
        )

        scores['composite_score'] = scores['composite_score'] * 100
        return scores
    
    
    
    # ========== Tabs for Different Views ==========
    tab_names = [
        L_current('tab_overview'),
        L_current('tab_campaigns'),
        L_current('tab_ads'),
        L_current('tab_demographics'),
        L_current('tab_devices'),
        L_current('winners_tab')
    ]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)
    
    # ========== Overview Tab ==========
    with tab1:
        st.subheader(L_current('daily_trends'))

        if not daily_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                # Clicks & Impressions over time
                fig_daily = go.Figure()
                fig_daily.add_trace(go.Scatter(
                    x=daily_df['date_start'],
                    y=daily_df['clicks'],
                    mode='lines+markers',
                    name='Clicks',
                    line=dict(color=PALETTE["series"]["clicks"], width=3)
                ))
                fig_daily.add_trace(go.Scatter(
                    x=daily_df['date_start'],
                    y=daily_df['impressions'],
                    mode='lines+markers',
                    name='Impressions',
                    yaxis='y2',
                    line=dict(color=PALETTE["series"]["impressions"], width=3)
                ))
                fig_daily.update_layout(
                    title='Clicks & Impressions Over Time',
                    xaxis_title='Date',
                    yaxis_title='Clicks',
                    yaxis2=dict(title='Impressions', overlaying='y', side='right'),
                )
                style_fig(fig_daily, height=400)
                st.plotly_chart(fig_daily, use_container_width=True)
            
            with col2:
                # Metric over time
                if metric_key == "landing_page_views":
                    if 'landing_page_view' in daily_df.columns:
                        fig_conv = px.bar(
                            daily_df, x='date_start', y='landing_page_view',
                            title=L_current('lpv_by_day'),
                            labels={'date_start': L_current('date_label'), 'landing_page_view': L_current('lpv_label')}
                        )
                        fig_conv.update_traces(marker_color=PALETTE["series"]["lpv"])
                        style_fig(fig_conv, height=400)
                        st.plotly_chart(fig_conv, use_container_width=True)
                elif metric_key  == "total_conversations":
                    if 'messaging_conversation_starts' in daily_df.columns:
                        fig_msg = px.bar(
                            daily_df, x='date_start', y='messaging_conversation_starts',
                            title='Messaging Conversations Started by Day',
                            labels={'date_start': 'Date', 'messaging_conversation_starts': 'Conversations'}
                        )
                        fig_msg.update_traces(marker_color=PALETTE["series"]["conversations"])
                        style_fig(fig_msg, height=400)
                        st.plotly_chart(fig_msg, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Platform Distribution
        if not platform_df.empty and 'publisher_platform' in platform_df.columns:
            st.subheader(L_current('performance_by_platform'))

            col1, col2 = st.columns(2)

            with col1:
                fig_platform = px.pie(
                    platform_df,
                    values='clicks',
                    names='publisher_platform',
                    title='Clicks by Platform',
                    labels={'publisher_platform': 'Platform', 'clicks': 'Clicks'},
                    hole=0.7,
                    color='publisher_platform',
                    color_discrete_map=platform_colors
                )
                fig_platform.update_layout(height=400)
                st.plotly_chart(fig_platform, use_container_width=True)
            
            with col2:
                fig_platform_imp = px.pie(
                    platform_df,
                    values='impressions',
                    names='publisher_platform',
                    title='Impressions by Platform',
                    labels={'publisher_platform': 'Platform', 'impressions': 'Impressions'},
                    hole=0.7,
                    color='publisher_platform',
                    color_discrete_map=platform_colors
                )
                fig_platform_imp.update_layout(height=400)
                st.plotly_chart(fig_platform_imp, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)
        
            # ========== Alerts and anomalies ==========
            st.subheader("Alerts and anomalies")

            anomalies_df = detect_anomalies(daily_df, campaigns_df)

            if anomalies_df.empty:
                st.success("No major anomalies found for this date range.")
            else:
                # Simple color hint through level text
                level_icons = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢",
                }
                anomalies_df = anomalies_df.copy()
                anomalies_df["level"] = anomalies_df["level"].apply(
                    lambda x: f"{level_icons.get(str(x).lower(), '⚪')} {x.capitalize()}"
                )

                st.dataframe(
                    anomalies_df,
                    use_container_width=True,
                    height=min(320, 60 + 30 * len(anomalies_df)),
                )

                with st.expander("How to read these alerts"):
                    st.markdown(
                        """
                        These checks are simple guardrails:
                        
                        - CTR alerts compare the last day against recent average.
                        - Spend alerts look for sudden jumps in daily spend.
                        - Conversion alerts flag days with zero landing page views or messages after a period with activity.
                        - Frequency alerts highlight campaigns that show the same ads too often.
                        - CPM alerts flag campaigns that are much more expensive than the account median.
                        """
                    )

        
        
    # ========== Campaigns Tab ==========
    with tab2:
        st.subheader(L_current('campaign_details'))

        if not campaigns_df.empty:
            # Display metrics
            display_df = campaigns_df[[
                'campaign_name', 'spend', 'reach', 'impressions', 'clicks',
                'landing_page_view', 'messaging_conversation_starts', 'cost_per_lpv', 'ctr', 'cpc'
            ]].copy()

            display_df.columns = [
                'Campaign', 'Spend (RM)', 'Reach', 'Impressions', 'Clicks',
                'Landing Page Views', 'Conversations', 'Cost/LPV (RM)', 'CTR (%)', 'CPC (RM)'
            ]

            if 'Spend (RM)' in display_df.columns:
                display_df['Spend (RM)'] = format_currency(display_df['Spend (RM)'])
            if 'Cost/LPV (RM)' in display_df.columns:
                display_df['Cost/LPV (RM)'] = format_currency(display_df['Cost/LPV (RM)'])
            if 'CPC (RM)' in display_df.columns:
                display_df['CPC (RM)'] = format_currency(display_df['CPC (RM)'])
            if 'CTR (%)' in display_df.columns:
                display_df['CTR (%)'] = format_percentage(display_df['CTR (%)'])

            for col in ['Reach', 'Impressions', 'Clicks', 'Landing Page Views']:
                if col in display_df.columns:
                    display_df[col] = format_int(display_df[col])

            st.dataframe(display_df, use_container_width=True, height=400)

            # Visualizations
            st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                fig_spend = px.bar(
                    campaigns_df, x='campaign_name', y='spend',
                    title='Spend by Campaign',
                    labels={'campaign_name': 'Campaign', 'spend': 'Spend (RM)'}
                )
                fig_spend.update_traces(marker_color=PALETTE["series"]["spend"])
                style_fig(fig_spend, height=600)
                st.plotly_chart(fig_spend, use_container_width=True)
            
            with col2:
                if metric_key == "landing_page_views":
                    if 'landing_page_view' in campaigns_df.columns:
                        fig_conv = px.bar(
                            campaigns_df, x='campaign_name', y='landing_page_view',
                            title=L_current('lpv_by_campaign'),
                            labels={'campaign_name': 'Campaign', 'landing_page_view': 'Landing Page Views'}
                        )
                        fig_conv.update_traces(marker_color=PALETTE["series"]["lpv"])
                        style_fig(fig_conv, height=600)
                        st.plotly_chart(fig_conv, use_container_width=True)
                elif metric_key == "total_conversations":
                    if 'messaging_conversation_starts' in campaigns_df.columns:
                        fig_conv = px.bar(
                            campaigns_df, x='campaign_name', y='messaging_conversation_starts',
                            title='Conversations by Campaign',
                            labels={'campaign_name': 'Campaign', 'messaging_conversation_starts': 'Conversations'}
                        )
                        fig_conv.update_traces(marker_color=PALETTE["series"]["conversations"])
                        style_fig(fig_conv, height=600)
                        st.plotly_chart(fig_conv, use_container_width=True)

        else:
            st.warning("No campaign data available for the selected date range.")

        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== Ads Tab ==========
    with tab3:
        st.subheader(L_current('ads_details'))

        if not ads_df.empty:
            # Display metrics
            # --- Ads Tab table (robust, no length-mismatch) ---
            base_cols = [
                'ad_id', 'ad_name', 'adset_name', 'campaign_name', 'impressions',
                'clicks', 'landing_page_view', 'messaging_conversation_starts', 'ctr', 'cpc'
            ]
            available_cols = [c for c in base_cols if c in ads_df.columns]
            display_df = ads_df[available_cols].copy()

            # Map preview URL (built earlier as preview_map)
            if 'ad_id' in display_df.columns:
                display_df['Preview'] = display_df['ad_id'].astype(str).map(preview_map).fillna("")

            # Reorder so Preview is first; only keep those that exist
            desired_order = [
                'Preview', 'ad_name', 'adset_name', 'campaign_name',
                'impressions', 'clicks', 'landing_page_view',
                'messaging_conversation_starts', 'ctr', 'cpc'
            ]
            display_df = display_df[[c for c in desired_order if c in display_df.columns]]

            # Rename with a dict (no length mismatch)
            rename_map = {
                'ad_name': 'Ad Name',
                'adset_name': 'Ad Set',
                'campaign_name': 'Campaign',
                'impressions': 'Impressions',
                'clicks': 'Clicks',
                'landing_page_view': 'Landing Page Views',
                'messaging_conversation_starts': 'Conversations',
                'ctr': 'CTR (%)',
                'cpc': 'CPC (RM)',
            }
            display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})

            for col in ['Impressions', 'Clicks', 'Landing Page Views']:
                if col in display_df.columns:
                    display_df[col] = format_int(display_df[col])
            if 'CTR (%)' in display_df.columns:
                display_df['CTR (%)'] = format_percentage(display_df['CTR (%)'])
            if 'CPC (RM)' in display_df.columns:
                display_df['CPC (RM)'] = format_currency(display_df['CPC (RM)'])

            # Show with image column
            st.dataframe(
                display_df,
                use_container_width=True,
                height=460,
                column_config={
                    "Preview": st.column_config.ImageColumn("Preview", help="Ad preview image", width="small")
                }
            )



            # Top performing ads
            st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(L_current('top_10_clicks'))
                top_clicks = ads_df.nlargest(10, 'clicks')[['ad_name', 'clicks']]
                fig_top_clicks = px.bar(
                    top_clicks, x='clicks', y='ad_name', orientation='h',
                    labels={'ad_name': 'Ad Name', 'clicks': 'Clicks'}
                )
                fig_top_clicks.update_traces(marker_color=PALETTE["series"]["clicks"])
                style_fig(fig_top_clicks, height=500, showlegend=False)
                st.plotly_chart(fig_top_clicks, use_container_width=True)
            
            with col2:
                if metric_key == "landing_page_views":
                    if 'landing_page_view' in ads_df.columns:
                        st.subheader(L_current('top_10_lpv'))
                        top_conv = ads_df.nlargest(10, 'landing_page_view')[['ad_name', 'landing_page_view']]
                        fig_top_conv = px.bar(
                            top_conv, x='landing_page_view', y='ad_name', orientation='h',
                            labels={'ad_name': 'Ad Name', 'landing_page_view': 'Landing Page Views'}
                        )
                        fig_top_conv.update_traces(marker_color=PALETTE["series"]["lpv"])
                        style_fig(fig_top_conv, height=500, showlegend=False)
                        st.plotly_chart(fig_top_conv, use_container_width=True)
                elif metric_key == "total_conversations":
                    if 'messaging_conversation_starts' in ads_df.columns:
                        st.subheader(L_current('top_10_conv'))
                        top_conv = ads_df.nlargest(10, 'messaging_conversation_starts')[['ad_name', 'messaging_conversation_starts']]
                        fig_top_conv = px.bar(
                            top_conv, x='messaging_conversation_starts', y='ad_name', orientation='h',
                            labels={'ad_name': 'Ad Name', 'messaging_conversation_starts': 'Conversations'}
                        )
                        fig_top_conv.update_traces(marker_color=PALETTE["series"]["conversations"])
                        style_fig(fig_top_conv, height=500, showlegend=False)
                        st.plotly_chart(fig_top_conv, use_container_width=True)
        else:
            st.warning(L_current('no_ads'))

        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== Demographics Tab ==========
    with tab4:
        st.subheader(L_current('audience_demographics'))

        col1, col2 = st.columns(2)

        with col1:
            if not age_df.empty and 'age' in age_df.columns:
                st.subheader(L_current('age_distribution'))
                fig_age = px.bar(
                    age_df, x='age', y='clicks',
                    title='Clicks by Age Group',
                    labels={'age': 'Age Group', 'clicks': 'Clicks'}
                )
                fig_age.update_traces(marker_color=PALETTE["series"]["clicks"])
                style_fig(fig_age, height=400)
                st.plotly_chart(fig_age, use_container_width=True)
                
                # Data table
                age_display = age_df[['age', 'clicks', 'impressions']].copy()
                age_display.columns = ['Age Group', 'Clicks', 'Impressions']
                for col in ['Clicks', 'Impressions']:
                    age_display[col] = format_int(age_display[col])
                st.dataframe(age_display, use_container_width=True)

        with col2:
            if not gender_df.empty and 'gender' in gender_df.columns:
                st.subheader(L_current('gender_distribution'))
                fig_gender = px.pie(
                    gender_df, values='clicks', names='gender',
                    title='Clicks by Gender'
                )
                # Use default categorical sequence already set, or force first two:
                # fig_gender.update_traces(marker=dict(colors=PALETTE["categorical"][:len(gender_df["gender"].unique())]))
                style_fig(fig_gender, height=400)
                st.plotly_chart(fig_gender, use_container_width=True)
                
                # Data table
                gender_display = gender_df[['gender', 'clicks', 'impressions']].copy()
                gender_display.columns = ['Gender', 'Clicks', 'Impressions']
                for col in ['Clicks', 'Impressions']:
                    gender_display[col] = format_int(gender_display[col])
                st.dataframe(gender_display, use_container_width=True)

        if (
            age_df.empty or 'age' not in age_df.columns
        ) and (
            gender_df.empty or 'gender' not in gender_df.columns
        ):
            st.info("No demographic breakdown available for the selected period.")

        st.markdown('</div>', unsafe_allow_html=True)
        
    # ========== Devices & Platforms Tab ==========
    with tab5:
        st.subheader(L_current('device_platform_analytics'))

        col1, col2 = st.columns(2)
        
        with col1:
            if not device_df.empty and 'impression_device' in device_df.columns:
                st.subheader(L_current('device_breakdown'))
                fig_device = px.pie(
                    device_df, values='clicks', names='impression_device',
                    title='Clicks by Device'
                )
                style_fig(fig_device, height=400)
                st.plotly_chart(fig_device, use_container_width=True)
                
                # Data table
                device_display = device_df[['impression_device', 'clicks', 'impressions']].copy()
                device_display.columns = ['Device', 'Clicks', 'Impressions']
                for col in ['Clicks', 'Impressions']:
                    device_display[col] = format_int(device_display[col])
                st.dataframe(device_display, use_container_width=True)
        
        with col2:
            if not platform_df.empty and 'publisher_platform' in platform_df.columns:
                st.subheader(L_current('platform_performance'))
                
                # Calculate CTR for platforms
                # CTR with divide-by-zero guard
                platform_display = platform_df.copy()
                if 'impressions' in platform_display.columns and 'clicks' in platform_display.columns:
                    platform_display['ctr'] = (
                        (platform_display['clicks'] / platform_display['impressions'])
                        .replace([float('inf')], 0)
                        .fillna(0) * 100
                    ).round(2)

                # Long-form for stable colors by label
                _long = platform_display.melt(
                    id_vars=['publisher_platform'],
                    value_vars=['clicks', 'impressions'],
                    var_name='Metric',
                    value_name='Value'
                )
                # Pretty legend labels
                label_map = {'clicks': 'Clicks', 'impressions': 'Impressions'}
                _long['Metric'] = _long['Metric'].map(label_map)

                fig_platform_bar = px.bar(
                    _long,
                    x='publisher_platform',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    title='Clicks vs Impressions by Platform',
                    labels={'publisher_platform': 'Platform', 'Value': 'Count', 'Metric': 'Metric'},
                    color_discrete_map={
                        'Clicks': SERIES_COLOR_MAP['Clicks'],
                        'Impressions': SERIES_COLOR_MAP['Impressions'],
                    },
                    category_orders={'Metric': ['Impressions', 'Clicks']}  # optional: fix legend/order
                )
                style_fig(fig_platform_bar, height=400)
                st.plotly_chart(fig_platform_bar, use_container_width=True)

                
                # Data table
                display_cols = ['publisher_platform', 'clicks', 'impressions']
                if 'ctr' in platform_display.columns:
                    display_cols.append('ctr')
                
                platform_table = platform_display[display_cols].copy()
                platform_table.columns = ['Platform', 'Clicks', 'Impressions', 'CTR (%)'][:len(display_cols)]
                
                for col in ['Clicks', 'Impressions']:
                    if col in platform_table.columns:
                        platform_table[col] = format_int(platform_table[col])
                if 'CTR (%)' in platform_table.columns:
                    platform_table['CTR (%)'] = format_percentage(platform_table['CTR (%)'])

                st.dataframe(platform_table, use_container_width=True)

        if (
            device_df.empty or 'impression_device' not in device_df.columns
        ) and (
            platform_df.empty or 'publisher_platform' not in platform_df.columns
        ):
            st.info("No device or platform analytics available for the selected period.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== Winners vs Underperformers Tab ==========
    with tab6:
        if not ads_df.empty:
            st.subheader(L_current('winners_tab'))

            # Calculate scores
            scores_df = score_ads(ads_df)

            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_impressions = st.number_input(
                    L_current('min_impressions'),
                    min_value=0,
                    value=100,
                    step=100,
                    help="Filter out ads with fewer impressions"
                )
            
            with col2:
                min_clicks = st.number_input(
                    L_current('min_clicks'),
                    min_value=0,
                    value=10,
                    step=10,
                    help="Filter out ads with fewer clicks"
                )
            
            with col3:
                campaign_filter = st.multiselect(
                    L_current('filter_campaign'),
                    options=sorted(scores_df['campaign_name'].unique()),
                    help="Select specific campaigns to analyze"
                )

            # Apply filters
            mask = (
                (ads_df['impressions'] >= min_impressions) &
                (ads_df['clicks'] >= min_clicks)
            )
            if campaign_filter:
                mask &= scores_df['campaign_name'].isin(campaign_filter)

            filtered_scores = scores_df[mask].copy()

            if len(filtered_scores) > 0:
                # Split into winners and underperformers
                median_score = filtered_scores['composite_score'].median()
                winners = filtered_scores[filtered_scores['composite_score'] >= median_score].sort_values('composite_score', ascending=False)
                underperformers = filtered_scores[filtered_scores['composite_score'] < median_score].sort_values('composite_score')

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(L_current('top_performers'))
                    metric_cols = [
                        'ad_name', 'campaign_name', 'composite_score',
                        'ctr', 'lpv_rate', 'conv_rate',
                        'cost_per_click', 'cost_per_lpv',
                        'reach_rate', 'frequency'
                    ]
                    winners_display = winners[metric_cols].copy()
                    winners_display.columns = [
                        'Ad Name', 'Campaign', 'Score',
                        'CTR %', 'LPV Rate %', 'Conv Rate %',
                        'CPC (RM)', 'Cost/LPV (RM)',
                        'Reach Rate %', 'Frequency'
                    ]
                    percentage_cols = ['Score', 'CTR %', 'LPV Rate %', 'Conv Rate %', 'Reach Rate %']
                    cost_cols = ['CPC (RM)', 'Cost/LPV (RM)']
                    winners_display['Score'] = format_percentage(winners_display['Score'], suffix="")
                    for col in percentage_cols[1:]:
                        winners_display[col] = format_percentage(winners_display[col])
                    for col in cost_cols:
                        winners_display[col] = format_currency(winners_display[col], prefix="RM ")
                    winners_display['Frequency'] = format_frequency(winners_display['Frequency'])
                    st.dataframe(winners_display, use_container_width=True, height=300)

                with col2:
                    st.subheader(L_current('needs_improve'))
                    underp_display = underperformers[metric_cols].copy()
                    underp_display.columns = winners_display.columns  # Use same column names
                    underp_display['Score'] = format_percentage(underp_display['Score'], suffix="")
                    for col in percentage_cols[1:]:
                        underp_display[col] = format_percentage(underp_display[col])
                    for col in cost_cols:
                        underp_display[col] = format_currency(underp_display[col], prefix="RM ")
                    underp_display['Frequency'] = format_frequency(underp_display['Frequency'])
                    st.dataframe(underp_display, use_container_width=True, height=300)

                # Performance Distribution Chart
                st.markdown('<div class="card-divider"></div>', unsafe_allow_html=True)
                st.subheader(L_current('winners_tab') + ' - Performance Score Distribution')
                
                fig_scores = px.bar(
                    filtered_scores.sort_values('composite_score', ascending=False),
                    x='ad_name',
                    y='composite_score',
                    color='composite_score',
                    labels={'ad_name': 'Ad', 'composite_score': 'Performance Score'},
                    title='Ads Ranked by Performance Score',
                    color_continuous_scale='RdYlGn'  # Red to Yellow to Green scale
                )
                fig_scores.add_hline(
                    y=median_score,
                    line_dash="dash",
                    line_color="white",
                    annotation_text="Median Score"
                )
                style_fig(fig_scores, height=500)
                st.plotly_chart(fig_scores, use_container_width=True)

                # Explanation of scoring
                with st.expander(L_current('winners_tab') + ' - How Score is Calculated'):
                    st.markdown(
                        """
                        The performance score (0-100) is calculated using four key components:

                        1. **Engagement Quality** (40% of total score)
                           - Click-Through Rate (40%)
                           - Landing Page View Rate (30%)
                           - Conversation Rate (30%)

                        2. **Cost Efficiency** (30% of total score)
                           - Cost per Click optimization (50%)
                           - Cost per Landing Page View optimization (50%)

                        3. **Volume & Reach** (20% of total score)
                           - Click Volume (40%)
                           - Audience Reach (30%)
                           - Reach Rate (30%)

                        4. **Frequency Optimization** (10% of total score)
                           - Optimal impression frequency score
                           - Penalizes excessive ad frequency

                        Each component is normalized against the best performing ad in the filtered set.
                        The final score balances engagement, efficiency, scale, and frequency optimization.
                        Higher scores indicate better overall performance across these dimensions.
                        """
                    )
            else:
                st.warning(L_current('no_match'))
        else:
            st.warning(L_current('no_ads'))

        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== Export Options ==========
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader(L_current('export_data'))

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not campaigns_df.empty:
            csv_campaigns = campaigns_df.to_csv(index=False)
            st.download_button(
                label=L_current('download_campaign'),
                data=csv_campaigns,
                file_name=f"campaigns_{start_str}_{end_str}.csv",
                mime="text/csv"
            )
    
    with col2:
        if not ads_df.empty:
            csv_ads = ads_df.to_csv(index=False)
            st.download_button(
                label=L_current('download_ads'),
                data=csv_ads,
                file_name=f"ads_{start_str}_{end_str}.csv",
                mime="text/csv"
            )
    
    with col3:
        if not daily_df.empty:
            csv_daily = daily_df.to_csv(index=False)
            st.download_button(
                label=L_current('download_daily'),
                data=csv_daily,
                file_name=f"daily_{start_str}_{end_str}.csv",
                mime="text/csv"
            )

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
