import json
import logging
import os
import smtplib
import sys
import time
import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import google.generativeai as genai


FEEDS: List[Tuple[str, str]] = [
    ("N12", "https://www.mako.co.il/rss-news-israel.xml"),
    ("Ynet", "https://www.ynet.co.il/Integration/StoryRss2.xml"),
    ("Walla", "https://rss.walla.co.il/feed/1?type=main"),
]

LOOKBACK_HOURS = 24


@dataclass
class Config:
    gemini_api_key: str
    email_sender: str
    email_password: str
    email_receiver: str
    smtp_host: str
    smtp_port: int
    gemini_model: str
    max_ai_retries: int
    ai_retry_base_seconds: float


@dataclass
class Article:
    source: str
    title: str
    link: str
    published: str
    summary: str


@dataclass
class AnalysisResult:
    relevant: bool
    category: str
    confidence: float
    reason: str


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def is_truthy_env(name: str) -> bool:
    value = (os.getenv(name) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def load_combined_secrets() -> Dict[str, str]:
    raw = (os.getenv("GEMINI_AND_EMAIL_SECRETS") or "").strip()
    if not raw:
        return {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("GEMINI_AND_EMAIL_SECRETS must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("GEMINI_AND_EMAIL_SECRETS must be a JSON object")

    normalized: Dict[str, str] = {}
    for key, value in payload.items():
        if value is None:
            continue
        normalized[str(key).strip().upper()] = str(value).strip()

    return normalized


def load_config() -> Config:
    combined = load_combined_secrets()

    required = {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY") or combined.get("GEMINI_API_KEY"),
        "EMAIL_SENDER": os.getenv("EMAIL_SENDER") or combined.get("EMAIL_SENDER"),
        "EMAIL_PASSWORD": os.getenv("EMAIL_PASSWORD") or combined.get("EMAIL_PASSWORD"),
        "EMAIL_RECEIVER": os.getenv("EMAIL_RECEIVER") or combined.get("EMAIL_RECEIVER"),
    }

    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    smtp_host = os.getenv("SMTP_HOST") or combined.get("SMTP_HOST") or "smtp.gmail.com"
    smtp_port_raw = os.getenv("SMTP_PORT") or combined.get("SMTP_PORT") or "587"
    gemini_model = os.getenv("GEMINI_MODEL") or combined.get("GEMINI_MODEL") or "gemini-1.5-flash"
    max_ai_retries_raw = os.getenv("MAX_AI_RETRIES") or combined.get("MAX_AI_RETRIES") or "3"
    ai_retry_base_seconds_raw = (
        os.getenv("AI_RETRY_BASE_SECONDS") or combined.get("AI_RETRY_BASE_SECONDS") or "2.0"
    )

    try:
        smtp_port = int(smtp_port_raw)
        max_ai_retries = int(max_ai_retries_raw)
        ai_retry_base_seconds = float(ai_retry_base_seconds_raw)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric env var value: {exc}") from exc

    return Config(
        gemini_api_key=required["GEMINI_API_KEY"] or "",
        email_sender=required["EMAIL_SENDER"] or "",
        email_password=required["EMAIL_PASSWORD"] or "",
        email_receiver=required["EMAIL_RECEIVER"] or "",
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        gemini_model=gemini_model,
        max_ai_retries=max_ai_retries,
        ai_retry_base_seconds=ai_retry_base_seconds,
    )


def safe_get_entry_field(entry: Dict[str, Any], key: str, default: str = "") -> str:
    value = entry.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def get_entry_datetime_utc(entry: Dict[str, Any]) -> Optional[datetime]:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed_time = entry.get(key)
        if not parsed_time:
            continue

        try:
            timestamp = calendar.timegm(parsed_time)
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except Exception:
            continue

    return None


def fetch_feed_articles(source: str, feed_url: str) -> List[Article]:
    articles: List[Article] = []
    now_utc = datetime.now(timezone.utc)
    oldest_allowed = now_utc - timedelta(hours=LOOKBACK_HOURS)

    try:
        parsed = feedparser.parse(feed_url)
    except Exception as exc:
        logging.exception("Failed to parse feed for %s (%s): %s", source, feed_url, exc)
        return articles

    if getattr(parsed, "bozo", False):
        bozo_exc = getattr(parsed, "bozo_exception", None)
        logging.warning("Feed parsing issue for %s (%s): %s", source, feed_url, bozo_exc)

    status = getattr(parsed, "status", None)
    if status and status >= 400:
        logging.error("HTTP error while reading %s feed (%s): status=%s", source, feed_url, status)
        return articles

    entries = getattr(parsed, "entries", [])
    if not entries:
        logging.warning("No entries found in %s feed (%s)", source, feed_url)
        return articles

    for entry in entries:
        try:
            title = safe_get_entry_field(entry, "title", "ללא כותרת")
            link = safe_get_entry_field(entry, "link", "")
            published = safe_get_entry_field(entry, "published", "")
            summary = safe_get_entry_field(entry, "summary", safe_get_entry_field(entry, "description", ""))
            published_dt = get_entry_datetime_utc(entry)

            if not link:
                logging.debug("Skipping entry with missing link from %s: title=%s", source, title)
                continue

            if not published_dt:
                logging.debug("Skipping %s entry without parseable timestamp: %s", source, title)
                continue

            if published_dt < oldest_allowed:
                continue

            if not published:
                published = published_dt.isoformat()

            articles.append(
                Article(
                    source=source,
                    title=title,
                    link=link,
                    published=published,
                    summary=summary,
                )
            )
        except Exception as exc:
            logging.exception("Failed to process one entry from %s feed: %s", source, exc)

    return articles


def deduplicate_articles(articles: List[Article]) -> List[Article]:
    unique: List[Article] = []
    seen = set()

    for article in articles:
        key = (article.link.strip().lower(), article.title.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)

    return unique


def collect_articles() -> List[Article]:
    all_articles: List[Article] = []
    for source, url in FEEDS:
        articles = fetch_feed_articles(source, url)
        logging.info("Fetched %d items from %s", len(articles), source)
        all_articles.extend(articles)

    unique_articles = deduplicate_articles(all_articles)
    logging.info("Total unique articles: %d", len(unique_articles))
    return unique_articles


def build_ai_prompt(article: Article) -> str:
    return f"""
אתה אנליסט תוכן חדשותי עבור ישראל.
קבל כותרת ותקציר כתבה, והחלט אם הכתבה מתארת אחד או יותר מהבאים:
1) חוסר משילות של המדינה
2) פשיעה חמורה
3) היעדר אכיפת חוק אפקטיבית בישראל

כללים:
- החזר relevant=true רק אם יש אינדיקציה ברורה לנושא.
- אם לא ברור, החזר relevant=false.
- השב אך ורק ב-JSON תקין בפורמט:
  {{
    "relevant": true/false,
    "category": "lack_of_governance|crime|law_enforcement_failure|other",
    "confidence": 0.0-1.0,
    "reason": "הסבר קצר בעברית"
  }}

מקור: {article.source}
כותרת: {article.title}
תקציר: {article.summary}
קישור: {article.link}
""".strip()


def _extract_json_blob(raw_text: str) -> Optional[str]:
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    return text[start : end + 1]


def _coerce_analysis_result(payload: Dict[str, Any]) -> AnalysisResult:
    relevant = bool(payload.get("relevant", False))
    category = str(payload.get("category", "other")).strip() or "other"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    reason = str(payload.get("reason", "")).strip() or "לא סופק נימוק."

    return AnalysisResult(
        relevant=relevant,
        category=category,
        confidence=confidence,
        reason=reason,
    )


def analyze_article(
    model: genai.GenerativeModel,
    article: Article,
    max_retries: int,
    retry_base_seconds: float,
) -> AnalysisResult:
    prompt = build_ai_prompt(article)

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                },
            )

            text = (response.text or "").strip()
            json_blob = _extract_json_blob(text)
            if not json_blob:
                raise ValueError("Model response did not contain valid JSON object")

            payload = json.loads(json_blob)
            return _coerce_analysis_result(payload)

        except Exception as exc:
            exc_text = str(exc)
            if "API_KEY_INVALID" in exc_text or "API key not valid" in exc_text:
                raise RuntimeError("Gemini API key is invalid. Update GEMINI_API_KEY.") from exc
            logging.warning(
                "AI analysis failed for '%s' (attempt %d/%d): %s",
                article.title,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                sleep_seconds = retry_base_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_seconds)

    return AnalysisResult(
        relevant=False,
        category="other",
        confidence=0.0,
        reason="כשל בניתוח AI לאחר מספר ניסיונות.",
    )


def build_email_content(relevant_items: List[Tuple[Article, AnalysisResult]]) -> Tuple[str, str, str]:
    report_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    subject = f"דוח יומי חוסר משילות | {report_date}"

    if not relevant_items:
        plain = (
            f"דוח יומי לתאריך {report_date}\n\n"
            "לא נמצאו היום כתבות רלוונטיות בנושא חוסר משילות/פשיעה/כשל אכיפה.\n"
        )
        html = f"""
<html>
  <body dir="rtl" style="font-family: Arial, sans-serif; line-height:1.7;">
    <h2>דוח יומי לתאריך {escape(report_date)}</h2>
    <p>לא נמצאו היום כתבות רלוונטיות בנושא חוסר משילות/פשיעה/כשל אכיפה.</p>
  </body>
</html>
""".strip()
        return subject, plain, html

    plain_lines = [
        f"דוח יומי לתאריך {report_date}",
        "",
        f"נמצאו {len(relevant_items)} כתבות רלוונטיות:",
        "",
    ]

    html_items = []
    for idx, (article, result) in enumerate(relevant_items, start=1):
        plain_lines.extend(
            [
                f"{idx}. {article.title}",
                f"   מקור: {article.source}",
                f"   קטגוריה: {result.category}",
                f"   ביטחון: {result.confidence:.2f}",
                f"   נימוק: {result.reason}",
                f"   קישור: {article.link}",
                "",
            ]
        )

        html_items.append(
            f"""
<li style="margin-bottom:14px;">
  <a href="{escape(article.link)}" target="_blank" rel="noopener noreferrer" style="font-size:16px;">
    {escape(article.title)}
  </a>
  <div>מקור: {escape(article.source)}</div>
  <div>קטגוריה: {escape(result.category)} | ביטחון: {result.confidence:.2f}</div>
  <div>נימוק: {escape(result.reason)}</div>
</li>
""".strip()
        )

    plain = "\n".join(plain_lines)
    html = f"""
<html>
  <body dir="rtl" style="font-family: Arial, sans-serif; line-height:1.7;">
    <h2>דוח יומי חוסר משילות - {escape(report_date)}</h2>
    <p>נמצאו {len(relevant_items)} כתבות רלוונטיות:</p>
    <ol>
      {''.join(html_items)}
    </ol>
  </body>
</html>
""".strip()

    return subject, plain, html


def send_email(
    smtp_host: str,
    smtp_port: int,
    sender: str,
    password: str,
    receiver: str,
    subject: str,
    plain_body: str,
    html_body: str,
) -> None:
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.set_content(plain_body, subtype="plain", charset="utf-8")
    msg.add_alternative(html_body, subtype="html", charset="utf-8")

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(sender, password)
            smtp.send_message(msg)
        logging.info("Email sent successfully to %s", receiver)
    except smtplib.SMTPException as exc:
        raise RuntimeError(f"SMTP error while sending email: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected email sending error: {exc}") from exc


def main() -> int:
    setup_logging()

    combined = load_combined_secrets()

    if is_truthy_env("DRY_RUN"):
        logging.info("Running in DRY_RUN mode (no email will be sent).")
        all_articles = collect_articles()

        api_key = os.getenv("GEMINI_API_KEY") or combined.get("GEMINI_API_KEY")
        if not api_key:
            logging.info("DRY_RUN completed with RSS ingestion only. Set GEMINI_API_KEY to test AI analysis.")
            return 0

        gemini_model = os.getenv("GEMINI_MODEL") or combined.get("GEMINI_MODEL") or "gemini-1.5-flash"
        max_ai_retries = int(os.getenv("MAX_AI_RETRIES") or combined.get("MAX_AI_RETRIES") or "3")
        ai_retry_base_seconds = float(
            os.getenv("AI_RETRY_BASE_SECONDS") or combined.get("AI_RETRY_BASE_SECONDS") or "2.0"
        )
        dry_run_max_articles = int(os.getenv("DRY_RUN_MAX_ARTICLES") or "10")

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(gemini_model)
        except Exception as exc:
            logging.error("Dry run failed to configure Gemini client: %s", exc)
            return 1

        sample_articles = all_articles[: max(0, dry_run_max_articles)]
        relevant_items: List[Tuple[Article, AnalysisResult]] = []
        try:
            for article in sample_articles:
                result = analyze_article(
                    model=model,
                    article=article,
                    max_retries=max_ai_retries,
                    retry_base_seconds=ai_retry_base_seconds,
                )
                if result.relevant:
                    relevant_items.append((article, result))
        except RuntimeError as exc:
            logging.error("Dry run AI failed: %s", exc)
            return 1

        logging.info(
            "Dry run AI completed: %d relevant articles out of %d analyzed.",
            len(relevant_items),
            len(sample_articles),
        )
        subject, plain_body, _ = build_email_content(relevant_items)
        logging.info("Dry run email subject: %s", subject)
        logging.info("Dry run email preview (first 500 chars): %s", plain_body[:500])
        return 0

    try:
        config = load_config()
    except Exception as exc:
        logging.error("Configuration error: %s", exc)
        return 1

    try:
        genai.configure(api_key=config.gemini_api_key)
        model = genai.GenerativeModel(config.gemini_model)
    except Exception as exc:
        logging.error("Failed to configure Gemini client: %s", exc)
        return 1

    all_articles = collect_articles()

    relevant_items: List[Tuple[Article, AnalysisResult]] = []
    try:
        for article in all_articles:
            result = analyze_article(
                model=model,
                article=article,
                max_retries=config.max_ai_retries,
                retry_base_seconds=config.ai_retry_base_seconds,
            )
            if result.relevant:
                relevant_items.append((article, result))
    except RuntimeError as exc:
        logging.error("AI analysis failed: %s", exc)
        return 1

    logging.info("Relevant articles found: %d", len(relevant_items))

    try:
        subject, plain_body, html_body = build_email_content(relevant_items)
        send_email(
            smtp_host=config.smtp_host,
            smtp_port=config.smtp_port,
            sender=config.email_sender,
            password=config.email_password,
            receiver=config.email_receiver,
            subject=subject,
            plain_body=plain_body,
            html_body=html_body,
        )
    except Exception as exc:
        logging.error("Failed to build/send daily report email: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
