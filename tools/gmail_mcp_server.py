"""Read-only Gmail MCP server.

Run modes:
  python gmail_mcp_server.py auth   # one-time OAuth bootstrap
  python gmail_mcp_server.py        # serve MCP over stdio (invoked by Claude Code)

Reads credentials.json and writes/refreshes token.json under ~/.gmail-mcp-readonly/.
Scope is hardcoded to gmail.readonly — the server cannot send, modify, or delete.
"""
from __future__ import annotations

import base64
import re
import sys
from html import unescape
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mcp.server.fastmcp import FastMCP


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

CREDS_DIR = Path.home() / ".gmail-mcp-readonly"
OAUTH_KEYS = CREDS_DIR / "credentials.json"
TOKEN_FILE = CREDS_DIR / "token.json"

mcp = FastMCP("gmail-readonly")


def _get_service():
    CREDS_DIR.mkdir(exist_ok=True)
    creds: Credentials | None = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not OAUTH_KEYS.exists():
                raise RuntimeError(
                    f"Missing {OAUTH_KEYS}. Place your Google OAuth client JSON there, "
                    f"then run: python gmail_mcp_server.py auth"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(OAUTH_KEYS), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _decode_body(data: str) -> str:
    return base64.urlsafe_b64decode(data.encode("ascii")).decode("utf-8", errors="replace")


def _extract_text_body(payload: dict[str, Any]) -> str:
    mime = payload.get("mimeType", "")
    body = payload.get("body", {})
    data = body.get("data")

    if data and mime.startswith("text/plain"):
        return _decode_body(data)

    for part in payload.get("parts", []) or []:
        text = _extract_text_body(part)
        if text:
            return text

    if data and mime.startswith("text/html"):
        html = _decode_body(data)
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(html))).strip()

    return ""


def _headers_dict(payload: dict[str, Any]) -> dict[str, str]:
    return {h["name"]: h["value"] for h in payload.get("headers", []) or []}


@mcp.tool()
def search_messages(query: str = "", max_results: int = 10) -> list[dict[str, Any]]:
    """Search Gmail using Gmail query syntax.

    Examples: 'from:foo@bar.com', 'subject:"Governance Watchdog" newer_than:7d',
    'is:unread', 'has:attachment'. Returns id, threadId, subject, from, date, snippet.
    """
    service = _get_service()
    resp = service.users().messages().list(
        userId="me", q=query, maxResults=max(1, min(max_results, 100))
    ).execute()
    out: list[dict[str, Any]] = []
    for m in resp.get("messages", []) or []:
        full = service.users().messages().get(
            userId="me",
            id=m["id"],
            format="metadata",
            metadataHeaders=["Subject", "From", "Date", "To"],
        ).execute()
        headers = _headers_dict(full.get("payload", {}))
        out.append({
            "id": full["id"],
            "threadId": full["threadId"],
            "subject": headers.get("Subject", ""),
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "date": headers.get("Date", ""),
            "snippet": full.get("snippet", ""),
        })
    return out


@mcp.tool()
def get_message(message_id: str) -> dict[str, Any]:
    """Fetch a single Gmail message by ID with headers and decoded plain-text body."""
    service = _get_service()
    msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    payload = msg.get("payload", {})
    headers = _headers_dict(payload)
    return {
        "id": msg["id"],
        "threadId": msg["threadId"],
        "subject": headers.get("Subject", ""),
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "cc": headers.get("Cc", ""),
        "date": headers.get("Date", ""),
        "snippet": msg.get("snippet", ""),
        "labels": msg.get("labelIds", []),
        "body": _extract_text_body(payload),
    }


@mcp.tool()
def get_profile() -> dict[str, Any]:
    """Return the authenticated Gmail account address and message/thread totals."""
    service = _get_service()
    return service.users().getProfile(userId="me").execute()


@mcp.tool()
def list_labels() -> list[dict[str, Any]]:
    """List all Gmail labels (system + user) with id, name, and type."""
    service = _get_service()
    resp = service.users().labels().list(userId="me").execute()
    return [
        {"id": label["id"], "name": label["name"], "type": label.get("type", "user")}
        for label in resp.get("labels", []) or []
    ]


def _bootstrap_auth() -> int:
    service = _get_service()
    profile = service.users().getProfile(userId="me").execute()
    print(f"Authenticated as: {profile.get('emailAddress')}")
    print(f"Token saved to: {TOKEN_FILE}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "auth":
        sys.exit(_bootstrap_auth())
    mcp.run()
