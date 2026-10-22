"""Fetch links shared in chat so the model can actually read them.

Without this, a shared URL reaches the model as raw text and the agent has no
way to open it. In practice the model then tries ``astrbot_execute_shell`` (which
is admin-only and gets denied) and, failing that, invents a reason such as "the
API probably needs a login" — which is worse than saying nothing.

This plugin runs as a high-priority message handler, extracts the first few URLs
from the message, fetches them and appends the readable text to
``event.message_str`` before the agent stage builds its prompt.

Security notes: chat participants control these URLs, so the fetcher resolves the
host first and refuses anything that points at a loopback, private, link-local,
reserved or multicast address. Redirects are followed but every hop is checked
the same way, and responses are capped in both time and size.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
import sys
from typing import Any
from urllib.parse import urlparse

import httpx
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Reply
from astrbot.api.star import Context, Star, register
from bs4 import BeautifulSoup

URL_RE = re.compile(r"https?://[^\s\u3000\"'<>()\[\]{}，。；！？、【】（）]+")
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
MAX_URLS = 2
MAX_BYTES = 1_500_000
MAX_CHARS = 2500
TIMEOUT = 8.0
MIN_MEANINGFUL_CHARS = 60

DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "iframe",
    "template",
)
TEXTUAL_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "application/json",
    "text/json",
)

# Hosts owned by astrbot_plugin_multi_parser. That plugin talks to each
# platform's own API, while a plain HTTP fetch here would only return an
# anti-bot page or a JavaScript shell, so defer to it instead of reporting a
# useless "cannot read" note. GitHub is deliberately absent: multi_parser only
# renders an OpenGraph card for it, whereas fetching the page lets the model
# actually read the README.
DELEGATED_HOSTS = (
    "bilibili.com",
    "b23.tv",
    "douyin.com",
    "iesdouyin.com",
    "fanqienovel.com",
    "xiaohongshu.com",
    "xiaohongshu.cn",
    "xhslink.com",
    "tieba.baidu.com",
    "weibo.com",
    "weibo.cn",
    "mp.weixin.qq.com",
    "weixin.qq.com",
    "xiaoheihe.cn",
    "zhihu.com",
    "pixiv.net",
    "pd.qq.com",
    "qzone.qq.com",
)


def _is_delegated(host: str) -> bool:
    """Report whether another plugin already handles this host.

    Args:
        host: Hostname from the URL.

    Returns:
        True when the host belongs to a platform parsed by multi_parser.
    """
    host = host.strip().lower().rstrip(".")
    return any(host == h or host.endswith(f".{h}") for h in DELEGATED_HOSTS)


def _is_blocked_ip(value: str) -> bool:
    """Report whether an IP literal belongs to a range we must not fetch.

    Args:
        value: IP address string.

    Returns:
        True when the address is loopback, private, link-local, reserved,
        multicast or otherwise not a public destination.
    """
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def _host_is_blocked(host: str) -> bool:
    """Resolve a hostname and decide whether fetching it is unsafe.

    Args:
        host: Hostname or IP literal from the URL.

    Returns:
        True when the host is empty, a local name, an IP literal in a blocked
        range, or resolves to any blocked address.
    """
    host = host.strip().lower().rstrip(".")
    if not host or host in {"localhost", "localhost.localdomain"}:
        return True
    if host.endswith((".local", ".internal", ".localhost", ".home.arpa")):
        return True
    if _is_blocked_ip(host):
        return True

    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except OSError:
        return False  # let the fetch itself report the DNS failure

    addresses = [info[4][0] for info in infos]
    if not addresses:
        return False
    # Block only when there is no public address at all. Requiring every record
    # to be public would reject legitimate hosts that also publish a Teredo or
    # link-local IPv6 record (e.g. news.ycombinator.com resolves to 2001::...,
    # which ipaddress reports as private).
    return all(_is_blocked_ip(addr) for addr in addresses)


def _extract_text(raw: bytes) -> tuple[str, str]:
    """Convert a fetched document into a title and readable body text.

    Args:
        raw: Raw response body, still undecoded so BeautifulSoup can honour the
            document's own charset declaration.

    Returns:
        Tuple of (title, body_text). Title is empty when the document has none.
    """
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(list(DROP_TAGS)):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    # Prefer the semantic containers; fall back to the whole body.
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)
    # Some pages embed markup inside script strings, which survives get_text as
    # literal "<style>...</style>". Strip those blocks so CSS never reaches the
    # model (Baidu's homepage is a good example).
    text = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<[^>]{1,200}>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return title, text.strip()


@register(
    "astrbot_plugin_link_reader",
    "Coe",
    "读取群聊里分享的网页链接，把正文喂给模型",
    "1.0.0",
)
class LinkReaderPlugin(Star):
    """Fetch URLs shared in chat and expose their text to the model."""

    def __init__(self, context: Context) -> None:
        """Store the plugin context.

        Args:
            context: AstrBot plugin context.
        """
        super().__init__(context)

    async def _fetch_one(self, url: str) -> tuple[str, str, str]:
        """Fetch a URL and report either its text or why it could not be read.

        Args:
            url: Absolute http(s) URL taken from the message.

        Returns:
            Tuple of (reason, title, body). ``reason`` is empty on success and a
            short human-readable explanation otherwise, so the model can tell the
            user what actually happened instead of inventing a cause.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "不是 http(s) 链接", "", ""
        if await _host_is_blocked(parsed.hostname or ""):
            logger.warning(f"[link-reader] 拒绝访问内网/保留地址: {parsed.hostname}")
            return "该地址指向内网或保留地址，出于安全考虑不允许访问", "", ""
        return await self._download(parsed.geturl())

    async def _download(self, url: str) -> tuple[str, str, str]:
        """Perform the HTTP request with size and time caps.

        Args:
            url: Absolute http(s) URL that already passed the host check.

        Returns:
            Tuple of (reason, title, body); reason is empty only on success.
        """
        headers = {
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        try:
            async with (
                httpx.AsyncClient(
                    timeout=TIMEOUT,
                    follow_redirects=True,
                    max_redirects=4,
                    headers=headers,
                ) as client,
                client.stream("GET", url) as resp,
            ):
                # Redirect targets are chat-controlled too, so re-check the
                # final host instead of trusting the first hop.
                final_host = urlparse(str(resp.url)).hostname or ""
                if await _host_is_blocked(final_host):
                    logger.warning(
                        f"[link-reader] 重定向到内网地址，拒绝: {final_host}"
                    )
                    return "重定向到了内网地址，已拒绝", "", ""
                if resp.status_code >= 400:
                    logger.debug(f"[link-reader] {url} -> HTTP {resp.status_code}")
                    hint = (
                        "（站点有反爬验证或需要登录）"
                        if resp.status_code in (401, 403, 412, 429)
                        else ""
                    )
                    return f"服务器返回 HTTP {resp.status_code}{hint}", "", ""

                ctype = resp.headers.get("content-type", "").lower()
                if ctype and not any(t in ctype for t in TEXTUAL_TYPES):
                    logger.debug(f"[link-reader] {url} 跳过非文本类型 {ctype}")
                    return f"返回的不是网页文本（content-type: {ctype}）", "", ""

                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_BYTES:
                        break
                    chunks.append(chunk)
                raw = b"".join(chunks)
        except Exception as exc:  # noqa: BLE001 - network failures are expected
            logger.debug(f"[link-reader] {url} 抓取失败: {exc}")
            return f"网络请求失败（{type(exc).__name__}）", "", ""

        if not raw:
            return "服务器返回了空响应", "", ""

        if "json" in ctype:
            try:
                payload: Any = json.loads(raw.decode("utf-8", "ignore"))
            except json.JSONDecodeError:
                return "", "", raw.decode("utf-8", "ignore")
            return "", "", json.dumps(payload, ensure_ascii=False)[:MAX_CHARS]

        try:
            title, text = _extract_text(raw)
        except Exception as exc:  # noqa: BLE001 - malformed markup must not crash
            logger.debug(f"[link-reader] {url} 解析失败: {exc}")
            return f"页面解析失败（{type(exc).__name__}）", "", ""
        return "", title, text

    @filter.event_message_type(filter.EventMessageType.ALL, priority=sys.maxsize)
    async def read_links(self, event: AstrMessageEvent) -> None:
        """Append fetched link text to the message before the agent runs.

        Args:
            event: Incoming message event.
        """
        text = event.message_str or ""
        if not text.strip():
            return
        # Leave bot commands alone: they never carry a link worth fetching.
        if text.lstrip().startswith(("~", "ww")):
            return

        # 被引用的消息里也可能有链接（QQ 引用时正文只有引用的摘要，链接在 Reply 段里）
        scan_text = text
        for seg in event.get_messages():
            if isinstance(seg, Reply):
                quoted = getattr(seg, "message_str", "") or ""
                if quoted:
                    scan_text += "\n" + quoted

        urls = []
        for url in URL_RE.findall(scan_text):
            cleaned = url.rstrip(".,;:!?)]}>'\"")
            if cleaned not in urls:
                urls.append(cleaned)
        # multi_parser 现在会把解析结果注入消息（不再发卡片），平台域名统一交给它，
        # 这里跳过以免重复抓取。
        urls = [u for u in urls if not _is_delegated(urlparse(u).hostname or "")]
        if not urls:
            return

        blocks: list[str] = []
        for url in urls[:MAX_URLS]:
            reason, title, body = await self._fetch_one(url)
            if reason:
                blocks.append(
                    f"[链接内容] {url}\n读取失败：{reason}。"
                    "不要猜测页面内容，把这个原因如实告诉对方。"
                )
                continue
            if len(body) < MIN_MEANINGFUL_CHARS:
                blocks.append(
                    f"[链接内容] {url}\n"
                    "页面没有可直接读取的正文（很可能是需要 JavaScript 渲染的单页应用）。"
                    "不要猜测页面内容，直接告诉对方这个链接打不开、请他把内容复制或截图发过来。"
                )
                continue
            head = f"[链接内容] {url}"
            if title:
                head += f"\n标题: {title}"
            blocks.append(f"{head}\n正文:\n{body[:MAX_CHARS]}")

        if not blocks:
            return
        event.message_str = f"{text}\n\n" + "\n\n".join(blocks)
        logger.info(
            f"[link-reader] 处理 {len(blocks)} 个链接，消息增至 {len(event.message_str)} 字"
        )
