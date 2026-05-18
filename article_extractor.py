from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING
from urllib.parse import urlparse

import requests

if TYPE_CHECKING:
    from lxml.html import HtmlElement


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
MOBILE_USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.0 Mobile/15E148 Safari/604.1"
)
REMOVE_XPATH = (
    ".//script|.//style|.//noscript|.//svg|.//nav|.//header|.//footer|"
    ".//aside|.//form|.//button|.//input|.//select|.//textarea|"
    ".//*[contains(translate(concat(' ', @class, ' ', @id), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), ' ad ')]|"
    ".//*[contains(translate(concat(' ', @class, ' ', @id), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), ' ads ')]|"
    ".//*[contains(translate(concat(' ', @class, ' ', @id), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), ' advert')]|"
    ".//*[contains(translate(concat(' ', @class, ' ', @id), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'promo')]|"
    ".//*[contains(translate(concat(' ', @class, ' ', @id), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'newsletter')]|"
    ".//*[contains(translate(concat(' ', @class, ' ', @id), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'subscribe')]|"
    ".//*[contains(translate(concat(' ', @class, ' ', @id), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'related')]|"
    ".//*[contains(translate(concat(' ', @class, ' ', @id), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'comment')]|"
    ".//*[contains(translate(concat(' ', @class, ' ', @id), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'share')]|"
    ".//*[contains(translate(concat(' ', @class, ' ', @id), "
    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'cookie')]"
)
TEXT_BLOCK_XPATH = ".//p|.//li|.//blockquote|.//h2|.//h3"
CONTAINER_TAGS = {"article", "main", "section", "div", "body"}
MIN_EXTRACTED_CHARS = 120
ARTICLE_MIN_CHARS = 300
MAX_OUTPUT_CHARS = 50_000
TRUNCATED_NOTE = "\n\n[内容已截断，超过约 50000 字符。]"
NAV_LIKE_LINES = {
    "home",
    "menu",
    "search",
    "subscribe",
    "login",
    "sign in",
    "share",
    "more",
    "next",
    "previous",
    "首页",
    "菜单",
    "搜索",
    "登录",
    "注册",
    "订阅",
    "分享",
    "更多",
    "下一页",
    "上一页",
    "返回",
}
BOILERPLATE_PATTERNS = (
    r"广告",
    r"赞助",
    r"相关阅读",
    r"相关文章",
    r"推荐阅读",
    r"延伸阅读",
    r"点击.*阅读",
    r"扫码",
    r"关注.*公众号",
    r"订阅",
    r"注册",
    r"登录",
    r"版权所有",
    r"copyright",
    r"all rights reserved",
    r"share this",
    r"follow us",
    r"sign up",
    r"newsletter",
    r"cookie",
)


@dataclass(frozen=True)
class ArticleExtraction:
    title: str
    text: str
    podcast_text: str
    source_url: str


def extract_article_from_url(url: str, timeout: int = 15) -> ArticleExtraction:
    try:
        from lxml import etree, html
    except ImportError as exc:
        raise ValueError("缺少网页解析依赖 lxml，请先运行: pip install -r requirements.txt") from exc

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("仅支持 http:// 或 https:// 开头的文章链接。")

    response = _fetch_article_page(url, timeout)

    try:
        document = html.fromstring(response.content)
    except (etree.ParserError, ValueError) as exc:
        raise ValueError("无法解析该页面内容，请确认链接返回的是公开 HTML 文章页面。") from exc
    document.make_links_absolute(url)
    _remove_unwanted_nodes(document)

    title = _extract_title(document)
    text = _extract_best_text(document)
    podcast_text = clean_article_for_podcast(title, text)

    if len(text) < MIN_EXTRACTED_CHARS:
        raise ValueError("未能从该链接提取到足够的正文内容，请确认链接是公开可访问的文章页面。")

    return ArticleExtraction(title=title, text=text, podcast_text=podcast_text, source_url=url)


def _fetch_article_page(url: str, timeout: int) -> requests.Response:
    attempts = (
        _browser_headers(url, USER_AGENT),
        _browser_headers(url, MOBILE_USER_AGENT),
    )
    last_error: requests.HTTPError | None = None
    for headers in attempts:
        response = requests.get(url, headers=headers, timeout=timeout)
        try:
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            if response.status_code != 403:
                raise
            last_error = exc

    raise ValueError(
        "该网站拒绝自动抓取正文（HTTP 403）。可以尝试换一个公开文章链接，"
        "或直接复制网页正文到“手动输入”。"
    ) from last_error


def _browser_headers(url: str, user_agent: str) -> dict[str, str]:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": origin,
        "Upgrade-Insecure-Requests": "1",
    }


def clean_article_for_podcast(title: str, text: str) -> str:
    """Remove webpage boilerplate and return text shaped for spoken recording."""

    clean_title = _clean_inline(title)
    raw_lines = re.split(r"\n+", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines: list[str] = []

    for raw_line in raw_lines:
        line = _clean_inline(raw_line)
        if not line:
            continue
        if clean_title and _same_text(line, clean_title):
            continue
        if _is_navigation_like(line) or _is_boilerplate_line(line):
            continue
        lines.append(line)

    cleaned = _join_paragraphs(lines)
    if clean_title:
        cleaned = f"{clean_title}\n\n{cleaned}" if cleaned else clean_title
    return _cap_text(cleaned)


def _remove_unwanted_nodes(document: "HtmlElement") -> None:
    for node in document.xpath(REMOVE_XPATH):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)


def _extract_title(document: "HtmlElement") -> str:
    for xpath in (
        "//meta[translate(@property, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='og:title']/@content",
        "//meta[translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='og:title']/@content",
        "//h1[normalize-space()][1]",
        "//title[normalize-space()][1]",
    ):
        values = document.xpath(xpath)
        for value in values:
            title = _clean_inline(value if isinstance(value, str) else value.text_content())
            if title:
                return title
    return "未命名文章"


def _extract_best_text(document: "HtmlElement") -> str:
    article_text = _best_semantic_text(document)
    if len(article_text) >= ARTICLE_MIN_CHARS:
        return article_text

    scored_text = _best_scored_container_text(document)
    if len(scored_text) > len(article_text):
        text = scored_text
    else:
        text = article_text

    if len(text) < MIN_EXTRACTED_CHARS:
        body = document.xpath("//body")
        if body:
            body_text = _text_from_element(body[0])
            if len(body_text) > len(text):
                text = body_text

    return _cap_text(text)


def _best_semantic_text(document: "HtmlElement") -> str:
    candidates = document.xpath(
        "//article|//main|//*[@role='main']|"
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' article ')]|"
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' content ')]"
    )
    best = ""
    for candidate in candidates:
        text = _text_from_element(candidate)
        if len(text) > len(best):
            best = text
    return _cap_text(best)


def _best_scored_container_text(document: "HtmlElement") -> str:
    scores: dict["HtmlElement", int] = {}
    for paragraph in document.xpath("//p[normalize-space()]"):
        paragraph_text = _clean_inline(paragraph.text_content())
        if _is_navigation_like(paragraph_text):
            continue
        length = len(paragraph_text)
        if length < 20:
            continue
        for ancestor in _container_ancestors(paragraph):
            scores[ancestor] = scores.get(ancestor, 0) + length

    if not scores:
        return ""

    best_container = max(scores, key=scores.get)
    return _cap_text(_text_from_element(best_container))


def _container_ancestors(element: "HtmlElement") -> Iterable["HtmlElement"]:
    current = element
    while current is not None:
        if isinstance(current.tag, str) and current.tag.lower() in CONTAINER_TAGS:
            yield current
        current = current.getparent()


def _text_from_element(element: "HtmlElement") -> str:
    blocks = element.xpath(TEXT_BLOCK_XPATH)
    lines: list[str] = []

    for block in blocks:
        line = _clean_inline(block.text_content())
        if line and not _is_navigation_like(line):
            lines.append(line)

    if not lines:
        raw_lines = element.text_content().splitlines()
        lines = [
            line
            for line in (_clean_inline(raw_line) for raw_line in raw_lines)
            if line and not _is_navigation_like(line)
        ]

    return _cap_text(_join_paragraphs(lines))


def _join_paragraphs(lines: Iterable[str]) -> str:
    paragraphs: list[str] = []
    previous = ""
    for line in lines:
        if line == previous:
            continue
        paragraphs.append(line)
        previous = line
    return "\n\n".join(paragraphs).strip()


def _clean_inline(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _is_navigation_like(line: str) -> bool:
    normalized = line.strip().lower()
    if not normalized:
        return True
    if normalized in NAV_LIKE_LINES:
        return True
    if len(normalized) <= 3:
        return True
    if len(normalized) <= 12 and normalized.rstrip(" >»").lower() in NAV_LIKE_LINES:
        return True
    return False


def _is_boilerplate_line(line: str) -> bool:
    normalized = line.strip().lower()
    if not normalized:
        return True
    if len(normalized) <= 18 and re.search(r"^(图|图片|来源|作者|编辑|责任编辑)[:：]", line):
        return True
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in BOILERPLATE_PATTERNS)


def _same_text(left: str, right: str) -> bool:
    normalize = lambda value: re.sub(r"\W+", "", value, flags=re.UNICODE).lower()
    return bool(normalize(left)) and normalize(left) == normalize(right)


def _cap_text(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[: MAX_OUTPUT_CHARS - len(TRUNCATED_NOTE)].rstrip() + TRUNCATED_NOTE
