import unittest
from unittest.mock import patch

import requests

from article_extractor import extract_article_from_url


class FakeResponse:
    def __init__(self, html_text: str | bytes, status_code: int = 200):
        self.content = html_text if isinstance(html_text, bytes) else html_text.encode("utf-8")
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)
        return None


def fake_get_factory(html_text):
    def fake_get(url, headers=None, timeout=None):
        fake_get.calls.append((url, headers, timeout))
        return FakeResponse(html_text)

    fake_get.calls = []
    return fake_get


def fake_sequence_factory(responses):
    def fake_get(url, headers=None, timeout=None):
        fake_get.calls.append((url, headers, timeout))
        return responses[len(fake_get.calls) - 1]

    fake_get.calls = []
    return fake_get


class ArticleExtractorTests(unittest.TestCase):
    def test_extracts_og_title_and_article_text(self):
        html = """
        <html>
          <head>
            <meta property="og:title" content="OG Article Title">
            <title>Fallback Title</title>
            <script>ignored()</script>
          </head>
          <body>
            <nav>Home Search Login</nav>
            <article>
              <h1>Visible H1</h1>
              <p>This is the first substantial paragraph for the article body with enough detail to keep.</p>
              <p>This is the second substantial paragraph for the article body and it should be preserved.</p>
              <p>This is the third substantial paragraph that pushes the cleaned article above the minimum length.</p>
            </article>
          </body>
        </html>
        """
        fake_get = fake_get_factory(html)

        with patch("article_extractor.requests.get", fake_get):
            result = extract_article_from_url("https://example.com/post", timeout=7)

        self.assertEqual(result.title, "OG Article Title")
        self.assertEqual(result.source_url, "https://example.com/post")
        self.assertNotIn("Home Search Login", result.text)
        self.assertIn("first substantial paragraph", result.text)
        self.assertIn("OG Article Title", result.podcast_text)
        self.assertIn("first substantial paragraph", result.podcast_text)
        self.assertIn("\n\n", result.text)
        self.assertEqual(fake_get.calls[0][2], 7)
        self.assertIn("Mozilla/5.0", fake_get.calls[0][1]["User-Agent"])
        self.assertIn("Accept-Language", fake_get.calls[0][1])

    def test_scores_best_paragraph_container_when_article_is_missing(self):
        html = """
        <html>
          <head><title>Document Title</title></head>
          <body>
            <div class="sidebar"><p>Subscribe</p><p>Share</p></div>
            <section id="story">
              <p>Alpha paragraph with meaningful article content that belongs to the main story extraction.</p>
              <p>Beta paragraph with more meaningful article content and enough words to win container scoring.</p>
              <p>Gamma paragraph continues the same story so the extracted text is long enough for validation.</p>
            </section>
          </body>
        </html>
        """

        with patch("article_extractor.requests.get", fake_get_factory(html)):
            result = extract_article_from_url("http://example.com/story")

        self.assertEqual(result.title, "Document Title")
        self.assertIn("Alpha paragraph", result.text)
        self.assertNotIn("Subscribe", result.text)

    def test_builds_podcast_text_without_ads_related_links_or_duplicate_title(self):
        html = """
        <html>
          <head><meta charset="utf-8"><title>Clean Podcast Title</title></head>
          <body>
            <main>
              <h1>Clean Podcast Title</h1>
              <p>核心正文第一段，包含足够多的信息，适合被保留到播客稿件里继续朗读。</p>
              <p>广告：这是页面广告，不应该进入播客稿。</p>
              <p>核心正文第二段，继续解释背景和原因，让抽取结果保持完整自然。</p>
              <p>相关阅读：另一个链接标题。</p>
              <p>核心正文第三段，补充更多上下文，使文本长度满足正文抽取要求。</p>
            </main>
          </body>
        </html>
        """

        with patch("article_extractor.requests.get", fake_get_factory(html)):
            result = extract_article_from_url("https://example.com/clean")

        self.assertTrue(result.podcast_text.startswith("Clean Podcast Title\n\n核心正文第一段"))
        self.assertEqual(result.podcast_text.count("Clean Podcast Title"), 1)
        self.assertNotIn("页面广告", result.podcast_text)
        self.assertNotIn("相关阅读", result.podcast_text)
        self.assertIn("核心正文第三段", result.podcast_text)

    def test_rejects_non_http_url(self):
        with self.assertRaisesRegex(ValueError, "仅支持"):
            extract_article_from_url("file:///tmp/article.html")

    def test_raises_when_text_is_too_short(self):
        html = "<html><head><title>Tiny</title></head><body><main><p>Too short.</p></main></body></html>"

        with patch("article_extractor.requests.get", fake_get_factory(html)):
            with self.assertRaisesRegex(ValueError, "足够的正文内容"):
                extract_article_from_url("https://example.com/tiny")

    def test_wraps_empty_page_parse_errors(self):
        with patch("article_extractor.requests.get", fake_get_factory(b"")):
            with self.assertRaisesRegex(ValueError, "无法解析"):
                extract_article_from_url("https://example.com/empty")

    def test_retries_403_with_mobile_headers(self):
        html = """
        <html><head><title>Retry Title</title></head><body><article>
          <p>First paragraph with enough article content for extraction after a mobile retry succeeds.</p>
          <p>Second paragraph with enough article content for extraction after a mobile retry succeeds.</p>
          <p>Third paragraph with enough article content for extraction after a mobile retry succeeds.</p>
        </article></body></html>
        """
        fake_get = fake_sequence_factory([FakeResponse("Forbidden", status_code=403), FakeResponse(html)])

        with patch("article_extractor.requests.get", fake_get):
            result = extract_article_from_url("https://example.com/retry")

        self.assertEqual(len(fake_get.calls), 2)
        self.assertIn("iPhone", fake_get.calls[1][1]["User-Agent"])
        self.assertIn("First paragraph", result.text)

    def test_reports_clear_error_when_403_persists(self):
        fake_get = fake_sequence_factory([
            FakeResponse("Forbidden", status_code=403),
            FakeResponse("Forbidden", status_code=403),
        ])

        with patch("article_extractor.requests.get", fake_get):
            with self.assertRaisesRegex(ValueError, "HTTP 403"):
                extract_article_from_url("https://example.com/blocked")

        self.assertEqual(len(fake_get.calls), 2)


if __name__ == "__main__":
    unittest.main()
