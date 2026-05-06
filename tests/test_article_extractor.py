import unittest
from unittest.mock import patch

from article_extractor import extract_article_from_url


class FakeResponse:
    def __init__(self, html_text: str | bytes):
        self.content = html_text if isinstance(html_text, bytes) else html_text.encode("utf-8")

    def raise_for_status(self):
        return None


def fake_get_factory(html_text):
    def fake_get(url, headers=None, timeout=None):
        fake_get.calls.append((url, headers, timeout))
        return FakeResponse(html_text)

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
        self.assertIn("\n\n", result.text)
        self.assertEqual(fake_get.calls[0][2], 7)
        self.assertIn("Mozilla/5.0", fake_get.calls[0][1]["User-Agent"])

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


if __name__ == "__main__":
    unittest.main()
