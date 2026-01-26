from __future__ import annotations

from dataclasses import dataclass

from .schemas import Category


@dataclass(frozen=True)
class FeedSource:
    category: Category
    name: str
    feed_url: str


# Curated, high-quality RSS feeds. Verify URLs and adjust as needed.
FEED_SOURCES: list[FeedSource] = [
    FeedSource("product", "OpenAI News", "https://openai.com/news/rss.xml"),
    FeedSource("product", "Google DeepMind Blog", "https://deepmind.com/blog/rss.xml"),
    FeedSource("product", "Hugging Face Blog", "https://huggingface.co/blog/feed.xml"),
    FeedSource("product", "AWS ML Blog", "https://aws.amazon.com/blogs/machine-learning/feed/"),
    FeedSource("product", "TechCrunch", "https://techcrunch.com/feed/"),
    FeedSource("product", "WIRED AI", "https://www.wired.com/feed/tag/ai/latest/rss"),
    FeedSource("product", "GitHub Blog", "https://github.com/blog.atom"),
    FeedSource("product", "Deno Blog", "https://deno.com/blog/rss.xml"),
    FeedSource("product", "GitHub Release Radar", "https://github.blog/tag/release-radar/feed/"),
    FeedSource("product", "Simon Willison Blog", "https://simonwillison.net/atom/everything/"),
    FeedSource("research", "Papers with Code", "https://paperswithcode.com/rss"),
    FeedSource("research", "arXiv cs.AI", "https://export.arxiv.org/rss/cs.AI"),
    FeedSource("research", "arXiv cs.LG", "https://export.arxiv.org/rss/cs.LG"),
    FeedSource("infra", "SemiAnalysis", "https://semianalysis.com/feed"),
    FeedSource("infra", "Latent Space", "https://www.latent.space/feed"),
    FeedSource("infra", "NVIDIA Developer Blog", "https://developer.nvidia.com/blog/rss/"),
]
