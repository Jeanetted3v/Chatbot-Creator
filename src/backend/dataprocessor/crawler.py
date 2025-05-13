"""To run
python -m src.backend.dataprocessor.crawler
"""
import os
import re
import asyncio
import logging
import hydra
from omegaconf import DictConfig
from pydantic_ai import Agent
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from src.backend.utils.settings import SETTINGS


logger = logging.getLogger(__name__)


async def crawl(crawl_data_dir: str, raw_crawled_file_name: str) -> str:
    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=3,
            include_external=False,
            max_pages=100,
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True
    )

    results = []
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(
            SETTINGS.WEBSITE,
            config=config
        )
    depth_counts = {}
    for result in results:
        depth = result.metadata.get("depth", 0)
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    logger.info("Pages crawled by depth:")
    for depth, count in sorted(depth_counts.items()):
        logger.info(f"  Depth {depth}: {count} pages")
    
    os.makedirs(crawl_data_dir, exist_ok=True)
    raw_crawled_file_name = "raw_crawl_results.md"
    output_file_path = os.path.join(crawl_data_dir, raw_crawled_file_name)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for result in results:
            depth = result.metadata.get("depth", 0)
            f.write(result.markdown + "\n\n")
    return results.markdown


async def extract(
    crawl_data_dir: str,
    raw_crawled_file_name: str,
    extracted_crawled_file_name: str,
    system_prompt: str,
    user_prompt: str,
    llm_model: str
) -> None:
    extraction_agent = Agent(
        model=llm_model,
        result_type=str,
        system_prompt=system_prompt,
    )
    raw_markdown_filepath = os.path.join(crawl_data_dir, raw_crawled_file_name)
    output_filepath = os.path.join(crawl_data_dir, extracted_crawled_file_name)
    with open(raw_markdown_filepath, "r", encoding="utf-8") as f:
        website_data = f.read()

    result = await extraction_agent.run(
        user_prompt.format(website_content=website_data)
    )
    extracted_info = result.data

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(extracted_info)
    logger.info(f"Extracted info: {extracted_info}")


async def clean_text(input_file_path: str, output_file_path: str) -> None:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    cleaned_content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # remove images
    cleaned_content = re.sub(r'\[.*?\]\(.*?\)', '', cleaned_content)  # remove links
    cleaned_content = re.sub(r'^\s*[-*•]\s+', '', cleaned_content, flags=re.MULTILINE)  # remove bullet points
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
    

async def translate(
    crawl_data_dir: str,
    extracted_crawled_file_name: str,
    data_ingest_dir: str,
    translated_crawled_file_name: str,
    system_prompt: str,
    user_prompt: str,
    llm_model: str
) -> str:
    translation_agent = Agent(
        model=llm_model,
        result_type=str,
        system_prompt=system_prompt,
    )
    extracted_filepath = os.path.join(
        crawl_data_dir, extracted_crawled_file_name)
    translated_filepath = os.path.join(
        data_ingest_dir, translated_crawled_file_name)
    with open(extracted_filepath, "r", encoding="utf-8") as f:
        website_data = f.read()
    result = await translation_agent.run(
        user_prompt.format(website_content=website_data)
    )
    translated_info = result.data
    with open(translated_filepath, "w", encoding="utf-8") as f:
        f.write(translated_info)
    logger.info(f"Translated info: {translated_info}")
    return translated_info


async def crawler_main(cfg: DictConfig) -> None:
    # await crawl(
    #     cfg.crawler.crawl_data_dir,
    #     cfg.crawler.raw_crawled_file_name
    # )
    input_file_path = os.path.join(
        cfg.crawler.crawl_data_dir, cfg.crawler.raw_crawled_file_name
    )
    output_file_path = os.path.join(
        cfg.crawler.crawl_data_dir, cfg.crawler.cleaned_file_name
    )
    await clean_text(
        input_file_path,
        output_file_path
    )
    await extract(
        cfg.crawler.crawl_data_dir,
        cfg.crawler.raw_crawled_file_name,
        cfg.crawler.extracted_crawled_file_name,
        cfg.crawler_prompts.extraction_agent.system_prompt,
        cfg.crawler_prompts.extraction_agent.user_prompt,
        cfg.crawler.llm
    )
    await translate(
        cfg.crawler.crawl_data_dir,
        cfg.crawler.raw_crawled_file_name,
        cfg.crawler.data_ingest_dir,
        cfg.crawler.translated_crawled_file_name,
        cfg.crawler_prompts.translation_agent.system_prompt,
        cfg.crawler_prompts.translation_agent.user_prompt,
        cfg.crawler.llm
    )


@hydra.main(
    version_base=None,
    config_path="../../../config",
    config_name="data_ingest")
def main(cfg) -> None:
    asyncio.run(crawler_main(cfg))


if __name__ == "__main__":
    main()