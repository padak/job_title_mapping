"""
Keboola Component: Job Title Mapping using Claude LLM

Maps raw job titles to standardized titles using Claude API.
"""

import json
import logging
from pathlib import Path

import anthropic
import pandas as pd
from keboola.component import CommonInterface
from pydantic import BaseModel

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models that support structured outputs
STRUCTURED_OUTPUT_MODELS = {
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250514",
    "claude-opus-4-1",
    "claude-opus-4-1-20250414",
}


class MappedTitle(BaseModel):
    """Schema for a single mapped job title."""
    id: str
    job_title: str
    standardized_job_title: str | None
    department: str | None
    seniority: float


class MappedTitlesResponse(BaseModel):
    """Schema for the batch response."""
    mappings: list[MappedTitle]


def create_taxonomy_reference(df_taxonomy: pd.DataFrame) -> str:
    """Create a formatted taxonomy reference for the LLM prompt."""
    lines = []
    for _, row in df_taxonomy.iterrows():
        title = row["standardized_job_title"]
        seniority = row["seniority"]
        department = row["department"]
        lines.append(f"- {title} | seniority: {seniority} | department: {department}")
    return "\n".join(lines)


def get_json_schema() -> dict:
    """Return JSON schema for structured output."""
    return {
        "type": "object",
        "properties": {
            "mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "job_title": {"type": "string"},
                        "standardized_job_title": {"type": ["string", "null"]},
                        "department": {"type": ["string", "null"]},
                        "seniority": {"type": "number"}
                    },
                    "required": ["id", "job_title", "standardized_job_title", "department", "seniority"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["mappings"],
        "additionalProperties": False
    }


def build_system_prompt(taxonomy_ref: str) -> str:
    """Build the system prompt with taxonomy (cacheable)."""
    return f"""You are a job title mapping assistant. Your task is to map raw job titles to standardized titles from a taxonomy.

## Taxonomy (standardized titles with seniority and department):
{taxonomy_ref}

## Instructions:
1. For each job title, find the BEST matching standardized title from the taxonomy above.
2. If no good match exists (similarity < 50%), use null for standardized_job_title and -1 for seniority.
3. Use the seniority and department values from the matched taxonomy entry.
4. Return the results in the specified JSON format.

Always return a JSON object with "mappings" array containing all mapped titles."""


def build_user_prompt(job_titles: list[dict]) -> str:
    """Build the user prompt with job titles (dynamic, not cached)."""
    titles_list = "\n".join([
        f"{i+1}. ID: {t['id']} | Title: {t['job_title']}"
        for i, t in enumerate(job_titles)
    ])

    return f"""Map these job titles to the taxonomy:

{titles_list}"""


def map_titles_with_structured_output(
    client: anthropic.Anthropic,
    job_titles: list[dict],
    taxonomy_ref: str,
    model: str
) -> tuple[list[dict], dict]:
    """Map titles using structured outputs (beta feature) with prompt caching."""
    system_prompt = build_system_prompt(taxonomy_ref)
    user_prompt = build_user_prompt(job_titles)

    response = client.beta.messages.create(
        model=model,
        max_tokens=8192,
        betas=["structured-outputs-2025-11-13", "prompt-caching-2024-07-31"],
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
        output_format={
            "type": "json_schema",
            "schema": get_json_schema()
        }
    )

    # Extract usage stats
    usage_stats = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
    }

    result = json.loads(response.content[0].text)
    return result.get("mappings", []), usage_stats


def map_titles_without_structured_output(
    client: anthropic.Anthropic,
    job_titles: list[dict],
    taxonomy_ref: str,
    model: str
) -> tuple[list[dict], dict]:
    """Map titles using regular JSON output (fallback) with prompt caching."""
    system_prompt = build_system_prompt(taxonomy_ref) + """

## Output format (JSON):
{
  "mappings": [
    {"id": "...", "job_title": "original", "standardized_job_title": "matched or null", "department": "...", "seniority": number}
  ]
}

Return ONLY valid JSON, no markdown formatting or explanation."""

    user_prompt = build_user_prompt(job_titles)

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{"role": "user", "content": user_prompt}]
    )

    # Extract usage stats
    usage_stats = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
    }

    response_text = response.content[0].text.strip()

    # Handle potential markdown code blocks
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        result = json.loads(response_text)
        return result.get("mappings", []), usage_stats
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Response was: {response_text[:500]}")
        return [], usage_stats


def map_titles_batch(
    client: anthropic.Anthropic,
    job_titles: list[dict],
    taxonomy_ref: str,
    model: str,
    use_structured_output: bool
) -> tuple[list[dict], dict]:
    """Map a batch of job titles to standardized titles."""
    if use_structured_output and model in STRUCTURED_OUTPUT_MODELS:
        return map_titles_with_structured_output(client, job_titles, taxonomy_ref, model)
    else:
        if use_structured_output and model not in STRUCTURED_OUTPUT_MODELS:
            logger.warning(f"Model {model} doesn't support structured outputs, using fallback")
        return map_titles_without_structured_output(client, job_titles, taxonomy_ref, model)


def process_contacts(
    df_contacts: pd.DataFrame,
    df_taxonomy: pd.DataFrame,
    api_key: str,
    model: str,
    batch_size: int,
    use_structured_output: bool
) -> tuple[pd.DataFrame, dict]:
    """Process all contacts and map their job titles."""
    client = anthropic.Anthropic(api_key=api_key)
    taxonomy_ref = create_taxonomy_reference(df_taxonomy)

    # Prepare job titles for processing
    titles_to_map = []
    for _, row in df_contacts.iterrows():
        job_title = row.get("job_title", "")
        if pd.isna(job_title) or not str(job_title).strip():
            continue
        titles_to_map.append({
            "id": str(row["contact_id"]),
            "job_title": str(job_title).strip()
        })

    logger.info(f"Processing {len(titles_to_map)} job titles in batches of {batch_size}...")
    logger.info(f"Model: {model}")
    logger.info(f"Structured outputs: {'enabled' if use_structured_output and model in STRUCTURED_OUTPUT_MODELS else 'disabled'}")

    # Process in batches
    all_results = []
    total_batches = (len(titles_to_map) + batch_size - 1) // batch_size

    # Track usage stats
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    for i in range(0, len(titles_to_map), batch_size):
        batch = titles_to_map[i:i + batch_size]
        batch_num = i // batch_size + 1

        results, usage_stats = map_titles_batch(client, batch, taxonomy_ref, model, use_structured_output)
        all_results.extend(results)

        # Accumulate usage stats
        for key in total_usage:
            total_usage[key] += usage_stats.get(key, 0)

        cache_status = "CACHE HIT" if usage_stats.get("cache_read_input_tokens", 0) > 0 else "CACHE MISS"
        logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} titles) - {cache_status}")

    # Create results DataFrame
    df_results = pd.DataFrame(all_results)

    if df_results.empty:
        logger.warning("No results returned from API")
        return pd.DataFrame(), total_usage

    # Merge with original contacts to preserve other columns
    df_output = df_contacts[["contact_id", "job_title", "email", "contact_type"]].copy()
    df_output["contact_id"] = df_output["contact_id"].astype(str)
    df_results["id"] = df_results["id"].astype(str)

    df_output = df_output.merge(
        df_results[["id", "standardized_job_title", "department", "seniority"]],
        left_on="contact_id",
        right_on="id",
        how="left"
    )
    df_output = df_output.drop(columns=["id"], errors="ignore")

    # Fill missing values for contacts without job titles
    df_output["seniority"] = df_output["seniority"].fillna(-1)

    return df_output, total_usage


def main():
    """Main entry point for Keboola component."""
    ci = CommonInterface()
    params = ci.configuration.parameters

    # Get parameters with defaults
    api_key = params.get("#ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Missing required parameter: #ANTHROPIC_API_KEY")

    model = params.get("model", "claude-sonnet-4-5")
    batch_size = int(params.get("batch_size", 100))
    use_structured_output = params.get("use_structured_output", True)

    logger.info("=== Job Title Mapping Component ===")
    logger.info(f"Model: {model}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Structured outputs: {use_structured_output}")

    # Fixed input paths (Table Input Mapping)
    contacts_path = Path(ci.data_folder_path) / "in/tables/contact.csv"
    taxonomy_path = Path(ci.data_folder_path) / "in/tables/standardized_job_titles.csv"

    if not contacts_path.exists():
        raise FileNotFoundError(f"Contacts file not found: {contacts_path}")
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

    # Load data
    df_contacts = pd.read_csv(contacts_path)
    df_taxonomy = pd.read_csv(taxonomy_path)

    logger.info(f"Loaded {len(df_contacts)} contacts and {len(df_taxonomy)} taxonomy entries")

    # Process contacts
    df_output, usage_stats = process_contacts(
        df_contacts,
        df_taxonomy,
        api_key=api_key,
        model=model,
        batch_size=batch_size,
        use_structured_output=use_structured_output
    )

    if df_output.empty:
        logger.warning("No results to save")
        return

    # Write output to fixed path
    output_path = Path(ci.data_folder_path) / "out/tables/contact_taxonomy_seniority.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)

    logger.info(f"Results saved: {len(df_output)} contacts mapped")
    logger.info(f"Output: {output_path}")

    # Log token usage summary
    logger.info("=== Token Usage Summary ===")
    logger.info(f"Input tokens: {usage_stats['input_tokens']:,}")
    logger.info(f"Output tokens: {usage_stats['output_tokens']:,}")
    logger.info(f"Cache creation tokens: {usage_stats['cache_creation_input_tokens']:,}")
    logger.info(f"Cache read tokens: {usage_stats['cache_read_input_tokens']:,}")
    total_tokens = usage_stats['input_tokens'] + usage_stats['output_tokens']
    logger.info(f"Total tokens: {total_tokens:,}")

    # Calculate cache savings
    cache_read = usage_stats['cache_read_input_tokens']
    cache_creation = usage_stats['cache_creation_input_tokens']
    if cache_read > 0:
        cache_hit_rate = (cache_read / (cache_read + cache_creation)) * 100
        logger.info(f"Cache hit rate: {cache_hit_rate:.1f}%")


if __name__ == "__main__":
    main()
