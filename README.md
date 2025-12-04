# Job Title Mapping - Keboola Component

Keboola component that maps raw job titles to a standardized taxonomy using Claude AI.

## Overview

Given input like:
- "Senior Vice President, Commercial Sales"
- "BI Engineer"
- "CEO & Co-Founder"

The component maps them to standardized titles:
- "vice president of sales" (seniority: 7, department: Executive C-Level)
- "analytics engineer (level II)" (seniority: 3, department: Data & Analytics)
- "co-founder" (seniority: 7, department: Executive C-Level)

## Features

- **LLM-powered mapping** - Uses Claude's semantic understanding for accurate matching
- **Structured outputs** - Guarantees valid JSON responses (beta feature for supported models)
- **Prompt caching** - Caches taxonomy in system prompt for 90% input token savings
- **Batch processing** - Processes multiple titles per API call for efficiency
- **Configurable** - Model, batch size, and features configurable via component parameters

## Configuration

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `#ANTHROPIC_API_KEY` | Yes | - | Anthropic API key (encrypted) |
| `model` | No | `claude-sonnet-4-5` | Claude model to use |
| `batch_size` | No | `100` | Titles processed per API call |
| `use_structured_output` | No | `true` | Use structured outputs beta feature |

### Supported Models

| Model | Structured Outputs | Notes |
|-------|-------------------|-------|
| `claude-sonnet-4-5` | ✅ Yes | Recommended - fast, accurate |
| `claude-opus-4-1` | ✅ Yes | Best quality, slower |
| `claude-haiku-3-5-20241022` | ❌ No | Fastest, cheapest |

## Input Tables

### contact.csv (required)

| Column | Type | Description |
|--------|------|-------------|
| `contact_id` | string | Unique identifier |
| `job_title` | string | Raw job title to map |
| `email` | string | Contact email |
| `contact_type` | string | Lead/Contact type |

### standardized_job_titles.csv (required)

| Column | Type | Description |
|--------|------|-------------|
| `standardized_job_title` | string | Canonical job title |
| `seniority` | number | Seniority level (1-7) |
| `department` | string | Department category |

## Output Table

### contact_taxonomy_seniority.csv

| Column | Type | Description |
|--------|------|-------------|
| `contact_id` | string | Original contact ID |
| `job_title` | string | Original job title |
| `email` | string | Contact email |
| `contact_type` | string | Lead/Contact type |
| `standardized_job_title` | string | Mapped standardized title |
| `department` | string | Department from taxonomy |
| `seniority` | number | Seniority score (-1 if no match) |

## How It Works

1. Loads contacts and taxonomy from input tables
2. Creates system prompt with taxonomy (cached after first call)
3. Batches job titles (default: 100 per batch)
4. Sends each batch to Claude - taxonomy is cached, only titles are sent fresh
5. Claude returns structured JSON with best-match mappings
6. Results are merged and written to output table

## Cost Estimation

With default settings (Sonnet 4.5, batch size 100, prompt caching enabled):
- First batch: ~$0.02 (cache creation)
- Subsequent batches: ~$0.005 (90% savings from cache hits)
- **~$8-12 per 143,000 titles** (vs ~$57 without caching)

## Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run locally (requires Keboola data folder structure)
python main.py
```

## Repository Structure

```
├── main.py                 # Component entry point
├── requirements.txt        # Python dependencies
├── data/                   # Local test data
│   ├── contact.csv
│   └── standardized_job_titles.csv
└── README.md
```
