#!/usr/bin/env python3
"""
Virus metadata processing script.

This script processes virus metadata from a TSV file using rule-based processing
(and optionally AI assistance through the Gemini API).
"""

import argparse
import logging
import os
import re
import json
import time
import yaml
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google.generativeai not available. AI processing will be skipped.")

def canonicalize(seq_id):
    # Do not remove version numbers or alter the accession beyond stripping whitespace
    return str(seq_id).strip()


@dataclass
class ProcessedRecord:
    """Class for storing processed virus metadata records."""
    id: int
    standardized_host: str
    host_category: str
    standardized_location: str
    zoonotic: bool
    confidence: str  # Either "rule", "pass1", or "ai"

class VirusDataProcessor:
    """Process virus metadata using rules and optionally AI."""
    
    def __init__(self, input_file: str, config_file: str, output_dir: str, api_key: str = None):
        self.input_file = input_file
        self.config_file = config_file
        self.output_dir = output_dir
        self.api_key = api_key
        self.load_config()
        self.setup_logging()
        self.gemini_enabled = False
        if api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.gemini_enabled = True
            self.logger.info("Gemini AI initialized")
        elif api_key and not GEMINI_AVAILABLE:
            self.logger.warning("Gemini API key provided but google.generativeai module not available. Install with 'pip install google-generativeai'")
        self.host_patterns = self.config.get("host_patterns", {})
        self.virus_patterns = self.config.get("virus_patterns", [])
        self.host_mapping = self.config.get("host_mapping", {})
        self.category_mapping = self.config.get("category_mapping", {})
        self.dengue_keywords = self.config.get("dengue_keywords", [])
        self.known_zoonotic_viruses = self.config.get("known_zoonotic_viruses", [])
    
    def load_config(self):
        try:
            with open(self.config_file, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading config file {self.config_file}: {str(e)}")
    
    def setup_logging(self):
        os.makedirs(self.output_dir, exist_ok=True)
        log_file = os.path.join(self.output_dir, f"virus_processing_{datetime.now():%Y%m%d_%H%M}.log")
        logger = logging.getLogger("VirusDataProcessor")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.hasHandlers():
            logger.handlers.clear()
        file_handler = logging.FileHandler(log_file)
        json_formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(simple_formatter)
        logger.addHandler(stream_handler)
        self.logger = logger
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def process_dataset(self) -> pd.DataFrame:
        self.logger.info(f"Starting processing of {self.input_file}")
        df = pd.read_csv(self.input_file, sep="\t")
        df.fillna("", inplace=True)
        possible_accession_cols = ["accession", "Accession", "accession_number", "Accession Number"]
        found = False
        for col in possible_accession_cols:
            if col in df.columns:
                df.rename(columns={col: "accession"}, inplace=True)
                found = True
                break
        if not found:
            self.logger.warning("No accession column found in input; an empty column will be created.")
        df["standardized_host"] = ""
        df["host_category"] = ""
        df["standardized_location"] = ""
        df["zoonotic"] = False
        df["processing_method"] = ""
        df["gemini_annotated"] = False
        
        self.logger.info("Pass 1: Direct Homo sapiens processing")
        processed_pass1 = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Pass 1"):
            record = self._record_from_row(idx, row)
            result = self.process_record_human(record)
            if result:
                df.loc[idx, "standardized_host"] = result.standardized_host
                df.loc[idx, "host_category"] = result.host_category
                df.loc[idx, "standardized_location"] = result.standardized_location
                df.loc[idx, "zoonotic"] = result.zoonotic
                df.loc[idx, "processing_method"] = result.confidence
                processed_pass1 += 1
        self.logger.info(f"Pass 1 complete. Processed {processed_pass1} records.")
        
        self.logger.info("Pass 2: Rules-based processing")
        processed_pass2 = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Pass 2"):
            if df.loc[idx, "standardized_host"] in ("", None):
                record = self._record_from_row(idx, row)
                result = self.process_record_rules(record)
                if result:
                    df.loc[idx, "standardized_host"] = result.standardized_host
                    df.loc[idx, "host_category"] = result.host_category
                    df.loc[idx, "standardized_location"] = result.standardized_location
                    df.loc[idx, "zoonotic"] = result.zoonotic
                    df.loc[idx, "processing_method"] = result.confidence
                    processed_pass2 += 1
        self.logger.info(f"Pass 2 complete. Processed {processed_pass2} records.")
        
        unprocessed_mask = df["standardized_host"].isna() | (df["standardized_host"] == "")
        if unprocessed_mask.sum() > 0:
            if self.gemini_enabled:
                self.logger.info("Pass 3: Processing remaining records with Gemini AI")
                self.process_with_gemini(df, unprocessed_mask)
            else:
                self.logger.info(f"Skipping Gemini AI processing. {unprocessed_mask.sum()} records remain unprocessed.")
        else:
            self.logger.info("No records require Gemini AI processing.")
        
        for idx, row in df.iterrows():
            if row["standardized_host"] == "":
                original = self._record_from_row(idx, row)
                orig_host = original.get("host", "")
                orig_location = original.get("location", "")
                df.loc[idx, "standardized_host"] = orig_host if orig_host != "" else "Unknown"
                df.loc[idx, "host_category"] = "Unknown"
                df.loc[idx, "standardized_location"] = orig_location
                df.loc[idx, "processing_method"] = "fallback"
        
        additional_columns = {
            "genus": "",
            "isolation_date": "",
            "lab_culture": "",
            "wastewater_sewage": "",
            "is_segmented": "",
            "segment_label": "",
            "manual_update": ""
        }
        for col, default in additional_columns.items():
            if col not in df.columns:
                df[col] = default
        # Here, we trust the processed metadata: the host column in the output TSV should already be binary.
        df["host"] = df["standardized_host"]
        ordered_columns = [
            "accession",
            "host",
            "genus",
            "isolation_date",
            "strain_name",
            "location",
            "virus_name",
            "isolation_source",
            "lab_culture",
            "wastewater_sewage",
            "standardized_host",
            "host_category",
            "standardized_location",
            "zoonotic",
            "processing_method",
            "gemini_annotated",
            "is_segmented",
            "segment_label",
            "manual_update"
        ]
        for col in ordered_columns:
            if col not in df.columns:
                df[col] = ""
        df = df[ordered_columns]
        self.generate_summary(df)
        output_file = os.path.join(self.output_dir, "processed_virus_data.tsv")
        df.to_csv(output_file, index=False, sep="\t")
        self.logger.info(f"Saved processed data to {output_file}")
        return df
    
    def _record_from_row(self, idx: int, row: pd.Series) -> dict:
        field_mappings = {
            "host": ["host", "host_name", "host_organism", "host_species"],
            "virus": ["virus_name", "organism_name", "virus", "organism", "species"],
            "location": ["location", "geo_location", "geography", "country", "origin"],
            "source": ["isolation_source", "source", "tissue_specimen_source", "specimen", "sample_source"],
            "strain": ["strain_name", "strain", "isolate", "variant", "genotype"]
        }
        result = {"id": idx}
        columns_lower = {col.lower(): col for col in row.index}
        for field, possible_names in field_mappings.items():
            value = ""
            for name in possible_names:
                if name.lower() in columns_lower:
                    actual_col = columns_lower[name.lower()]
                    value = row.get(actual_col, "")
                    if value and str(value) != "nan":
                        break
            result[field] = str(value) if value else ""
        return result
    
    def process_record_human(self, record: dict) -> ProcessedRecord:
        host_value = record.get("host", "").lower()
        human_patterns = self.host_patterns.get("human", [])
        for pattern, standard_name in human_patterns:
            if re.search(pattern, host_value):
                return ProcessedRecord(
                    id=record["id"],
                    standardized_host=standard_name,
                    host_category=self.category_mapping.get(standard_name, "Mammal"),
                    standardized_location=self._standardize_location(record.get("location", "")),
                    zoonotic=self._check_zoonotic(record.get("virus", "")),
                    confidence="pass1",
                )
        return None
    
    def process_record_rules(self, record: dict) -> ProcessedRecord:
        record = {k: str(v) for k, v in record.items()}
        for vp in self.virus_patterns:
            pattern = vp.get("pattern")
            processor_name = vp.get("processor")
            match = re.search(pattern, record.get("strain", ""))
            if match:
                processor_func = self._get_processor_func(processor_name)
                if processor_func:
                    return processor_func(match, record)
        host_value = record.get("host", "").lower()
        for host_key, patterns in self.host_patterns.items():
            for pattern, std_name in patterns:
                if re.search(pattern, host_value):
                    return ProcessedRecord(
                        id=record["id"],
                        standardized_host=std_name,
                        host_category=self.category_mapping.get(std_name, "Unknown"),
                        standardized_location=self._standardize_location(record.get("location", "")),
                        zoonotic=self._check_zoonotic(record.get("virus", "")),
                        confidence="rule",
                    )
        dengue_result = self._process_dengue_hosting(record)
        if dengue_result:
            return dengue_result
        return None
    
    def _get_processor_func(self, name: str):
        mapping = {
            "process_influenza_pattern": self._process_influenza_pattern,
            "process_generic_pattern": self._process_generic_pattern,
            "process_dengue_pattern": self._process_dengue_pattern,
        }
        return mapping.get(name)
    
    def _process_pattern(self, match, record, standardized_host: str, host_category: str, zoonotic: bool, confidence="rule") -> ProcessedRecord:
        location = record.get("location", "")
        return ProcessedRecord(
            id=record["id"],
            standardized_host=standardized_host,
            host_category=host_category,
            standardized_location=self._standardize_location(location),
            zoonotic=zoonotic,
            confidence=confidence,
        )
    
    def _process_generic_pattern(self, match, record) -> ProcessedRecord:
        host = record.get("host", "").lower()
        std_host = self.host_mapping.get(host, "Unknown")
        host_category = self.category_mapping.get(std_host, "Unknown")
        return self._process_pattern(match, record, std_host, host_category, True)
    
    def _process_influenza_pattern(self, match, record) -> ProcessedRecord:
        subtype = match.group(1)
        location = match.group(2) if (match.lastindex and match.lastindex >= 2 and match.group(2)) else record.get("location", "")
        is_zoonotic = any(marker in subtype.upper() for marker in ["H5", "H7", "H9"])
        if "Influenza" in subtype:
            is_zoonotic = True
        host_value = record.get("host", "").lower()
        std_host = None
        for patterns in self.host_patterns.values():
            for pattern, standard_name in patterns:
                if re.search(pattern, host_value):
                    std_host = standard_name
                    break
            if std_host:
                break
        if not std_host:
            std_host = "Unknown"
        host_category = self.category_mapping.get(std_host, "Unknown")
        return ProcessedRecord(
            id=record["id"],
            standardized_host=std_host,
            host_category=host_category,
            standardized_location=self._standardize_location(location),
            zoonotic=is_zoonotic,
            confidence="rule",
        )
    
    def _process_dengue_pattern(self, match, record) -> ProcessedRecord:
        return ProcessedRecord(
            id=record["id"],
            standardized_host="aedes",
            host_category="insect",
            standardized_location=self._standardize_location(record.get("location", "")),
            zoonotic=True,
            confidence="rule",
        )
    
    def _process_dengue_hosting(self, record: dict) -> ProcessedRecord:
        if record.get("host", "").strip() and record.get("host", "").strip().lower() != "unknown":
            return None
        combined_text = " ".join([record.get("virus", ""), record.get("strain", ""), record.get("source", ""), record.get("location", "")]).lower()
        for keyword in self.dengue_keywords:
            if re.search(keyword.lower(), combined_text):
                return ProcessedRecord(
                    id=record["id"],
                    standardized_host="aedes",
                    host_category="insect",
                    standardized_location=self._standardize_location(record.get("location", "")),
                    zoonotic=True,
                    confidence="rule",
                )
        return None
    
    def _standardize_location(self, location: str) -> str:
        parts = location.split(",")
        if len(parts) >= 2:
            city = parts[0].strip()
            country = parts[-1].strip()
            return f"{city}, {country}"
        return location.strip()
    
    def _check_zoonotic(self, virus_name: str) -> bool:
        if not virus_name:
            return False
        for known in self.known_zoonotic_viruses:
            if known.lower() in virus_name.lower():
                return True
        for marker in ["H5", "H7", "H9"]:
            if marker in virus_name:
                return True
        return False
    
    def process_with_gemini(self, df: pd.DataFrame, mask: pd.Series):
        if not self.gemini_enabled:
            self.logger.warning("Gemini AI not enabled. Skipping processing.")
            return
        batch_size = 15
        max_retries = 3
        unprocessed_indices = df[mask].index.tolist()
        all_processed = set()
        all_failed = set()
        self.logger.info(f"Total records needing Gemini processing: {len(unprocessed_indices)}")
        for start in tqdm(range(0, len(unprocessed_indices), batch_size), desc="Gemini processing"):
            batch_indices = unprocessed_indices[start : start + batch_size]
            remaining = list(batch_indices)
            retries = 0
            while remaining and retries < max_retries:
                batch_df = df.loc[remaining].copy()
                try:
                    results = self._get_gemini_analysis(batch_df)
                    processed_in_batch = set()
                    for res in results:
                        gemini_id = res.get("id")
                        if gemini_id not in remaining:
                            self.logger.warning(f"Invalid ID {gemini_id} from Gemini; skipping.")
                            continue
                        df.loc[gemini_id, "standardized_host"] = res.get("standardized_host", "Unknown")
                        df.loc[gemini_id, "host_category"] = res.get("host_category", "Unknown")
                        df.loc[gemini_id, "standardized_location"] = res.get("standardized_location", "")
                        df.loc[gemini_id, "zoonotic"] = res.get("zoonotic", False)
                        df.loc[gemini_id, "processing_method"] = "ai"
                        df.loc[gemini_id, "gemini_annotated"] = True
                        processed_in_batch.add(gemini_id)
                        all_processed.add(gemini_id)
                    remaining = [idx for idx in remaining if idx not in processed_in_batch]
                    if remaining:
                        self.logger.warning(f"Incomplete Gemini response in batch; {len(remaining)} records remaining.")
                        retries += 1
                        time.sleep(2)
                    else:
                        break
                except Exception as e:
                    self.logger.error(f"Error during Gemini processing: {str(e)}")
                    retries += 1
                    time.sleep(2)
            if remaining:
                all_failed.update(remaining)
                self.logger.error(f"Failed to process {len(remaining)} records in batch after {max_retries} retries.")
        if all_failed:
            failed_df = df.loc[list(all_failed)].copy()
            failed_file = os.path.join(self.output_dir, "failed_records.tsv")
            failed_df.to_csv(failed_file, sep="\t", index=True)
            self.logger.error(f"Exported {len(failed_df)} failed records to {failed_file}")
        else:
            self.logger.info("All records processed successfully by Gemini.")
    
    def _get_gemini_analysis(self, batch_df: pd.DataFrame) -> list:
        records = []
        for idx, row in batch_df.iterrows():
            rec = {
                "id": idx,
                "host": str(row.get("host", "")),
                "virus": str(row.get("virus_name", "")),
                "location": str(row.get("location", "")),
                "source": str(row.get("isolation_source", "")),
                "strain": str(row.get("strain_name", "")),
            }
            records.append(rec)
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                prompt = self._get_gemini_prompt(records)
                response = self.model.generate_content(prompt)
                response_text = response.text
                if response_text.startswith("```"):
                    start_idx = response_text.find("\n") + 1
                    end_idx = response_text.rfind("```")
                    response_text = response_text[start_idx:end_idx].strip()
                parsed_response = json.loads(response_text)
                validated = []
                for rec in parsed_response:
                    validated.append({
                        "id": rec.get("id", 0),
                        "standardized_host": rec.get("standardized_host", "Unknown"),
                        "host_category": rec.get("host_category", "Unknown"),
                        "standardized_location": rec.get("standardized_location", ""),
                        "zoonotic": bool(rec.get("zoonotic", False)),
                    })
                if len(validated) == len(records):
                    return validated
                else:
                    self.logger.warning(f"Incomplete Gemini response: expected {len(records)} records, got {len(validated)}.")
            except Exception as e:
                self.logger.error(f"Error in Gemini API call: {str(e)}")
            retries += 1
            time.sleep(2)
        self.logger.error("Failed to get a valid Gemini response after maximum retries.")
        return []
    
    def _get_gemini_prompt(self, records: list) -> str:
        prompt = (
            "Please analyze these virus records and return ONLY a JSON array with the following information for each record. "
            "Do not include any explanatory text or markdown formatting. Be complete.\n\n"
            "For ALL records, analyze and return:\n"
            "1. Standardized host name (scientific name if possible)\n"
            "2. Host category: One of [Mammal, Avian, Insect, Crustacean, Plant, Bacteria, Fungi, Reptile, Fish, Amphibian, Other]\n"
            "3. Location in \"City, Country\" format\n"
            "4. Zoonotic potential: boolean true/false based on known virus behaviors\n\n"
            "Required JSON format:\n"
            "[\n"
            "  {\n"
            "    \"id\": 1,\n"
            "    \"standardized_host\": \"scientific_name\",\n"
            "    \"host_category\": \"category\",\n"
            "    \"standardized_location\": \"City, Country\",\n"
            "    \"zoonotic\": boolean\n"
            "  }\n"
            "]\n\n"
            "Input Records:\n" + json.dumps(records, indent=2)
        )
        return prompt
    
    def generate_summary(self, df: pd.DataFrame):
        summary = []
        summary.append("=== Virus Data Analysis Summary ===\n")
        summary.append("Host Categories:")
        cat_counts = df["host_category"].value_counts()
        for cat, count in cat_counts.items():
            summary.append(f"{cat}: {count}")
        summary.append("\nProcessing Methods:")
        method_counts = df["processing_method"].value_counts()
        for method, count in method_counts.items():
            summary.append(f"{method}: {count}")
        gemini_count = df["gemini_annotated"].sum()
        summary.append(f"\nGemini Annotations:\nRecords processed by Gemini: {gemini_count}\nRecords processed by rules: {len(df) - gemini_count}")
        zoonotic_count = df["zoonotic"].sum()
        summary.append(f"\nZoonotic Potential:\nZoonotic: {zoonotic_count}\nNon-zoonotic: {len(df) - zoonotic_count}")
        virus_column = None
        possible_columns = ["virus_name", "Organism_Name", "Species", "GenBank_Title"]
        for col in possible_columns:
            if col in df.columns:
                virus_column = col
                break
        if virus_column:
            summary.append("\nTop Viruses by Host Category:")
            for cat in df["host_category"].unique():
                cat_df = df[df["host_category"] == cat]
                top_viruses = cat_df[virus_column].value_counts().head(5)
                summary.append(f"\n{cat}:")
                for virus, count in top_viruses.items():
                    summary.append(f"  {virus}: {count}")
        summary_file = os.path.join(self.output_dir, "virus_analysis_summary.txt")
        with open(summary_file, "w") as f:
            f.write("\n".join(summary))
        self.logger.info(f"Generated summary report: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Virus Data Processor")
    parser.add_argument("--input", required=True, help="Input TSV file with virus data")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--output_dir", required=True, help="Directory for output files")
    parser.add_argument("--api_key", help="Google Gemini API key (optional)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    processor = VirusDataProcessor(args.input, args.config, args.output_dir, args.api_key)
    processor.process_dataset()
    print(f"Processing complete. Check output files in {args.output_dir}.")

if __name__ == "__main__":
    main()
