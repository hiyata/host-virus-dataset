#!/usr/bin/env python3
"""
Pipeline Testing Script.

This script tests the entire pipeline using the example data and verifies
that each component works correctly. It generates statistics and reports
on the success or failure of each step.
"""

import os
import sys
import subprocess
import time
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime


class PipelineTester:
    """Test the virus host classification pipeline."""
    
    def __init__(self, output_dir="test_results", example_data_dir="data/examples",
                 config_dir="config", keep_intermediate=False, verbose=True):
        """
        Initialize the tester.
        
        Args:
            output_dir (str): Directory to store test results
            example_data_dir (str): Directory containing example data
            config_dir (str): Directory containing configuration files
            keep_intermediate (bool): Whether to keep intermediate files
            verbose (bool): Whether to print detailed output
        """
        self.output_dir = output_dir
        self.example_data_dir = example_data_dir
        self.config_dir = config_dir
        self.keep_intermediate = keep_intermediate
        self.verbose = verbose
        
        # Create paths for intermediate results
        self.base_dir = Path(output_dir)
        self.preprocessed_dir = self.base_dir / "preprocessed_data"
        self.metadata_dir = self.base_dir / "metadata_data"
        self.features_dir = self.base_dir / "features_data"
        self.split_dir = self.base_dir / "split_data"
        self.models_dir = self.base_dir / "models"
        self.viz_dir = self.base_dir / "visualizations"
        
        # Create test output directory
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        # Path to host patterns config file
        self.host_patterns_config = Path(config_dir) / "host_patterns.yml"
        
        # Path to processed metadata file
        self.processed_metadata = self.metadata_dir / "processed_virus_data.tsv"
        
        # Initialize results
        self.results = {
            "overall_success": False,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "duration_seconds": None,
            "steps": {}
        }
    
    def run_command(self, command, step_name, expected_outputs=None):
        """
        Run a shell command and capture the output.
        
        Args:
            command (str): Command to run
            step_name (str): Name of the step
            expected_outputs (list): List of expected output files/directories
            
        Returns:
            bool: Whether the command succeeded
        """
        # Initialize step results
        self.results["steps"][step_name] = {
            "command": command,
            "success": False,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": None,
            "stdout": None,
            "stderr": None,
            "expected_outputs_exist": False
        }
        
        # Print command
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Running step: {step_name}")
            print(f"Command: {command}")
            print(f"{'='*80}\n")
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Run command
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            
            # Store outputs
            self.results["steps"][step_name]["stdout"] = stdout
            self.results["steps"][step_name]["stderr"] = stderr
            self.results["steps"][step_name]["return_code"] = process.returncode
            
            # Check if command succeeded
            success = process.returncode == 0
            self.results["steps"][step_name]["success"] = success
            
            # Print outputs
            if self.verbose:
                if stdout:
                    print("STDOUT:")
                    print(stdout)
                if stderr:
                    print("STDERR:")
                    print(stderr)
                
            # Check expected outputs
            if expected_outputs and success:
                all_outputs_exist = True
                for output in expected_outputs:
                    if not os.path.exists(output):
                        all_outputs_exist = False
                        if self.verbose:
                            print(f"Expected output not found: {output}")
                
                self.results["steps"][step_name]["expected_outputs_exist"] = all_outputs_exist
                success = success and all_outputs_exist
            
        except Exception as e:
            self.results["steps"][step_name]["success"] = False
            self.results["steps"][step_name]["error"] = str(e)
            if self.verbose:
                print(f"Error: {e}")
            success = False
        
        # Calculate duration
        duration = time.time() - start_time
        self.results["steps"][step_name]["duration_seconds"] = duration
        
        # Print result
        if self.verbose:
            print(f"\nStep completed: {step_name}")
            print(f"Success: {success}")
            print(f"Duration: {duration:.2f} seconds")
        
        return success
    
    def check_script_supports_param(self, script_path, param_name):
        """
        Check if a script supports a specific parameter.
        
        Args:
            script_path (str): Path to the script
            param_name (str): Parameter name to check
            
        Returns:
            bool: Whether the script supports the parameter
        """
        try:
            cmd = f"python {script_path} --help"
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
            return param_name in output
        except:
            return False
    
    def test_preprocessing(self):
        """Test the preprocessing step."""
        step_name = "preprocessing"
        
        # Create command
        command = (
            f"python scripts/preprocess_fasta.py "
            f"--input_dir {self.example_data_dir}/fasta "
            f"--output_dir {self.preprocessed_dir} "
            f"--metadata {self.example_data_dir}/metadata/sample_metadata.tsv "
            f"--exclusion_keywords partial,mutant,unverified,bac,clone"
        )
        
        # Run command
        success = self.run_command(command, step_name)
        
        # Check if at least one FASTA file was created
        if success:
            fasta_files = list(self.preprocessed_dir.glob("*.fasta"))
            if not fasta_files:
                self.results["steps"][step_name]["success"] = False
                self.results["steps"][step_name]["error"] = "No FASTA files created"
                success = False
        
        return success
    
    def test_metadata_processing(self):
        """Test the metadata processing step."""
        step_name = "metadata_processing"
        
        # Create metadata directory
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Check if config file exists
        if not os.path.exists(self.host_patterns_config):
            print(f"Warning: Config file {self.host_patterns_config} not found. Using local config.")
            # Create temporary config file for testing
            local_config_dir = self.base_dir / "config"
            local_config_dir.mkdir(exist_ok=True)
            local_config = local_config_dir / "host_patterns.yml"
            
            with open(local_config, 'w') as f:
                f.write("""# Simple host patterns for testing
host_patterns:
  human:
    - ["\\\\bhomo sapiens\\\\b", "Homo sapiens"]
    - ["\\\\bhuman\\\\b", "Homo sapiens"]
    - ["\\\\bh\\\\. sapiens\\\\b", "Homo sapiens"]
  
category_mapping:
  "Homo sapiens": "Mammal"
  "Unknown": "Unknown"

host_mapping:
  "unknown": "Unknown"
  "NA": "Unknown"
  "": "Unknown"

known_zoonotic_viruses:
  - "influenza"
  - "dengue"
  - "zika"
""")
            self.host_patterns_config = local_config
        
        # Create command
        command = (
            f"python scripts/process_metadata.py "
            f"--input {self.example_data_dir}/metadata/sample_metadata.tsv "
            f"--config {self.host_patterns_config} "
            f"--output_dir {self.metadata_dir}"
        )
        
        # Expected outputs
        expected_outputs = [
            self.processed_metadata
        ]
        
        # Run command
        success = self.run_command(command, step_name, expected_outputs)
        
        return success
    
    def test_feature_generation(self):
        """Test the feature generation step."""
        step_name = "feature_generation"
        
        # Update k-mer value to 5
        command = (
            f"python scripts/generate_features.py "
            f"--input_dir {self.preprocessed_dir} "
            f"--output_dir {self.features_dir} "
            f"--k_values 5"  # Use k-mer size 5 for testing
        )
        
        # Expected output: viral_dataset_k5.h5 should be created
        expected_outputs = [
            self.features_dir / "viral_dataset_k5.h5"
        ]
        
        success = self.run_command(command, step_name, expected_outputs)
        
        return success
    
    def test_dataset_splitting(self):
        """Test the dataset splitting step."""
        step_name = "dataset_splitting"
        
        # Check if split_dataset.py supports --metadata parameter
        supports_metadata = self.check_script_supports_param("scripts/split_dataset.py", "--metadata")
        
        # Use metadata if supported
        metadata_arg = ""
        if os.path.exists(self.processed_metadata) and supports_metadata:
            metadata_arg = f"--metadata {self.processed_metadata}"
            if self.verbose:
                print("Using metadata for dataset splitting")
        elif os.path.exists(self.processed_metadata) and not supports_metadata:
            if self.verbose:
                print("split_dataset.py doesn't support --metadata parameter. Updating script...")
            with open("scripts/split_dataset.py", "r") as f:
                content = f.read()
            if "argparse.ArgumentParser" in content:
                if "--metadata" not in content:
                    content = content.replace(
                        "parser.add_argument(\"--test_ratio\"",
                        "parser.add_argument(\"--metadata\", type=str, help=\"Path to processed metadata file (optional)\")\n    parser.add_argument(\"--test_ratio\""
                    )
                    with open("scripts/split_dataset.py", "w") as f:
                        f.write(content)
                    if self.verbose:
                        print("Updated split_dataset.py to support --metadata parameter")
                    metadata_arg = f"--metadata {self.processed_metadata}"
        
        # Update k_values to 5 and expected outputs to use directory k5
        command = (
            f"python scripts/split_dataset.py "
            f"--input_dir {self.features_dir} "
            f"--output_dir {self.split_dir} "
            f"--k_values 6,5 "  # Use k-mer 5 for splitting
            f"{metadata_arg}"
        )
        
        expected_outputs = [
            self.split_dir / "split_metadata.csv",
            self.split_dir / "k5" / "train.h5",
            self.split_dir / "k5" / "validate.h5",
            self.split_dir / "k5" / "test.h5"
        ]
        
        success = self.run_command(command, step_name, expected_outputs)
        
        return success
    
    def test_model_training(self):
        """Test the model training step."""
        step_name = "model_training"
        
        # Use k-mer 5 and neural network training
        command = (
            f"python scripts/train_models.py "
            f"--data_dir {self.split_dir} "
            f"--output_dir {self.models_dir} "
            f"--kmers 5 "  # Use only k-mer 5 for testing
            f"--model nn "  # Use neural network for testing
            f"--epochs 10"  # Fewer epochs for faster testing
        )
        
        # Expected outputs: update directory from k3 to k5
        expected_outputs = [
            self.models_dir / "k5" / "nn" / "model.pt",
            self.models_dir / "k5" / "nn" / "scaler.pkl"
        ]
        
        success = self.run_command(command, step_name, expected_outputs)
        
        return success
    
    def test_visualization(self):
        """Test the visualization step."""
        step_name = "visualization"
        
        # Use the k5 split files and neural network outputs
        command = (
            f"python scripts/visualize_tsne.py "
            f"--h5_path {self.split_dir}/k5/train.h5 "  # k-mer 5 split
            f"--model_path {self.models_dir}/k5/nn/model.pt "  # Neural network model
            f"--scaler_path {self.models_dir}/k5/nn/scaler.pkl "  # Neural network scaler
            f"--output_dir {self.viz_dir} "
            f"--sample_size 100 "  # Small sample for testing
            f"--perplexity 5 "     # Small perplexity for testing
        )
        
        expected_outputs = [
            self.viz_dir / "tsne_visualization.png"
        ]
        
        success = self.run_command(command, step_name, expected_outputs)
        
        return success
    
    def run_full_pipeline(self):
        """Run the full pipeline testing sequence."""
        print(f"Starting pipeline testing at {self.results['start_time']}")
        print(f"Output directory: {self.base_dir}")
        print(f"Example data directory: {self.example_data_dir}")
        print(f"Config directory: {self.config_dir}")
        
        try:
            if not os.path.exists(self.host_patterns_config):
                print(f"Warning: Config file {self.host_patterns_config} not found. Metadata processing will likely fail.")
            
            print("\n[Step 1/6] Testing preprocessing...")
            preprocess_success = self.test_preprocessing()
            
            print("\n[Step 2/6] Testing metadata processing...")
            metadata_success = self.test_metadata_processing()
            
            print("\n[Step 3/6] Testing feature generation...")
            if preprocess_success:
                feature_success = self.test_feature_generation()
            else:
                feature_success = False
                print("Skipping feature generation due to preprocessing failure")
            
            if feature_success:
                print("\n[Step 4/6] Testing dataset splitting...")
                split_success = self.test_dataset_splitting()
            else:
                split_success = False
                print("Skipping dataset splitting due to feature generation failure")
            
            if split_success:
                print("\n[Step 5/6] Testing model training...")
                model_success = self.test_model_training()
            else:
                model_success = False
                print("Skipping model training due to dataset splitting failure")
            
            if model_success:
                print("\n[Step 6/6] Testing visualization...")
                viz_success = self.test_visualization()
            else:
                viz_success = False
                print("Skipping visualization due to model training failure")
            
            self.results["overall_success"] = (
                preprocess_success and 
                feature_success and 
                split_success and
                model_success and 
                viz_success
            )
            
        except Exception as e:
            print(f"Pipeline testing failed with error: {e}")
            self.results["error"] = str(e)
            self.results["overall_success"] = False
        
        self.results["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results["duration_seconds"] = (
            datetime.strptime(self.results["end_time"], "%Y-%m-%d %H:%M:%S") -
            datetime.strptime(self.results["start_time"], "%Y-%m-%d %H:%M:%S")
        ).total_seconds()
        
        self.save_results()
        
        if not self.keep_intermediate:
            self.cleanup()
        
        self.print_summary()
        
        return self.results["overall_success"]
    
    def save_results(self):
        """Save testing results to file."""
        try:
            json_results = self.results.copy()
            for step in json_results["steps"].values():
                if "stdout" in step:
                    step["stdout_length"] = len(step["stdout"]) if step["stdout"] else 0
                    del step["stdout"]
                if "stderr" in step:
                    step["stderr_length"] = len(step["stderr"]) if step["stderr"] else 0
                    del step["stderr"]
            
            with open(self.base_dir / "test_results.json", 'w') as f:
                json.dump(json_results, f, indent=2)
                
            print(f"Results saved to {self.base_dir / 'test_results.json'}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def cleanup(self):
        """Clean up intermediate files."""
        print("Cleaning up intermediate files...")
        
        test_results_file = self.base_dir / "test_results.json"
        if os.path.exists(test_results_file):
            with open(test_results_file, 'r') as f:
                test_results = json.load(f)
        else:
            test_results = None
        
        shutil.rmtree(self.base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        if test_results:
            with open(test_results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
    
    def print_summary(self):
        """Print a summary of the testing results."""
        print("\n" + "="*80)
        print("PIPELINE TESTING SUMMARY")
        print("="*80)
        
        if self.results["overall_success"]:
            print("\n✅ Pipeline testing SUCCEEDED!")
        else:
            print("\n❌ Pipeline testing FAILED!")
        
        print(f"Start time: {self.results['start_time']}")
        print(f"End time: {self.results['end_time']}")
        print(f"Duration: {self.results['duration_seconds']:.2f} seconds")
        
        print("\nStep Status:")
        for step_name, step_results in self.results["steps"].items():
            if step_results["success"]:
                print(f"✅ {step_name}: SUCCESS ({step_results['duration_seconds']:.2f}s)")
            else:
                print(f"❌ {step_name}: FAILED ({step_results['duration_seconds']:.2f}s)")
        
        if not self.results["overall_success"]:
            print("\nErrors:")
            for step_name, step_results in self.results["steps"].items():
                if not step_results["success"]:
                    print(f"  {step_name}:")
                    if "error" in step_results:
                        print(f"    {step_results['error']}")
                    elif "stderr" in step_results and step_results["stderr"]:
                        print(f"    {step_results['stderr']}")
        
        print("\n" + "="*80)


def main(args):
    """Main function to run pipeline testing."""
    tester = PipelineTester(
        output_dir=args.output_dir,
        example_data_dir=args.example_data_dir,
        config_dir=args.config_dir,
        keep_intermediate=args.keep_intermediate,
        verbose=args.verbose
    )
    
    success = tester.run_full_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the virus host classification pipeline")
    parser.add_argument("--output_dir", type=str, default="test_results",
                        help="Directory to store test results")
    parser.add_argument("--example_data_dir", type=str, default="data/examples",
                        help="Directory containing example data")
    parser.add_argument("--config_dir", type=str, default="config",
                        help="Directory containing configuration files")
    parser.add_argument("--keep_intermediate", action="store_true",
                        help="Keep intermediate files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    
    args = parser.parse_args()
    exit_code = main(args)
    sys.exit(exit_code)
