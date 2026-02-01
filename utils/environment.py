"""
Environment validation and setup utilities.
Provides functions to check environment configuration and dependencies.
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnvironmentCheck:
    """Result of an environment check."""
    name: str
    passed: bool
    message: str
    required: bool = True


class EnvironmentValidator:
    """Validates the environment for the RAG system."""

    def __init__(self):
        """Initialize the validator."""
        self.checks: List[EnvironmentCheck] = []

    def check_python_version(self) -> EnvironmentCheck:
        """Check Python version."""
        version = sys.version_info
        passed = version.major >= 3 and version.minor >= 9
        message = f"Python {version.major}.{version.minor}.{version.micro} - OK" if passed else f"Python 3.9+ required, got {version.major}.{version.minor}"
        return EnvironmentCheck(
            name="Python Version",
            passed=passed,
            message=message,
            required=True
        )

    def check_openai_api_key(self) -> EnvironmentCheck:
        """Check if OpenAI API key is available."""
        api_key = os.getenv("OPENAI_API_KEY")
        passed = api_key is not None and len(api_key) > 0
        message = "OpenAI API key configured" if passed else "OpenAI API key not found (set OPENAI_API_KEY)"
        return EnvironmentCheck(
            name="OpenAI API Key",
            passed=passed,
            message=message,
            required=True
        )

    def check_hf_token(self) -> EnvironmentCheck:
        """Check if HuggingFace token is available."""
        token = os.getenv("HF_TOKEN")
        passed = token is not None and len(token) > 0
        message = "HuggingFace token configured" if passed else "HuggingFace token not found (set HF_TOKEN) - optional"
        return EnvironmentCheck(
            name="HuggingFace Token",
            passed=passed,
            message=message,
            required=False
        )

    def check_data_directories(self) -> EnvironmentCheck:
        """Check if required data directories exist."""
        dirs = ["data", "models", "data/faiss_index"]
        missing = []
        for d in dirs:
            if not os.path.exists(d):
                missing.append(d)

        passed = len(missing) == 0
        if not passed:
            os.makedirs("data", exist_ok=True)
            os.makedirs("data/faiss_index", exist_ok=True)
            passed = True

        message = "Data directories ready" if passed else f"Missing directories: {missing}"
        return EnvironmentCheck(
            name="Data Directories",
            passed=passed,
            message=message,
            required=True
        )

    def check_dependencies(self) -> EnvironmentCheck:
        """Check if required packages are installed."""
        required_packages = [
            "langchain",
            "langchain_openai",
            "langchain_community",
            "faiss",
            "sentence_transformers",
            "datasets",
            "transformers",
            "numpy",
            "python_dotenv",
        ]

        missing = []
        for package in required_packages:
            package_name = package.replace("-", "_")
            try:
                __import__(package_name)
            except ImportError:
                missing.append(package)

        passed = len(missing) == 0
        message = "All dependencies installed" if passed else f"Missing packages: {', '.join(missing)}"
        return EnvironmentCheck(
            name="Dependencies",
            passed=passed,
            message=message,
            required=True
        )

    def run_all_checks(self) -> Tuple[bool, List[EnvironmentCheck]]:
        """Run all environment checks."""
        checks = [
            self.check_python_version(),
            self.check_openai_api_key(),
            self.check_hf_token(),
            self.check_data_directories(),
            self.check_dependencies(),
        ]

        self.checks = checks
        all_passed = all(
            check.passed or not check.required
            for check in checks
        )

        return all_passed, checks

    def print_report(self):
        """Print a formatted environment report."""
        print("=" * 60)
        print("Environment Validation Report")
        print("=" * 60)

        all_passed, checks = self.run_all_checks()

        for check in checks:
            status = "✓" if check.passed else "✗"
            required = "(required)" if check.required else "(optional)"
            print(f"  {status} {check.name} {required}")
            print(f"    {check.message}")

        print("=" * 60)
        if all_passed:
            print("✓ Environment is ready for RAG system")
        else:
            print("✗ Some checks failed. Please resolve the issues above.")
        print("=" * 60)


def setup_environment():
    """Setup the environment for the RAG system."""
    print("Setting up environment...")
    validator = EnvironmentValidator()
    validator.print_report()
    return validator.run_all_checks()


def get_required_env_vars() -> Dict[str, str]:
    """Get list of required environment variables."""
    return {
        "OPENAI_API_KEY": "OpenAI API key for the LLM",
        "HF_TOKEN": "HuggingFace token for dataset access (optional)",
    }