#!/usr/bin/env python3
"""
Verification script to check if the system is properly set up.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

def check_imports():
    """Check if all imports are working correctly."""
    print("Checking imports...")

    try:
        # Check providers
        print("  ✓ Checking providers...")
        from providers.config import get_config
        from providers.qdrant import QdrantProvider
        from providers.embeddings import EmbeddingProvider
        from providers.database import DatabaseProvider
        from providers.llm import LLMProvider
        print("    ✅ Providers imports successful")
    except ImportError as e:
        print(f"    ❌ Providers import failed: {e}")
        return False

    try:
        # Check retrieval modules
        print("  ✓ Checking retrieval modules...")
        from retrieval.sql_rag.sql_retrieval import SQLRetrieval
        from retrieval.sql_rag.prompts.database_manager import get_sql_generation_prompt
        from retrieval.doc_rag.doc_retrieval import DocRetrieval
        print("    ✅ Retrieval modules imports successful")
    except ImportError as e:
        print(f"    ❌ Retrieval modules import failed: {e}")
        return False

    try:
        # Check unified retrieval
        print("  ✓ Checking unified retrieval...")
        from retrieval.core.unified_retrieval import UnifiedRetrievalSystem
        print("    ✅ Unified retrieval import successful")
    except ImportError as e:
        print(f"    ❌ Unified retrieval import failed: {e}")
        return False

    try:
        # Check common modules
        print("  ✓ Checking common modules...")
        from common.logger import setup_logger, get_logger
        print("    ✅ Common modules imports successful")
    except ImportError as e:
        print(f"    ❌ Common modules import failed: {e}")
        return False

    return True

def check_files():
    """Check if all required files exist."""
    print("\nChecking required files...")

    required_files = [
        "providers/config.toml",
        "providers/.env.dev",
        "retrieval/sql_rag/prompts/database_manager.py",
        "common/logger.py"
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} not found")
            all_exist = False

    return all_exist

def check_test_files():
    """Check if test files are properly located."""
    print("\nChecking test files...")

    test_files = [
        "tests/run_simple_test.py",
        "tests/test_retrieval_system.py",
        "tests/test_system_validation.py"
    ]

    all_exist = True
    for file_path in test_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} not found")
            all_exist = False

    return all_exist

def main():
    """Main verification function."""
    print("="*60)
    print("DBGpt Testing System - Setup Verification")
    print("="*60)

    # Check imports
    imports_ok = check_imports()

    # Check files
    files_ok = check_files()

    # Check test files
    tests_ok = check_test_files()

    print("\n" + "="*60)
    if imports_ok and files_ok and tests_ok:
        print("✅ System setup is complete and verified!")
        print("\nTo run the tests:")
        print("  cd tests")
        print("  python3 run_simple_test.py")
        print("  python3 test_retrieval_system.py")
    else:
        print("❌ Setup verification failed. Please address the issues above.")

        if not imports_ok:
            print("\n⚠️ Missing dependencies. Install them with:")
            print("  pip install -r requirements.txt")

        if not files_ok:
            print("\n⚠️ Missing configuration files. Ensure you have:")
            print("  - providers/config.toml")
            print("  - providers/.env.dev")

    print("="*60)

if __name__ == "__main__":
    main()