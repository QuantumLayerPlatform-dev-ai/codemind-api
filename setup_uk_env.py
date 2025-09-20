#!/usr/bin/env python3
"""
Setup script for CodeMind UK environment
Helps configure credentials and test connectivity
"""

import os
import subprocess
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists and prompt for credentials"""
    env_file = Path(".env")

    if not env_file.exists():
        print("❌ .env file not found!")
        print("📝 Please copy .env.template to .env and fill in your credentials")
        return False

    # Check if template values are still present
    with open(env_file, 'r') as f:
        content = f.read()

    if "your_aws_access_key_here" in content or "your_azure_openai_api_key_here" in content:
        print("⚠️ .env file contains template values!")
        print("📝 Please edit .env and replace template values with your actual credentials")
        print("\nRequired credentials:")
        print("1. AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (from AWS Console)")
        print("2. AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY (from Azure Portal)")
        return False

    print("✅ .env file looks configured")
    return True

def setup_port_forwarding():
    """Setup port forwarding for K8s services"""
    print("🔌 Setting up port forwarding for K8s services...")

    port_forwards = [
        ("postgresql", "5432:5432"),
        ("redis-master", "6379:6379"),
        ("qdrant-simple", "6333:6333"),
        ("nats", "4222:4222"),
        ("temporal-frontend", "7233:7233"),
    ]

    print("\nTo access your K8s services, run these commands in separate terminals:")
    print("(These will run in the background)")

    for service, ports in port_forwards:
        cmd = f"kubectl port-forward svc/{service} {ports} -n codemind"
        print(f"  {cmd}")

    response = input("\nWould you like me to start port forwarding automatically? (y/n): ")

    if response.lower() == 'y':
        print("🚀 Starting port forwarding...")

        for service, ports in port_forwards:
            cmd = ["kubectl", "port-forward", f"svc/{service}", ports, "-n", "codemind"]
            try:
                # Start in background
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"  ✅ {service} -> localhost:{ports.split(':')[0]}")
            except Exception as e:
                print(f"  ❌ Failed to start {service}: {e}")

        print("\n🔥 Port forwarding started! Services should be accessible on localhost")
        print("💡 Tip: Use 'kubectl get pods -n codemind' to check service status")

    return True

def validate_uk_region():
    """Validate UK region configuration"""
    print("🇬🇧 Validating UK region configuration...")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    aws_region = os.getenv('AWS_REGION', '')
    azure_region = os.getenv('AZURE_REGION', '')

    if aws_region != 'eu-west-2':
        print(f"⚠️ AWS_REGION is '{aws_region}', should be 'eu-west-2' for London")
        return False

    if azure_region != 'uksouth':
        print(f"⚠️ AZURE_REGION is '{azure_region}', should be 'uksouth' for UK")
        return False

    print("✅ UK regions configured correctly")
    print(f"  AWS: {aws_region} (London)")
    print(f"  Azure: {azure_region} (UK South)")

    return True

def check_credentials():
    """Check if credentials are properly set"""
    print("🔑 Checking credentials...")

    from dotenv import load_dotenv
    load_dotenv()

    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_key = os.getenv('AZURE_OPENAI_API_KEY')

    issues = []

    if not aws_key or len(aws_key) < 16:
        issues.append("AWS_ACCESS_KEY_ID missing or too short")

    if not aws_secret or len(aws_secret) < 32:
        issues.append("AWS_SECRET_ACCESS_KEY missing or too short")

    if not azure_endpoint or 'openai.azure.com' not in azure_endpoint:
        issues.append("AZURE_OPENAI_ENDPOINT missing or invalid")

    if not azure_key or len(azure_key) < 32:
        issues.append("AZURE_OPENAI_API_KEY missing or too short")

    if issues:
        print("❌ Credential issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("✅ All credentials look properly configured")
    return True

def main():
    """Main setup function"""
    print("🇬🇧 CodeMind UK Environment Setup")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Please run this script from the /apps/api directory")
        sys.exit(1)

    success = True

    # Step 1: Check environment file
    success &= check_env_file()

    if not success:
        print("\n❌ Please fix the .env configuration and run again")
        sys.exit(1)

    # Step 2: Validate UK regions
    success &= validate_uk_region()

    # Step 3: Check credentials
    success &= check_credentials()

    # Step 4: Setup port forwarding
    setup_port_forwarding()

    if success:
        print("\n🎉 Environment setup complete!")
        print("\nNext steps:")
        print("1. Wait 10 seconds for port forwarding to establish")
        print("2. Run: python3 simple_test.py")
        print("3. If tests pass, start the API: python3 main.py")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")

if __name__ == "__main__":
    main()