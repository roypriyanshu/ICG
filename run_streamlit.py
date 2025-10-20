#!/usr/bin/env python3
"""
Launcher script for Streamlit app with proper setup
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        # Try to install basic requirements first
        basic_packages = [
            "streamlit",
            "torch",
            "transformers", 
            "pillow",
            "pandas",
            "plotly",
            "requests"
        ]
        
        for package in basic_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✓ {package} already installed")
            except ImportError:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        return True
    except Exception as e:
        print(f"Installation error: {e}")
        return False

def check_streamlit():
    """Check if streamlit is available"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def main():
    """Main launcher function"""
    print("🖼️ Image Captioning System Launcher")
    print("=" * 40)
    
    # Check and install requirements
    if not check_streamlit():
        print("Installing required packages...")
        if not install_requirements():
            print("❌ Failed to install requirements")
            return
    
    # Check if streamlit_app.py exists
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py not found in current directory")
        return
    
    print("✅ All requirements satisfied")
    print("🚀 Starting Streamlit app...")
    print("\nThe app will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("-" * 40)
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    main()