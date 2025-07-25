# Android UI Analyzer 🔍

An intelligent Android UI flow analyzer powered by Google's Gemini 2.0 Flash AI model. This tool automatically analyzes Android app screen recordings to understand user interactions, UI patterns, and navigation flows.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-red.svg)

## ✨ Features

- 🎬 **Video Analysis**: Process MP4, AVI, MOV, and MKV screen recordings
- 🧠 **AI-Powered**: Uses Google's Gemini 2.0 Flash for multimodal analysis
- 📱 **UI Pattern Recognition**: Identifies interaction patterns like login flows, form filling, navigation
- 🔄 **Screen Transition Mapping**: Tracks navigation between different app screens
- 👆 **Action Detection**: Recognizes user actions (tap, swipe, scroll, type)
- 📊 **Detailed JSON Output**: Generates comprehensive analysis reports
- ⚡ **Intelligent Frame Selection**: Optimizes frame extraction based on video duration

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))
- Android screen recording (MP4 format recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/android-ui-analyzer.git
   cd android-ui-analyzer
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv ui_analyzer_env
   source ui_analyzer_env/bin/activate  # On Windows: ui_analyzer_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```

5. **Prepare your video**:
   ```bash
   mkdir test_data
   # Place your Android screen recording in the test_data folder
   ```

### Usage

```bash
python ui_analyzer.py
```

The analyzer will:
1. 🔍 Find video files in the `test_data` folder
2. 📸 Extract key frames using intelligent sampling
3. 🧠 Analyze UI flow with Gemini 2.0 Flash
4. 📊 Generate a detailed JSON report

## 📊 Output Format

The analyzer generates a comprehensive JSON report with the following structure:

```json
{
  "video_analysis": {
    "video_path": "test_data/your_video.mp4",
    "duration_seconds": 127.76,
    "frames_analyzed": 80
  },
  "ui_flow_analysis": {
    "flow_summary": "The user navigates through the app to create a new email account...",
    "interaction_patterns": [
      "Account Creation",
      "Form Filling", 
      "CAPTCHA Interaction"
    ],
    "screen_transitions": [
      "Inbox",
      "Navigation Drawer",
      "Settings",
      "Add Account",
      "Email Entry",
      "Password Entry"
    ],
    "user_actions": [
      {
        "timestamp": 0.0,
        "action_type": "tap",
        "target": "Menu Button",
        "description": "User taps on the menu button to open navigation drawer"
      }
    ]
  },
  "summary": {
    "total_actions_identified": 16,
    "total_patterns_found": 3,
    "total_screen_transitions": 12,
    "analysis_quality": "focused_and_accurate"
  }
}
```

## 🎯 Use Cases

- **📱 Mobile App Testing**: Automate UI flow documentation
- **🔄 User Journey Analysis**: Understand user behavior patterns
- **📋 QA Documentation**: Generate test case documentation
- **🎨 UX Research**: Analyze interaction patterns and usability
- **🤖 Test Automation**: Create automated test scripts from recordings
- **📊 App Analytics**: Track user flows and identify bottlenecks
