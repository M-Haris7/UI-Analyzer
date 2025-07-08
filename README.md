# Android UI Analyzer

An AI-powered tool for analyzing Android app screen recordings and screenshots to understand user interaction patterns, UI flows, and interface elements using OpenAI's GPT-4o and computer vision.

## 🚀 Features

- **Video Analysis**: Analyze Android screen recordings (MP4, AVI, MOV, MKV)
- **Smart Frame Extraction**: Dynamic frame selection based on video duration
- **UI Flow Detection**: Identify user interaction patterns and navigation flows
- **Screen Transition Analysis**: Track how users move between app screens
- **Action Recognition**: Detect taps, swipes, scrolls, and text input
- **Multi-language Support**: Analyze apps in different languages (e.g., Spanish) with translation
- **Structured Output**: Generate JSON reports with detailed analysis

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Video files of Android screen recordings

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd android-ui-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Create test data directory**
```bash
mkdir test_data
```

5. **Add your video files**
Place your Android screen recording videos in the `test_data` folder.

## 🎯 Usage

### Basic Video Analysis

```bash
python android_ui_analyzer.py
```

The script will automatically:
- Find video files in the `test_data` directory
- Extract key frames using dynamic selection
- Analyze UI flow and interaction patterns
- Generate a comprehensive analysis report


## 🔍 Analysis Capabilities

### UI Flow Detection
- **Navigation patterns**: How users move between screens
- **Interaction sequences**: Order and timing of user actions
- **Screen transitions**: What triggers each screen change
- **User journey mapping**: Complete flow from start to finish

### Action Recognition
- **Tap actions**: Button clicks, field selections
- **Swipe gestures**: Screen navigation, scrolling
- **Text input**: Form filling, search queries

### Smart Frame Extraction
The analyzer uses duration-based frame selection:
- **Short videos (0-5s)**: 6 frames
- **Medium videos (5-20s)**: 8-10 frames
- **Long videos (20-60s)**: 20-35 frames
- **Extended videos (60s+)**: Up to 80 frames

## 🎛️ Configuration

### Model Selection
Change the OpenAI model in the analyzer initialization:
```python
analyzer = AndroidUIAnalyzer(
    openai_api_key="your_key",
    model_name="gpt-4o"  # or "gpt-4-turbo", "gpt-4"
)
```

### Analysis Parameters
Adjust analysis focus by modifying the prompts in the `AndroidUIAnalyzer` class:
- `ui_analysis_prompt`: For screenshot analysis
- `flow_analysis_prompt`: For interaction flow detection

## 📁 Project Structure

```
android-ui-analyzer/
├── android_ui_analyzer.py    # Main analyzer class
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .env                     # Environment variables (create this)
├── test_data/               # Video files directory (create this)
│   └── your_video.mp4
└── video_analysis_results.json  # Generated analysis output
```

## 🔧 Advanced Features

### Custom Frame Extraction
```python
# Extract specific number of frames
frames, timestamps = analyzer.extract_key_frames_from_video("video.mp4")

# Analyze with custom frame selection
flow_analysis = analyzer.analyze_video_flow_directly(frames, timestamps)
```

### Batch Processing
```python
# Process multiple videos
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = []

for video_path in video_paths:
    result = analyzer.process_screen_recording(video_path)
    results.append(result)
```



### Performance Tips

- **Optimize video quality**: Use compressed videos for faster processing
- **Batch processing**: Process multiple short videos rather than one long video
- **Frame selection**: The dynamic frame selection already optimizes for accuracy vs. speed
