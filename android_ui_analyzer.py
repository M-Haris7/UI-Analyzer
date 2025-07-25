import cv2
import base64
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io
import re
from datetime import datetime

load_dotenv()

# Data models for structured output
@dataclass
class UIElement:
    """Represents a detected UI element"""
    element_type: str
    coordinates: Tuple[int, int, int, int]  # x1, y1, x2, y2
    text_content: str
    confidence: float
    properties: Dict[str, any]

class UserAction(BaseModel):
    """Represents a user action in the recording"""
    timestamp: float = Field(description="Time in seconds when action occurred")
    action_type: str = Field(description="Type of action (tap, swipe, scroll, type)")
    target: str = Field(description="Description of target UI element")
    description: str = Field(description="Human-readable action description")

class VideoAnalysisResults(BaseModel):
    """Complete video analysis results matching the expected format"""
    video_analysis: Dict = Field(description="Video metadata and analysis info")
    ui_flow_analysis: Dict = Field(description="Detailed UI flow analysis")
    summary: Dict = Field(description="Analysis summary statistics")

class AndroidUIAnalyzer:
    """Main class for analyzing Android UI recordings and screenshots"""
    
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-2.5-pro"):
        """
        Initialize with Gemini 2.0 Flash
        Available models:
        - gemini-2.0-flash-exp (latest experimental model)
        - gemini-1.5-flash (stable version)
        - gemini-1.5-pro (more capable, multimodal)
        """
        try:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            print(f"✅ Initialized with {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to initialize {model_name}, falling back to gemini-1.5-flash")
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self.model_name = "gemini-1.5-flash"
        
        # Generation config for consistent responses
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=8000,  # Increased for detailed responses
            candidate_count=1
        )

    def numpy_to_pil(self, frame: np.ndarray) -> Image.Image:
        """Convert numpy frame to PIL Image"""
        # Convert BGR to RGB (OpenCV uses BGR, PIL expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def calculate_optimal_frames(self, duration: float) -> int:
        """Calculate optimal number of frames based on video duration"""
        if duration <= 5:          # Very short: 0-5 seconds
            return 8
        elif duration <= 10:       # Short: 5-10 seconds  
            return 12
        elif duration <= 20:       # Medium: 10-20 seconds
            return 20
        elif duration <= 30:       # Medium-long: 20-30 seconds
            return 30
        elif duration <= 60:       # Long: 30-60 seconds (1 minute)
            return 50
        elif duration <= 120:      # Very long: 1-2 minutes
            return 80
        else:                      # Extra long: 2+ minutes
            return 100

    def extract_key_frames_from_video(self, video_path: str) -> Tuple[List[np.ndarray], List[float]]:
        """Extract key frames from video for multimodal analysis"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / frame_rate
        
        # Calculate optimal frame count dynamically
        optimal_frames = self.calculate_optimal_frames(duration)
        
        print(f"📊 Video: {duration:.1f}s → Selecting {optimal_frames} frames")
        
        # Intelligent frame selection for better analysis
        if duration <= 10:
            # For short videos, sample evenly
            frame_indices = [int(i * total_frames / optimal_frames) for i in range(optimal_frames)]
        elif duration <= 60:
            # For medium videos, focus on key moments
            beginning_frames = int(optimal_frames * 0.3)
            middle_frames = int(optimal_frames * 0.4)
            end_frames = optimal_frames - beginning_frames - middle_frames
            
            start_indices = [int(i * total_frames * 0.25 / beginning_frames) for i in range(beginning_frames)]
            middle_indices = [int(total_frames * 0.25 + i * total_frames * 0.5 / middle_frames) for i in range(middle_frames)]
            end_indices = [int(total_frames * 0.75 + i * total_frames * 0.25 / end_frames) for i in range(end_frames)]
            
            frame_indices = start_indices + middle_indices + end_indices
        else:
            # For long videos, use segmented approach
            segments = min(8, int(duration / 15))
            frames_per_segment = optimal_frames // segments
            remaining_frames = optimal_frames % segments
            
            frame_indices = []
            for segment in range(segments):
                segment_start = int(segment * total_frames / segments)
                segment_end = int((segment + 1) * total_frames / segments)
                segment_frames = frames_per_segment + (1 if segment < remaining_frames else 0)
                
                for i in range(segment_frames):
                    frame_idx = segment_start + int(i * (segment_end - segment_start) / segment_frames)
                    frame_indices.append(frame_idx)
        
        frame_indices = sorted(list(set(frame_indices)))
        
        # Extract frames
        key_frames = []
        timestamps = []
        
        print(f"🎯 Frame selection strategy: {len(frame_indices)} frames")
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                key_frames.append(frame)
                timestamp = frame_idx / frame_rate
                timestamps.append(timestamp)
                if i < 5 or i >= len(frame_indices) - 2:
                    print(f"   📸 Frame {i+1}: {timestamp:.1f}s")
                elif i == 5:
                    print(f"   📸 ... (extracting {len(frame_indices)-7} more frames) ...")
        
        cap.release()
        print(f"✅ Successfully extracted {len(key_frames)} frames")
        return key_frames, timestamps

    def create_detailed_analysis_prompt(self, frames_count: int, timestamps: List[float]) -> str:
        """Create a comprehensive prompt for detailed analysis"""
        return f"""You are an expert Android UI/UX analyst with deep experience in mobile app flow analysis. Analyze this sequence of {frames_count} screenshots from an Android app recording.

**VIDEO DETAILS:**
- Total frames: {frames_count}
- Duration: {timestamps[-1] - timestamps[0]:.1f} seconds  
- Timestamps: {[f"{t:.1f}s" for t in timestamps[:10]]}{'...' if len(timestamps) > 10 else ''}

**ANALYSIS REQUIREMENTS:**

Provide a comprehensive analysis in the EXACT JSON format below. Be extremely detailed and accurate:

```json
{{
  "video_analysis": {{
    "video_path": "test_data/video_name.mp4",
    "duration_seconds": {timestamps[-1] - timestamps[0]:.1f},
    "frames_analyzed": {frames_count}
  }},
  "ui_flow_analysis": {{
    "flow_summary": "Detailed 2-3 sentence summary of what the user accomplishes in this video",
    "interaction_patterns": [
      "Pattern1 (e.g., Account Creation, Form Filling, Navigation)",
      "Pattern2", 
      "Pattern3"
    ],
    "screen_transitions": [
      "Screen1 (starting screen)",
      "Screen2",
      "Screen3",
      "etc..."
    ],
    "user_actions": [
      {{
        "timestamp": 0.0,
        "action_type": "tap|swipe|scroll|type|long_press",
        "target": "Specific UI element name",
        "description": "Detailed description of what the user does"
      }}
    ]
  }},
  "summary": {{
    "total_actions_identified": 0,
    "total_patterns_found": 0,
    "total_screen_transitions": 0,
    "analysis_quality": "focused_and_accurate|comprehensive|detailed"
  }}
}}
```

**DETAILED ANALYSIS INSTRUCTIONS:**

1. **FLOW SUMMARY**: Write a clear, detailed summary explaining the overall user journey and what they accomplish.

2. **INTERACTION PATTERNS**: Identify high-level patterns like:
   - Account Creation, Login Flow, Form Filling
   - Navigation, Search, Content Browsing  
   - Settings Configuration, Profile Setup
   - CAPTCHA Interaction, Security Setup
   - Shopping Flow, Payment Process

3. **SCREEN TRANSITIONS**: List each distinct screen/view in chronological order:
   - Start with the initial screen shown
   - Include intermediate screens/dialogs
   - End with the final screen reached
   - Use clear, descriptive names

4. **USER ACTIONS**: For each user interaction, provide:
   - **timestamp**: Estimated time when action occurs (based on frame timing)
   - **action_type**: tap, swipe, scroll, type, long_press
   - **target**: Specific UI element (button name, field name, menu item)
   - **description**: Clear explanation of what the user does

**IMPORTANT NOTES:**
- The app content may be in Spanish - translate UI elements to English for clarity
- Focus on user intent and actions, not just screen content
- Estimate timestamps based on frame sequence and timing
- Be specific about UI elements (button names, field labels, etc.)
- Identify CAPTCHAs, form validations, error states
- Note any security or verification steps

Return ONLY the JSON response with no additional text or formatting."""

    def clean_and_parse_json_response(self, response_text: str) -> Dict:
        """Clean and parse JSON response from Gemini"""
        try:
            # Remove any markdown formatting
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Try to find JSON content between braces
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                return json.loads(json_content)
            else:
                # Try parsing the entire response
                return json.loads(response_text)
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            print(f"Response preview: {response_text[:200]}...")
            return self.create_fallback_analysis(response_text)

    def create_fallback_analysis(self, response_text: str) -> Dict:
        """Create a fallback analysis structure when JSON parsing fails"""
        return {
            "video_analysis": {
                "video_path": "test_data/analyzed_video.mp4",
                "duration_seconds": 0.0,
                "frames_analyzed": 0
            },
            "ui_flow_analysis": {
                "flow_summary": "Analysis parsing failed. Raw response captured for review.",
                "interaction_patterns": ["Parsing Error"],
                "screen_transitions": ["Unknown"],
                "user_actions": [
                    {
                        "timestamp": 0.0,
                        "action_type": "unknown",
                        "target": "Unknown",
                        "description": f"Raw response: {response_text[:200]}..."
                    }
                ]
            },
            "summary": {
                "total_actions_identified": 0,
                "total_patterns_found": 1,
                "total_screen_transitions": 1,
                "analysis_quality": "parsing_failed"
            }
        }

    def analyze_video_flow_with_gemini(self, frames: List[np.ndarray], timestamps: List[float], video_path: str) -> Dict:
        """Analyze video frames with Gemini 2.0 Flash for detailed JSON output"""
        
        print(f"🧠 Analyzing UI flow using {self.model_name}...")
        
        # Convert frames to PIL Images
        pil_images = [self.numpy_to_pil(frame) for frame in frames]
        
        # Create detailed analysis prompt
        prompt = self.create_detailed_analysis_prompt(len(frames), timestamps)
        
        try:
            # Create content with text and images
            content = [prompt] + pil_images
            
            print("🔍 Sending frames to Gemini for analysis...")
            
            # Generate response
            response = self.model.generate_content(
                content,
                generation_config=self.generation_config
            )
            
            # Parse the response
            response_text = response.text
            print("📝 Parsing Gemini response...")
            
            analysis_result = self.clean_and_parse_json_response(response_text)
            
            # Update video path with actual path
            if "video_analysis" in analysis_result:
                analysis_result["video_analysis"]["video_path"] = video_path
                analysis_result["video_analysis"]["duration_seconds"] = timestamps[-1] - timestamps[0]
                analysis_result["video_analysis"]["frames_analyzed"] = len(frames)
            
            # Ensure summary statistics are correct
            if "ui_flow_analysis" in analysis_result and "summary" in analysis_result:
                ui_analysis = analysis_result["ui_flow_analysis"]
                analysis_result["summary"]["total_actions_identified"] = len(ui_analysis.get("user_actions", []))
                analysis_result["summary"]["total_patterns_found"] = len(ui_analysis.get("interaction_patterns", []))
                analysis_result["summary"]["total_screen_transitions"] = len(ui_analysis.get("screen_transitions", []))
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Error during Gemini analysis: {e}")
            return self.create_fallback_analysis(f"Analysis failed: {str(e)}")

    def process_screen_recording(self, video_path: str) -> Dict:
        """Process video recording using Gemini 2.0 Flash with detailed output"""
        
        print("🎬 Starting comprehensive video analysis with Gemini 2.0 Flash...")
        
        # Extract key frames
        key_frames, timestamps = self.extract_key_frames_from_video(video_path)
        
        # Analyze UI flow with detailed JSON output
        print("🔍 Performing detailed UI flow analysis...")
        detailed_analysis = self.analyze_video_flow_with_gemini(key_frames, timestamps, video_path)
        
        return detailed_analysis

def main():
    print("🔍 Android UI Analyzer - Gemini 2.0 Flash Version")
    print("=" * 55)
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API key found. Please set GEMINI_API_KEY in your .env file.")
        print("💡 Get your free API key at: https://aistudio.google.com/app/apikey")
        return
    
    # Look for video files in test_data folder
    test_data_dir = Path("test_data")
    if not test_data_dir.exists():
        print("❌ test_data folder not found. Please create it and add your MP4 file.")
        return
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(test_data_dir.glob(f"*{ext}"))
        video_files.extend(test_data_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print("❌ No video files found in test_data folder.")
        print("Supported formats: MP4, AVI, MOV, MKV")
        return
    
    # Use the first video file found
    video_path = str(video_files[0])
    print(f"📹 Found video: {video_path}")
    
    try:
        # Initialize the analyzer with Gemini 2.0 Flash
        print("🚀 Initializing Android UI Analyzer with Gemini 2.0 Flash...")
        analyzer = AndroidUIAnalyzer(gemini_api_key=api_key)
        print("✅ Analyzer initialized successfully!")
        print(f"🎯 Using {analyzer.model_name} for comprehensive image analysis")
        
        # Process the video
        print(f"\n🔄 Analyzing video: {video_path}")
        results = analyzer.process_screen_recording(video_path)
        
        print("\n🎉 ANALYSIS COMPLETED!")
        print("=" * 55)
        
        # Display essential results
        video_analysis = results.get("video_analysis", {})
        ui_analysis = results.get("ui_flow_analysis", {})
        summary = results.get("summary", {})
        
        print(f"📱 Video Duration: {video_analysis.get('duration_seconds', 0):.1f} seconds")
        print(f"🔍 Frames Analyzed: {video_analysis.get('frames_analyzed', 0)}")
        print(f"👆 Actions Identified: {summary.get('total_actions_identified', 0)}")
        print(f"🔄 Screen Transitions: {summary.get('total_screen_transitions', 0)}")
        print(f"📋 Patterns Found: {summary.get('total_patterns_found', 0)}")
        
        # Print detailed flow analysis
        print(f"\n📋 FLOW SUMMARY:")
        print(f"   {ui_analysis.get('flow_summary', 'No summary available')}")
        
        patterns = ui_analysis.get('interaction_patterns', [])
        if patterns:
            print(f"\n🔍 INTERACTION PATTERNS:")
            for i, pattern in enumerate(patterns, 1):
                print(f"   {i}. {pattern}")
        
        transitions = ui_analysis.get('screen_transitions', [])
        if transitions:
            print(f"\n🔄 SCREEN TRANSITIONS:")
            for i, transition in enumerate(transitions, 1):
                print(f"   {i}. {transition}")
        
        actions = ui_analysis.get('user_actions', [])
        if actions:
            print(f"\n👆 USER ACTIONS:")
            for i, action in enumerate(actions, 1):
                timestamp = action.get('timestamp', 0)
                action_type = action.get('action_type', 'unknown')
                description = action.get('description', 'No description')
                print(f"   {i}. [{timestamp:.1f}s] {action_type}: {description}")
        
        # Save detailed results
        output_file = "gemini_2_detailed_analysis_results.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        print(f"✨ Analysis completed using {analyzer.model_name}")
        print("🎯 Generated comprehensive JSON matching your required format!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
