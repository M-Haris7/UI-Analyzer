import cv2
import base64
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from langchain.schema import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
from pathlib import Path

# Data models for structured output
@dataclass
class UIElement:
    """Represents a detected UI element"""
    element_type: str
    coordinates: Tuple[int, int, int, int]  # x1, y1, x2, y2
    text_content: str
    confidence: float
    properties: Dict[str, any]

class UIAnalysisResult(BaseModel):
    """Structured result from UI analysis"""
    ui_elements: List[Dict] = Field(description="List of detected UI elements with their properties")
    screen_type: str = Field(description="Type of screen (login, home, settings, etc.)")
    interaction_areas: List[Dict] = Field(description="Areas where user can interact")
    text_content: List[str] = Field(description="All readable text on screen")
    layout_description: str = Field(description="Overall layout description")

class UserAction(BaseModel):
    """Represents a user action in the recording"""
    timestamp: float = Field(description="Time in seconds when action occurred")
    action_type: str = Field(description="Type of action (tap, swipe, scroll, type)")
    target_element: str = Field(description="Description of target UI element")
    coordinates: Tuple[int, int] = Field(description="Action coordinates")
    description: str = Field(description="Human-readable action description")

class InteractionFlow(BaseModel):
    """Represents the complete interaction flow"""
    actions: List[UserAction] = Field(description="Sequence of user actions")
    flow_summary: str = Field(description="Summary of the interaction flow")
    patterns: List[str] = Field(description="Identified interaction patterns")
    screen_transitions: List[str] = Field(description="Screen transition sequence")

class AndroidUIAnalyzer:
    """Main class for analyzing Android UI recordings and screenshots"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.1,
            max_tokens=4000
        )
        
        # Output parsers for structured responses
        self.ui_parser = PydanticOutputParser(pydantic_object=UIAnalysisResult)
        self.flow_parser = PydanticOutputParser(pydantic_object=InteractionFlow)
        
        # Prompts for different analysis tasks
        self.ui_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Android UI analyzer. Analyze the provided screenshot and identify:
            1. All UI elements (buttons, text fields, images, CAPTCHA handling, navigation elements)
            2. Their locations and properties
            3. Screen type and layout
            4. Interactive areas
            5. All readable text content
            
            Be precise with coordinates and comprehensive in element detection.
            
            {format_instructions}"""),
            ("human", "Analyze this Android screenshot and provide detailed UI analysis.")
        ])
        
        self.flow_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in mobile UX analysis. Analyze the sequence of screenshots from a screen recording to understand:
            1. User interaction patterns
            2. Action sequences and timing
            3. Screen transitions
            4. UI flow logic
            5. Common interaction patterns
            
            Focus on understanding the logical flow of user actions and identifying repeatable patterns.
            
            {format_instructions}"""),
            ("human", "Analyze this sequence of Android screenshots to understand the user interaction flow.")
        ])

    def encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert numpy frame to base64 for API consumption"""
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8')

    def calculate_optimal_frames(self, duration: float) -> int:
        """Calculate optimal number of frames based on video duration"""
        if duration <= 5:          # Very short: 0-5 seconds
            return 6
        elif duration <= 10:       # Short: 5-10 seconds  
            return 8
        elif duration <= 20:       # Medium: 10-20 seconds
            return 10
        elif duration <= 30:       # Medium-long: 20-30 seconds
            return 20
        elif duration <= 60:       # Long: 30-60 seconds (1 minute)
            return 35
        elif duration <= 120:      # Very long: 1-2 minutes
            return 50
        else:                      # Extra long: 2+ minutes
            return 80

    def extract_key_frames_from_video(self, video_path: str) -> Tuple[List[np.ndarray], List[float]]:
        """Dynamic frame extraction based on video duration and content"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / frame_rate
        
        # Calculate optimal frame count dynamically
        optimal_frames = self.calculate_optimal_frames(duration)
        
        print(f"📊 Video: {duration:.1f}s → Selecting {optimal_frames} frames")
        
        # Dynamic frame selection strategy
        if duration <= 10:
            # Short videos: Even distribution
            frame_indices = [int(i * total_frames / optimal_frames) for i in range(optimal_frames)]
            
        elif duration <= 30:
            # Medium videos: Focus on beginning and end with some middle coverage
            beginning_frames = int(optimal_frames * 0.4)  # 40% at beginning
            middle_frames = int(optimal_frames * 0.3)     # 30% in middle
            end_frames = optimal_frames - beginning_frames - middle_frames  # 30% at end
            
            start_indices = [int(i * total_frames * 0.3 / beginning_frames) for i in range(beginning_frames)]
            middle_indices = [int(total_frames * 0.35 + i * total_frames * 0.3 / middle_frames) for i in range(middle_frames)]
            end_indices = [int(total_frames * 0.7 + i * total_frames * 0.3 / end_frames) for i in range(end_frames)]
            
            frame_indices = start_indices + middle_indices + end_indices
            
        else:
            # Long videos: Segmented approach with consistent sampling
            segments = min(6, int(duration / 10))  # One segment per ~10 seconds, max 6 segments
            frames_per_segment = optimal_frames // segments
            remaining_frames = optimal_frames % segments
            
            frame_indices = []
            for segment in range(segments):
                segment_start = int(segment * total_frames / segments)
                segment_end = int((segment + 1) * total_frames / segments)
                segment_frames = frames_per_segment + (1 if segment < remaining_frames else 0)
                
                # Sample evenly within each segment
                for i in range(segment_frames):
                    frame_idx = segment_start + int(i * (segment_end - segment_start) / segment_frames)
                    frame_indices.append(frame_idx)
        
        # Remove duplicates and sort
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
                if i < 5 or i >= len(frame_indices) - 2:  # Show first 5 and last 2
                    print(f"   📸 Frame {i+1}: {timestamp:.1f}s")
                elif i == 5:
                    print(f"   📸 ... (extracting {len(frame_indices)-7} more frames) ...")
        
        cap.release()
        print(f"✅ Successfully extracted {len(key_frames)} frames")
        return key_frames, timestamps

    # Remove the motion detection methods since we're not using them anymore

    def analyze_video_flow_directly(self, frames: List[np.ndarray], timestamps: List[float]) -> InteractionFlow:
        """Analyze video frames directly with focus on accuracy and essential information"""
        
        print("🧠 Analyzing UI flow and interaction patterns...")
        
        # Encode frames to base64
        encoded_frames = []
        for i, frame in enumerate(frames):
            base64_frame = self.encode_frame_to_base64(frame)
            encoded_frames.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_frame}",
                    "detail": "high"
                }
            })
        
        # Enhanced prompt for accuracy - focus on essential UI analysis
        enhanced_prompt = f"""You are an expert Android UI/UX analyst. Analyze this sequence of {len(frames)} screenshots from an Android app recording. The app recording is in Spanish so detect the UI flow and translate the with high accuracy.

FRAME SEQUENCE: {len(frames)} frames over {timestamps[-1] - timestamps[0]:.1f} seconds
TIMESTAMPS: {[f"{t:.1f}s" for t in timestamps]}

ANALYZE FOR:

1. **UI FLOW & NAVIGATION**:
   - What screens/views are shown in sequence?
   - How does the user navigate between screens?
   - What is the logical flow of the app usage?

2. **USER ACTIONS**:
   - What specific actions does the user perform?
   - What buttons does the user click?
   - When do these actions occur in the sequence?
   - What triggers each screen transition?

3. **INTERACTION PATTERNS**:
   - What type of app interaction pattern is this? (login flow, search, navigation, form filling, etc.)
   - What UI components are being used? (buttons, forms, lists, menus, etc.)
   - Are there any CAPTCHA that the user interacts with?
   - What is the user trying to accomplish?

4. **SCREEN TRANSITIONS**:
   - What are the distinct screens or states shown?
   - How do screens change from one to another?
   - What causes each transition?

Be specific and accurate. Focus on the actual UI elements and user journey, not technical details.

{self.flow_parser.get_format_instructions()}"""
        
        # Create message content
        message_content = [enhanced_prompt]
        message_content.extend(encoded_frames)
        
        messages = [HumanMessage(content=message_content)]
        
        response = self.llm.invoke(messages)
        
        try:
            content_str = response.content if isinstance(response.content, str) else json.dumps(response.content)
            return self.flow_parser.parse(content_str)
        except Exception as e:
            print(f"❌ Error parsing analysis: {e}")
            return InteractionFlow(
                actions=[],
                flow_summary="Analysis failed - please try again",
                patterns=[],
                screen_transitions=[]
            )

    def analyze_screenshot(self, image_path: str) -> UIAnalysisResult:
        """Analyze a single screenshot for UI elements"""
        base64_image = self.encode_image_to_base64(image_path)
        
        # Format the prompt with parser instructions
        formatted_prompt = self.ui_analysis_prompt.format_messages(
            format_instructions=self.ui_parser.get_format_instructions()
        )
        
        # Add the image to the message
        message_content = [
            {"type": "text", "text": formatted_prompt[1].content},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            }
        ]
        
        messages = [
            formatted_prompt[0],  # System message
            HumanMessage(content=message_content)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            return self.ui_parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing UI analysis: {e}")
            return UIAnalysisResult(
                ui_elements=[],
                screen_type="unknown",
                interaction_areas=[],
                text_content=[],
                layout_description="Analysis failed"
            )

    def analyze_interaction_flow(self, screenshots: List[str], motion_data: Optional[List] = None) -> InteractionFlow:
        """Analyze sequence of screenshots to understand interaction flow"""
        
        # Encode all screenshots
        encoded_images = []
        for screenshot_path in screenshots:
            base64_image = self.encode_image_to_base64(screenshot_path)
            encoded_images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            })
        
        # Format the prompt
        formatted_prompt = self.flow_analysis_prompt.format_messages(
            format_instructions=self.flow_parser.get_format_instructions()
        )
        
        # Create message content with all images
        message_content = [{"type": "text", "text": formatted_prompt[1].content}]
        message_content.extend(encoded_images)
        
        # Add motion data if available
        if motion_data:
            message_content.append({
                "type": "text",
                "text": f"Motion detection data: {json.dumps(motion_data)}"
            })
        
        messages = [
            formatted_prompt[0],  # System message
            HumanMessage(content=message_content)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            return self.flow_parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing flow analysis: {e}")
            return InteractionFlow(
                actions=[],
                flow_summary="Analysis failed",
                patterns=[],
                screen_transitions=[]
            )

    def process_screen_recording(self, video_path: str) -> Dict:
        """Streamlined processing pipeline focused on accuracy and essential information"""
        
        print("🎬 Starting focused video analysis...")
        
        # Extract key frames with dynamic selection
        key_frames, timestamps = self.extract_key_frames_from_video(video_path)
        
        # Analyze UI flow directly (no motion detection complexity)
        print("🔍 Analyzing UI flow and patterns...")
        flow_analysis = self.analyze_video_flow_directly(key_frames, timestamps)
        
        # Get basic video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()
        
        return {
            "video_path": video_path,
            "video_duration_seconds": duration,
            "frames_analyzed": len(key_frames),
            "interaction_flow": flow_analysis,
            "summary": {
                "total_actions": len(flow_analysis.actions),
                "total_patterns": len(flow_analysis.patterns),
                "total_screen_transitions": len(flow_analysis.screen_transitions),
                "analysis_method": "focused_ui_analysis"
            }
        }

    def process_screenshots(self, screenshot_paths: List[str]) -> Dict:
        """Process a list of screenshots"""
        print(f"Analyzing {len(screenshot_paths)} screenshots...")
        
        ui_analyses = []
        for i, screenshot_path in enumerate(screenshot_paths):
            print(f"Analyzing screenshot {i+1}/{len(screenshot_paths)}")
            analysis = self.analyze_screenshot(screenshot_path)
            ui_analyses.append({
                "screenshot_path": screenshot_path,
                "analysis": analysis
            })
        
        print("Analyzing interaction flow...")
        flow_analysis = self.analyze_interaction_flow(screenshot_paths)
        
        return {
            "screenshots": screenshot_paths,
            "ui_analyses": ui_analyses,
            "interaction_flow": flow_analysis,
            "summary": {
                "screens_analyzed": len(screenshot_paths),
                "actions_identified": len(flow_analysis.actions),
                "patterns_found": flow_analysis.patterns
            }
        }

def main():
    print("🔍 Android UI Analyzer - Auto Video Analysis")
    print("=" * 50)
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found. Please set OPENAI_API_KEY in your .env file.")
        return
    
    # Look for MP4 files in test_data folder
    test_data_dir = Path("test_data")
    if not test_data_dir.exists():
        print("❌ test_data folder not found. Please create it and add your MP4 file.")
        return
    
    # Find all MP4 files
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
        # Initialize the analyzer
        print("🚀 Initializing Android UI Analyzer...")
        analyzer = AndroidUIAnalyzer(openai_api_key=api_key)
        print("✅ Analyzer initialized successfully!")
        
        # Process the video with focused analysis
        print(f"\n🔄 Analyzing video: {video_path}")
        results = analyzer.process_screen_recording(video_path)
        
        print("\n🎉 ANALYSIS COMPLETED!")
        print("=" * 50)
        
        # Essential results only
        print(f"📱 Video Duration: {results['video_duration_seconds']:.1f} seconds")
        print(f"🔍 Frames Analyzed: {results['frames_analyzed']}")
        print(f"👆 Actions Identified: {results['summary']['total_actions']}")
        print(f"🔄 Screen Transitions: {results['summary']['total_screen_transitions']}")
        print(f"📋 Patterns Found: {results['summary']['total_patterns']}")
        
        # Print interaction flow
        flow = results['interaction_flow']
        print(f"\n📋 FLOW SUMMARY:")
        print(f"   {flow.flow_summary}")
        
        if flow.patterns:
            print(f"\n🔍 INTERACTION PATTERNS:")
            for i, pattern in enumerate(flow.patterns, 1):
                print(f"   {i}. {pattern}")
        
        if flow.screen_transitions:
            print(f"\n🔄 SCREEN TRANSITIONS:")
            for i, transition in enumerate(flow.screen_transitions, 1):
                print(f"   {i}. {transition}")
        
        if flow.actions:
            print(f"\n👆 USER ACTIONS:")
            for i, action in enumerate(flow.actions, 1):
                print(f"   {i}. [{action.timestamp:.1f}s] {action.action_type}: {action.description}")
        
        # Save clean, essential results only
        output_file = "video_analysis_results.json"
        with open(output_file, "w") as f:
            clean_results = {
                "video_analysis": {
                    "video_path": results["video_path"],
                    "duration_seconds": results["video_duration_seconds"],
                    "frames_analyzed": results["frames_analyzed"]
                },
                "ui_flow_analysis": {
                    "flow_summary": flow.flow_summary,
                    "interaction_patterns": flow.patterns,
                    "screen_transitions": flow.screen_transitions,
                    "user_actions": [
                        {
                            "timestamp": action.timestamp,
                            "action_type": action.action_type,
                            "target": action.target_element,
                            "description": action.description
                        } for action in flow.actions
                    ]
                },
                "summary": {
                    "total_actions_identified": len(flow.actions),
                    "total_patterns_found": len(flow.patterns),
                    "total_screen_transitions": len(flow.screen_transitions),
                    "analysis_quality": "focused_and_accurate"
                }
            }
            json.dump(clean_results, f, indent=2)
        
        print(f"\n💾 Clean results saved to: {output_file}")
        print(f"✨ Analysis focused on essential UI flow information")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()