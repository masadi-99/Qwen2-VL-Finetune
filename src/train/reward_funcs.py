import os
import re
from datetime import datetime

# Optional import for mathematical verification
try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    print("Warning: math_verify module not found. Falling back to string matching for reward verification.")
    MATH_VERIFY_AVAILABLE = False
    
    # Create dummy functions to prevent errors
    def parse(content):
        return content
    
    def verify(answer, solution):
        return 0.0  # Always fail verification, will use string matching fallback

def accuracy_reward(completions, assistant, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    solution = [a['content'] for a in assistant]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        # Try symbolic verification first (only if math_verify is available)
        if MATH_VERIFY_AVAILABLE:
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

        # If symbolic verification failed or unavailable, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def length_reward(completions, **kwargs):
    """Reward function that gives higher scores to responses within an optimal length range."""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        length = len(content.split())
        
        # Optimal range: 50-300 words
        if 50 <= length <= 300:
            reward = 1.0
        elif 20 <= length < 50 or 300 < length <= 500:
            reward = 0.7  # Slightly too short or long
        elif 10 <= length < 20 or 500 < length <= 1000:
            reward = 0.3  # Too short or long
        else:
            reward = 0.1  # Way too short or too long
            
        rewards.append(reward)
    
    return rewards


def helpfulness_reward(completions, **kwargs):
    """Simple reward function that gives positive rewards to helpful-looking responses."""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    # Keywords that indicate helpful responses
    helpful_patterns = [
        r'\bhow to\b', r'\bsteps?\b', r'\bexample\b', r'\bfor instance\b',
        r'\bspecifically\b', r'\bdetailed?\b', r'\bexplanation\b', r'\bclearly?\b',
        r'\buseful\b', r'\bhelpful\b', r'\bpractical\b', r'\btip\b'
    ]
    
    # Patterns that might indicate unhelpful responses
    unhelpful_patterns = [
        r'\bi don\'t know\b', r'\bcan\'t help\b', r'\bsorry.*can\'t\b',
        r'\bnot sure\b', r'\bno idea\b'
    ]
    
    for content in completion_contents:
        content_lower = content.lower()
        
        helpful_score = sum(1 for pattern in helpful_patterns 
                          if re.search(pattern, content_lower))
        unhelpful_score = sum(1 for pattern in unhelpful_patterns 
                            if re.search(pattern, content_lower))
        
        # Base reward
        if len(content.strip()) > 20:  # Not too short
            reward = 0.5
        else:
            reward = 0.1
            
        # Boost for helpful indicators
        reward += helpful_score * 0.2
        
        # Penalize for unhelpful indicators
        reward -= unhelpful_score * 0.3
        
        # Clamp between 0 and 1
        reward = max(0.0, min(1.0, reward))
        rewards.append(reward)
    
    return rewards


def simple_text_reward(completions, **kwargs):
    """Simple default reward function for general text-only GRPO training."""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        # Basic quality indicators
        word_count = len(content.split())
        char_count = len(content.strip())
        
        # Start with base reward
        reward = 0.5
        
        # Length scoring (prefer moderate length responses)
        if 30 <= word_count <= 200:
            reward += 0.3
        elif 10 <= word_count < 30 or 200 < word_count <= 400:
            reward += 0.1
        else:
            reward -= 0.2
            
        # Basic structure scoring
        if '.' in content or '!' in content or '?' in content:
            reward += 0.1  # Has sentence endings
            
        if len(content.split('\n')) > 1:
            reward += 0.1  # Has multiple lines/paragraphs
            
        # Clamp between 0 and 1
        reward = max(0.0, min(1.0, reward))
        rewards.append(reward)
    
    return rewards
