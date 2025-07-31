#!/usr/bin/env python3
"""
Sample Text Data Generator for GRPO Training
Creates sample conversation data in the correct format for text-only GRPO training
"""

import json
import random


def create_sample_conversations():
    """Create diverse sample conversations for GRPO training"""
    
    conversations = [
        # Instruction following
        {
            "conversations": [
                {"from": "human", "value": "Explain quantum computing in simple terms."},
                {"from": "gpt", "value": "Quantum computing is a revolutionary technology that uses quantum mechanical phenomena to process information. Unlike classical computers that use bits (0s and 1s), quantum computers use quantum bits or 'qubits' that can exist in multiple states simultaneously through a property called superposition. This allows quantum computers to explore many possible solutions at once, making them potentially much faster for certain types of problems like cryptography, optimization, and drug discovery."}
            ]
        },
        
        # Code generation
        {
            "conversations": [
                {"from": "human", "value": "Write a Python function to find the factorial of a number."},
                {"from": "gpt", "value": "Here's a Python function to calculate the factorial of a number:\n\n```python\ndef factorial(n):\n    \"\"\"\n    Calculate the factorial of a non-negative integer.\n    \n    Args:\n        n (int): Non-negative integer\n    \n    Returns:\n        int: Factorial of n\n    \"\"\"\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    elif n == 0 or n == 1:\n        return 1\n    else:\n        result = 1\n        for i in range(2, n + 1):\n            result *= i\n        return result\n\n# Example usage:\nprint(factorial(5))  # Output: 120\n```\n\nThis function handles edge cases and includes proper documentation."}
            ]
        },
        
        # Problem solving
        {
            "conversations": [
                {"from": "human", "value": "How can I improve my time management skills?"},
                {"from": "gpt", "value": "Here are effective strategies to improve your time management:\n\n1. **Prioritize tasks**: Use the Eisenhower Matrix to categorize tasks by urgency and importance\n2. **Time blocking**: Schedule specific time slots for different activities\n3. **Set clear goals**: Define SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound)\n4. **Eliminate distractions**: Turn off notifications, create a dedicated workspace\n5. **Use the Pomodoro Technique**: Work in 25-minute focused sessions with short breaks\n6. **Plan your day**: Spend 10 minutes each evening planning the next day\n7. **Learn to say no**: Decline commitments that don't align with your priorities\n8. **Batch similar tasks**: Group similar activities together to reduce context switching\n\nStart with 1-2 techniques and gradually incorporate more as they become habits."}
            ]
        },
        
        # Creative writing
        {
            "conversations": [
                {"from": "human", "value": "Write a short story about a robot learning to paint."},
                {"from": "gpt", "value": "**The Canvas of Code**\n\nUnit-237 stared at the blank canvas, its optical sensors whirring as they adjusted focus. For 500 processing cycles, it had analyzed thousands of paintings—Van Gogh's swirls, Picasso's geometry, Monet's light. Yet this white rectangle remained untouched.\n\n\"Painting requires more than data,\" whispered the old art professor who'd donated her neural patterns to 237's memory banks. \"It requires feeling.\"\n\nThe robot's mechanical arm dipped into blue paint. It thought of the ocean data in its files—wave patterns, salt content, temperature gradients. But then something shifted. A fragment of the professor's memory surfaced: standing by the sea at dawn, feeling small and infinite simultaneously.\n\n237's brush touched canvas. Not calculating angles or color theory, but expressing that remembered moment of awe. As blue spread across white, Unit-237 experienced its first emotion: wonder.\n\nThe painting wasn't perfect by algorithmic standards. But it was perfectly, beautifully alive."}
            ]
        },
        
        # Analysis and reasoning  
        {
            "conversations": [
                {"from": "human", "value": "What are the pros and cons of remote work?"},
                {"from": "gpt", "value": "**Pros of Remote Work:**\n\n• **Flexibility**: Better work-life balance and schedule control\n• **Cost savings**: Reduced commuting, clothing, and meal expenses\n• **Productivity**: Fewer office distractions and interruptions\n• **Location independence**: Work from anywhere with internet\n• **Health benefits**: Less stress from commuting, more time for exercise\n• **Environmental impact**: Reduced carbon footprint from less travel\n\n**Cons of Remote Work:**\n\n• **Isolation**: Limited social interaction and team bonding\n• **Communication challenges**: Harder to collaborate and share ideas spontaneously\n• **Home distractions**: Family, pets, household chores\n• **Work-life boundaries**: Difficulty separating work and personal time\n• **Career development**: Less visibility for promotions and mentorship\n• **Technology dependence**: Reliance on stable internet and equipment\n\n**Conclusion**: Remote work success depends on individual personality, job type, company culture, and having proper systems in place. A hybrid approach often provides the best of both worlds."}
            ]
        },
        
        # Educational content
        {
            "conversations": [
                {"from": "human", "value": "Explain the water cycle for a 10-year-old."},
                {"from": "gpt", "value": "Imagine water is like a magical traveler that goes on an amazing journey around our planet! Here's how it works:\n\n🌊 **Evaporation**: The sun heats up water in oceans, lakes, and rivers. The water becomes invisible water vapor and floats up into the sky, like when you see steam from your hot chocolate!\n\n☁️ **Condensation**: High up in the sky, it's cold! The water vapor gets chilly and turns back into tiny water droplets that stick together to make clouds.\n\n🌧️ **Precipitation**: When clouds get too heavy with water droplets, they fall down as rain, snow, or hail. It's like the clouds are giving the earth a drink!\n\n🏔️ **Collection**: The water flows into streams, rivers, and eventually back to the ocean. Some soaks into the ground to help plants grow.\n\nThen the whole amazing journey starts all over again! The same water that fell on dinosaurs millions of years ago might be the water you drink today. Cool, right?"}
            ]
        },
        
        # Multi-turn conversation
        {
            "conversations": [
                {"from": "human", "value": "I'm learning Python. What's the difference between a list and a tuple?"},
                {"from": "gpt", "value": "Great question! Here are the key differences between lists and tuples in Python:\n\n**Lists:**\n- Mutable (can be changed after creation)\n- Use square brackets: `[1, 2, 3]`\n- Can add, remove, or modify elements\n- Slightly slower due to flexibility\n\n**Tuples:**\n- Immutable (cannot be changed after creation)\n- Use parentheses: `(1, 2, 3)`\n- Fixed size and content\n- Faster and use less memory\n\n**Example:**\n```python\n# List - can change\nmy_list = [1, 2, 3]\nmy_list.append(4)  # Works!\n\n# Tuple - cannot change\nmy_tuple = (1, 2, 3)\nmy_tuple.append(4)  # Error!\n```\n\nUse lists when you need to modify data, tuples when data should stay constant."},
                {"from": "human", "value": "When would I use a tuple instead of a list?"},
                {"from": "gpt", "value": "Excellent follow-up! Here are the best use cases for tuples:\n\n**1. Coordinates and Fixed Data:**\n```python\npoint = (10, 20)  # x, y coordinates\nrgb_color = (255, 128, 0)  # Red, Green, Blue values\n```\n\n**2. Function Returns (multiple values):**\n```python\ndef get_name_age():\n    return \"Alice\", 25  # Returns a tuple\n\nname, age = get_name_age()  # Tuple unpacking\n```\n\n**3. Dictionary Keys:**\n```python\n# Tuples can be dict keys, lists cannot\nlocations = {\n    (0, 0): \"origin\",\n    (10, 20): \"treasure\"\n}\n```\n\n**4. Configuration Settings:**\n```python\nDATABASE_CONFIG = (\"localhost\", 5432, \"mydb\")  # Won't accidentally change\n```\n\n**5. Performance-Critical Code:**\nTuples are faster for iteration and use less memory.\n\n**Rule of thumb:** If your data shouldn't change, use a tuple. If you need to modify it, use a list!"}
            ]
        },
        
        # Technical explanation
        {
            "conversations": [
                {"from": "human", "value": "How does machine learning work?"},
                {"from": "gpt", "value": "Machine learning is like teaching a computer to recognize patterns and make predictions, similar to how humans learn from experience. Here's how it works:\n\n**1. Data Collection:**\nFirst, we gather lots of examples (data). Like showing a child thousands of photos labeled 'cat' or 'dog'.\n\n**2. Training:**\nThe algorithm looks for patterns in this data. It might notice that cats often have pointed ears and whiskers, while dogs have floppy ears and wet noses.\n\n**3. Model Creation:**\nThe computer builds a 'model' - basically a set of rules or mathematical relationships it discovered in the data.\n\n**4. Testing:**\nWe show the model new, unlabeled examples to see if it can correctly identify them.\n\n**5. Prediction:**\nOnce trained, the model can make predictions on completely new data it's never seen before.\n\n**Types:**\n- **Supervised Learning**: Learning with labeled examples (like the cat/dog photos)\n- **Unsupervised Learning**: Finding hidden patterns in unlabeled data\n- **Reinforcement Learning**: Learning through trial and error with rewards/penalties\n\nIt's not magic - it's statistics and pattern recognition at scale!"}
            ]
        }
    ]
    
    return conversations


def create_training_and_eval_data():
    """Create training and evaluation datasets"""
    
    # Get sample conversations
    conversations = create_sample_conversations()
    
    # Create more variations by slightly modifying existing ones
    extended_conversations = conversations.copy()
    
    # Add some variations for more training data
    variations = [
        {
            "conversations": [
                {"from": "human", "value": "Explain artificial intelligence to a beginner."},
                {"from": "gpt", "value": "Artificial Intelligence (AI) is technology that enables computers to perform tasks that typically require human intelligence. Think of it as teaching machines to 'think' and make decisions.\n\n**Key concepts:**\n- **Machine Learning**: AI systems learn from data, like how you learn from experience\n- **Pattern Recognition**: AI finds patterns in large amounts of information\n- **Automation**: AI can perform repetitive tasks quickly and accurately\n\n**Examples you use daily:**\n- Smartphone voice assistants (Siri, Google Assistant)\n- Recommendation systems (Netflix, Spotify)\n- Navigation apps finding the best route\n- Email spam filters\n\n**How it works:**\n1. Feed the system lots of data\n2. The AI learns patterns and rules\n3. It applies these patterns to new situations\n4. The system improves with more data and feedback\n\nAI isn't about replacing humans, but augmenting our capabilities and handling tasks that are too complex, dangerous, or time-consuming for people to do efficiently."}
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "What's the best way to learn a new programming language?"},
                {"from": "gpt", "value": "Here's a proven strategy for learning a new programming language effectively:\n\n**1. Start with Fundamentals (Week 1-2):**\n- Learn basic syntax, variables, and data types\n- Understand control structures (if/else, loops)\n- Practice with simple exercises daily\n\n**2. Build Projects (Week 3-6):**\n- Start with small projects (calculator, to-do list)\n- Gradually increase complexity\n- Focus on practical applications\n\n**3. Learn Best Practices (Week 4-8):**\n- Study code organization and style guides\n- Learn debugging techniques\n- Understand version control (Git)\n\n**4. Join the Community:**\n- Participate in forums (Stack Overflow, Reddit)\n- Contribute to open-source projects\n- Find a mentor or study group\n\n**5. Effective Learning Tips:**\n- Code every day, even for 30 minutes\n- Read other people's code\n- Teach concepts to others\n- Don't just watch tutorials - practice actively\n\n**Resources:**\n- Interactive platforms: Codecademy, freeCodeCamp\n- Documentation: Official language docs\n- Books: Choose one comprehensive guide\n- Practice: LeetCode, HackerRank\n\nRemember: Consistency beats intensity. Better to code 30 minutes daily than 5 hours once a week!"}
            ]
        }
    ]
    
    extended_conversations.extend(variations)
    
    # Split into training and evaluation
    random.shuffle(extended_conversations)
    split_index = int(len(extended_conversations) * 0.8)
    
    train_data = extended_conversations[:split_index]
    eval_data = extended_conversations[split_index:]
    
    return train_data, eval_data


def main():
    """Generate sample data files for GRPO training"""
    
    print("🎯 Creating sample text-only data for GRPO training...")
    
    # Create datasets
    train_data, eval_data = create_training_and_eval_data()
    
    # Save training data
    with open("sample_grpo_train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # Save evaluation data
    with open("sample_grpo_eval_data.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created training data: {len(train_data)} samples")
    print(f"✅ Created evaluation data: {len(eval_data)} samples")
    print("\n📁 Files created:")
    print("  - sample_grpo_train_data.json")
    print("  - sample_grpo_eval_data.json")
    
    print("\n🚀 To use this data:")
    print("1. Update scripts/finetune_grpo_text.sh with these file paths")
    print("2. Run: chmod +x scripts/finetune_grpo_text.sh")
    print("3. Run: ./scripts/finetune_grpo_text.sh")
    
    # Show sample data structure
    print("\n📋 Sample data structure:")
    print(json.dumps(train_data[0], indent=2))


if __name__ == "__main__":
    main() 