# Import the classes you need from the VisionAgent package
from vision_agent.agent import VisionAgentCoderV2
from vision_agent.models import AgentMessage

# Enable verbose output 
agent = VisionAgentCoderV2(verbose=True)

# Add your prompt (content) and image file (media)
code_context = agent.generate_code(
    [
        AgentMessage(
            role="user",
            content="Describe the image",
            media=["friends.jpg"]
        )
    ]
)

# Write the output to a file
with open("generated_code.py", "w") as f:
    f.write(code_context.code + "\n" + code_context.test)