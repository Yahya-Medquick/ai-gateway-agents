import os
import sys
from crewai import Agent, Task, Crew, LLM

# 1. Initialize Gemini Flash as the core reasoning engine
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Parse incoming payload data from n8n forwarded by the GitHub Action
try:
    user_input_payload = sys.argv[1]
except IndexError:
    user_input_payload = "Please convert the text into clean exam notes."

# 2. Define your Role-Based Cloud Agents
analyst_agent = Agent(
    role="Core Concept Analyst",
    goal="Extract every key equation, strict scientific term, and core concept from raw text.",
    backstory="You are an advanced text distillation tool. Your layout strips out conversational text and filters for absolute data.",
    llm=llm,
    verbose=False
)

exam_coach_agent = Agent(
    role="Exam Preparation Coach",
    goal="Structure raw technical data blocks into concise, high-yield structured study modules.",
    backstory="You are an expert physics and mathematics academic tutor. You structure arguments cleanly into notes optimized for standard student exam marking criteria.",
    llm=llm,
    verbose=False
)

# 3. Define the Agent Execution Pipeline Tasks
extract_task = Task(
    description=f"Thoroughly analyze this raw text and pull every single vital fact, formula, and definition: {user_input_payload}",
    expected_output="A clean bulleted list containing only technical scientific facts, concepts, and formulas.",
    agent=analyst_agent
)

format_task = Task(
    description="Take the extracted structured points and reformat them into a gorgeous, highly scannable Markdown study guide with logical section headers.",
    expected_output="A beautiful, comprehensive Markdown document optimized for studying.",
    agent=exam_coach_agent
)

# 4. Assemble the Autonomous Cloud Crew
crew = Crew(
    agents=[analyst_agent, exam_coach_agent],
    tasks=[extract_task, format_task]
)

# Run the crew and output the result
result = crew.kickoff()
print(result)