from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Read Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")

# Configure Gemini LLM for CrewAI
gemini_llm = LLM(
    model="gemini/gemini-2.5-pro",  # You can change to another Gemini model if needed
    api_key=GEMINI_API_KEY,
    temperature=0.3,  # Slight creativity, but still stable
)

app = FastAPI()

# CORS configuration
# Allow all origins for now (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Basic Routes ----------

@app.get("/")
def read_root():
    return {"message": "Study Material Generator API is running 🚀"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---------- Request Model for Summarization ----------

class SummarizeRequest(BaseModel):
    text: str

# ---------- Agentic AI: Simple Summarizer Agent ----------

@app.post("/summarize")
def summarize_text(request: SummarizeRequest):
    """Use a CrewAI Agent + Gemini to summarize study material."""
    user_text = request.text

    # 1. Define the agent (like a role in a team)
    summarizer_agent = Agent(
        role="Subject Teacher",
        goal="Create short, exam-focused summaries of study material.",
        backstory=(
            "You are a very good college teacher. "
            "You read the given content and write clear, simple, point-wise notes "
            "that help students revise before exams."
        ),
        llm=gemini_llm,
        verbose=True,  # prints logs in terminal - helpful for debugging
    )

    # 2. Define the task assigned to this agent
    summarize_task = Task(
        description=(
            "Read the following study material and create a short, clear, "
            "exam-focused summary in simple language. Use bullet points where possible.\n\n"
            f"CONTENT:\n{user_text}"
        ),
        expected_output=(
            "A concise summary of the content, with bullet points, "
            "covering all important concepts for exam preparation."
        ),
        agent=summarizer_agent,
    )

    # 3. Create a crew with this single agent (later we'll add more agents)
    crew = Crew(
        agents=[summarizer_agent],
        tasks=[summarize_task],
        verbose=True,
    )

    # 4. Run the crew (kickoff = start the work)
    result = crew.kickoff()

    # 5. Return result back to frontend / client
    return {
        "summary": str(result)
    }

class StudyMaterialRequest(BaseModel):
    text: str
    num_mcqs: int = 5  # default 5 MCQs, can change from frontend or docs

@app.post("/generate_study_material")
def generate_study_material(request: StudyMaterialRequest):
    """
    Use multiple CrewAI agents + Gemini to:
    1) Analyze topics
    2) Generate notes
    3) Generate MCQs

    We run them step-by-step and return all three parts separately.
    """
    content = request.text
    num_mcqs = request.num_mcqs

    # ---- 1. Define Agents ----

    topic_analyzer = Agent(
        role="Topic Analyzer",
        goal="Identify the most important topics and subtopics from the given study material.",
        backstory=(
            "You are an expert at reading long chapters and extracting only the most "
            "important headings and subheadings that students should study for exams."
        ),
        llm=gemini_llm,
        verbose=True,
    )

    notes_maker = Agent(
        role="Notes Maker",
        goal="Write short, exam-focused notes in simple language.",
        backstory=(
            "You are a friendly college teacher. You explain concepts in very simple terms "
            "and create bullet-point notes that students can revise quickly before exams."
        ),
        llm=gemini_llm,
        verbose=True,
    )

    mcq_maker = Agent(
        role="MCQ Creator",
        goal="Create clear MCQs based on the notes and topics, with 4 options and correct answer.",
        backstory=(
            "You are an experienced question paper setter. You create fair and clear MCQs "
            "that directly test understanding of the notes and topics."
        ),
        llm=gemini_llm,
        verbose=True,
    )

    # ---- 2. TOPIC TASK: First crew -> only topics ----

    topic_task = Task(
        description=(
            "Read the following study material and extract the MOST IMPORTANT topics and subtopics "
            "for exam preparation and give full length description or points.\n\n"
            f"CONTENT:\n{content}\n\n"
            "OUTPUT FORMAT (VERY IMPORTANT):\n"
            "- Do NOT use any Markdown formatting (no *, no #, no **, no ```).\n"
            "- Use only plain text.\n"
            "- Write in this style:\n"
            "  Main Topic 1:\n"
            "    - Subtopic 1\n"
            "    - Subtopic 2\n"
            "  Main Topic 2:\n"
            "    - Subtopic 1\n"
            "    - Subtopic 2\n"
        ),
        expected_output=(
            "A bullet list of main topics and their subtopics, focused only on what is actually important for exams."
        ),
        agent=topic_analyzer,
    )

    topics_crew = Crew(
        agents=[topic_analyzer],
        tasks=[topic_task],
        verbose=True,
    )

    topics_result = topics_crew.kickoff()
    topics_text = str(topics_result)

    # ---- 3. NOTES TASK: second crew -> use topics + original content ----

    notes_task = Task(
        description=(
            "You are creating exam-focused notes for a B.Tech CSE student.\n"
            "Use the topics and subtopics listed below, plus the original content, "
            "to write SHORT, SCORING NOTES for university exams.\n\n"
            "TOPICS AND SUBTOPICS:\n"
            f"{topics_text}\n\n"
            "ORIGINAL CONTENT:\n"
            f"{content}\n\n"
            "VERY IMPORTANT RULES:\n"
            "- DO NOT use Markdown (no *, no #, no **, no ```).\n"
            "- Use ONLY plain text.\n"
            "- Write in clean headings and bullet points.\n"
            "- Target answers that can directly be written in 6–8 mark questions.\n"
            "- For each main topic, include:\n"
            "  1) Definition (1–2 lines)\n"
            "  2) Important points / properties (point-wise)\n"
            "  3) Important operations / algorithms (in short)\n"
            "  4) Advantages / disadvantages (if applicable)\n"
            "  5) Applications / examples (if useful)\n"
            "- Avoid long stories or over-explanation.\n"
            "- Keep language simple, as if explaining to an average student before exam.\n\n"
            "OUTPUT FORMAT (example style, but adapt to content):\n"
            "Array:\n"
            "  - Definition: ...\n"
            "  - Important points:\n"
            "    - ...\n"
            "    - ...\n"
            "  - Operations and time complexity:\n"
            "    - Traversal: ...\n"
            "    - Insertion: ...\n"
            "  - Advantages:\n"
            "    - ...\n"
            "  - Disadvantages:\n"
            "    - ...\n"
            "  - Applications:\n"
            "    - ...\n"
        ),
        expected_output=(
            "Plain text, point-wise exam notes (no markdown) for each main topic, "
            "good enough to write 6–8 mark answers directly."
        ),
        agent=notes_maker,
    )


    notes_crew = Crew(
        agents=[notes_maker],
        tasks=[notes_task],
        verbose=True,
    )

    notes_result = notes_crew.kickoff()
    notes_text = str(notes_result)

    # ---- 4. MCQ TASK: third crew -> use notes as base ----

    mcq_task = Task(
        description=(
            "Based on the notes below, generate MCQs for exam preparation.\n\n"
            f"NOTES:\n{notes_text}\n\n"
            f"Create around {num_mcqs} MCQs.\n\n"
            "RULES:\n"
            "- Each question must have 4 options: (a), (b), (c), (d).\n"
            "- Clearly mention the correct answer after each question.\n"
            "- Questions should directly test understanding of the notes.\n"
            "- Avoid too tricky or confusing questions.\n\n"
            "OUTPUT FORMAT (very important):\n"
            "Q1. <question text>\n"
            "(a) option 1\n"
            "(b) option 2\n"
            "(c) option 3\n"
            "(d) option 4\n"
            "Answer: <option letter>\n\n"
            "Q2. ... and so on."
        ),
        expected_output=(
            f"A list of about {num_mcqs} MCQs in the specified format, each with options and correct answer."
        ),
        agent=mcq_maker,
    )

    mcq_crew = Crew(
        agents=[mcq_maker],
        tasks=[mcq_task],
        verbose=True,
    )

    mcq_result = mcq_crew.kickoff()
    mcq_text = str(mcq_result)

    # ---- 5. Return all parts separately ----

    return {
        "topics": topics_text,
        "notes": notes_text,
        "mcqs": mcq_text,
    }

