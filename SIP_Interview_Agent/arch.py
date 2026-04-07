import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from pydantic import BaseModel
from baml_client import b
from resumeMD import convert_resume_to_markdown
load_dotenv()  # Load environment variables from .env file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_DSA_MODE = "prompt"
ENV_RESUME_PATH = "SIP_RESUME_FILE"
ENV_DSA_MODE = "SIP_DSA_MODE"


class AnswerAnalysis(TypedDict):
    """Analysis of a candidate's answer."""
    clarity: int
    correctness: int
    depth: int
    review: str
    uncovered_gaps: List[str]

class ProjectPastContext(BaseModel):
    """Past context for a specific project."""

    weak_areas: List[str]
    strong_areas: List[str]
    unanswered_concepts: List[str]
    coverage_score: float

class ProjectState(TypedDict):
    """State of a specific project under discussion."""

    name: str
    description: List[str]
    claimed_stack: List[str]
    complexity: int

    question_ids: Dict[str, str]  # question_id -> question string
    followup_count: Dict[str, int]  # parent_question_id -> number of followups asked
    parent_questions: Dict[str, str]  # question_id -> parent question_id
    per_question_answer_analysis: Dict[str, AnswerAnalysis]  # question_id -> answer analysis

    overall_analysis: ProjectPastContext

class InterviewDecision(TypedDict):
    intent: Literal[
        "ProbeDepth",
        "FollowUp",
    "Clarify",
    "Challenge",
    "MoveOn",
        "SwitchProject",
        "WrapUp",
    ]
    urgency: float
    confidence: float
    rationale: str

class InterviewState(TypedDict):
    stage: Literal["Intro", "DeepDive", "FollowUp", "WrapUp"]
    turn_index: int
    max_turns: int

    max_followups_per_question: int
    max_questions_per_project: int

    projects_done: Dict[str, bool]

    active_project_id: str
    active_question_id: str

    projects: Dict[str, ProjectState]
    decision: InterviewDecision
    last_answer: Optional[str]

def node1_question_generator(state: InterviewState) -> InterviewState:
    project = state["projects"][state["active_project_id"]]
    decision = state["decision"]

    # Hard guard: enforce question budget per project
    if len(project["question_ids"]) >= state["max_questions_per_project"]:
        state["decision"] = {
            "intent": "SwitchProject",
            "urgency": 1.0,
            "confidence": 1.0,
            "rationale": "Question budget exhausted for this project",
        }
        return state

    state2 = state.copy()

    question = b.GenerateQuestion(
    state=state2,
    intent=decision["intent"],
    urgency=decision["urgency"],
    )

    qid = question.question_id
    project["question_ids"][qid] = question.question

    # Parent assignment logic for root and follow-up questions.
    if question.parent_question_id is not None:
        project["parent_questions"][qid] = question.parent_question_id
    elif state.get("active_question_id") and state["active_question_id"] != qid:
        project["parent_questions"][qid] = state["active_question_id"]
    else:
        project["parent_questions"][qid] = qid

    root = project["parent_questions"][qid]
    if root != qid:
        project["followup_count"].setdefault(root, 0)
        project["followup_count"][root] += 1
    project["followup_count"].setdefault(qid, 0)

    state["active_question_id"] = qid
    return state


def node2_answer_input(state: InterviewState) -> InterviewState:
    if state["decision"]["intent"] in ["SwitchProject", "WrapUp"]:
        return state

    project = state["projects"][state["active_project_id"]]
    if not state["active_question_id"] or state["active_question_id"] not in project["question_ids"]:
        return state

    q = project["question_ids"][state["active_question_id"]]

    state["turn_index"] += 1
    state["last_answer"] = input(f"\nQ: {q}\nA: ")
    return state

def node3_answer_analysis(state: InterviewState) -> InterviewState:
    if not state.get("active_project_id"):
        state["active_project_id"] = next(iter(state["projects"].keys()))

    project = state["projects"][state["active_project_id"]]
    qid = state["active_question_id"]
    question = project["question_ids"][qid]
    answer = state["last_answer"]

    res = b.AnalyzeAnswer(
        question=question,
        answer=answer,
        state=state,
    )

    project["per_question_answer_analysis"][qid] = res.analysis
    project["overall_analysis"]["weak_areas"] = res.pastContext.weak_areas
    project["overall_analysis"]["strong_areas"] = res.pastContext.strong_areas
    project["overall_analysis"]["unanswered_concepts"] = res.pastContext.unanswered_concepts
    project["overall_analysis"]["coverage_score"] = res.pastContext.coverage_score

    state["decision"] = res.decision.model_dump()

    if state["stage"] == "FollowUp" and state["decision"]["intent"] == "MoveOn":
        state["stage"] = "DeepDive"
    elif state["stage"] == "Intro" and state["decision"]["intent"] in ["Challenge", "Clarify"]:
        state["stage"] = "FollowUp"
    elif state["decision"]["intent"] == "FollowUp":
        state["stage"] = "FollowUp"
    else:
        state["stage"] = "DeepDive"

    if state["decision"]["intent"] == "SwitchProject":
        state["projects_done"][state["active_project_id"]] = True
        state["active_project_id"] = None
        state["active_question_id"] = None

        for proj_id, proj_state in state["projects"].items():
            if not state["projects_done"].get(proj_id, False):
                state["active_project_id"] = proj_id
                state["active_question_id"] = None
                state["stage"] = "Intro"
                state["decision"] = {
                    "intent": "ProbeDepth",
                    "urgency": 0.3,
                    "confidence": 0.6,
                    "rationale": f"Starting interview for {proj_state['name']}",
                }
                break

        if state["active_project_id"] is None:
            state["decision"] = {
                "intent": "WrapUp",
                "urgency": 1.0,
                "confidence": 1.0,
                "rationale": "All projects covered",
            }

    if state["decision"]["intent"] == "FollowUp" and state.get("active_question_id"):
        project = state["projects"][state["active_project_id"]]
        root = project["parent_questions"].get(state["active_question_id"], state["active_question_id"])
        if project["followup_count"].get(root, 0) >= state["max_followups_per_question"]:
            state["decision"]["intent"] = "MoveOn"
            state["decision"]["rationale"] = "Followup limit reached, moving on"

    if state["turn_index"] >= state["max_turns"]:
        state["decision"] = {
            "intent": "WrapUp",
            "urgency": 1.0,
            "confidence": 1.0,
            "rationale": "Max turns reached",
        }

    return state


def node4_final_performance_analysis(state: InterviewState) -> None:
    """Provides final analysis and recommendations after the interview."""
    final_report = b.FinalAnalysis(state)
    print("\n===== FINAL ANALYSIS AND RECOMMENDATIONS =====\n")
    print(final_report)


graph = StateGraph(InterviewState)

graph.add_node("question_generator", node1_question_generator)
graph.add_node("answer_input", node2_answer_input)
graph.add_node("answer_analysis", node3_answer_analysis)
graph.add_node("final_analysis", node4_final_performance_analysis)

graph.add_edge("question_generator", "answer_input")
graph.add_edge("answer_input", "answer_analysis")


def route_from_analysis(state: InterviewState) -> str:
    intent = state["decision"]["intent"]

    if intent == "WrapUp":
        return "final_analysis"

    if intent == "SwitchProject":
        return "question_generator"

    return "question_generator"


graph.add_conditional_edges(
    "answer_analysis",
    route_from_analysis,
    {
        "final_analysis": "final_analysis",
        "question_generator": "question_generator",
    },
)

graph.add_edge("final_analysis", END)
graph.set_entry_point("question_generator")

compiled_graph = graph.compile()


def initialise(projects: List[Dict[str, Any]]) -> InterviewState:
    project_states: Dict[str, ProjectState] = {}
    projects_done: Dict[str, bool] = {}

    for idx, proj in enumerate(projects):
        pid = f"project_{idx}"

        project_states[pid] = {
            "name": proj.name,
            "description": proj.description,
            "claimed_stack": proj.tech_stack,
            "complexity": proj.complexity,
            "question_ids": {},
            "followup_count": {},
            "parent_questions": {},
            "per_question_answer_analysis": {},
            "overall_analysis": {
                "weak_areas": [],
                "strong_areas": [],
                "unanswered_concepts": proj.tech_stack.copy(),
                "coverage_score": 0.0,
            },
        }

        projects_done[pid] = False

    initial_state: InterviewState = {
        "stage": "Intro",
        "turn_index": 0,
        "max_turns": 20,
        "max_followups_per_question": 3,
        "max_questions_per_project": 6,
        "projects_done": projects_done,
        "active_project_id": next(iter(project_states.keys())),
        "active_question_id": None,
        "projects": project_states,
        "decision": {
            "intent": "ProbeDepth",
            "urgency": 0.3,
            "confidence": 0.6,
            "rationale": "Initial probing",
        },
        "last_answer": None,
    }

    return initial_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run project interview using resume PDF, with optional DSA round"
    )
    parser.add_argument(
        "--resume-file",
        help=(
            f"Path to resume PDF. If omitted, reads ${ENV_RESUME_PATH} "
            "or auto-detects a single PDF in current directory."
        ),
    )
    parser.add_argument(
        "--dsa-mode",
        choices=["prompt", "always", "never"],
        default=os.getenv(ENV_DSA_MODE, DEFAULT_DSA_MODE),
        help="Control DSA round behavior: prompt (default), always, or never.",
    )
    return parser.parse_args()


def resolve_resume_file(cli_resume_file: Optional[str]) -> Path:
    candidate = cli_resume_file or os.getenv(ENV_RESUME_PATH)

    if candidate:
        resume_path = Path(candidate).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume file not found: {resume_path}")
        if resume_path.suffix.lower() != ".pdf":
            raise ValueError(f"Resume file must be a PDF: {resume_path}")
        return resume_path

    pdf_files = sorted(Path.cwd().glob("*.pdf"))
    if len(pdf_files) == 1:
        return pdf_files[0].resolve()
    if len(pdf_files) == 0:
        raise FileNotFoundError(
            f"No PDF found in {Path.cwd()}. Provide --resume-file or set {ENV_RESUME_PATH}."
        )

    found = ", ".join(p.name for p in pdf_files)
    raise ValueError(
        f"Multiple PDFs found ({found}). Provide --resume-file or set {ENV_RESUME_PATH}."
    )


def run_resume_interview(resume_file: Path) -> bool:
    logger.info("Using resume: %s", resume_file)
    markdown = convert_resume_to_markdown(str(resume_file))
    if not markdown:
        logger.error("Markdown conversion failed for %s", resume_file)
        return False

    projects = b.ExtractProjects(markdown)
    projects.sort(key=lambda x: x.complexity, reverse=True)

    if not projects:
        logger.error("No projects found in resume.")
        return False

    initial_state = initialise(projects)
    compiled_graph.invoke(initial_state, config={"recursion_limit": 100})
    print("\n===== RESUME INTERVIEW COMPLETE =====\n")
    return True


async def run_dsa_round() -> None:
    from dsa_client import DSABotClient

    dsa_client = DSABotClient()
    await dsa_client.run_interactive_session()


def main() -> int:
    args = parse_args()

    try:
        resume_file = resolve_resume_file(args.resume_file)
    except (FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        return 1

    if not run_resume_interview(resume_file):
        return 1

    if args.dsa_mode == "always":
        asyncio.run(run_dsa_round())
    elif args.dsa_mode == "prompt":
        answer = input("\nWould you like to continue with a DSA coding round? (y/n): ").strip().lower()
        if answer == "y":
            asyncio.run(run_dsa_round())
        else:
            print("\nAll done! Good luck with your interviews.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
