from dotenv import load_dotenv
from typing import *
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from baml_client import b
from resumeMD import convert_resume_to_markdown
import json
import os

load_dotenv()  # Load environment variables from .env file

# Ensure debug log directory exists
# _DEBUG_LOG_DIR = '/Users/umanggoel/Desktop/Projects/Project1/.cursor'
# _DEBUG_LOG_PATH = '/Users/umanggoel/Desktop/Projects/Project1/.cursor/debug.log'
# os.makedirs(_DEBUG_LOG_DIR, exist_ok=True)

## reminder to add descision policy 


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
    name : str
    description: List[str]
    claimed_stack: List[str]
    complexity: int

    question_ids: Dict[str, str]  # question_id -> question string
    followup_count: Dict[str, int]  # parent_question_id -> number of followups asked
    parent_questions: Dict[str, str]  # question_id -> parent question_id
    per_question_answer_analysis : Dict[str, AnswerAnalysis] # question_id -> answer analysis

    overall_analysis: ProjectPastContext

class InterviewDecision(TypedDict):
    intent: Literal[
        "ProbeDepth",
        "FollowUp",
        "Clarify",
        "Challenge",
        "MoveOn",  
        "SwitchProject",
        "WrapUp"
    ]
    urgency: float        # 0–1
    confidence: float     # confidence refer to how sure the model is regarding its intent decision -> this helps in deciding whether to take risky actions like switching project or wrapping up
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
    # #region agent log
    #with open(_DEBUG_LOG_PATH, 'a') as f:
        #f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"arch.py:74","message":"node1_entry","data":{"active_project_id":state.get("active_project_id"),"stage":state.get("stage"),"decision_intent":state.get("decision",{}).get("intent") if state.get("decision") else None,"active_question_id":state.get("active_question_id")},"timestamp":int(__import__("time").time()*1000)})+'\n')
    # #endregion
    project = state["projects"][state["active_project_id"]]
    decision = state["decision"]

    # Hard guard: enforce question budget per project
    if len(project["question_ids"]) >= state["max_questions_per_project"]:
        state["decision"] = {
            "intent": "SwitchProject",
            "urgency": 1.0,
            "confidence": 1.0,
            "rationale": "Question budget exhausted for this project"
        }
        return state

    #deep copy of state
    state2 = state.copy()
    #here state is a dict
    #first time decision is dict 
    # outputs of function are objects

    # #region agent log
    #with open(_DEBUG_LOG_PATH, 'a') as f:
        #f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"arch.py:84","message":"before_GenerateQuestion","data":{"state2_active_project":state2.get("active_project_id"),"intent":decision["intent"],"urgency":decision["urgency"]},"timestamp":int(__import__("time").time()*1000)})+'\n')
    # #endregion

    question = b.GenerateQuestion( 
        state=state2,
        intent=decision["intent"],
        urgency=decision["urgency"], 
    )
    # this generates question based on intent and urgency. Here urgency helps in deciding how deep or challenging the question should be. 
    # question contains question string, question_id, parent_question_id

    qid = question.question_id
    project["question_ids"][qid] = question.question
    # #region agent log
    #with open(_DEBUG_LOG_PATH, 'a') as f:
        #f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"arch.py:92","message":"before_parent_assignment","data":{"qid":qid,"parent_question_id":question.parent_question_id},"timestamp":int(__import__("time").time()*1000)})+'\n')
    # #endregion
    # Fix: If parent_question_id is None, it's a root question (parent should be None or qid itself for tracking)
    # If parent_question_id exists, use it. Otherwise, if we have an active_question_id, this is a followup
    if question.parent_question_id is not None:
        project["parent_questions"][qid] = question.parent_question_id
    elif state.get("active_question_id") and state["active_question_id"] != qid:
        # This is a followup to the current active question
        project["parent_questions"][qid] = state["active_question_id"]
    else:
        # Root question - parent is itself for tracking purposes
        project["parent_questions"][qid] = qid

    # #region agent log
    #with open(_DEBUG_LOG_PATH, 'a') as f:
        #f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"arch.py:95","message":"after_parent_assignment","data":{"qid":qid,"assigned_parent":project["parent_questions"][qid]},"timestamp":int(__import__("time").time()*1000)})+'\n')
    # #endregion

    # Fix: Only count followups, not root questions themselves
    root = project["parent_questions"][qid]
    # If parent is the question itself, it's a root question - don't count it as a followup
    # Otherwise, increment the followup count for the root question
    if root != qid:
        project["followup_count"].setdefault(root, 0)
        project["followup_count"][root] += 1
    # For root questions, we still want to initialize the count (but don't increment yet)
    project["followup_count"].setdefault(qid, 0)
    # #region agent log
    #with open(_DEBUG_LOG_PATH, 'a') as f:
        #f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"arch.py:100","message":"followup_count_updated","data":{"root":root,"followup_count":project["followup_count"][root]},"timestamp":int(__import__("time").time()*1000)})+'\n')
    # #endregion

    state["active_question_id"] = qid
    # #region agent log
    #with open(_DEBUG_LOG_PATH, 'a') as f:
        #f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"arch.py:103","message":"node1_exit","data":{"active_question_id":qid,"question":question.question[:50]},"timestamp":int(__import__("time").time()*1000)})+'\n')
    # #endregion
    return state


def node2_answer_input(state: InterviewState) -> InterviewState:
    # If the generator decided to switch project or wrap up, skip input
    if state["decision"]["intent"] in ["SwitchProject", "WrapUp"]:
        return state

    project = state["projects"][state["active_project_id"]]
    # Ensure active_question_id belongs to the current project
    if not state["active_question_id"] or state["active_question_id"] not in project["question_ids"]:
         return state

    q = project["question_ids"][state["active_question_id"]]

    state["turn_index"] += 1
    state["last_answer"] = input(f"\nQ: {q}\nA: ")  
    return state


def node3_answer_analysis(state: InterviewState) -> InterviewState:
    # #region agent log
    # with open(_DEBUG_LOG_PATH, 'a') as f:
        # f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"arch.py:114","message":"node3_entry","data":{"active_project_id":state.get("active_project_id"),"active_question_id":state.get("active_question_id"),"stage":state.get("stage"),"turn_index":state.get("turn_index")},"timestamp":int(__import__("time").time()*1000)})+'\n')
    # #endregion
    
    # Fix: Check active_project_id BEFORE accessing project to avoid KeyError
    if not state.get("active_project_id"):
        state["active_project_id"] = next(iter(state["projects"].keys()))
    
    project = state["projects"][state["active_project_id"]]
    qid = state["active_question_id"]
    question = project["question_ids"][qid]
    answer = state["last_answer"]

    # Get analysis from LLM
    res = b.AnalyzeAnswer(
        question=question,
        answer=answer,
        state=state,
    )
    
    # Update project state with results
    project["per_question_answer_analysis"][qid] = res.analysis
    project["overall_analysis"]["weak_areas"] = res.pastContext.weak_areas
    project["overall_analysis"]["strong_areas"] = res.pastContext.strong_areas
    project["overall_analysis"]["unanswered_concepts"] = res.pastContext.unanswered_concepts
    project["overall_analysis"]["coverage_score"] = res.pastContext.coverage_score
    
    state["decision"] = res.decision.model_dump()
    
    # #region agent log
    #with open(_DEBUG_LOG_PATH, 'a') as f:
        #f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"arch.py:137","message":"decision_received","data":{"intent":state["decision"]["intent"],"urgency":state["decision"]["urgency"],"confidence":state["decision"]["confidence"],"coverage":res.pastContext.coverage_score},"timestamp":int(__import__("time").time()*1000)})+'\n')
    # #endregion

    # Handle stage transitions
    old_stage = state["stage"]
    if state["stage"] == "FollowUp" and state["decision"]["intent"] == "MoveOn":
        state["stage"] = "DeepDive"
    elif state["stage"] == "Intro" and state["decision"]["intent"] in ["Challenge", "Clarify"]:
        state["stage"] = "FollowUp"
    elif state["decision"]["intent"] == "FollowUp":
        state["stage"] = "FollowUp"
    else:
        state["stage"] = "DeepDive"
    
    # #region agent log
    #with open(_DEBUG_LOG_PATH, 'a') as f:
        #f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"arch.py:145","message":"stage_transition","data":{"old_stage":old_stage,"new_stage":state["stage"],"intent":state["decision"]["intent"]},"timestamp":int(__import__("time").time()*1000)})+'\n')
    # #endregion

    # Handle project switch
    if state["decision"]["intent"] == "SwitchProject":
        # #region agent log
        #with open(_DEBUG_LOG_PATH, 'a') as f:
            #f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"arch.py:148","message":"before_project_switch","data":{"old_project_id":state["active_project_id"],"old_stage":state["stage"],"old_decision_intent":state["decision"]["intent"]},"timestamp":int(__import__("time").time()*1000)})+'\n')
        # #endregion
        
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
                    "rationale": f"Starting interview for {proj_state['name']}"
                }
                # #region agent log
                #with open(_DEBUG_LOG_PATH, 'a') as f:
                    #f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"arch.py:156","message":"after_project_switch","data":{"new_project_id":proj_id,"stage_after_switch":state["stage"],"decision_after_switch":state["decision"]["intent"]},"timestamp":int(__import__("time").time()*1000)})+'\n')
                # #endregion
                break

        # Guard: if all projects are done, force WrapUp to prevent crash
        if state["active_project_id"] is None:
            state["decision"] = {
                "intent": "WrapUp",
                "urgency": 1.0,
                "confidence": 1.0,
                "rationale": "All projects covered"
            }

    # Hard guard: enforce followup limit
    if state["decision"]["intent"] == "FollowUp" and state.get("active_question_id"):
        project = state["projects"][state["active_project_id"]]
        root = project["parent_questions"].get(state["active_question_id"], state["active_question_id"])
        if project["followup_count"].get(root, 0) >= state["max_followups_per_question"]:
            state["decision"]["intent"] = "MoveOn"
            state["decision"]["rationale"] = "Followup limit reached, moving on"

    # Hard safety: force wrapup if max turns reached
    if state["turn_index"] >= state["max_turns"]:
        state["decision"] = {
            "intent": "WrapUp",
            "urgency": 1.0,
            "confidence": 1.0,
            "rationale": "Max turns reached"
        }
    
    # #region agent log
    #with open(_DEBUG_LOG_PATH, 'a') as f:
        #f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"arch.py:172","message":"node3_exit","data":{"active_project_id":state.get("active_project_id"),"stage":state.get("stage"),"decision_intent":state.get("decision",{}).get("intent") if state.get("decision") else None},"timestamp":int(__import__("time").time()*1000)})+'\n')
    # #endregion

    return state



def node4_final_performance_analysis(state: InterviewState) -> None:
    """Provides final analysis and recommendations after the interview."""
    final_report = b.FinalAnalysis(state)
    print("\n===== FINAL ANALYSIS AND RECOMMENDATIONS =====\n")
    print(final_report)


# if after node3 turn >= max_turn then END else 
# graph definition
graph = StateGraph(InterviewState)

# Nodes
graph.add_node("question_generator", node1_question_generator)
graph.add_node("answer_input", node2_answer_input)
graph.add_node("answer_analysis", node3_answer_analysis)
graph.add_node("final_analysis", node4_final_performance_analysis)

# Linear edges
graph.add_edge("question_generator", "answer_input")
graph.add_edge("answer_input", "answer_analysis")

# Conditional routing from node3
def route_from_analysis(state: InterviewState) -> str:
    intent = state["decision"]["intent"]

    if intent == "WrapUp":
        return "final_analysis"

    if intent == "SwitchProject":
        return "question_generator"

    # default loop
    return "question_generator"


graph.add_conditional_edges(
    "answer_analysis",
    route_from_analysis,
    {
        "final_analysis": "final_analysis",
        "question_generator": "question_generator",
    }
)

# End after final analysis
graph.add_edge("final_analysis", END)

# Entry point
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
            "overall_analysis":  {
                
                "weak_areas": [],
                "strong_areas": [],
                "unanswered_concepts": proj.tech_stack.copy(),
                "coverage_score": 0.0,
            }
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
            "rationale": "Initial probing"
        },
        "last_answer": None,
    }

    return initial_state



if __name__ == "__main__":
    import asyncio
    from dsa_client import DSABotClient

    file_path = "Primary_revised2.pdf"
    markdown = convert_resume_to_markdown(file_path)
    if markdown:
        print("Markdown Conversion Successful:")
        projects = b.ExtractProjects(markdown)
        projects.sort(  # sort by complexity descending
            key=lambda x: x.complexity,
            reverse=True
        )

        if not projects:
            print("No projects found.")
            exit(1)

        initial_state = initialise(projects)
        # Run resume/project interview
        final_state = compiled_graph.invoke(
            initial_state, 
            config={"recursion_limit": 100}
        )
        print("\n===== RESUME INTERVIEW COMPLETE =====\n")

        # ── Optional DSA round ────────────────────────────────────────────
        answer = input("\nWould you like to continue with a DSA coding round? (y/n): ").strip().lower()
        if answer == "y":
            dsa_client = DSABotClient()
            asyncio.run(dsa_client.run_interactive_session())
        else:
            print("\n👋  All done! Good luck with your interviews.")

    else:
        print("Markdown Conversion Failed.")


# from here these are few structs that can be in the final llm call output structure 

# class SkillProfile(TypedDict):
#     """Candidate's skill profile with ratings."""
#     system_design: float
#     implementation: float
#     debugging: float
#     tradeoffs: float
#     scalability: float
#     fundamentals: float


# class InterviewerNotes(TypedDict):
#     """Interviewer notes and observations."""
#     red_flags: List[str]
#     green_flags: List[str]
#     consistency_issues: List[str]



# def node3_answer_analysis(state: InterviewState, question: str, answer: str) -> None:
#     """Analyzes the candidate's answer."""
#     analysis, next_action = b.AnalyzeAnswer(question, answer)
#     # analysis contains clarity, correctness, depth, review, uncovered_gaps
#     project = state["projects"][state["active_project_id"]]
#     project["per_question_answer_analysis"][state["active_question_id"]] = analysis
#     state["next_action"] = next_action

#     if state["max_turns"] <= state["turn_index"]:
#         state["next_action"] = "wrapup"
#         # if next action is wrapup then we give final analysis and recommendations from node 3





# def node1_question_generator(state: InterviewState) -> None:
#     """Generates questions based on the current state of the interview."""
#     #when i am writing a node ->1. prompt and output structure ->2. we update the required state -> conditionally call next node (this is decided by enum chosen by the llm) ->3. finally return the output
#     project = state["projects"][state["active_project_id"]]
    
#     if state["next_action"] == "ask_question":
#         parent_question_list = [project["question_ids"][pqid] for qid, pqid in project["parent_questions"].items()]

#         question = b.GenerateQuestion(state) # generate question based on current project and past parent questions so that no repetition and focus on uncovered areas
#         # question contains question string, question_id, parent_question_id
#         project["question_ids"].append(question["question_id"], question["question"])
#         project["followup_count"][question["question_id"]] = 0
#         project["parent_questions"][question["question_id"]] = question["question_id"]  # new question's parent is itself 
#         state["active_question_id"] = question["question_id"]
        
    
#     elif state["next_action"] == "followup":
#         # if 
#         if(state["projects"][state["active_project_id"]]["followup_count"][project["parent_questions"][state["active_question_id"]]] > state["max_followups_per_question"]):
#             state["next_action"] = "ask_question"
#             node1_question_generator(state)
#         # now here do i make a edge in graph as well or this is fine?

#         followup_question = b.GenerateQuestion(state) # based on parent question and answer analysis generate followup question
#         project["question_ids"].append(followup_question["question_id"], followup_question["question"])
#         project["parent_questions"].append(followup_question["question_id"], project["parent_questions"][state["active_question_id"]])  # new followup question's parent is current active question
#         project["followup_count"][project["parent_questions"][followup_question["question_id"]]] += 1 # increment followup count for the parent question/ root question
#         state["active_question_id"] = followup_question["question_id"]     
        

#     elif state["next_action"] == "switch_project":
#         # Logic to switch to the next project
#         state["projects_done"][state["active_project_id"]] = True
#         state["active_project_id"] = None
#         state["active_question_id"] = None

#         for proj_id, proj_state in state["projects"].items():
#             if not state["projects_done"].get(proj_id, False):
#                 state["active_project_id"] = proj_id
#                 state["active_question_id"] = None
#                 state["next_action"] = "ask_question"
#                 break
#         return node1_question_generator(state)

# class InterviewState(TypedDict):
#     # control
#     stage: Literal["intro", "deep_dive", "follow_up", "wrap_up"]
#     turn_index: int # current turn index so that we can limit total turns and by total turns we mean all projects combined
#     max_turns: int 
#     # lets do per project max turns and per question id max followups but the problem is how will i track folloup of the followup question? 
#     max_followups_per_question: int = 2
#     max_questions_per_project: int = 5
#     # progress
#     projects_done: Dict[str, bool]  # project_id -> done status
#     # active focus
#     active_project_id: Optional[str]
#     active_question_id: Optional[str]
#     # memory
#     projects: Dict[str, ProjectState]   # project_id -> state
#     # routing
#     next_action: Optional[
#         Literal["ask_question", "followup", "switch_project", "wrapup"]
#     ]


