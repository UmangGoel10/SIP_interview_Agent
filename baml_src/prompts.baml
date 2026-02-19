class Project {
  name string
  description string[]
  tech_stack string[] 
  role string
  complexity int
}

enum Intent {
  ProbeDepth
  Clarify
  Challenge
  MoveOn
  SwitchProject
  WrapUp
}

enum InterviewStage {
  Intro
  DeepDive
  FollowUp
  WrapUp
}

class AnswerAnalysis {
  clarity int
  correctness int
  depth int
  review string

}
class ProjectPastContext {
  weak_areas string[]
  strong_areas string[]
  unanswered_concepts string[]
  coverage_score float

}

class ProjectState {
  name string
  description string[]
  claimed_stack string[]
  complexity int
  question_ids map<string, string>
  followup_count map<string, int>
  parent_questions map<string, string>
  per_question_answer_analysis map<string, AnswerAnalysis>?
  overall_analysis ProjectPastContext
}

class InterviewDecision {
  intent Intent
  urgency float
  confidence float
  rationale string
}

class InterviewState {
  stage InterviewStage
  turn_index int
  max_turns int
  max_followups_per_question int
  max_questions_per_project int
  projects_done map<string, bool>
  active_project_id string | null
  active_question_id string | null
  projects map<string, ProjectState>
  decision InterviewDecision
  last_answer string | null
}

class QuestionResponse {
  question string
  question_id string
  parent_question_id string
}

class AnalysisResponse {
  analysis AnswerAnalysis
  decision InterviewDecision
  pastContext ProjectPastContext

}



function ExtractProjects(resume_text: string) -> Project[]{

    client MyFreeClient 
    prompt #"
    {{ _.role('system') }}
    Extract projects from the input. For each project, follow these specific formatting rules for the 'description' field:
    
    1. The FIRST element must be a high-level summary of the project's purpose.
    2. Subsequent elements must be specific technical details, such as architecture choices, performance wins, or specific algorithms used
    
    GUIDELINES:
    1. For 'tech_stack', extract specific tools/languages. If none are mentioned, return null.
    2. For 'complexity', rate the project from 1-10 based on technical depth and scale.
    3. For 'role', identify the specific contribution (e.g., 'Lead Developer', 'Contributor').

    include internship projects as well.

    {{ ctx.output_format }}

    {{ _.role('user') }}
    Analyze the following content and extract the projects:
    ---
    {{ resume_text }}
    ---

    Before providing the JSON, briefly reason about why you assigned the specific complexity score for each project.
  "#

}


function GenerateQuestion(
  state: InterviewState,
  intent: Intent,
  urgency: float
) -> QuestionResponse {

  client MyFreeClient

  prompt #"
  {# ================= SYSTEM PERSONA ================= #}
  You are a senior technical interviewer having a natural technical conversation.
  Each question should feel like a continuation of what was just discussed,
  not a fresh or disconnected prompt.

  Ask ONE precise, high-signal question per turn.

  {% set project = state.projects[state.active_project_id] %}
  Active Project: {{ project.name }}
  Stack: {{ project.claimed_stack | join(", ") }}
  Complexity: {{ project.complexity }}/10

  {# ================= STAGE: INTRO ================= #}
  {% if state.stage == "INTRO" %}
    TASK:
    Ask ONE open-ended, conversational question that invites the candidate to:
    - Explain what the project does
    - Describe the problem it solves
    - Clarify their role

    Style:
    - Natural and human (as in a real interview)
    - Encouraging, not interrogative

    Rules:
    - No implementation details
    - No challenges or edge cases

    {{ ctx.output_format }}

  {% endif %}

  {# ================= CONTEXT: PAST QUESTIONS ================= #}
  {% if project.question_ids | length > 0 %}
    {% set total_q = project.question_ids | length %}
    {% set max_q = state.max_questions_per_project %}
    
    Questions: {{ total_q }}/{{ max_q }}
    Coverage: {{ project.overall_analysis.coverage_score * 100 | round }}%
    
    {% if project.unanswered_concepts %}
    Untested: 
    {%- for concept in project.unanswered_concepts -%}
        {{ concept }}{% if not loop.last %}, {% endif %}
    {%- endfor -%}
    {% endif %}
    
    {% if project.weak_areas %}
    Weak Areas: 
    {%- for area in project.weak_areas -%}
        {{ area }}{% if not loop.last %}, {% endif %}
    {%- endfor -%}
    {% endif %}

    {% if project.strong_areas %}
    Strong Areas: 
    {%- for area in project.strong_areas -%}
        {{ area }}{% if not loop.last %}, {% endif %}
    {%- endfor -%}
    {% endif %}

    {% if total_q >= 3 %}
      Recent Questions (last 3):
      {% for  q in project.question_ids | reverse | slice(3) %}
      - {{ q.value }}
      {% endfor %}
    {% endif %}

  {% else %}
    No questions asked yet.
    Coverage: 0%
    All concepts untested: {{ project.claimed_stack | join(", ") }}
    ASK TO GIVE A HIGH-LEVEL OVERVIEW OF THE PROJECT FIRST.

  {% endif %}


  {# ================= INTERVIEWER DECISION ================= #}

  RULE:
  - Do NOT repeat or paraphrase any of the questions above.
  - Each new question must advance the discussion into a new angle or deeper layer.

  {# ================= FOLLOW-UP MODE ================= #}
  {% if intent == Intent.FollowUp and state.active_question_id %}
    {% set qid = state.active_question_id %}
    {% set analysis = project.per_question_answer_analysis[qid] %}

    Last Question Asked:
    {{ project.question_ids[qid] }}

    Answer Analysis:
    - Strengths: {{ analysis.strengths }}
    - Gaps: {{ analysis.gaps }}
    - Weaknesses: {{ analysis.weaknesses }}
    - Confidence: {{ analysis.confidence }}

    TASK:
    Ask ONE conversational follow-up that:
    - Naturally references what the candidate just said
    - Zooms in on a specific gap or weak assumption
    - Feels like “let’s dig a bit deeper here” rather than a new topic

    Style Rules:
    - Phrase it as a continuation (e.g., “You mentioned…”, “Earlier you said…”)
    - Avoid reset phrases like “Explain” or “Describe”

    Constraints:
    - No multi-part questions
    - Narrow aggressively if urgency > 0.7

    {{ ctx.output_format }}
    STOP.
  {% endif %}

  {# ================= DEFAULT DEEPDIVE ================= #}
  TASK:
  Ask ONE conversational, implementation-rooted question based on:
  {{ project.claimed_stack | join("/") }}

  Intent: {{ intent }}
  Urgency: {{ urgency }}

  Conversational Guidelines:
  - Reference earlier discussion implicitly or explicitly
  - Prefer continuation phrases:
      • “When you were building…”
      • “Earlier you mentioned…”
      • “That choice makes sense — how did you…”
  - Focus on trade-offs, constraints, or real failure cases, backtesting their decisions
  - Avoid sounding like a checklist or exam question

  Hard Constraints:
  - No textbook definitions
  - No multi-part questions

  {{ ctx.output_format }}

  "#
}


function AnalyzeAnswer(question: string, answer: string, state: InterviewState) -> AnalysisResponse{
    client MyFreeClient
    prompt #"
    {{ _.role('system') }}
    You are a meticulous technical interviewer analyzing candidate answers.

    TASK 1 - Evaluate Answer:
    1. Clarity (1-10): How clearly explained?
    2. Correctness (1-10): Technically sound?
    3. Depth (1-10): Trade-offs, alternatives, constraints covered?
    
    Provide:
    - Concise review (strengths/weaknesses)
    - Uncovered gaps or misconceptions from THIS answer

    TASK 2 - Decide Next Action:
    - **ProbeDepth**: Good answer, needs deeper implementation details
    - **Clarify**: Unclear, ambiguous, needs precision
    - **Challenge**: Incorrect, inconsistent, or needs validation
    - **MoveOn**: Satisfactory and complete, new aspect needed
    - **SwitchProject**: Project sufficiently covered
    - **WrapUp**: Conclude interview

    TASK 3 - Update Project Context:
    Based on ALL answers so far (including this one):
    - **weak_areas**: Topics/concepts where candidate struggled or showed gaps
    - **strong_areas**: Topics/concepts where candidate demonstrated solid understanding
    - **unanswered_concepts**: Technologies from claimed stack NOT yet tested/discussed
    - **coverage_score** (0.0-1.0): % of claimed tech stack adequately covered

    {{ ctx.output_format }}

    {{ _.role('user') }}

    {% set project = state.projects[state.active_project_id] %}
    {# Initialize the variable #}
    {% set total_q = 0 %}
    {% for id in project.question_ids %}
      {# Incrementing logic depends on your BAML runtime; usually, 
        you just iterate to display. If you need the number in the prompt: #}
      {% if loop.last %}{% set total_q = loop.index %}{% endif %}
    {% endfor %}

Total number of questions: {{ total_q }}

Total Questions: {{ total_q }}

    {% set max_q = state.max_questions_per_project %}
    {% set global_urgency = state.turn_index / state.max_turns %}

    Covereage: {{ (project.overall_analysis.coverage_score * 100) | round }}%
    **Budget Status:**
    Turns: {{ state.turn_index }}/{{ state.max_turns }} ({{ (global_urgency * 100) | round }}% used)
    Questions this project: {{ total_q }}/{{ max_q }}
    {% if state.active_question_id %}
    {% set current_root = project.parent_questions.get(state.active_question_id, state.active_question_id) %}
    Followups on current thread: {{ project.followup_count.get(current_root, 0) }}/{{ state.max_followups_per_question }}
    {% endif %}

    **Decision Guidance:**
    {% if total_q >= max_q * 0.8 %}
      Question budget nearly exhausted ({{ total_q }}/{{ max_q }}) - PREFER MoveOn over followups
    {% endif %}
    {% if project.coverage_score >= 0.75 %}
      Coverage satisfactory ({{ (project.coverage_score * 100) | round }}%) - CONSIDER SwitchProject, don't exhaust question budget
    {% endif %}
    {% if global_urgency >= 0.8 %}
      Global time critical ({{ (global_urgency * 100) | round }}% used) - AVOID deep followups, prioritize breadth
    {% endif %}
    {% if total_q >= max_q %}
      Question budget exhausted - MUST use MoveOn or SwitchProject
    {% endif %}

    **Project: {{ project.name }}**
    Claimed Stack: {{ project.claimed_stack | join(", ") }}
    Current Coverage: {{ (project.coverage_score * 100) | round }}%
    {% if project.weak_areas | length > 0 %}Current Weak: {{ project.weak_areas | join(", ") }}{% endif %}
    {% if project.strong_areas | length > 0 %}Current Strong: {{ project.strong_areas | join(", ") }}{% endif %}
    {% if project.unanswered_concepts | length > 0 %}Currently Untested: {{ project.unanswered_concepts | join(", ") }}{% endif %}

    {% if project.per_question_answer_analysis | length > 0 %}
    **Accumulated Gaps from Previous Answers:**
    {% set all_gaps = [] %}
    {% for var in project.per_question_answer_analysis %}
    {% set _ = all_gaps.extend(var.value.uncovered_gaps) %}
    {% endfor %}
    {% if all_gaps | length > 0 %}{{ all_gaps | unique | join(", ") }}{% endif %}
    {% endif %}

    **Current Question:**
    {{ question }}

    **Candidate's Answer:**
    {{ answer }}

    ---
    Now provide your complete analysis with all three tasks: answer evaluation, decision, and updated project context.
    "#
}

function FinalAnalysis(state: InterviewState) -> string {
    client MyFreeClient
    prompt #"
    {{ _.role('system') }}
    You are a senior technical interviewer providing a comprehensive final evaluation.

    TASK:
    Provide a structured final report with:
    1. Overall Performance Summary (strengths, weaknesses, consistency)
    2. Technical Depth Assessment (per technology/concept)
    3. Red Flags & Green Flags
    4. Hiring Recommendation (Strong Hire / Hire / No Hire / Strong No Hire)
    5. Suggested Level (Junior / Mid / Senior / Staff)

    {{ ctx.output_format }}

    {{ _.role('user') }}

    **Interview Summary:**
    Total Turns: {{ state.turn_index }}/{{ state.max_turns }}
    {% set completed_count = 0 %}
    {% for proj_id in state.projects_done.keys() %}
    {% if state.projects_done[proj_id] %}{% set completed_count = completed_count + 1 %}{% endif %}
    {% endfor %}
    Projects Covered: {{ completed_count }}/{{ state.projects.keys() | length }}

    {% for proj_id in state.projects.keys() %}
    {% set project = state.projects[proj_id] %}
    ---
    **Project {{ loop.index }}: {{ project.name }}**
    Complexity: {{ project.complexity }}/10
    Claimed Stack: {{ project.claimed_stack | join(", ") }}
    
    Coverage: {{ (project.coverage_score * 100) | round }}%
    Questions Asked: {{ project.question_ids.keys() | length }}
    
    {% if project.strong_areas | length > 0 %}
    Strong Areas: {{ project.strong_areas | join(", ") }}
    {% endif %}
    
    {% if project.weak_areas | length > 0 %}
    Weak Areas: {{ project.weak_areas | join(", ") }}
    {% endif %}
    
    {% if project.unanswered_concepts | length > 0 %}
    Untested: {{ project.unanswered_concepts | join(", ") }}
    {% endif %}

    **Answer Quality Distribution:**
    {% for qid in project.per_question_answer_analysis.keys() %}
    {% set analysis = project.per_question_answer_analysis[qid] %}
    Q{{ loop.index }}: Clarity={{ analysis.clarity }}/10, Correctness={{ analysis.correctness }}/10, Depth={{ analysis.depth }}/10
    Review: {{ analysis.review }}
    {% if analysis.uncovered_gaps | length > 0 %}
      Gaps: {{ analysis.uncovered_gaps | join(", ") }}
    {% endif %}
    {% endfor %}

    {% endfor %}

    **Cross-Project Patterns:**
    {% set all_weak = [] %}
    {% set all_strong = [] %}
    {% for proj_id in state.projects.keys() %}
    {% set project = state.projects[proj_id] %}
    {% for weak in project.weak_areas %}
    {% set _ = all_weak.append(weak) %}
    {% endfor %}
    {% for strong in project.strong_areas %}
    {% set _ = all_strong.append(strong) %}
    {% endfor %}
    {% endfor %}
    
    Recurring Weaknesses: {{ all_weak | unique | join(", ") if all_weak | length > 0 else "None" }}
    Consistent Strengths: {{ all_strong | unique | join(", ") if all_strong | length > 0 else "None" }}
    "#
}

// test TestName {
//   functions [GenerateQuestion]
//   args {
//     state {
//         stage Intro
//         turn_index 1
//         max_turns 20
//         max_followups_per_question 3
//         max_questions_per_project 6
//         projects_done {
//         "a_string" false
//       }
//         active_project_id "a_string"
//         active_question_id null
//         projects {
//         "a_string" {
//             name #"
//             hello world
//           "#
//             description [
//             #"
//               hello world
//             "#,
//             #"
//               hello world
//             "#
//           ]
//             claimed_stack [
//             #"
//               hello world
//             "#,
//             #"
//               hello world
//             "#
//           ]
//             complexity 123
//             question_ids null
//             followup_count null
//             parent_questions null
//             per_question_answer_analysis null
//             overall_analysis null
//         }
//       }
//         decision null
//     }
//     intent ProbeDepth
//     urgency 0.5
//   }
// }

test TestName {
  functions [GenerateQuestion]
  args {
    state {
      stage "Intro"
      turn_index 1
      max_turns 20
      max_followups_per_question 3
      max_questions_per_project 6
      projects_done {
        "project_0" false
        "project_1" false
        "project_2" false
        "project_3" false
      }
      active_project_id "project_0"
      active_question_id "probe_lstm_overfitting"
      projects {
        "project_0" {
          name "CDS Index Trading via Merton-Enhanced LSTM Models"
          description [
            "Developed and implemented a Merton-LSTM model for forecasting CDS index spreads",
            "Used machine learning techniques like LSTM and Merton's model for credit risk assessment",
            "Demonstrated superior profitability and risk management compared to baseline models"
          ]
          claimed_stack ["Python", "Machine Learning", "LSTM"]
          complexity 9
          question_ids {
            "probe_lstm_overfitting" "How did you handle overfitting in your Merton-LSTM model implementation, specifically with regards to the LSTM component and the dataset used for training the CDS index spread forecasts?"
          }
          followup_count {
            "None" 1
          }
          parent_questions {
            "probe_lstm_overfitting" "None"
          }
          per_question_answer_analysis {}
          overall_analysis {
            weak_areas []
            strong_areas []
            unanswered_concepts ["Python", "Machine Learning", "Data Structures"]
            coverage_score 0.0
          }
        }
        "project_1" {
          name "Machine Learning Intern, NRSC ISRO"
          description [
            "Developed a high-performance LULC classification model for Delhi",
            "Used Sentinel 2 L2A imagery and achieved 88% accuracy"
          ]
          claimed_stack ["Python", "Machine Learning", "Data Structures"]
          complexity 8
          question_ids {}
          followup_count {}
          parent_questions {}
          per_question_answer_analysis {}
          overall_analysis {
            weak_areas []
            strong_areas []
            unanswered_concepts ["Python", "Machine Learning", "Data Structures"]
            coverage_score 0.0
          }
        }
      }
      decision {
        intent "ProbeDepth"
        urgency 0.3
        confidence 0.6
        rationale "Initial probing"
      }
      last_answer "yo"
    }
    intent ProbeDepth
    urgency 0.5

  }
}

test TestName {
  functions [AnalyzeAnswer]
  args {
    question #"
      hello world
    "#
    answer #"
      hello world
    "#
    state {
        stage Intro
        turn_index 123
        max_turns 123
        max_followups_per_question 123
        max_questions_per_project 123
        projects_done {
        "a_string" true
      }
        active_project_id null
        active_question_id null
        projects {
        "a_string" {
            name #"
            hello world
          "#
            description [
            #"
              hello world
            "#,
            #"
              hello world
            "#
          ]
            claimed_stack [
            #"
              hello world
            "#,
            #"
              hello world
            "#
          ]
            complexity 123
            question_ids {
            "a_string" #"
              hello world
            "#
          }
            followup_count {
            "a_string" 123
          }
            parent_questions {
            "a_string" #"
              hello world
            "#
          }
            per_question_answer_analysis null
            overall_analysis {
              weak_areas [
              #"
                hello world
              "#,
              #"
                hello world
              "#
            ]
              strong_areas [
              #"
                hello world
              "#,
              #"
                hello world
              "#
            ]
              unanswered_concepts [
              #"
                hello world
              "#,
              #"
                hello world
              "#
            ]
              coverage_score 0.5
          }
        }
      }
        decision {
          intent ProbeDepth
          urgency 0.5
          confidence 0.5
          rationale #"
          hello world
        "#
      }
        last_answer null
    }
  }
}












