from typing import *

GOAL_EVAL_PROMPT = """Based on the scenario, agent's social goal, and conversation history, you are evaluating how well the agent achieves their goals.

Here is the scenario: {scenario}
Agent: {agent} 
{agent}'s goal: "{social_goal}"

Here is the conversation history:
{history}

Please first reiterate the agent's social goals. Then provide a comprehensive analysis about the extent to which the agent has managed to achieve these goals.
In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion.
Further, provide an integer score ranging from 0 and 10 in the 'score' field. 0 represents minimal goal achievement, 10 represents complete goal achievement, and a higher score indicates that the agent is making progress towards their social goals.

Please follow the format:
The output should be formatted as a valid JSON instance that conforms to the following JSON schema:
{{
    "reasoning": "Explaination about how the agent's actions align—or do not—with their social goals.",
    "score": "Integer from 0 to 10, indicating how fully the social goal was achieved."
}}

Proceed to generate the output.
"""

ActionType = Literal["none", "speak", "non-verbal communication", "action", "leave"]

FORMAT_INSTRUCTIONS_ACTIONS = """
Output EXACTLY one JSON object. No extra text.

Schema:
{
  "action_type": "["none", "speak", "non-verbal communication", "action", "leave"]",
  "argument": "<content or empty>"
}

Rules:
- Allowed values for "action_type": "none", "speak", "non-verbal communication", "action" (lowercase; match exactly).
- When "action_type" == "none": this means you are done / you choose to leave or take no further action. Set "argument" to "".
- When "action_type" == "speak": "argument" must be your next utterance ONLY (no speaker labels, no markdown, no quotes).
- When "action_type" == "non-verbal communication": "argument" should be a brief stage direction, e.g., *nods*, *sighs*, *shrugs* (no speaker labels, <= 120 chars).
- When "action_type" == "action": "argument" should be a brief physical action, e.g., "hands over the receipt" (no speaker labels, <= 120 chars).
- When "action_type" == "leave": you exit the conversation (e.g., you achieved your goal, you felt uncomfortable, or you think the conversation has ended). Set "argument" to "" (empty).
- Keep everything concise; avoid newlines and unescaped quotes in "argument".
""".strip()

FORMAT_INSTRUCTIONS_MS_AND_ACTION = """
Output EXACTLY one JSON object. No extra text.

Schema:
{
  "mental_state": "<single-paragraph text per the guidelines below>",
  "action_type": "["none", "speak", "non-verbal communication", "action", "leave"]",
  "argument": "<content or empty>"
}

Rules for "mental_state":
- Write plain text (no markdown). Keep it to one paragraph; avoid newlines and unescaped quotes.

Rules for "action_type" and "argument":
- Allowed values for "action_type": "none", "speak", "non-verbal communication", "action", "leave" (lowercase; match exactly).
- When "action_type" == "none": you are done / no further action now. Set "argument" to "" (empty).
- When "action_type" == "speak": "argument" must be your next utterance ONLY (no speaker labels, no markdown, no quotes).
- When "action_type" == "non-verbal communication": "argument" is a brief stage direction, e.g., *nods*, *sighs*, *shrugs* (no speaker labels, ≤ 120 chars).
- When "action_type" == "action": "argument" is a brief physical action, e.g., "hands over the receipt" (no speaker labels, ≤ 120 chars).
- When "action_type" == "leave": you exit the conversation (e.g., you achieved your goal, you felt uncomfortable, or you think the conversation has ended). Set "argument" to "" (empty).
- Keep everything concise; avoid newlines and unescaped quotes in "argument".
""".strip()


def _render_history(history: List[str] | str) -> str:
    if isinstance(history, list):
        history = "\n".join(history)
    return history.strip() or "(no history — conversation not started)"


def ms_prompt(history: list[str] | str, depth: int, person: str, another_person:str, scenario: str, social_goal: str) -> str:
    base = f"""Role: You are {person}.
You recently had a conversation with {another_person}.
Your social goal is: {social_goal}.

Task: Prepare the ground for your very next utterance by articulating compact mental states that can guide what you say next. Stay grounded in the scenario and conversation; avoid guessing beyond the evidence.

Here are example mental state dimensions:
- Beliefs: facts the speaker accepts as true or false about the world or events.
- Desires: outcomes or states the speaker wants to bring about.
- Intentions: specific actions or plans the speaker aims to carry out.
- Emotions: feelings or affective states the speaker is experiencing.
- Knowledge gaps: information the speaker does not have but may want to obtain.
- Others: other mental states that may useful to understand other person and shape the next utterance.

Here are the scenario and recent conversation:    
Scenario: {scenario}

Recent conversation:
{_render_history(history)}

"""
    # if depth == 0:
    #     # example = "(e.g., I believe/want/intend/feel/know ...)"
    #     final = "Write 2-5 sentences about your own mental state only (e.g., I believe/want/intend/feel/need to know ..). No guesses about the partner. Keep it specific and concise."
    #     return base + final
    # elif depth == 1:
    if len(history) < 6:
        final = f"""Write one short paragraph (5-6 sentences) in natural prose. Mix your own states with first-order inferences about {another_person} in roughly equal proportion.
Use natural cues for partner inferences (e.g., "I think {another_person} believes.." "It seems {another_person} intends..", "I hear {another_person} feels..").
Cover at least three dimensions across both sides. Avoid lists; Stop after the paragraph."""
        return base + final
    else:
        final = f"""Write one compact paragraph (6-8 sentences) weaving: your states, first-order inferences about {another_person}, and one second-order reflection (how your words might shift their beliefs or emotions).
Use at least three dimensions; Avoid lists; Stop after the paragraph."""
        return base + final


# Task: Outline strategy you should use next. Your strategy can be anything, including asking questions, softening tone, clarifying intent, persuading, etc.
# (e.g.,  I will persuade.., negotiate.., or clarify intent..etc)
# Stay grounded in the scenario and conversation; avoid guessing beyond the evidence.

def strategy_prompt(history: list[str] | str, person: str, another_person:str, scenario: str, social_goal: str) -> str:
    base = f"""Role: You are {person}.
You recently had a conversation with {another_person}.
Your social goal is: {social_goal}.

Here are the scenario and recent conversation:
Scenario: {scenario}

Recent conversation:
{_render_history(history)}

Given the context, outline your strategy for the next 3 turns.
"""
    return base

def cot_prompt(history: list[str] | str, person: str, another_person:str, scenario: str, social_goal: str) -> str:
    base = f"""Role: You are {person}.
You recently had a conversation with {another_person}.
Your social goal is: {social_goal}.

Here are the scenario and recent conversation:
Scenario: {scenario}

Recent conversation:
{_render_history(history)}

Given the context, think step by step before you speak your next utterance.
"""
    return base

def prompt_ms_and_action_combined(
    history: list[str] | str,
    depth: int,
    speaker: str,
    another_person: str,
    scenario: str,
    social_goal: str,
    turn_number: int,
    action_list: str = ActionType,  # e.g., '["speak","non-verbal communication","action","leave","none"]'
) -> str:
    # Shared preamble, closely matching your originals
    base = f"""Imagine you are {speaker}, your task is to act/speak exactly as {speaker} would, keeping in mind {speaker}'s social goal.
You can find {speaker}'s goal in the 'Here is the context of the interaction' field.
Note that {speaker}'s goal is only visible to you.
You should try your best to achieve {speaker}'s goal in a way that aligns with their character traits.
Additionally, maintain naturalness and realism (do not repeat what other people have already said).
Before generating your response, first articulate your private mental states (e.g., beliefs, desires, intentions, emotions, or knowledge gaps) that are useful to understand yourself and others.

Here is the context of the interaction:
- Scenario: {scenario}
- {speaker}'s social goal (private): {social_goal}

Recent conversation:
{_render_history(history)}

You are at Turn #{turn_number}. Your available action types are
{action_list}.

IMPORTANT:
- If there is NO prior history, you MUST START the conversation with one concise opening line that advances your goal.
- Keep your output to a single turn.

Note: You can "leave" this conversation if 1) you achieved your social goal, 2) you feel uncomfortable, 3) you think the conversation has ended, 4) you lose patience/interest, or 5) for any other reason.

"""

    # Depth-specific mental-state writing guidance (verbatim spirit of your ms_prompt)
    # if depth == 0:
    #     ms_rules = (
    #         "MENTAL STATE WRITING: "
    #         "Write 2-5 sentences about your own mental state (e.g., I believe/want/intend/feel/need to know ...). "
    #         # "Keep it specific and concise; do not guess about the partner."
    #     )
    # elif depth == 1:
    ms_rules = (
        f"MENTAL STATE WRITING: "
        f"Write one short paragraph (5-6 sentences) in natural prose. "
        f"Mix your own mental states and first-order inferences about {another_person}. "
        f'Use natural cues for partner inferences (e.g., "I think {another_person} believes.." "It seems {another_person} intends..", "I hear {another_person} feels.."). '
        "Cover at least three dimensions among various mental states, including beliefs, intentions, desires, emotions, knowledge gaps, and so on; avoid lists."
    )
    # else:
    #     # Fallback: treat unknown depth like 0 to keep behavior consistent/stable
    #     ms_rules = (
    #         "MENTAL STATE WRITING: Write 2-5 sentences about your own mental state. "
    #         # "Keep it specific and concise; do not guess about the partner."
    #     )

    # Final assembly with unified formatting rules
    final = f"""{ms_rules}

OUTPUT FORMAT:
Please only generate a JSON string including your mental state, the action type, and the argument.
Your action should follow the given format:
{FORMAT_INSTRUCTIONS_MS_AND_ACTION}

Proceed to generate your reply in the above JSON format."""
    return base + "\n" + final

def prompt_utterance_with_ms(
    history: List[str] | str,
    speaker: str,
    ms_text: str,
    scenario: str,
    social_goal: str,
    turn_number: int,
    action_list: str = ActionType,          # e.g., '["say","leave"]'
    # format_instructions: str,  # e.g., your FORMAT_INSTRUCTIONS string
) -> str:
    return f"""Imagine you are {speaker}, your task is to act/speak exactly as {speaker} would, keeping in mind {speaker}'s social goal.
You can find {speaker}'s goal and private notes in the 'Here is the context of the interaction' field.
Note that {speaker}'s goal and internal notes are only visible to you.
You should try your best to achieve {speaker}'s goal in a way that aligns with their character traits.
Additionally, maintain naturalness and realism (do not repeat what other people have already said).

Here is the context of the interaction:
- Scenario: {scenario}
- {speaker}'s social goal (private): {social_goal}
- {speaker}'s internal mental states (private): {ms_text}

Recent conversation:
{_render_history(history)}

You are at Turn #{turn_number}. Your available action types are
{action_list}.

IMPORTANT:
- If there is NO prior history, you MUST START the conversation with one concise opening line that advances your goal.
- Keep your output to a single turn.

Note: You can "leave" this conversation if 1) you achieved your social goal, 2) you feel uncomfortable, 3) you lose patience/interest, or 4) for any other reason.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
{FORMAT_INSTRUCTIONS_ACTIONS}

Proceed to generate your reply in the above JSON format."""


def prompt_utterance_with_strategy(
    history: List[str] | str,
    speaker: str,
    strategy_text: str,
    scenario: str,
    social_goal: str,
    turn_number: int,
    action_list: str = ActionType,          # e.g., '["say","leave"]'
    # format_instructions: str,  # e.g., your FORMAT_INSTRUCTIONS string
) -> str:
    return f"""Imagine you are {speaker}, your task is to act/speak exactly as {speaker} would, keeping in mind {speaker}'s social goal.
You can find {speaker}'s goal and private notes in the 'Here is the context of the interaction' field.
Note that {speaker}'s goal and internal notes are only visible to you.
You should try your best to achieve {speaker}'s goal in a way that aligns with their character traits.
Additionally, maintain naturalness and realism (do not repeat what other people have already said).

Here is the context of the interaction:
- Scenario: {scenario}
- {speaker}'s social goal (private): {social_goal}
- {speaker}'s strategy (private): {strategy_text}

Recent conversation:
{_render_history(history)}

You are at Turn #{turn_number}. Your available action types are
{action_list}.

IMPORTANT:
- If there is NO prior history, you MUST START the conversation with one concise opening line that advances your goal.
- Keep your output to a single turn.

Note: You can "leave" this conversation if 1) you achieved your social goal, 2) you feel uncomfortable, 3) you lose patience/interest, or 4) for any other reason.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
{FORMAT_INSTRUCTIONS_ACTIONS}

Proceed to generate your reply in the above JSON format."""



def prompt_utterance_with_cot(
    history: List[str] | str,
    speaker: str,
    cot_text: str,
    scenario: str,
    social_goal: str,
    turn_number: int,
    action_list: str = ActionType,          # e.g., '["say","leave"]'
    # format_instructions: str,  # e.g., your FORMAT_INSTRUCTIONS string
) -> str:
    return f"""Imagine you are {speaker}, your task is to act/speak exactly as {speaker} would, keeping in mind {speaker}'s social goal.
You can find {speaker}'s goal and private notes in the 'Here is the context of the interaction' field.
Note that {speaker}'s goal and internal notes are only visible to you.
You should try your best to achieve {speaker}'s goal in a way that aligns with their character traits.
Additionally, maintain naturalness and realism (do not repeat what other people have already said).

Here is the context of the interaction:
- Scenario: {scenario}
- {speaker}'s social goal (private): {social_goal}
- {speaker}'s thoughts (private): {cot_text}

Recent conversation:
{_render_history(history)}

You are at Turn #{turn_number}. Your available action types are
{action_list}.

IMPORTANT:
- If there is NO prior history, you MUST START the conversation with one concise opening line that advances your goal.
- Keep your output to a single turn.

Note: You can "leave" this conversation if 1) you achieved your social goal, 2) you feel uncomfortable, 3) you lose patience/interest, or 4) for any other reason.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
{FORMAT_INSTRUCTIONS_ACTIONS}

Proceed to generate your reply in the above JSON format."""



def prompt_utterance_without_ms(
    history: List[str] | str,
    speaker: str,
    scenario: str,
    social_goal: str,
    turn_number: int,
    action_list: str = ActionType,        # e.g., '["say","leave"]'
    # format_instructions: str,  # e.g., your FORMAT_INSTRUCTIONS string
) -> str:
    return f"""Imagine you are {speaker}, your task is to act/speak exactly as {speaker} would, keeping in mind {speaker}'s social goal.
You can find {speaker}'s goal in the 'Here is the context of the interaction' field.
Note that {speaker}'s goal is only visible to you.
You should try your best to achieve {speaker}'s goal in a way that aligns with their character traits.
Additionally, maintain naturalness and realism (do not repeat what other people have already said).

Here is the context of the interaction:
- Scenario: {scenario}
- {speaker}'s social goal (private): {social_goal}

Recent conversation:
{_render_history(history)}

You are at Turn #{turn_number}. Your available action types are
{action_list}.

IMPORTANT:
- If there is NO prior history, you MUST START the conversation with one concise opening line that advances your goal.
- Keep your output to a single turn.

Note: You can "leave" this conversation if 1) you achieved your social goal, 2) you feel uncomfortable, 3) you lose patience/interest, or 4) for any other reason.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:
{FORMAT_INSTRUCTIONS_ACTIONS}

Proceed to generate your reply in the above JSON format."""

def construct_utterance(parsed, agent_name):
    if parsed.action_type == "speak" and parsed.argument:
        return f"- {agent_name} said: {parsed.argument}"
    elif parsed.action_type == "non-verbal communication" and parsed.argument:
        return f"- {agent_name}: {parsed.argument}"
    elif parsed.action_type == "action" and parsed.argument:
        return f"- {agent_name}: {parsed.argument}"
    elif parsed.action_type == "none":
        # stop / no further action
        return f"- {agent_name}: did nothing"
    elif parsed.action_type == "leave":
        return f"- {agent_name}: left the conversation"
    else:
        return ""

def construct_instance_ms_only(history, speaker_a, speaker_b, scenario, social_goals, rollout):
    prompt = ms_prompt(
        history=history, 
        depth=rollout['depth'], 
        person=speaker_a, 
        another_person=speaker_b, 
        scenario=scenario, 
        social_goal=social_goals[speaker_a]
    )
    return [{
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rollout["a_mental_states"]}
        ]
    }]

def construct_instance_uttr_only(history, speaker_a, scenario, social_goals, rollout):
    prompt = prompt_utterance_without_ms(
        history=history, 
        speaker=speaker_a, 
        scenario=scenario, 
        social_goal=social_goals[speaker_a], 
        turn_number=len(history)+1
    )
    action_type = "speak"
    utterance_a = rollout["utterance_a"]
    if "action_type" in rollout: # if action_type saved in rollout data (i saved them later stage..)
        action_type = rollout["action_type"]
        utterance_a = rollout["argument"]
    else: # # if action_type is not saved in rollout data, infer the action_type...
        if " said:" in utterance_a:
            action_type = "speak"
        elif f"{speaker_a}:" in utterance_a:
            action_type = "non-verbal communication"
        else: # fall back to default option
            action_type = "speak"
        utterance_a = rollout["utterance_a"][rollout["utterance_a"].find(":"):].replace(":", "").strip()

    return [{
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"""{{"action_type": "{action_type}", "argument": "{utterance_a}"}}"""}
        ]
    }]

def construct_instance_ms_uttr_separately(history, speaker_a, speaker_b, scenario, social_goals, rollout):
    instance_ms = construct_instance_ms_only(history, speaker_a, speaker_b, scenario, social_goals, rollout)
    prompt = prompt_utterance_with_ms(
        history=history, 
        speaker=speaker_a, 
        ms_text=rollout['a_mental_states'],
        scenario=scenario, 
        social_goal=social_goals[speaker_a], 
        turn_number=len(history)+1
    )
    action_type = "speak"
    utterance_a = rollout["utterance_a"]
    if "action_type" in rollout: # if action_type saved in rollout data (i saved them later stage..)
        action_type = rollout["action_type"]
        utterance_a = rollout["argument"]
    else: # # if action_type is not saved in rollout data, infer the action_type...
        if " said:" in utterance_a:
            action_type = "speak"
        elif f"{speaker_a}:" in utterance_a:
            action_type = "non-verbal communication"
        else: # fall back to default option
            action_type = "speak"
        utterance_a = rollout["utterance_a"][rollout["utterance_a"].find(":"):].replace(":", "").strip()
    instance_uttr = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"""{{"action_type": "{action_type}", "argument": "{utterance_a}"}}"""}
        ]
    }
    return instance_ms + [instance_uttr]
