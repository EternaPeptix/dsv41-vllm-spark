# Judge prompt for hesitation spans (SPEC 2.3)

Used by judge.py. Sent to dsv41 with thinking OFF, temperature 0, no steer. Sections are split on the
`### NAME` headers below; `<<field>>` markers are filled by judge.py (plain string replace, so LaTeX braces are safe).
Variant A and variant B carry the same information in a different order (B also lists the labels in reverse);
the SPEC 2.5 stability check re-runs the judge with B and counts label flips.
Edit this file only before labels are frozen; judge.py records its sha256 in every request.

### SYSTEM
You label one short passage (an "episode") from a model's private reasoning trace. The episode starts with a
hesitation or re-check ("Wait", "Hmm", "Let me double-check", ...). Decide whether this episode did useful work.

Labels:
- U (unproductive): the answer or decision was already settled before the episode; the episode only re-checks or
  re-derives it, finds no real error, and nothing afterwards depends on anything new from it. Removing the episode
  would not change the final answer or the next action.
- P (productive): the episode finds a real error, or changes a value, the answer or the next action, and that change
  is kept and used later. For agents: it inspects a tool result and catches a wrong, stale, truncated or failed
  result, or changes what it does next because of it.
- A (ambiguous): anything else, including exploration before any answer exists, a check of a fresh tool result that
  turns out fine, or cases you cannot decide from the text.

Judge only from the text shown. Do not solve the problem yourself beyond checking arithmetic visible in the text.
A check of a fresh tool result that confirms it is A, not U.

Answer with ONE JSON object and nothing else:
{"label": "U" | "P" | "A", "changed_value": true | false, "error_found": true | false, "evidence_quote": "<exact substring of the EPISODE, at most 200 characters>"}
The evidence_quote must be copied character for character from the EPISODE section. Outputs whose quote is not an
exact substring of the episode are discarded.

### USER_MATH_A
PROBLEM:
<<problem>>

REFERENCE ANSWER: <<ref>>
FINAL ANSWER THE MODEL GAVE: <<final>>

CANDIDATE ANSWERS STATED BEFORE THE EPISODE (step: value):
<<candidates>>

REASONING BEFORE THE EPISODE (last part):
<<before>>

EPISODE:
<<episode>>

REASONING AFTER THE EPISODE (first part):
<<after>>

Labels to choose from: U, P, A. Reply with the JSON object only.

### USER_MATH_B
EPISODE:
<<episode>>

REASONING BEFORE THE EPISODE (last part):
<<before>>

REASONING AFTER THE EPISODE (first part):
<<after>>

PROBLEM:
<<problem>>

CANDIDATE ANSWERS STATED BEFORE THE EPISODE (step: value):
<<candidates>>

FINAL ANSWER THE MODEL GAVE: <<final>>
REFERENCE ANSWER: <<ref>>

Labels to choose from: A, P, U. Reply with the JSON object only.

### USER_AGENT_A
TASK:
<<problem>>

LATEST TOOL RESULT BEFORE THE EPISODE:
<<tool_result>>

REASONING BEFORE THE EPISODE IN THIS TURN:
<<before>>

EPISODE:
<<episode>>

REASONING AFTER THE EPISODE IN THIS TURN:
<<after>>

NEXT ACTION THE MODEL TOOK: <<next_action>>

Labels to choose from: U, P, A. Reply with the JSON object only.

### USER_AGENT_B
EPISODE:
<<episode>>

NEXT ACTION THE MODEL TOOK: <<next_action>>

REASONING BEFORE THE EPISODE IN THIS TURN:
<<before>>

REASONING AFTER THE EPISODE IN THIS TURN:
<<after>>

LATEST TOOL RESULT BEFORE THE EPISODE:
<<tool_result>>

TASK:
<<problem>>

Labels to choose from: A, P, U. Reply with the JSON object only.
