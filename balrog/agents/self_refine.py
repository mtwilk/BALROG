import re
from balrog.agents.symbolic import SymbolicAgent
from balrog.prompt_builder.history import Message

FEEDBACK_PROMPT = """

You are evaluating an attempt of a player at a text game. Before their answer is sent to the game, you need to make sure 
their answer is valid, consistent, efficient, and aligned with the overall goals of the game. Finally, you need to provide
a pass/fail verdict on the answer. Review in order:

Consistency: Check if the SUMMARY and MAP are consistent with what the game master actually said.
Objective: Check if the ACTION is aligned with the goals and the game map, and if it is a valid game command.
Repetition and loops: Check if the ACTION repeats a command that already failed, or the player repeats suboptimal actions without a clear reason.

Answer in this format:
FEEDBACK: <short review of the player behavior: point out any mistakes, inconsistencies in the map, or suboptimal actions, and suggest improvements. If everything is fine, return "No issues found.">
GRADE: PASS if the answer is correct, or FAIL if it must be improved
""".strip()

REFINE_PROMPT = """
An expert game player has reviewed your answer and provided the following feedback:
{feedback}

Rewrite your answer from scratch, improved according to this feedback, in exactly the same format as before (SUMMARY, MAP, REFLECTION, GOALS, ACTION).
""".strip()


class SelfRefineAgent(SymbolicAgent):
    """Symbolic agent with Self-Refine. The same LLM writes the draft answer, criticises it, and rewrites it if the answer does not pass a check."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.last_feedback = None
        self.last_refined = False

    def act(self, obs, prev_action=None):
        response = super().act(obs, prev_action)
        # Keep the feedback in the logged reasoning, to inspect it afterwards
        reasoning = (
            f"{response.reasoning}\nFEEDBACK: {self.last_feedback}\nREFINED: {'yes' if self.last_refined else 'no'}"
        )
        return response._replace(reasoning=reasoning)

    def query_llm(self, messages):

        """ This function overwrites the vanila SymbolicAgent's generation call to add a feedback and refinement step"""

        draft = self.client.generate(messages)
        conversation = messages + [Message(role="assistant", content=draft.completion)]

        feedback = self.client.generate(conversation + [Message(role="user", content=FEEDBACK_PROMPT)])
        self.last_feedback, self.last_refined = self._extract_feedback(feedback.completion)

        calls = [draft, feedback]
        final = draft
        if self.last_refined:
            refine_prompt = REFINE_PROMPT.format(feedback=self.last_feedback)
            final = self.client.generate(conversation + [Message(role="user", content=refine_prompt)])
            calls.append(final)

        # Count the tokens of all the calls of this step
        return final._replace(
            input_tokens=sum(call.input_tokens for call in calls),
            output_tokens=sum(call.output_tokens for call in calls),
        )

    def _extract_feedback(self, response_text):
        """Split the LLM's review into its feedback text and its grade.
        Returns:
            tuple: The feedback text, and True if the model asks to revise the answer.
        """
        feedback_match = re.search(
            r"^\W*FEEDBACK\W*:[\s*]*(.*?)(?=^\W*GRADE\W*:|\Z)",
            response_text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        feedback = (feedback_match.group(1) if feedback_match else response_text).strip()
        redo = bool(re.search(r"^\W*GRADE\W*:[\s*]*FAIL", response_text, re.IGNORECASE | re.MULTILINE))
        return feedback, redo

