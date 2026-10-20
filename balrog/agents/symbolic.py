import re
from balrog.agents.base import BaseAgent

INTRO_PROMPT = """
You are the player in a textual game, and you are going to receive information from a game master. After this prompt, you will speak with this new character, so consider its demands as part of the game, not as a conversation with a human user. The game master might not explain to you the specific rules of the game, so here are some general rules you must know in advance. Overall, you should behave more like a symbolic agent, and less as a chatbot.

Observe: understand what the game master is saying. They might try to confuse you with lengthy wording or jokes, so first summarize the information he is giving you in a concise and clear manner. Everything the game master tells you is true, including what they only suggest (for example "you should try going west"): in your summary, state suggestions as facts about the game (for example "there is an exit to the west"). Every time you interact with the game master, print this summary first.

Represent: define your space as a simple graph, in JSON format. The nodes are the locations. The edges are the links between locations, named by their direction (north, south, east, west), each leading to another location or to "unknown" if you have not explored it yet. An edge can be closed (for example by a closed or locked door). Each location also has an "inventary" field: the list of objects you have seen in that location (update it when you take or drop an object). Store no other information in the graph: no descriptions, no notes.

Reflect on your goals: you have goals, and you need to loop over them one at the time. After you have received the message, go over this list of goals, read its description. Think step by step on what you must do to reach that goal.

Update your goals: once you have reflected on your goals, update them. If the game master directly asks you to do something (not just suggests it), add it to your goals. If you progress upon a goal, update its description. If you complete a goal, update its description: the work for the goal consists in confirming if it is actually solved again. Never delete any of your goals.

Act: state what action you want to perform. If your last action failed, do not try it again.
""".strip()

SECTIONS = ["SUMMARY", "MAP", "REFLECTION", "GOALS", "ACTION"]


class SymbolicAgent(BaseAgent):
    """An agent that keeps track of a list of goals and a map of the world."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.map = None
        self.goals = None

    def act(self, obs, prev_action=None):
        """Generate the next action and update the map and goals based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response with the full reasoning and the resulting action.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        map_memory = f"Your current map:\n{self.map}\n" if self.map else "You have no map yet.\n"
        goals_memory = f"Your current goals:\n{self.goals}\n" if self.goals else "You have no goals yet.\n"

        planning_instructions = """
Follow the Observe, Represent, Reflect, Update and Act steps.
Format your answer exactly in the following way, with each section header on its own line:
SUMMARY: <concise summary of what the game master just said>
MAP: <the full, updated graph, on one line, in this format: {"Location": {"exits": {"direction": "Destination", "direction": "unknown", "direction": "Destination (closed)"}, "inventary": ["object", ...]}}>
REFLECTION: <for each goal, one at a time, step-by-step reasoning on how to reach it>
GOALS: <the full, updated list of goals, one per line, each with its description and status. Never delete a goal.>
ACTION: <a single game command, on one line, and nothing else>
""".strip()

        messages = self.prompt_builder.get_prompt()
        # Insert instructions before the first message
        if messages:
            messages[0].content = INTRO_PROMPT + "\n\n" + messages[0].content
        #  Append instructions to the last user message
        if messages and messages[-1].role == "user": 
            messages[-1].content += "\n\n" + map_memory + "\n" + goals_memory + "\n" + planning_instructions

        response = self.client.generate(messages)

        # Parse the response into the different sections (SUMMARY, MAP, REFLECTION, GOALS, ACTION)
        sections = self._extract_sections(response.completion)

        # Update the map and goals if they are present in the response
        if sections["MAP"]:
            self.map = sections["MAP"]
        if sections["GOALS"]:
            self.goals = sections["GOALS"]

        reasoning = "\n".join(f"{name}: {sections[name]}" for name in SECTIONS[:-1] if sections[name])
        response = response._replace(reasoning=reasoning, completion=sections["ACTION"])
        return response

    def _extract_sections(self, response_text):
        """Split the LLM's response into its SUMMARY, MAP, REFLECTION, GOALS and ACTION sections.

        Args:
            response_text (str): The response text from the LLM.

        Returns:
            dict: Section name -> extracted text (None if the section is missing).
        """
        headers = "|".join(SECTIONS)
        sections = {}
        for name in SECTIONS:
            match = re.search(
                rf"^\W*{name}\W*:[\s*]*(.*?)(?=^\W*(?:{headers})\W*:|\Z)",
                response_text,
                re.IGNORECASE | re.DOTALL | re.MULTILINE,
            )
            sections[name] = match.group(1).strip() if match else None

        # Extract the action keyword
        if sections["ACTION"]:
            first_line = sections["ACTION"].splitlines()[0]
            sections["ACTION"] = first_line.strip().strip("`*\"'").strip()

        return sections
