# Deals with time of day, social setting, weather, budget, etc.

class ContextManager:
    def enrich(self, base_mood: str, context: dict) -> str:
        if context.get("time") == "morning":
            return "light"
        if context.get("company") == "partner":
            return "romantic"
        return base_mood
    