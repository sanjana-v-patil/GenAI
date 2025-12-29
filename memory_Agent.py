class MemoryAgent(Agent):
    """Agent responsible for maintaining conversation history and context"""
    def __init__(self, max_history: int = 20):
        super().__init__("MemoryAgent")
        self.conversation_history = deque(maxlen=max_history)
        self.context_window = []
        self.max_context_length = 4000  # Approximate token limit for context window

    def add_interaction(self, user_input: str, system_response: str):
        """Add a user-system interaction to memory"""
        self.conversation_history.append({
            'user': user_input,
            'system': system_response,
            'timestamp': time.time()
        })

    def get_recent_history(self, num_interactions: int = 3) -> List[Dict[str, str]]:
        """Get the most recent interactions"""
        return list(self.conversation_history)[-num_interactions:]

    def update_context_window(self, new_context: str):
        """Update the context window with new information"""
        tokens = new_context.split()
        if len(tokens) > self.max_context_length:
            new_context = ' '.join(tokens[-self.max_context_length:])

        self.context_window.append(new_context)
        if len(self.context_window) > 5:
            self.context_window = self.context_window[-5:]

    def get_context_summary(self) -> str:
        """Generate a summary of the current context"""
        if not self.conversation_history:
            return "No conversation history available."

        recent = self.get_recent_history(3)
        summary = "Recent conversation history:\n"
        for i, interaction in enumerate(recent, 1):
            summary += f"{i}. User: {interaction['user']}\n   System: {interaction['system']}\n"

        if self.context_window:
            summary += "\nAdditional context:\n" + "\n".join(f"- {ctx}" for ctx in self.context_window)

        return summary

    def process(self, interaction: Dict[str, str]) -> str:
        """Process an interaction and return context summary"""
        self.add_interaction(interaction['user'], interaction['system'])
        return self.get_context_summary()
