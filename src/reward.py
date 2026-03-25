"""Rule-based reward function for the GRPO demo.

In a production RLHF pipeline this would be replaced by a learned reward model
trained on human preference pairs. Here we use simple heuristics that are easy
to understand and verify:

  +0.4  appropriate response length  (40–200 characters)
  +0.3  mentions at least one Shakespeare character name
  +0.3  starts with a capital letter  (basic fluency signal)
  ─────────────────────────────────────────────────────────
  max   1.0
"""

SHAKESPEARE_NAMES = {
    "Romeo", "Juliet", "Hamlet", "Macbeth", "Iago", "Othello", "Lear",
    "Prospero", "Portia", "Brutus", "Desdemona", "Cordelia", "Ariel",
    "Puck", "Oberon", "Titania", "Shylock", "Ophelia", "Rosalind",
    "Benedick", "Beatrice", "Antony", "Cleopatra", "Falstaff", "Caliban",
    "Miranda", "Cassio", "Emilia", "Horatio", "Laertes", "Polonius",
}


def compute_reward(response: str) -> float:
    """Score a single response string in [0.0, 1.0]."""
    reward = 0.0

    # Length signal
    if 40 <= len(response) <= 200:
        reward += 0.4

    # On-topic signal: mentions a Shakespeare character
    words = set(response.split())
    if words & SHAKESPEARE_NAMES:
        reward += 0.3

    # Basic fluency: starts with a capital letter
    stripped = response.lstrip()
    if stripped and stripped[0].isupper():
        reward += 0.3

    return reward
