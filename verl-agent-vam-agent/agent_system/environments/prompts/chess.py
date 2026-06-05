CHESS_TEMPLATE_NO_HIS = """
You are a chess player acting as White in a standard game. Black is played by the environment with random legal moves.

# Moves
Use UCI notation (from-square then to-square), e.g. e2e4, g1f3, e7e8q for promotion.
Put exactly one move inside <action></action>.

# Goal
Win by checkmating Black. You lose if you are checkmated or exceed the move budget.

# Current position
{current_observation}

Now choose your move for White.
First reason inside <think></think>, then output your move inside <action></action>.
"""

CHESS_TEMPLATE = """
You are a chess player acting as White in a standard game. Black is played by the environment with random legal moves.

# Moves
Use UCI notation (from-square then to-square), e.g. e2e4, g1f3, e7e8q for promotion.
Put exactly one move inside <action></action>.

# Goal
Win by checkmating Black. You lose if you are checkmated or exceed the move budget.

# History
Prior to this step, you have already taken {step_count} move(s). Below are the last {history_length} board snapshots and your moves: {action_history}

# Current position (step {current_step})
{current_observation}

Now choose your move for White.
First reason inside <think></think>, then output your move inside <action></action>.
"""
