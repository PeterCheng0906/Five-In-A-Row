# Five-In-A-Row
Five In A Row Project with Minimax Alpha-Beta Pruning

Introduction:
Five-in-a-Row, also known as Gomoku, is a game played by two players where the goal is to create a line of five consecutive pieces on a 15 by 15 Go board. The two players, black and white, take alternating turns placing stones on empty intersections of the board. Black plays first and once pieces are put down, they may not be moved or removed. If the entire board gets filled with pieces and neither player has five pieces in a row vertically, horizontally, or diagonally, then the outcome is a draw between the two players. Our objective was to implement a search-based AI algorithm using Minimax with Alpha-Beta Pruning, enabling our agent to evaluate game states and make optimal decisions based on predefined heuristics. In this report, we will start with a literature review of previous works or projects related to simulating the game Five-in-a-Row, next, we will discuss the dataset that we leveraged to train the agent, the baseline we used, our main approach, evaluation metrics, and results, an analysis of the results, and finally, we will conclude with a brief discussion of the ethical considerations associated with this project.

Dataset:
For our project, since we are employing a minimax strategy with alpha-beta pruning, we did not leverage an external dataset. Our project doesn’t rely on a pre-existing dataset because our agent dynamically generates different levels of game states and then evaluates it using alpha-beta pruning with minimax. These game states serve as data points for our agent’s decision-making process. A strength of this method is that it can evolve to dynamically changing board configurations during each player turn. 

Baseline:
Our baseline is when the strategy is to move randomly. This occurs through the random selection of one possible move, which is the placement of a piece in one of the empty spaces left on the board. Benefits include high efficiency and low computational costs. In this strategy, there are no defensive considerations, leading to a relatively poor performance against other players. 

Main Approach:
Our approach for the Five-in-a-Row Project uses Minimax with Alpha-Beta Pruning to evaluate possible moves, simulate future board states, and choose the move that maximizes the AI agent’s advantage while minimizing the opponent’s opportunities.

Input
A 2D array representing the board state, where each cell represents a position on the board and the value of each cell indicates whether it’s empty, occupied by the player, or occupied by the opponent.

board = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

Here, 0 means empty, 1 means the AI's piece, and 2 means the opponent's piece. Current Player (AI or Opponent): Represented by an enum (e.g., player = 1 for AI, player = 0 for the opponent).

Output
The output of main_game.py of our project is the winner state and the final board. In the case that one player wins, say player 1, the outcome of our python script is a message including the player that won and the board representation. One example of what this output could look like is as follows:  
          
            Player 1 won!
            board = [
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
           ]
If the outcome is a draw game, the python script will print “Draw!” as well as the board representation. 

The output of our script main_evaluator.py is a set of evaluation metrics that analyze the agent’s performance over multiple simulated games. These include the win rate, or the percentage of games the agent wins, the optimal move selection rate, or the percentage of moves were the agent selects the optimal move, the average time to victory, or the average number of turns it takes for the agent to win, the average memory usage, and the average computation time per move. The output will look be in the format outlined below: 

Win Rate: print(f"Win Rate: {self.results['winrate'] * 100:.2f}%")
Optimal Move Selection Rate: print(f"Optimal Move Selection: {self.results['optimal_moves'] * 100:.2f}%")
Average Time to Victory: print(f"Average Time to Victory: {self.results['avg_time_to_victory']:.2f} turns")
Average Memory Usage: print(f"Average Computation Time per Move: {self.results['avg_computation_time_per_move']:.4f} seconds")
Average Computation Time per Move: print(f"Average Memory Usage: {self.results['avg_memory_usage'] / 1024:.2f} KB")


Variables 
The primary variable in this project is the game state, which is composed of the board and the current player. The game state represents the current state of the game and provides all the necessary data to simulate moves, evaluate positions, and determine winners. The board variable is a 2-dimensional array (NumPy array) containing numerical integers. 0 represents empty cells, 1 represents pieces placed by the AI agent (Player 1), and 2 represents pieces placed by the opponent (Player 2). The current player variable is an integer (1 or 2) which indicates which player’s turn it is and alternates with each new turn. 

We also have a heuristic function that evaluates each potential move that the AI agent discovered to assign it a score based on its strategic importance. The key factors in the heuristic function include offensive scoring, defensive scoring, and the win condition. For offensive scoring, we assign higher scores to moves that advance the AI agent’s pieces closer to forming five in a row. Practically, this means assigning rewards for moves that create lines of 2, 3, or 4 consecutive pieces. For defensive scoring, we penalize moves that leave the AI agent’s pieces vulnerable to immediate threats from the opponent. For example, there is a penalty associated with the opponent placing three or more consecutive pieces on the board. The win condition, or a move that immediately results in five consecutive pieces, is given the highest possible score.

Minimax Depth: Determines the number of turns the AI simulates in the future.
A higher depth leads to more accurate decisions but increases computation time. We typically set between 2 and 4 to balance performance and speed.
Alpha and Beta Values: 
Used in the Alpha-Beta Pruning process to optimize Minimax by eliminating unnecessary branches. Helps speed up Minimax by avoiding evaluating branches that won’t influence the final decision.
Evaluation Metrics Values: Tracks the AI's performance during simulations for optimization.
Candidate Moves: A subset of possible moves the AI evaluates during its turn. 
It limited our AI to focus only on empty cells within a specified radius of existing pieces. 
Reduces the number of moves the AI evaluates, optimizing the decision-making process. Otherwise it took so long to generate outputs.



Factors 
Offensive Factors: the goal is to increase the AI’s chances of winning.
Consecutive Pieces: The number of AI pieces in a row (horizontal, vertical, diagonal).
Higher focus on 3 or 4 consecutive pieces as they are closer to forming five.
Open Ends: Lines of consecutive pieces with open ends (empty at either end) are more valuable.
Example: A line of 4 with an open end.
Potential Lines: Moves that create multiple intersecting lines 
Example: A piece is part of both a horizontal and diagonal line.
Winning Moves: A move that directly leads to five consecutive pieces has the highest priority.
Defensive Factors: the goal is to prevent the opponent from winning.
Blocking Moves: Stopping the opponent from forming 4 or 5 consecutive pieces.
Threat Identification: Moves that minimize threats posed by the opponent’s positions.
Double Threats: High importance defensive actions against positions where the opponent can win in multiple directions.
Computational Factors: the goal is to optimize AI decision-making
Candidate Moves:
This could prioritize to evaluate moves near active areas of the board.
Minimax Depth:
Trade-off between computational efficiency and decision accuracy. Deeper the depth, longer computation time.
Alpha-Beta Pruning:
Eliminates unnecessary branches in the decision tree to speed up calculations.


States
Board State: The current configuration of the board.
Track the placement of pieces.
Identify potential winning or blocking moves.
Player State: Tracks whose turn it is to play.
Alternate turns between AI and opponent.
Determine maximizing or minimizing player
Game Progress State: Tracks the overall progress and status of the game.
Winner Check: Detect if either player has won the game.
Draw Check: Determine if the board is full with no winner.
Move Count: Tracks the number of turns it tooks so far.



Evaluation Metrics
1. Win Rate: measures the percentage of games the AI wins against human players or other AI opponents.
Formula:
winrate=#wins#total games played100
2. Optimal Move Selection: tracks the percentage of moves where the AI selects the theoretically best possible move.
Formula:
Optimal Move Rate=#optimal moves#total moves100
3. Time to Victory: measures the number of moves or the time (in seconds) required for the AI to win a game. This helps evaluate efficiency in gameplay.
Formula:
Time to Victory(Moves)=#total moves in Game
Time to Victory(Seconds)=End Time-Start Time
4. Memory Usage: monitors the memory used by the AI during gameplay to ensure resource efficiency.
5. Computation Time per Move: measures the time in sec the AI takes to compute its next move.
Formula:
computation time=End Time - Start Time
