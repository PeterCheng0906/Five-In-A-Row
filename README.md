# Five-In-A-Row
Five In A Row Project with Minimax Alpha-Beta Pruning
Five-in-a-Row, also known as Gomoku, is a game played by two players where the goal is to create a line of five consecutive pieces on a 15 by 15 Go board, as seen in Figure 1.
 
Figure 1: White player wins Gomoku with five in a row on diagonal 

The two players, black and white, take alternating turns placing stones on empty intersections of the board. Black plays first and once pieces are put down, they may not be moved or removed. If the entire board gets filled with pieces and neither player has five pieces in a row vertically, horizontally, or diagonally, then the outcome is a draw between the two players. Our objective was to implement a search-based AI algorithm using Minimax with Alpha-Beta Pruning, enabling our agent to evaluate game states and make optimal decisions based on predefined heuristics. In this report, we will start with a literature review of previous works or projects related to simulating the game Five-in-a-Row, next, we will discuss the dataset that we leveraged to train the agent, the baseline we used, our main approach, evaluation metrics, and results, an analysis of the results, and finally, we will conclude with a brief discussion of the ethical considerations associated with this project.
