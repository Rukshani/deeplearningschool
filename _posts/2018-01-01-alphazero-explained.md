---
published: true
---


![go.jpg]({{site.baseurl}}/media/go.jpg)


If you follow the AI world, you've probably heard about AlphaGo. 

The ancient Chinese game of Go was once thought impossible for machines to play. It has more board positions ($$10^{17010170}$$) than there are atoms in the universe. The top grandmasters regularly trounced the best computer Go programs with absurd (10 or 15 stone!) handicaps, justifying their decisions in terms of abstract strategic concepts -- _joseki, fuseki, sente, tenuki, balance_ -- that they believed computers would never be able to learn. 

Demis Hassabis and his team at DeepMind believed otherwise. And they spent three years painstaking years trying to prove this belief; collecting Go data from expert databases, tuning deep neural network architectures, and developing hybrid strategies honed against people as well as machines. Eventually, their efforts culminated in a dizzyingly complex, strategic program they called AlphaGo, trained using millions of hours of CPU and TPU time, able to compete with the best of the best Go players. They set up a match between AlphaGo and grandmaster Lee Sedol ... and the rest is history.

![go.jpg]({{site.baseurl}}/media/alphago.jpg)

{:.image-caption}
The highly publicized match between Lee Sedol, 9th dan Go grandmaster, and AlphaGo. Google DeepMind's program won 4 out of 5 games.

But I'm not here to talk about AlphaGo. I'm here to discuss AlphaZero, the algorithm some DeepMind researchers released a year later. The algorithm that uses NO previous information or human-played games whatsoever, knowing nothing but the rules of the game. The algorithm that was able to handily beat the original version of AlphaGo in only four hours (?) of training time. The algorithm that can be applied without modification to chess, Shogi, and (AI researchers believe) almost any game with perfect information and no randomness.

AlphaGo is, at its heart, a story about human underdogs beating the odds -- applying new techniques 

The algorithm that is a **radical simplification** of AlphaGo, so much simpler that even a lowly blogger like me is able to explain it and teach YOU how to code it. At least, that's the idea.




### General Game-Playing and DFS

In game theory, rather than reason about specific games, mathematicians like to reason about a special class of games: turn-based, two-player games with _perfect information_. In these games, both players know everything relevant about the state of the game at any given time. Furthermore, there is no randomness or uncertainty in how making **moves** affects the game; making a given move will always result in the same final game state, one that both players know with complete certainty. We call games like these **classical games**.

These are examples of classical games:
- Tic-Tac-Toe
- Chess
- Go
- [Gomoku](https://en.wikipedia.org/wiki/Gomoku) (a 5-in a row game on a 19 by 19 go board)
- [Mancala](https://en.wikipedia.org/wiki/Mancala)

whereas these games are not:
- Poker (and most other card games)
- Rock-Paper-Scissors
- [The Prisoner's Dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma)
- Video games like Starcraft

When games have random elements and hidden states, it is much more difficult to design AI systems to play them, although there have been powerful poker and Starcraft AI developed. Thus, the AlphaZero algorithm is restricted to solving classical games only.

In a classical game, because both players have perfect information, every position is either winnable or unwinnable.  Either the player who is just about to make a move can win (given that they choose the right move) or they can't (because no matter what move they make, the game is winnable for the other player). When you add in the possibility of drawing (neither player wins) then there are three possible values for a given state: either it is a guaranteed loss **(-1)**, a guaranteed win **(+1)**, or a guaranteed draw **(0)**.

If this definition makes you shout "Recursion!", then your instincts are on the right track. In fact, it is easy to determine the value of a game state using a self-referential definition of winnability. We can write some Python code, using my [predefined](https://github.com/nikcheerla/alphazero/tree/master/games) `AbstractGame` class. Note that we handle the game using general methods, such as `make_move()`, `undo_move()`, and `over()`, that could apply to any game.


~~~ python
from games.games import AbstractGame
from games.tictactoe import TicTacToeGame
from games.chess import ChessGame
from games.gomoku import GomokuGame

r"""
Returns -1 if a game is a guaranteed loss for the player
just about to play, +1 is the game is a guaranteed victory
and 0 for a draw.
"""
def value(game):
    if game.over():
        return -game.score()
    
    state_values = []
    for move in game.valid_moves():
        game.make_move(move)
        # guaranteed win for P2 is loss for P1, so we flip values
        state_values.append(-value(game)) 
        game.undo_move()
	
    # The player always chooses the optimal game state
    # +1 (win) if possible, otherwise draw, then loss
    return max(state_values)
~~~

Now, how can we create an AI that always chooses the "best move"? We simply tell the AI to pick a move that results in the highest resultant score. We call this the DFS approach because, to choose a move, we are essentially doing depth-first search on the tree of possible game states. 

~~~ python

r"""
Chooses optimal move to play; ideally plays
moves that result in -1 valued states for the opponent 
(opponent loses), but will pick draws or opponent victories
if necessary.
"""
def ai_best_move(game):
	
    action_dict = {}
    for move in game.valid_moves():
        game.make_move(move)
        action_dict[move] = value(game)
        game.undo_move()

    return min(action_dict, key=action_dict.get)
~~~

In fact, this simple AI can play tic-tac-toe optimally - it will always either win or draw with anyone it plays. 


![out.gif]({{site.baseurl}}/media/out.gif)

{:.image-caption}
A simulated game between two AIs using DFS. Since both AIs always pick an optimal move, the game will end in a draw (Tic-Tac-Toe is an example of a game where the second player can always force a draw).



### Monte-Carlo Tree Search


So does this mean that we've solved all two-player classical games? Not quite.  Although the recursion above looks simple, it has to check all possible game states reachable from a given position in order to compute the value of a state. Thus, even though there do exist optimal strategies for complex games like chess and Go, their **game trees** are so intractably large that it would be impossible to find them.

![branching.jpg]({{site.baseurl}}/media/branching.jpg)

{:.image-caption}
Branching paths in the game of Go. There are about 150-250 moves on average playable from a given game state.

The reason for the slow progress of DFS is that when estimating the value of a given state in the search, both players must play **optimally**, choosing the move that gives them the best value (making complex recursion necessary). Maybe, instead of making the players choose optimal moves (which is extremely computationally expensive), we can compute the value of a state by making the players choose _random_ moves from there on, and seeing who wins. Or perhaps we could even use cheap computational heuristics to make the players more likely to choose good moves.

This is the basic idea between Monte Carlo Tree Search -- use random exploration to estimate the value of a state. We call a single random game a "playout"; if you play 1000 playouts from a given position $X$ and player 1 wins $60\%$ of the time, it's likely that that position $X$ is better for player 1 than player 2. Thus, we can create  a `monte_carlo_value()` function that estimates the value of a state using a given number of random playouts.

~~~ python
import random
import numpy as np
from games.games import AbstractGame

def playout(game):
    if game.over():
        return -game.score()
    
    state_values = []
    move = random.choice(game.valid_moves())
    game.make_move(move)
    value = -value(game))
    game.undo_move()
	
    return value

def monte_carlo_value(game, N=100):
    scores = [playout(game) for i in range(0, N)]
    return np.mean(scores)

def ai_best_move(game):
	
    action_dict = {}
    for move in game.valid_moves():
        game.make_move(move)
        action_dict[move] = -monte_carlo_value(game)
        game.undo_move()

    return max(action_dict, key=action_dict.get)
~~~



Clearly, Monte Carlo search doesn't choose the optimal move, but for many simple games, a large number of random playouts will suffice . key is that while some positions might be easy to win with against a random opponent, they are utterly undefensible against a _competent_ opponent. 


### Upper-Confidence Bounds Applied to Trees (UCT)

One way to fix this problem is to make the opponent move selections more intelligent. Rather than having the move choices within the playouts to be random, we want the opponent to choose their move using a heuristic approximation of **what moves are worth exploring**.

Looking at this problem from the perspective of the opponent, each move is a complete black box; almost like a slot machine with unknown payout probabilities. Some moves might result in only a $30\%$ win rate, other moves might result in a $70\%$ win rate, but crucially, you don't know any of this in advance. You need to balance exploring and testing the slot machines (and of course recording statistics) with actually choosing the best moves. That's what the UCT algorithm is for: balancing exploration and exploitation in a reasonable way.

Jeff Bradberry sums up this algorithm concisely in his great blog post on UCT:

> Imagine ... that you are faced with a row of slot machines, each with different (unknown) payout probabilities and amounts. As a rational person (if you are going to play them at all), you would prefer to use a strategy that will allow you to maximize your net gain. But how can you do that? ... Clearly, your strategy is going to have to balance playing all of the machines to gather that information yourself, with concentrating your plays on the observed best machine. One strategy, called UCB1, does this by constructing statistical confidence intervals for each machine.

> $$x_i \pm \sqrt{\frac{2\ln{N}}{n_i}}$$

>where:
> - $x_i$: the mean payout for machine i
> - $n_i$: the number of plays of machine ii
> - $N$: the total number of plays

> Then, your strategy is to pick the machine with the highest upper bound each time. As you do so, the observed mean value for that machine will shift and its confidence interval will become narrower, but all of the other machines' intervals will widen. Eventually, one of the other machines will have an upper bound that exceeds that of your current one, and you will switch to that one. This strategy has the property that your regret, the difference between what you would have won by playing solely on the actual best slot machine and your expected winnings under the strategy that you do use, grows only as $O(\ln⁡{n})$.

Whereas the previous two algorithms we worked with, DFS and MCTS, were static, UCT involves learning over time. The first time the UCT algorithm runs, it focuses more on exploring all game states within the playouts (looking a lot like MCTS). But as it collects more and more data, the random playouts become less random and more "heavy", exploring moves and paths that have already proven to be good choices and ignoring those that haven't. 

Thus, when we write code to represent UCT, we need to make a record of the states we visit and their values. We can let the algorithm play games against itself and watch it slowly improve.

~~~ python

class MCTSController(object):

	def __init__(self, manager, T=0.3, C=1.5):
		super().__init__()

		self.visits = manager.dict()
		self.differential = manager.dict()
		self.T = T
		self.C = C

	def record(self, game, score):
		self.visits["total"] = self.visits.get("total", 1) + 1
		self.visits[hashable(game.state())] = self.visits.get(hashable(game.state()), 0) + 1
		self.differential[hashable(game.state())] = self.differential.get(hashable(game.state()), 0) + score

	r"""
	Evaluates the "value" of a state as a bandit problem, using the value + exploration heuristic.
	"""
	def heuristic_value(self, game):
		N = self.visits.get("total", 1)
		Ni = self.visits.get(hashable(game.state()), 1e-9)
		V = self.differential.get(hashable(game.state()), 0)*1.0/Ni 
		return V + self.C*(np.log(N)/Ni)

	r"""
	Runs a single, random heuristic guided playout starting from a given state. This updates the 'visits' and 'differential'
	counts for that state, as well as likely updating many children states.
	"""
	def playout(self, game, expand=150):

		if expand == 0 or game.over():
			score = game.score()
			self.record(game, score)
			#print ('X' if game.turn==1 else 'O', score)
			return score

		action_mapping = {}

		for action in game.valid_moves():
			
			game.make_move(action)
			action_mapping[action] = self.heuristic_value(game)
			game.undo_move()
		
        # Instead of choosing a move randomly, we choose it using the exploration heuristic
		chosen_action = sample(action_mapping, T=self.T)
		game.make_move(chosen_action)
		score = -self.playout(game, expand=expand-1) #play branch
		game.undo_move()
		self.record(game, score)

		return score

	r"""
	Evaluates the "value" of a state by randomly playing out games starting from that state and noting the win/loss ratio.
	"""
	def value(self, game, playouts=100, steps=5):

		# play random playouts starting from that game value
		with Pool() as p:
			scores = p.map(self.playout, [game.copy() for i in range(0, playouts)])

		return self.differential[hashable(game.state())]*1.0/self.visits[hashable(game.state())]

	r"""
	Chooses the move that results in the highest value state.
	"""
	def best_move(self, game, playouts=100):

		action_mapping = {}

		for action in game.valid_moves():
			game.make_move(action)
			action_mapping[action] = self.value(game, playouts=playouts)
			game.undo_move()

		print ({a: "{0:.2f}".format(action_mapping[a]) for a in action_mapping})
		return max(action_mapping, key=action_mapping.get)

~~~

Thinking about the problem from a deeper theoretical basis, we see that 


A key component of the success of UCT is how it allows for the construction of **lopsided exploration trees**. In complex games like chess and Go, there are an incomprehensible number of states, but most of them are unimportant because they can only be reached if one or both players play extremely badly. Using UCT, you can avoid exploring out these "useless" states and focus most of your computational energy on simulating games in the interesting portion of the state space. For a visualization of this see below.


### AlphaZero: Deep Learning Game Heuristics

Given enough playouts, UCT will be able to explore all of the important game positions in any game (spending far less time on states that tend not to occur in intelligent play) and determine their values using the Monte Carlo Method. But the amount of playouts needed in chess and Go for this to happen is computationally infeasible, even with the UCT prioritization; thus, most viable MCTS engines for these games end up exploiting a lot of domain-specific knowledge and heuristics.

To see why, look at the diagram below. The first and second chess game are completely different in terms of the positions of pieces, but the essential components of the "situation" are the same; a black bishop supported by a black knight is checking a white king. The differences in symmetry and pawn ordering are mostly irrelevant. Yet, to the UCT algorithm, the two chess games are completely different states, and even if UCT has thoroughly "explored" game #1 and knows by the heuristic it is a +0.2 advantage for black, it has to start from scratch when exploring game #2.

Humans have no problem noting the similarities between different game positions and their "winnability"; in fact, that's how most of our intuition in playing games comes about! But how can we train an AI to do the same? 

We can encode 




### The Secret Factor: Compute Time





