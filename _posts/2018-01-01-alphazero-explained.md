---
published: true
---


![435117.jpg]({{site.baseurl}}/media/435117.jpg)


If you follow the AI world, you've probably heard about AlphaGo. 

The ancient Chinese game of Go was once thought impossible for machines to play. It has more board positions ($$10^{17010170}$$) than there are atoms in the universe. The top grandmasters regularly trounced the best computer Go programs with absurd (10 or 15 stone!) handicaps, justifying their decisions in terms of abstract strategic concepts -- _joseki, fuseki, sente, tenuki, balance_ -- that they believed computers would never be able to learn. 

Demis Hassabis and his team at DeepMind believed otherwise. And they spent three years painstaking years trying to prove this belief; collecting Go data from expert databases, tuning deep neural network architectures, and developing hybrid strategies honed against people as well as machines. Eventually, their efforts culminated in a dizzyingly complex, strategic program they called AlphaGo, trained using millions of hours of CPU and TPU time, able to compete with the best of the best Go players. They set up a match between AlphaGo and grandmaster Lee Sedol ... and the rest is history.

But I'm not here to talk about AlphaGo -- you can go watch [the movie](https://www.alphagomovie.com/) for that (it's like a classic sports underdog movie but with less sweat and more suits). I'm here to discuss AlphaZero, the algorithm some DeepMind researchers released a year later. The algorithm that uses NO previous information or human-played games whatsoever, knowing nothing but the rules of the game. The algorithm that was able to handily beat the original version of AlphaGo in only four hours (?) of training time. The algorithm that can be applied without modification to chess, Shogi, and (AI researchers believe) almost any game with perfect information and no randomness.

The algorithm that is a **radical simplification** of AlphaGo, so much simpler that even a lowly blogger like me is able to explain it and teach YOU how to code it. At least, that's the idea.




### General Game-Playing Terminology

In game theory, chess and Go are examples of turn-based, two-player games with _perfect information_; both players know everything relevant about the state of the game at any given time. Furthermore, there is no randomness or uncertainty in how making **moves** affects the game; making a given move will always result in the same final game state, one that both players know with complete certainty.  

Because both players have perfect information, it is clear that every position in a classical game is either **winnable** or **unwinnable**.  Either the player who is just about to make a move can win (given that they choose the right move) or they can't (because no matter what move they make, the game is winnable for the other player). When you add in the possibility of drawing (neither player wins) then there are three possible **values** for a given state: either it is a guaranteed loss, a guaranteed win, or a guaranteed draw.

If this definition makes you shout "Recursion!", then your instincts are on the right track. In fact, it is easy to determine the value of a game state using a self-referential definition of winnability. We can write some Python code, using the `AbstractGame` template that I've defined in this file, to do just this. We represent a guaranteed loss for the player about to play with $-1$, a guaranteed win with $1$, and a draw with $0$.  Note that we handle the game using general methods, such as `make_move()`, `undo_move()`, and `over()`, that could apply to any game, whether something as simple as Tic-Tac-Toe or as complex as chess or Go.


~~~ python
from games.games import AbstractGame

def value(game):
    if game.over():
        return game.score()
    
    state_values = []
    for move in game.valid_moves():
        game.make_move(move)
        # guaranteed win for P2 is guaranteed loss for P1, so we flip the values
        state_values.append(-value(game)) 
        game.undo_move()
	
    # The player always chooses the optimal move: the best possible achievable state
    return max(state_values)
~~~

Now, how can we create an AI that always chooses the "best move"? We simply tell the AI to pick a move that results in the highest resultant score.

~~~ python

def ai_best_move(game):
	
    action_dict = {}
    for move in game.valid_moves():
        game.make_move(move)
        action_dict[move] = -value(game)
        game.undo_move()

    return max(action_dict, key=action_dict.get)
~~~

In fact, this simple AI can play tic-tac-toe optimally - it will always either win or draw, as can be seen in the following GIF.




### Monte-Carlo Tree Search


So does this mean that we've solved all two-player classical games? Not quite.  Although the recursion above looks simple, it actually ends up checking all possible game states reachable from a given position in order to compute the value of a state. Thus, even though there do exist optimal strategies for complex games like chess and Go, their **game trees** are so intractably large that it would be impossible to find them.

(Representation of Go game tree)

We need a faster way to approximate the value of a given game state. What if, instead of making the players choose optimal moves, we computed the value of a state by making the players choose _random_ moves from there on, and seeing who wins more? 


~~~ python
import random
from games.games import AbstractGame

def value(game):
    if game.over():
        return game.score()
    
    state_values = []
    move = random.choice(game.valid_moves())
    game.make_move(move)
    state_values.append(-value(game)) 
    game.undo_move()
	
    # The player always chooses the optimal move: the best possible achievable state
    return max(state_values)

def ai_best_move(game):
	
    action_dict = {}
    for move in game.valid_moves():
        game.make_move(move)
        action_dict[move] = -value(game)
        game.undo_move()

    return max(action_dict, key=action_dict.get)
~~~

Now, clearly this estimate of value is inaccurate.



