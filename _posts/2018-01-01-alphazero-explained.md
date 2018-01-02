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

In game theory, chess and Go are examples of games with _perfect information_; both players know everything relevant about the state of the game at any given time. Furthermore, there is no randomness or uncertainty in how making **moves** affects the game; making a given move will always result in the same final game state, one that both players know with complete certainty.  

Because both players have perfect information, it is clear that every position in a classical game is either **winnable** or **unwinnable**.  Either the player just about to make a move can win (given that they choose the right move) or they can't (because no matter what move they make, the game is winnable for the other player). 

If this definition makes you shout "Recursion!", then your instincts are on the right track. In fact it is easy to determine whether a game is winnable using a recursive definition of winnability. We can write some Python code, using the `AbstractGame` template that I've defined in this file, to do just this. Note that we handle the game using general methods, such as `make_move()`, `undo_move()`, and `over()`, that could apply to any game, whether something as simple as Tic-Tac-Toe or as complex as chess or Go.


~~~ python
from games.games import AbstractGame

def winnable(game):

	if game.over():
    	return False
    
	for move in game.valid_moves():
    	game.make_move(move)
        p2win = winnable(game)
        game.undo_move()
        if not p2win:
        	return True
    
   	return False
~~~


