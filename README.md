# Peg solitaire
## How to use
An implementation of Peg solitaire in Python. You can use multiple different boards (but
you have to implement them directly in the code (that's just a matrix, not complicated) or
to create them in god mode).
You can go back and forth in history of the moves by pressing the left and right arrow keys.
You can enter god mode by pressing the 'g' key and modify the board.
You can compute a solution, if there is one, by pressing 's'.
You can test if there is a pagoda function proving that there is no solution by pressing 't'.
You can compute a solution faster by pressing 'r', but sometimes it can miss a solution.
## Explanations of algorithms
The slow algortihm computes solutions by a simple back-tracking algorithm.
A pagoda function is a function f that associates to each position a real number, such that
for any three consecutive positions a, b, c, we have f(a) + f(b) >= f(c). It means that the
sum of values of f at each pieces cannot increase by doing valid moves, and so if the value of
f at the center is greater than this sum, it means that there is no solution from this situation.
My impression was that finding a pagoda function, when there are still lots of pieces, means that
the situation is really bad, and so it was probably already impossible at the previous step, so the
fast algorithm is a back-tracking algorithm, but at each step it tries to find a pagoda function
proving that the position is not winnable, if it can find one, and there are not too few pieces,
it goes two steps back instead of just one. This simple idea works surprisingly well, and there
are probably other ideas worth exploring around this. To find these pagoda functions, I use
a linear optimization module.