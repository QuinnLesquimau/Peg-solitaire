# A peg solitaire, using oriented-object programming
import math
import pygame
import copy
N = 7
width_cross = math.ceil(N/3)
width_left_part = math.ceil((N - width_cross)/2)
width_right_part = N - width_cross - width_left_part
origin = (N//2, N//2)
top_line = [-1 if x < width_left_part or x >= width_left_part + width_cross else 1 for x in range(N)]
middle_line = [1 for x in range(N)]
standard_board = [top_line.copy() if x < width_left_part or x >= width_left_part + width_cross else middle_line.copy() for x in range(N)]
standard_board[origin[1]][origin[0]] = 0
standard_board[2][4] = 1
standard_board[3][4] = 1

"""winable_board = [x.copy() for x in standard_board]
for y,row in enumerate(winable_board):
    for x,cell in enumerate(row):
        if cell == 1:
            if not (x,y) in {(2,4),(3,1),(3,2),(3,3),(4,3),(5,3),(5,4)}:
                winable_board[y][x] = 0
    winable_board[3][3] = 1"""

class PegBoard:

    def __init__(self, board_state=standard_board, goal=origin):
        self.position_pieces = set()
        self.non_playable_squares = set()
        for y, row in enumerate(board_state):
            for x, cell in enumerate(row):
                if cell==1:
                    self.position_pieces.add((x,y))
                if cell==-1:
                    self.non_playable_squares.add((x,y))
        self.goal = goal
        self.history = []
        self.width = len(board_state[0])
        self.height = len(board_state)
        self.place_in_history = -1
        self.is_current = True

    def _get_square_type(self,x,y):
        if (x,y) in self.position_pieces:
            return 1
        if (x,y) in self.non_playable_squares:
            return -1
        return 0

    @profile
    def possible_moves(self):
        possible_moves = set()
        for piece in self.position_pieces:
            x = piece[0]
            y = piece[1]
            if (x-1,y) in self.position_pieces:
                if x > 1 and self._get_square_type(x-2,y) == 0:
                    possible_moves.add(((x,y),(x-2,y)))
                if x < self.width-1 and self._get_square_type(x+1,y) == 0:
                    possible_moves.add(((x-1,y),(x+1,y)))
            if (x,y-1) in self.position_pieces:
                if y > 1 and self._get_square_type(x,y-2) == 0:
                    possible_moves.add(((x,y),(x,y-2)))
                if y < self.height-1 and self._get_square_type(x,y+1) == 0:
                    possible_moves.add(((x,y-1),(x,y+1)))
        return possible_moves

    def adjacent_pieces(self):
        # Return the set of pairs of pieces which are adjacent, where each pair is in the lexicographic order.
        adjacent_pieces = set()
        for piece in self.position_pieces:
            if (x+1,y) in self.position_pieces:
                adjacent_pieces.add((x,y),(x+1,y))
            if (x,y+1) in self.position_pieces:
                adjacent_pieces.add((x,y),(x,y+1))
        return adjacent_pieces

    def adjacent_pieces_after_move(self,prev_adjacent_pieces:set,prev_move:tuple):
        pos1 = prev_move[0]
        pos2 = prev_move[1]
        if pos1[0] == pos2[0]:
            prev_adjacent_pieces.discard(((pos1[0]-1,pos1[1]),pos1))
            prev_adjacent_pieces.discard(((pos2[0]-1,pos2[1]),pos2))
            prev_adjacent_pieces.discard((pos1,(pos1[0]+1,pos1[1])))
            prev_adjacent_pieces.discard((pos2,(pos2[0]+1,pos2[1])))
            # Work in progress

        return True

    def _set(self,x,y,square_type):
        self.position_pieces.discard((x,y))
        self.non_playable_squares.discard((x,y))
        if square_type == 1:
            self.position_pieces.add((x,y))
        if square_type == -1:
            self.non_playable_squares.add((x,y))

    def get_square_type(self,x,y):
        return self._get_square_type(x,y)

    def can_move(self,x,y):
        if self._get_square_type(x,y) == 1:
            if x > 1 and self._get_square_type(x-1,y) == 1 and self._get_square_type(x-2,y) == 0:
                return True
            if x < self.width-2 and self._get_square_type(x+1,y) == 1 and self._get_square_type(x+2,y) == 0:
                return True
            if y > 1 and self._get_square_type(x,y-1) == 1 and self._get_square_type(x,y-2) == 0:
                return True
            if y < self.height-2 and self._get_square_type(x,y+1) == 1 and self._get_square_type(x,y+2) == 0:
                return True
        return False
    
    def is_over(self):
        for piece in self.position_pieces:
            if self.can_move(piece[0],piece[1]):
                return False
        return True

    def is_on_board(self,x,y):
        return x >= 0 and y >= 0 and x < self.width and y < self.height
        

    def has_won(self):
        for piece in self.position_pieces:
            if piece != self.goal:
                return False
        return True


    def _play(self,pos1,pos2,history=False):
        # Doesn't check that it is playable
        x_dif = pos2[0] - pos1[0]
        y_dif = pos2[1] - pos1[1]
        self._set(pos1[0],pos1[1],0)
        self._set(pos2[0],pos2[1],1)
        between_piece_x = pos1[0] + x_dif//2
        between_piece_y = pos1[1] + y_dif//2
        self._set(between_piece_x,between_piece_y,0)
        if history:
            if not self.is_current:
                self.history = self.history[:self.place_in_history + 1]
                self.is_current = True
            self.history.append((pos1,pos2))
            self.place_in_history += 1
        return True

    

    def play(self,pos1,pos2):
        if not self.is_on_board(pos1[0],pos1[1]) or not self.is_on_board(pos2[0],pos2[1]):
            return False
        x_dif = pos2[0] - pos1[0]
        y_dif = pos2[1] - pos1[1]
        if abs(x_dif) + abs(y_dif) != 2 or abs(x_dif) == 1:
            return False
        between_piece_x = pos1[0] + x_dif//2
        between_piece_y = pos1[1] + y_dif//2
        if self._get_square_type(pos1[0],pos1[1]) != 1 or self._get_square_type(pos2[0],pos2[1]) != 0:
            return False
        if self._get_square_type(between_piece_x,between_piece_y) != 1:
            return False
        self._set(pos1[0],pos1[1],0)
        self._set(pos2[0],pos2[1],1)
        self._set(between_piece_x,between_piece_y,0)
        if not self.is_current:
            self.history = self.history[:self.place_in_history + 1]
            self.is_current = True
        self.history.append((pos1,pos2))
        self.place_in_history += 1
        return True

    def go_back(self):
        if self.place_in_history < 0:
            return False
        last_move = self.history[self.place_in_history]
        self.place_in_history -= 1
        self.is_current = False
        pos1 = last_move[0]
        pos2 = last_move[1]
        between_piece_x = pos1[0] + (pos2[0] - pos1[0])//2
        between_piece_y = pos1[1] + (pos2[1] - pos1[1])//2
        self._set(pos1[0],pos1[1],1)
        self._set(pos2[0],pos2[1],0)
        self._set(between_piece_x,between_piece_y,1)

    def go_forward(self):
        if self.is_current:
            return False
        self.place_in_history += 1
        next_move = self.history[self.place_in_history]
        if self.place_in_history + 1 == len(self.history):
            self.is_current = True
        pos1 = next_move[0]
        pos2 = next_move[1]
        self._play(pos1,pos2)

    @profile
    def compute_solution(self,screen):
        computed_states = [set() for x in range(40)]
        count = 0
        history_computation = [self.possible_moves()]
        level = 0
        steps = [0]
        while level >= 0:
            count += 1
            if count > 100000:
                for i in computed_states:
                    print(len(i))
                pygame.quit()
                quit()
            if history_computation[level] != set():
                if steps[level] > 0 or not self.position_pieces in computed_states[level]:
                    next_move = history_computation[level].pop()
                    self._play(next_move[0],next_move[1],True)
                    level += 1
                    steps[-1] += 1
                    steps.append(0)
                    history_computation.append(self.possible_moves())
                    #draw_board(self,screen)
                    #pygame.display.update()
                else:
                    steps = steps[:-1]
                    history_computation = history_computation[:-1]
                    level -= 1
                    self.go_back()                            
            else:
                if steps[level] == 0:
                    if self.has_won():
                        print("Yay!")
                        return True
                else:
                    computed_states[level].add(frozenset(self.position_pieces))
                steps = steps[:-1]
                history_computation = history_computation[:-1]
                level -= 1
                self.go_back()
        self.go_forward()
        return False
        




def draw_board(board,screen,selected_square=None):
    screen.fill((255,255,255))
    (x,y) = screen.get_size()
    size_row = board.width
    size_column = board.height
    size_square = x/size_row
    goal = board.goal
    for i in range(size_column + 1):
        pygame.draw.line(screen,(0,0,0),(0,i*size_square),(x,i*size_square),2)
    for j in range(size_row + 1):
        pygame.draw.line(screen,(0,0,0),(j*size_square,0),(j*size_square,y),2)
    for i in range(size_column):
        for j in range(size_row):
            if board.get_square_type(j,i) == 1:
                pygame.draw.circle(screen,(0,36,215),((j + 0.5) * size_square + 1,(i + 0.5) * size_square + 1),size_square/2.5)
            elif board.get_square_type(j,i) == -1:
                screen.fill((0,0,0),pygame.Rect(j*size_square,i*size_square,size_square,size_square))
    if not selected_square == None:
        pygame.draw.circle(screen,(255,128,0),((selected_square[0] + 0.5) * size_square + 1,(selected_square[1] + 0.5) * size_square + 1),size_square/2.5)
    pygame.draw.rect(screen,(255,0,0),pygame.Rect(goal[0] * size_square, goal[1] * size_square,size_square + 2,size_square + 2),2)



if __name__ == '__main__':
    board = PegBoard()
    size_square = 35
    length_screen = size_square * board.width
    height_screen = size_square * board.height
    selected_square = None
    screen = pygame.display.set_mode((length_screen,height_screen))
    pygame.init()
    draw_board(board,screen,selected_square)
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x = pos[0] // size_square
                y = pos[1] // size_square
                if selected_square == None:
                    if board.get_square_type(x,y) == 1:
                        selected_square = (x,y)
                else:
                    if abs(selected_square[0] - x) + abs(selected_square[1] - y) == 2 and abs(selected_square[0] - x) != 1:
                        board.play(selected_square,(x,y))
                    selected_square = None
                draw_board(board,screen,selected_square)
                pygame.display.update()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                selected_square = None
                board.go_back()
                draw_board(board,screen,selected_square)
                pygame.display.update()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                selected_square = None
                board.go_forward()
                draw_board(board,screen,selected_square)
                pygame.display.update()
            if True:
            #event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                board.compute_solution(screen)
                draw_board(board,screen,selected_square)
                pygame.display.update()