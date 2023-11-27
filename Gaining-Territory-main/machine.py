import random
from itertools import combinations
from shapely.geometry import LineString, Point, Polygon
import sys
from scipy.spatial import ConvexHull
import math
########version1
# 그을 수 있는 선의 갯수 짝수 -> 후공 유리
# (num_dots - 1) * 2 - 1 + 내부 점의 갯수
# convex hull
# 초반 / 중반/ 후반 -> 다르게
# 초반 : 룰기반
# 중반 / 후반 -> minMaxTree
# Tree search -> connected 정보 잘 사용
# moveGenerate, evaluate, heauristic 잘 설정
# 어떤 룰을 candidate 할지 heauristic
# only heauristic -> x
class MACHINE():
    """
        [ MACHINE ]
        MinMax Algorithm을 통해 수를 선택하는 객체.
        - 모든 Machine Turn마다 변수들이 업데이트 됨

        ** To Do **
        MinMax Algorithm을 이용하여 최적의 수를 찾는 알고리즘 생성
           - class 내에 함수를 추가할 수 있음
           - 최종 결과는 find_best_selection을 통해 Line 형태로 도출
               * Line: [(x1, y1), (x2, y2)] -> MACHINE class에서는 x값이 작은 점이 항상 왼쪽에 위치할 필요는 없음 (System이 organize 함)
    """
    def __init__(self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]):
        self.id = "MACHINE"
        self.score = [0, 0] # USER, MACHINE
        self.drawn_lines = [] # Drawn Lines
        self.board_size = 7 # 7 x 7 Matrix
        self.num_dots = 0 # 초반부 후반부 -> 남은 그을 수 있는 선분 갯수 -> depth로
        self.whole_points = []
        self.location = []
        self.triangles = [] # [(a, b), (c, d), (e, f)]
        
        self.intersect_on = False
        self.has_run = False
        self.advantage = False
        self.heuristics = {}
        self.temp_score = []
#        available moves
        self.avail_moves = []
        self.intersect_lines = []
        self.alpha = 5
#        self.beta = 3
        self.gamma = 2
#    self.advantage = self.is_advantage()
  
# tree 구현하고 insert delete init... depth 설정  minmax 구현 해당 depth에서 가지치기
# 삼각형을 만들 수 있는지 체크
#point_distance_zero_vertical=0 #수직선과 거리가 0인 점들 초기화
#point_distance_zero_equation=0 #수직선이 아닌 직선과 거리가 0인 점들 초기화
    def find_best_selection(self):
        self.intersect_on = False
        print("현재 그려진 라인 : ", self.drawn_lines)
        available = self.get_available_moves()
        self.avail_moves = available
        self.temp_score = self.score
#       유리한지 불리한지 계산
        if not self.has_run:
            self.advantage = self.is_advantage()
            print("유리한지 여부 : ", self.advantage)
            self.has_run = True
        print("available moves -> ",self.avail_moves)
#       중 / 후반부
        if self.is_ended_begin():
            print("end the beginning")
            best_score = -float('inf')
            best_move = None
            triangle_line = None
            for line in self.avail_moves:
                self.drawn_lines.append(line)
                if self.can_form_triangle(line):
                    triangle_line = line
                score = self.minmax(0, False)
                self.drawn_lines.remove(line)

                if score > best_score:
                    best_score = score
                    best_line = line

            if triangle_line is not None:
                return triangle_line

            if best_line is None:
                return random.choice(available_lines)

            return best_line
            
#       초반부
        else:
            print("is beginning yet")
            triangle_line = None
            for line in self.avail_moves:
                self.drawn_lines.append(line)
                if self.can_form_triangle(line):
                    triangle_line = line
                self.drawn_lines.remove(line)
            if triangle_line is not None:
                return triangle_line
#           휴리스틱 기반 1. 중심에서 멀리 떨어진 것 부터 2. 교점이 없는 선분 부터
            center = self.board_size // 2
            best_move = [(center,center), (center, center)]
            for (point1, point2) in available:
                if (self.is_intersected([point1, point2]) is None and self.calc_center2far_score([point1, point2]) >= self.calc_center2far_score(best_move)):
                    best_move = [point1, point2]
            print("best moves : ", best_move)
            return best_move
# 선공, 후공에 따라 유리한지 확인
    def is_advantage(self):
        if(len(self.avail_moves) % 2 == 0):
            print("짝수")
            return len(self.drawn_lines) % 2 != 0
        else:
            print("홀수")
            return len(self.drawn_lines) % 2 == 0
            
# 교점이 없는 선분을 더 이상 그을 수 없을 때 -> 중반부 시작점 확인
    def is_ended_begin(self):
        intersectioned_points=[]
        for line in self.drawn_lines:
            for point in line:
                if point not in intersectioned_points:
                    intersectioned_points.append(point);
        if len(intersectioned_points)>=len(self.whole_points)-1: return True
        else: return False

# 교점이 있는 선분인지 확인
    def is_intersected(self, move):
        new_line = LineString(self.organize_points(move))
        for target_line in self.drawn_lines:
            if LineString(target_line).intersects(new_line):
                return target_line
        return None
        
    def organize_points(self, point_list):
        point_list.sort(key=lambda x: (x[0], x[1]))
        return point_list

# 좌표의 가운데에서 멀어질 수록 큰 점수
    def calc_center2far_score(self, move):
        center = (self.board_size // 2, self.board_size // 2)
        dist = LineString(move).distance(Point(center))
        score = 1 - ((self.board_size - dist) / self.board_size)
        return score

# 게임 후반부
#       사용할지 의문
    def calc_heuristics(self):
        points = [(x, y) for x in range(self.board_size) for y in range(self.board_size)]
        possible_moves = [list(move) for move in combinations(points, 2) if move not in self.drawn_lines]
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        for move in available:
            if not self.is_intersected(move):
                tri_score = self.calc_score(move) * 1.5
                adj_score = self.calc_adjacency_score(move) / 7
                col_score = self.calc_colinear_score(move)
                score = tri_score + col_score
                self.heuristics[tuple(move)] = score
#       새로운 line을 가지고 삼각형을 만들 수 있는지 검사
    def can_form_triangle(self, new_line):
        for line1, line2 in combinations(self.drawn_lines, 2):
            if new_line == line1 or new_line == line2:
                continue
            target_line = LineString(self.organize_points(line2))
            if LineString(line1).intersects(target_line):
                triangle = set(new_line + line1 + line2)
#                print("triangle : ", triangle)
                if len(triangle) == 3 and self.is_valid_triangle(triangle):
                    return True
        return False
#        삼각형이 맞는지 검사
    def is_valid_triangle(self, points):
        triangle = Polygon(points)
        for p in self.whole_points:
            if Point(p).within(triangle) and p not in points:
                return False
        return True
#        현재 그릴 수 있는 가능한 모든 선
    def get_available_moves(self):
        return [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
#        더이상 만들 수 있느 선이 없는지 검사
    def check_endgame(self):
        return not self.get_available_moves()
#        top-down방식으로 각 노드에서 값 평가
    def evaluate_score(self, is_maximizing_player):
        if self.can_form_triangle(self.drawn_lines[-1]):
            if is_maximizing_player: # MACHINE TURN
                self.temp_score[1] = self.temp_score[1] + 1
            else:                    # USER TURN
                self.temp_score[0] = self.temp_score[0] + 1
            return float('inf')
        a_val = self.alpha * (self.temp_score[1] - self.temp_score[0])
#        triangle_max = 0
#        for line in self.get_available_moves():
#            temp = self.possible_num_triangle(line)
#            if (triangle_max < temp):
#                triangle_max = temp
#        b_val = self.beta * temp
        c_val = self.gamma * (self.temp_score[0])
        return self.score[1] - self.score[0]
#        특정 노드에서 하나의 선분을 그어 만들 수 있는 최대의 삼각형 개수
#    def possible_num_triangle(self, new_line):
#        max_triangle = 0
#        for line1, line2 in combinations(self.drawn_lines, 2):
#            if new_line == line1 or new_line == line2:
#                continue
#            target_line = LineString(self.organize_points(line2))
#            if LineString(line1).intersects(target_line):
#                triangle = set(new_line + line1 + line2)
#                if len(triangle) == 3:
#                    self.triangles.append(triangle)
#                    if max_triangle < len(self.triangles):
#                        max_triangle = len(self.triangles)
#                    print(max_triangle)
#                    self.triangles.remove(triangle)
#        return max_triangle
#        minmax
    def minmax(self, depth, is_maximizing_player):
        if depth == 2 or self.check_endgame():
            return self.evaluate_score(is_maximizing_player)

        if is_maximizing_player:
            best_score = -float('inf')
            for line in self.get_available_moves():
                self.drawn_lines.append(line)
                score = self.minmax(depth + 1, False)
                self.drawn_lines.remove(line)
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for line in self.get_available_moves():
                self.drawn_lines.append(line)
                score = self.minmax(depth + 1, True)
                self.drawn_lines.remove(line)
                best_score = min(best_score, score)
            return best_score
#       가능한 line인지 검사
    def check_availability(self, line):
        line_string = LineString(line)

        # Must be one of the whole points
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)
        
        # Must not skip a dot
        condition2 = True
        for point in self.whole_points:
            if point==line[0] or point==line[1]:
                continue
            else:
                if bool(line_string.intersection(Point(point))):
                    condition2 = False

        # Must not cross another line
        condition3 = True
        for l in self.drawn_lines:
            if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                condition3 = False

        # Must be a new line
        condition4 = (line not in self.drawn_lines)

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False
#       point 정렬
    def organize_points(self, point_list):
        point_list.sort(key=lambda x: (x[0], x[1]))
        return point_list
