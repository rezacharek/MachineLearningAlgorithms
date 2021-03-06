class Point:
    def __init__(self, points):
        self.points = points
    
    def __sub__(self, other):
        array_of_substracted_points = []

        for i in range(len(self.points)):
            array_of_substracted_points.append(self.points[i] - other[i])
        return Point(array_of_substracted_points)
    
    def distance_from_origin(self):
        sum_of_squares_by_index = 0

        for x in self.points:
            sum_of_squares_by_index += x*x

        return sum_of_squares_by_index
    
    def distance_from_point(self, other):
        temp_point = Point(self.points.copy())
        temp_point = Point(temp_point - other)
        return temp_point.distance_from_origin()

    def array_form(self):
        return self.points

    def __getitem__(self, index):
        return self.points[index]
    
    def __lt__(self, other):
        return self.distance_from_origin() < other.distance_from_origin()